import numpy as np
from scipy import ndimage as nd

from operator import itemgetter
from FUNCS import FNS

class ArtVar:
    def __init__(self, period, bot_size, top_size, bot_para, top_para):
        self.bot_size = bot_size
        self.top_size = top_size
        self.alpha = 0.01
        self.epsilon = 0.01
        self.power = 1.1
        self.period = period

        self.corr_pred = []
        self.incorr_pred = []


        self.top = self.Top(top_size, top_para)
        self.bot = self.Bot(bot_size, bot_para)
        self.mid = self.Mid(bot_size[2], top_size[2])

        self.score = 0

    class Top:
        def __init__(self, top_size, top_para):
            self.beta = top_para[0]
            self.rho = top_para[1]
            self.top_stm = np.zeros(top_size[2])
            self.top_ltm = np.zeros((top_size[2], 2, top_size[0], top_size[1]))
            self.committ = np.zeros(top_size[2], dtype=int)
            self.sort_signal = []

    class Mid:
        def __init__(self, bot, top):  # specify top and bot dimensions from the parameters in __init__
            self.crs_stm = np.zeros(top)
            self.crs_ltm = np.zeros((bot, top))

    class Bot:
        def __init__(self, bot_size, bot_para):
            self.beta = bot_para[0]
            self.rho = bot_para[1]
            self.top_stm = np.zeros(bot_size[2])
            self.top_ltm = np.zeros((bot_size[2], 2, bot_size[0], bot_size[1]))
            self.committ = np.zeros(bot_size[2], dtype=int)
            self.sort_signal = []



class ArtFun:
    def __init__(self, ArtVar):
        self.Art = ArtVar
        self.FNS = FNS()

    # apply log polar, coarse code, and complement code transformation to input
    # preprocessing also includes first center the input before log polar then again center the input after log polar
    # and before coarse code
    def BotPrep(self, input, size):
        FNS = self.FNS
        period = self.Art.period

        # first center and shift
        centroid = nd.center_of_mass(input)
        first_transl = nd.shift(input, (size - centroid[0], size - centroid[1]), mode="nearest")
        logpol_map = FNS.logpol_map(first_transl, size)

        # second center and shift
        centroid = nd.center_of_mass(logpol_map)
        second_transl = nd.shift(logpol_map, (size - centroid[0], size - centroid[1]), mode="nearest")

        coarse_map = FNS.coarse_map(second_transl, period, size)
        comple_map = FNS.comple_map(coarse_map)
        out = comple_map

        return comple_map, coarse_map

    def TopPrep(self, input):
        FNS = self.FNS
        out = 1 - input  # do not use FNS.comple_map for discrete labels
        return out

    def TopChoc(self, input):
        size = self.Art.top_size[0] * self.Art.top_size[1]
        number = np.count_nonzero(self.Art.top.committ)
        choice_signal = np.zeros(number)

        for j in range(number):
            minimum = np.minimum(input, self.Art.top.top_ltm[j])
            choice_signal[j] = np.sum(minimum[0]) + np.sum(minimum[1]) + (1 - self.Art.alpha) * \
                                 (size - np.sum(self.Art.top.top_ltm[j][0]) - np.sum(self.Art.top.top_ltm[j][1]))

        signal = [(j, choice_signal[j]) for j in range(number)]
        signal.sort(key=itemgetter(1), reverse=True)
        self.Art.top.sort_signal = signal

    def BotChoc(self, input):
        size = self.Art.bot_size[0] * self.Art.bot_size[1]
        number = np.count_nonzero(self.Art.bot.committ)
        choice_signal = np.zeros(number)

        for j in range(number):
            minimum = np.minimum(input, self.Art.bot.top_ltm[j])
            choice_signal[j] = np.sum(minimum[0]) + np.sum(minimum[1]) + (1 - self.Art.alpha) * \
                                 (size - np.sum(self.Art.bot.top_ltm[j][0]) - np.sum(self.Art.bot.top_ltm[j][1]))

        signal = [(j, choice_signal[j]) for j in range(number)]
        signal.sort(key=itemgetter(1), reverse=True)
        self.Art.bot.sort_signal = signal


    def TopArt(self, input):
        size = self.Art.top_size[0] * self.Art.top_size[1]
        number = np.count_nonzero(self.Art.top.committ)
        if number == 0:
            self.Art.top.top_stm = np.zeros(self.Art.top_size[2])
            index = number
            self.Art.top.committ[index] = 1
            self.Art.top.top_stm[index] = 1.0
            self.Art.top.top_ltm[index] = input

        else:
            self.TopChoc(input)
            signal = self.Art.top.sort_signal

            counter = 0
            for j in range(number):
                self.Art.top.top_stm = np.zeros(self.Art.top_size[2])
                index = signal[j][0]
                value = signal[j][1]
                minimum = np.minimum(input, self.Art.top.top_ltm[index])
                match = (np.sum(minimum[0]) + np.sum(minimum[1])) - size * self.Art.top.rho

                # the chosen category node satisfies the matching condition
                if value > self.Art.alpha * size and match >= 0:
                    self.Art.top.top_stm[index] = 1.0
                    learned_weight = self.Art.top.top_ltm[index]
                    self.Art.top.top_ltm[index] = self.Art.top.beta * np.array(minimum) \
                                             + (1 - self.Art.top.beta) * np.array(learned_weight)
                    break  # remember to break out of the loop if the match condition is satisfied

                else:
                    counter += 1

            # after search, no committed category nodes satisfy the matching condition, then new committed category
            # nodes are added
            if counter == number and counter < self.Art.top_size[2]:
                self.Art.top.top_stm = np.zeros(self.Art.top_size[2])
                index = number
                self.Art.top.committ[index] = 1
                self.Art.top.top_stm[index] = 1.0
                self.Art.top.top_ltm[index] = input

    def MidArt(self, bot_input, top_input):
        size = self.Art.bot_size[0] * self.Art.bot_size[1]
        number = np.count_nonzero(self.Art.bot.committ)
        if number == 0:
            rho = self.Art.bot.rho
            self.Art.bot.top_stm = np.zeros(self.Art.bot_size[2])
            index = number
            self.Art.bot.committ[index] = 1
            self.Art.bot.top_stm[index] = 1.0
            self.Art.bot.top_ltm[index] = bot_input
            self.TopArt(top_input)
            self.Art.mid.crs_ltm[index] = self.Art.top.top_stm

        else:
            rho = self.Art.bot.rho
            self.BotChoc(bot_input)
            signal = self.Art.bot.sort_signal

            counter = 0
            for j in range(number):
                self.Art.bot.top_stm = np.zeros(self.Art.bot_size[2])
                index = signal[j][0]
                value = signal[j][1]
                minimum = np.minimum(bot_input, self.Art.bot.top_ltm[index])
                match = np.sum(minimum[0]) + np.sum(minimum[1]) - size * rho

                # the chosen category node satisfies the matching condition
                if value > self.Art.alpha * size and match >= 0:
                    self.Art.bot.top_stm[index] = 1.0
                    learned_weight = self.Art.bot.top_ltm[index]
                    crs_predict = self.Art.mid.crs_ltm[index]
                    self.TopArt(top_input)
                    top_predict = self.Art.top.top_stm
                    wta_compare = (crs_predict == top_predict)

                    # the chosen category node makes a correct class label prediction for a wtm code
                    if np.array(wta_compare).all() == True:
                        self.Art.bot.top_ltm[index] = self.Art.bot.beta * np.array(minimum) + \
                                                 (1 - self.Art.bot.beta) * np.array(learned_weight)

                        """
                        # check the class label prediction for a distributed code
                        self.DistArt(bot_input)
                        distr_predict = self.Art.top.top_stm
                        self.TopArt(top_input)
                        top_predict = self.Art.top.top_stm
                        distr_compare = (distr_predict == top_predict)

                        # the distributed code makes a correct and consistent class label prediction
                        if np.array(distr_compare).all() == True:
                            # after learning, vigilance is reset to baseline value and "break" is executed to exit
                            # the for loop to continue to the next input

                            break

                        # the distributed code makes an incorrect class label prediction
                        else:
                            rho = (np.sum(minimum[0]) + np.sum(minimum[1])) / size + self.Art.epsilon
                            counter += 1
                        """
                        break

                    # the chosen category node makes an incorrect class label prediction, and match tracking
                    # increases vigilance to search for more accurate category nodes within the current for loop

                    else:
                        rho = (np.sum(minimum[0]) + np.sum(minimum[1])) / size + self.Art.epsilon
                        counter += 1

                # the chosen category node does not satisfy the matching condition, then search through other
                # category nodes continues in the current for loop; match tracking is not implemented if the
                # matching condition is not satisfied, and it is only implemented if the predicted class label
                # is incorrect
                else:
                    counter += 1

            # after search, no committed category nodes satisfy the matching condition or make a correct label
            # class prediction, then new committed category nodes are added, and their corresponding weight
            # vector and prediction are updated, and full search is complete if counter equals the length of the
            # search list
            if counter == number and counter < self.Art.bot_size[2]:
                self.Art.bot.top_stm = np.zeros(self.Art.bot_size[2])
                index = number
                self.Art.bot.committ[index] = 1
                self.Art.bot.top_stm[index] = 1.0
                self.Art.bot.top_ltm[index] = bot_input
                self.TopArt(top_input)
                self.Art.mid.crs_ltm[index] = self.Art.top.top_stm

            # if counter is equal to self.Art.bot_size[2], then do nothing

    def weight(self, x):
        size = self.Art.bot_size[0] * self.Art.bot_size[1]
        return (1 / (size - x + 0.01)) ** self.Art.power


    def BotUnChoc(self, input):
        size = self.Art.bot_size[0] * self.Art.bot_size[1]
        number = np.count_nonzero(self.Art.bot.committ)
        choice_signal = np.zeros(number)

        for j in range(number):
            minimum = np.minimum(input, self.Art.bot.top_ltm[j])
            choice_signal[j] = np.sum(minimum[0]) + np.sum(minimum[1]) + (1 - self.Art.alpha) * \
                            (size - np.sum(self.Art.bot.top_ltm[j][0]) - np.sum(self.Art.bot.top_ltm[j][1]))

        signal = [(j, choice_signal[j]) for j in range(number)]

        return signal





