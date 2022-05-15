import numpy as np
import random, os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from FUNCS import FNS
from EXTRACT import CorxVar, CorxFun
from CLASSFY import ArtVar, ArtFun
from TRANSFM import SptVar, SptFun
from TESTS import BasTstVar, BasTstFun

# ---------------------------------------------------------------------------------------------------------------------
# combine cortx and artmap for real-time vision processing using c extensions and cuda

if __name__ == '__main__':
    length = 10000
    interval = 1
    size = 100
    period = 10  # it is slow enough not to cause too much asynchronization in the recorded video
    orient = 1

    new_size = int((2 * size) / period)
    bot_num = 100
    top_num = 5
    para = (0.9, 0.9)
    bot_size = (new_size, new_size, bot_num)
    top_size = (1, 2, top_num)
    bot_para = para
    top_para = para

    set01 = []
    set02 = []
    dirPath = os.path.join(os.path.abspath(".") + os.path.sep + '..', 'Resource')
    for i in range(9):
        file = "SAMPLE0{}".format(i)
        name = os.path.join(dirPath, file + '.png')
        if i < 4:
            set01.append(name)
        if i > 4:
            set02.append(name)


    top_zeros = np.zeros((top_num))
    top_zeros[0] = 1
    car_label = top_zeros

    top_zeros = np.zeros((top_num))
    top_zeros[1] = 1
    pla_label = top_zeros

    fig, ax = plt.subplots(2, 2)

    # ----------------------------------------------------------------------------------------------------------------
    # initialize variables
    SptVar = SptVar(size, 10, 1, 10)
    Spt = SptFun(SptVar)

    label = np.random.randint(0, 2)
    pick = FNS().delta_fn(label, 0) * random.choice(set01) + FNS().delta_fn(label, 1) * random.choice(set02)
    #img_load = Image.open(pick)
    #img_gray = img_load.convert('L')  # convert image to grayscale
    img_gray = random.choice(Spt.Preproc(set01[0]))  # first input is car to be compatible w/ the label definitions
    img_resize = img_gray.resize((2 * size, 2 * size))
    img_norm = np.array(img_resize) / 255  # normalize it btw 0 and 1


    CorxVar = CorxVar(img_norm, size, 2 * orient, 1, 2)
    Corx = CorxFun(CorxVar)

    ArtVar = ArtVar(period, bot_size, top_size, bot_para, top_para)
    Art = ArtFun(ArtVar)

    TstVar = BasTstVar(ax, period, size)
    Tst = BasTstFun(TstVar)

    # ----------------------------------------------------------------------------------------------------------------

    def pract(t):
        # update input image every 100 epochs
        if t > 0:
            CorxVar.label = np.random.randint(0, 2)
            label = CorxVar.label
            pick = FNS().delta_fn(label, 0) * random.choice(set01) + FNS().delta_fn(label, 1) * random.choice(set02)
            #img_load = Image.open(pick)
            #img_gray = img_load.convert('L')
            img_gray = random.choice(Spt.Preproc(pick))
            img_resize = img_gray.resize((2 * size, 2 * size))
            img_norm = np.array(img_resize) / 255
            CorxVar.img = img_norm

        Corx.LGN()
        Corx.Simple()
        Corx.Complex()

        img = CorxVar.img
        label = CorxVar.label
        top_input = FNS().delta_fn(label, 0) * np.array([(1, 0)]) + FNS().delta_fn(label, 1) * np.array([(0, 1)])
        bot_input = Art.BotPrep(img, size)[0]
        top_input = Art.TopPrep(top_input)
        #Art.TopArt(top_input)  # remember to organize the labels in addition to categorize the images
        Art.MidArt(bot_input, top_input)

        # input image, output image, coarse image, learned template; be careful coarse map need be based on centered
        # images
        input = CorxVar.img
        output = np.sum(CorxVar.cmpx.comb_map[0])
        coarse = Art.BotPrep(input, size)[1]
        # template is of the active category
        top_ind = 0
        bot_ind = 0

        if len(ArtVar.bot.sort_signal) > 0:
            bot_ind = ArtVar.bot.sort_signal[0][0]
            # prediction is based on the node w/ largest activity as during testing

        if len(ArtVar.top.sort_signal) > 0:
            top_ind = ArtVar.top.sort_signal[0][0]

        # keep in mind ltm are complement code, so access is either the on channel or off channel
        top_temp = ArtVar.top.top_ltm[top_ind][0]
        bot_temp = ArtVar.bot.top_ltm[bot_ind][1]

        data = input, output, coarse, bot_temp

        count = (t + 1)
        active = (np.count_nonzero(ArtVar.top.committ), np.count_nonzero(ArtVar.bot.committ))
        crs_predict = ArtVar.mid.crs_ltm[bot_ind]
        top_predict = ArtVar.top.top_stm

        # score should not depend on size of down-sampling
        ArtVar.score += FNS().delta_fn(crs_predict, top_predict)
        score = ArtVar.score

        # be careful, it depends if the first category is car or plane
        label = "set01" * FNS().delta_fn(crs_predict, car_label) + "set02" * FNS().delta_fn(crs_predict, pla_label)
        #label = "set01" * FNS().delta_fn(top_predict, car_label) + "set02" * FNS().delta_fn(top_predict, pla_label)

        text = count, active, score, label
        Tst.RealTime(data, text)


        return TstVar.input, TstVar.output, TstVar.coarse, TstVar.template, TstVar.text,



    ani = animation.FuncAnimation(fig, pract, frames=length, interval=100, blit=True)

    for i in range(2):
        for j in range(2):
            ax[i, j].invert_yaxis()

    plt.show()

"""
To-do:

0. write api for each file, e.g., input, output, and connect the files to be able to run the algorithm and get the
results 

1. generate sample pattern w/ no repeat, assign id, and save 5-10 samples; remove rectangle outline, insert hexagon
outline

2. apply translation, rotation, and resizing to stored samples for testing classification using ARTMAP

3. run ARTMAP on test samples and check accuracy

"""

