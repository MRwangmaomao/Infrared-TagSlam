import numpy as np
import random, os
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from FUNCS import FNS
from CLASSFY import ArtVar, ArtFun
from EXTRACT import ExtrVar, ExtrFun
from TESTS import BasTstVar, BasTstFun

# ---------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    length = 10000
    interval = 1
    size = np.array((480, 640))

    bot_num = 100
    top_num = 20
    para = (0.99, 0.99)
    bot_size = (size[0], size[1], bot_num)
    top_size = (1, 4, top_num)  # binary of 9 is 1001
    bot_para = para
    top_para = para

    set = []
    curr_path = os.getcwd()
    pare_path = os.path.dirname(curr_path)
    dest_path = os.path.join(pare_path, 'Resource')

    for i in range(9):
        file = "SAMPLE0{}".format(i)
        name = os.path.join(dest_path, file + '.png')
        set.append((name, np.binary_repr(i)))  # the binary code is the unique id of the image, and leave it as string
        # to be split into individual digit

    fig, ax = plt.subplots(2, 2)

    # ----------------------------------------------------------------------------------------------------------------
    # initialize variables

    name, id = random.choice(set)

    ExtrVar = ExtrVar(name)
    Ext = ExtrFun(ExtrVar)

    ArtVar = ArtVar(1, bot_size, top_size, bot_para, top_para)
    Art = ArtFun(ArtVar)

    TstVar = BasTstVar(ax, 1, size)
    Tst = BasTstFun(TstVar)

    # ----------------------------------------------------------------------------------------------------------------

    def pract(t):

        name, id = random.choice(set)

        ExtrVar.name = name
        Ext.CompCents()
        Ext.CompTransf()
        output = Ext.RestoreImg() / 255

        top_input = FNS().fill_arr(id)  # convert id into an array
        bot_input = np.array((output, 1 - output))  # input is complement code
        top_input = Art.TopPrep(top_input)  # both top and bot input are normalized

        Art.TopArt(top_input)  # remember to organize the labels in addition to categorize the images
        Art.MidArt(bot_input, top_input)

        # input image, output image, coarse image, learned template; be careful coarse map need be based on centered
        # images
        input = ExtrVar.input
        output = Ext.RestoreImg() / 255
        coarse = output
        # template is of the active category
        top_ind = 0
        bot_ind = 0

        if len(ArtVar.bot.sort_signal) > 0:
            bot_ind = ArtVar.bot.sort_signal[0][0]
            # prediction is based on the node w/ largest activity as during testing

        if len(ArtVar.top.sort_signal) > 0:
            top_ind = ArtVar.top.sort_signal[0][0]

        # keep in mind ltm are complement code, so access is either the on channel or off channel
        top_templ = ArtVar.top.top_ltm[top_ind][0]
        bot_templ = ArtVar.bot.top_ltm[bot_ind][1]

        data = input, output, coarse, bot_templ

        count = (t + 1)
        active = (np.count_nonzero(ArtVar.top.committ), np.count_nonzero(ArtVar.bot.committ))
        crs_predict = ArtVar.mid.crs_ltm[bot_ind]
        top_predict = ArtVar.top.top_stm

        # score should not depend on size of down-sampling
        ArtVar.score += FNS().delta_fn(crs_predict, top_predict)
        score = ArtVar.score

        # use decimal for display and binary for computation
        label = int('{}'.format(id), 2)


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

