import numpy as np
from FUNCS import FNS
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------------------------------------------------
# variable test class for vision; it matters if the test is static or dynamic; make them dynamic, each stage/layer
# is distinguished by calling a separate function, but they all update the same plot variables
class VisTstVar:
    def __init__(self, ax2d, size):
        self.ax2d = ax2d  # suppose plot is 2 x 3, or 2 x 4, where row is size and column is orient/disparity
        self.size = size

        for i in range(2):
            for j in range(4):
                self.ax2d[i, j].xaxis.set_major_locator(plt.NullLocator())
                self.ax2d[i, j].yaxis.set_major_locator(plt.NullLocator())
                #self.ax2d[i, j].set_xlim(0, 2 * size)
                #self.ax2d[i, j].set_ylim(0, 2 * size)


        self.size_S = np.empty(4, dtype=object)
        self.size_L = np.empty(4, dtype=object)

        input = np.zeros((2 * size, 2 * size))

        for j in range(4):
            self.size_S[j] = self.ax2d[0, j].imshow(input, cmap='gray', vmax=255, vmin=0)
            self.size_L[j] = self.ax2d[1, j].imshow(input, cmap='gray', vmax=255, vmin=0)



# method test class for vision
class VisTstFun:
    def __init__(self, VisTstVar):
        self.FNS = FNS()
        self.Vis = VisTstVar
        self.size = self.Vis.size

    def LGN(self, data):
        FNS = self.FNS
        sma, lar = data
        for j in range(4):
            img_S = FNS.img_normz(sma[j])
            self.Vis.size_S[j].set_array(img_S)

            img_L = FNS.img_normz(lar[j])
            self.Vis.size_L[j].set_array(img_L)

    def Simple(self, data):
        FNS = self.FNS
        sma, lar = data
        for j in range(4):
            img_S = FNS.img_normz(sma[j])
            self.Vis.size_S[j].set_array(img_S)

            img_L = FNS.img_normz(lar[j])
            self.Vis.size_L[j].set_array(img_L)

    def Complex(self, data):
        FNS = self.FNS
        sma, lar = data
        for j in range(4):
            img_S = FNS.img_normz(sma[j])
            self.Vis.size_S[j].set_array(img_S)

            img_L = FNS.img_normz(lar[j])
            self.Vis.size_L[j].set_array(img_L)

    def Hypercomplex(self, data):
        FNS = self.FNS
        sma, lar = data
        for j in range(4):
            img_S = FNS.img_normz(sma[j])
            self.Vis.size_S[j].set_array(img_S)

            img_L = FNS.img_normz(lar[j])
            self.Vis.size_L[j].set_array(img_L)


    def Bipole(self, data):
        FNS = self.FNS
        sma, lar = data
        for j in range(4):
            img_S = FNS.img_normz(sma[j])
            self.Vis.size_S[j].set_array(img_S)

            img_L = FNS.img_normz(lar[j])
            self.Vis.size_L[j].set_array(img_L)

# ---------------------------------------------------------------------------------------------------------------------

class ArtTstVar:
    def __init__(self, ax2d, period, size):
        self.ax2d = ax2d  # suppose plot is 2 x 4, where each row is original, logpolar, coarse code, learned prototype
        self.size = size

        new_size = int(2 * size / period)

        for i in range(2):
            for j in range(2):
                # original and logpolar have format 2 * size x 2 * size
                self.ax2d[i, j].xaxis.set_major_locator(plt.NullLocator())
                self.ax2d[i, j].yaxis.set_major_locator(plt.NullLocator())
                self.ax2d[i, j].set_xlim(0, 2 * size)
                self.ax2d[i, j].set_ylim(0, 2 * size)

                # coarse code and learned prototype have format 2 * size / period x 2 * size / period
                self.ax2d[i, j + 2].xaxis.set_major_locator(plt.NullLocator())
                self.ax2d[i, j + 2].yaxis.set_major_locator(plt.NullLocator())
                self.ax2d[i, j + 2].set_xlim(0, new_size)
                self.ax2d[i, j + 2].set_ylim(0, new_size)


        self.cat_A = np.empty(4, dtype=object)
        self.cat_B = np.empty(4, dtype=object)

        lar_inp = np.zeros((2 * size, 2 * size))
        sma_inp = np.zeros((new_size, new_size))

        for j in range(2):
            self.cat_A[j] = self.ax2d[0, j].imshow(lar_inp, cmap='gray', vmax=255, vmin=0)
            self.cat_A[j + 2] = self.ax2d[0, j + 2].imshow(sma_inp, cmap='gray', vmax=255, vmin=0)

            self.cat_B[j] = self.ax2d[1, j].imshow(lar_inp, cmap='gray', vmax=255, vmin=0)
            self.cat_B[j + 2] = self.ax2d[1, j + 2].imshow(sma_inp, cmap='gray', vmax=255, vmin=0)

class ArtTstFun:
    def __init__(self, ArtTstVar):
        self.FNS = FNS()
        self.Art = ArtTstVar
        self.size = self.Art.size

    def Orig_Map(self, input):
        FNS = self.FNS
        car, pla = input

        img = FNS.img_normz(car)
        self.Art.cat_A[0].set_array(img)

        img = FNS.img_normz(pla)
        self.Art.cat_B[0].set_array(img)

    def Logpol_Map(self, input):
        FNS = self.FNS
        car, pla = input

        img = FNS.img_normz(car)
        self.Art.cat_A[1].set_array(img)

        img = FNS.img_normz(pla)
        self.Art.cat_B[1].set_array(img)

    def Coarse_Map(self, input):
        FNS = self.FNS
        car, pla = input

        img = FNS.img_normz(car)
        self.Art.cat_A[2].set_array(img)

        img = FNS.img_normz(pla)
        self.Art.cat_B[2].set_array(img)

    def Proto_Map(self, input):
        FNS = self.FNS
        car, pla = input

        img = FNS.img_normz(car)
        self.Art.cat_A[3].set_array(img)

        img = FNS.img_normz(pla)
        self.Art.cat_B[3].set_array(img)

# ---------------------------------------------------------------------------------------------------------------------

class BasTstVar:
    def __init__(self, ax, period, size):
        self.ax = ax
        self.size = size  # suppose size = (480, 640)

        for i in range(2):
            for j in range(2):
                self.ax[i, j].xaxis.set_major_locator(plt.NullLocator())
                self.ax[i, j].yaxis.set_major_locator(plt.NullLocator())
                self.ax[1, j].set_xlim(0, self.size[1])
                self.ax[1, j].set_ylim(0, self.size[0] + 20)

        self.ax[0, 0].set_title('input image')
        self.ax[0, 1].set_title('output image')
        self.ax[1, 0].set_title('coarse image')
        self.ax[1, 1].set_title('learned template')

        X = np.arange(0, self.size[1], 1)
        Y = np.arange(0, self.size[0], 1)
        X, Y = np.meshgrid(X, Y)
        Z = np.random.rand(self.size[0], self.size[1])
        self.input = self.ax[0, 0].pcolor(X, Y, Z, shading='auto', cmap='gray')
        self.output = self.ax[0, 1].pcolor(X, Y, Z, shading='auto', cmap='gray')

        X = np.arange(0, self.size[1], 1)
        Y = np.arange(0, self.size[0], 1)
        X, Y = np.meshgrid(X, Y)
        Z = np.random.rand(self.size[0], self.size[1])
        self.coarse = self.ax[1, 0].pcolor(X, Y, Z, shading='auto', cmap='gray')
        self.template = self.ax[1, 1].pcolor(X, Y, Z, shading='auto', cmap='gray')

        self.text = self.ax[1, 1].text(0, self.size[0] + 15, "Count = {Count}; Active = {Active}; Score = {Score}; "
                                "Percent = {Percent}; Predict = {Predict}".format(Count=0, Active=(0, 0), Score=0, Percent=0, Predict=0))
        # text need be within the x_lim and y_lim of the axis, else it will not be displayed

class BasTstFun:
    def __init__(self, BasTstVar):
        self.FNS = FNS()
        self.Bas = BasTstVar
        self.size = self.Bas.size

    def RealTime(self, data, text):
        FNS = self.FNS
        input, output, coarse, template = data
        self.Bas.input.set_array(input.ravel())
        self.Bas.output.set_array(output.ravel())
        self.Bas.coarse.set_array(coarse.ravel())
        self.Bas.template.set_array(template.ravel())


        count, active, score, predict = text
        self.Bas.text.set_text("Count = {Count}; Active = {Active}; Score = {Score}; "
                                "Percent = {Percent}; Predict = {Predict}".format(Count=count, Active=active,
                                                             Score=score, Percent=round(score / float(count), 2), Predict=predict))
