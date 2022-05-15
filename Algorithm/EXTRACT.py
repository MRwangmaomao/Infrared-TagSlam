import numpy as np
from FUNCS import FNS
from scipy import ndimage as nd


# variable class
class CorxVar:
    def __init__(self, img_input, size, orient, scale, gain):
        self.img = img_input
        self.label = 0
        self.num = orient
        self.size = size
        self.scae = scale
        self.gain = gain


        self.lgnn = self.LGN(size)
        self.simp = self.Simple(size, orient, scale)
        self.cmpx = self.Complex(size, orient, scale)
        self.coord = self.CoordSys(size)

    class LGN:
        def __init__(self, size):
            self.on_map = np.zeros((2 * size, 2 * size))
            self.off_map = np.zeros((2 * size, 2 * size))


    class Simple:
        def __init__(self, size, orient, scale):
            self.left_on = np.empty((scale, orient), dtype=object)
            self.left_off = np.empty((scale, orient), dtype=object)
            self.right_on = np.empty((scale, orient), dtype=object)
            self.right_off = np.empty((scale, orient), dtype=object)

            for s in range(scale):
                for k in range(orient):
                    self.left_on[s, k] = np.zeros((2 * size, 2 * size))
                    self.left_off[s, k] = np.zeros((2 * size, 2 * size))
                    self.right_on[s, k] = np.zeros((2 * size, 2 * size))
                    self.right_off[s, k] = np.zeros((2 * size, 2 * size))



    class Complex:
        def __init__(self, size, orient, scale):
            self.comb_map = np.empty((scale, orient), dtype=object)

            for s in range(scale):
                for k in range(orient):
                    self.comb_map[s, k] = np.zeros((2 * size, 2 * size))

    class CoordSys:
        def __init__(self, size):
            self.sum_map = np.zeros((2 * size, 2 * size))
            self.invert_map = np.zeros((2 * size, 2 * size))

            self.rot_ang = 0

            # it is a pair of points in x-direction and y-direction
            self.max_x = np.zeros((2, 2))
            self.max_y = np.zeros((2, 2))

            # recover the 4 points w/ holes



class CorxFun:
    def __init__(self, CorxVar):
        self.Corx = CorxVar
        self.size = CorxVar.size
        self.num = CorxVar.num
        self.scae = CorxVar.scae
        self.gain = CorxVar.gain
        self.FNS = FNS()

    def LGN(self):
        FNS = self.FNS
        size = self.size
        num = self.num
        image = self.Corx.img
        decay = 1
        spont = 2
        on_size = 1
        off_size = 2
        on_coeff = 5
        off_coeff = 3

        on_conv = FNS.lgnn_map(image, on_size, num, size)
        off_conv = FNS.lgnn_map(image, off_size, num, size)

        on_top_sum = FNS.thresh_fn(on_coeff * on_conv - off_coeff * off_conv, 0)
        off_top_sum = FNS.thresh_fn(decay * spont + (off_coeff * off_conv - on_coeff * on_conv), 0)

        bot_sum = decay + on_conv + off_conv

        self.Corx.lgnn.on_map = on_top_sum / bot_sum
        self.Corx.lgnn.off_map = off_top_sum / bot_sum

    def Simple(self):
        FNS = self.FNS
        size = self.size
        num = self.num
        scale = self.scae
        gain = self.gain
        on_map = self.Corx.lgnn.on_map
        off_map = self.Corx.lgnn.off_map
        contrast = 1.1

        for s in range(scale):
            for k in range(num):
                left_on = FNS.simp_map(on_map, -1, (s + 1) * gain, k, num, size)
                left_off = FNS.simp_map(off_map, -1, (s + 1) * gain, k, num, size)

                right_on = FNS.simp_map(on_map, +1, (s + 1) * gain, k, num, size)
                right_off = FNS.simp_map(off_map, +1, (s + 1) * gain, k, num, size)

                self.Corx.simp.left_on[s, k] = FNS.thresh_fn(left_on - contrast * right_on, 0)
                self.Corx.simp.left_off[s, k] = FNS.thresh_fn(left_off - contrast * right_off, 0)

                self.Corx.simp.right_on[s, k] = FNS.thresh_fn(right_on - contrast * left_on, 0)
                self.Corx.simp.right_off[s, k] = FNS.thresh_fn(right_off - contrast * left_off, 0)


    def Complex(self):
        FNS = self.FNS
        num = self.num
        scale = self.scae

        for s in range(scale):
            for k in range(num):
                self.Corx.cmpx.comb_map[s, k] = self.Corx.simp.left_on[s, k] + self.Corx.simp.left_off[s, k] + \
                                            self.Corx.simp.right_on[s, k] + self.Corx.simp.right_off[s, k]


    # extract the coordinate system and compute the angle of rotation
    def CoordSys(self):
        FNS = self.FNS
        size = self.size
        num = self.num
        input = sum(self.Corx.cmpx.comb_map[0][o] for o in range(num))
        decay = 1
        on_size = (size / 30) * 2
        off_size = 2 * on_size
        on_coeff = 5
        off_coeff = 3

        on_conv = FNS.circ_map(input, on_size, size)
        off_conv = FNS.circ_map(input, off_size, size)

        top_sum = FNS.thresh_fn(on_coeff * on_conv - off_coeff * off_conv, 0.0)

        bot_sum = decay + on_conv + off_conv

        comb_rat = top_sum / bot_sum
        comb_max = np.max(comb_rat)

        sum_map = FNS.thresh_fn((comb_rat / comb_max) ** 2, 0.55)

        self.Corx.coord.sum_map = sum_map


        # extract the coordinate system

        # find points w/ largest distance in x-direction and y-direction, e.g., this requires comparing each point
        # in the image w/ every point in another copy of the image, each update stores a pair of points
        for j in range(2 * size):
            for i in range(2 * size):
                if sum_map[j][i] > 0.01:
                    for q in range(2 * size):
                        for p in range(2 * size):
                            if sum_map[q][p] > 0.01:
                                pass



        # compute the angle of rotation


    def CoordSys01(self):
        input = self.Corx.img
        FNS = self.FNS
        size = self.size
        on_size = (size / 30) * 2
        self.Corx.coord.invert_map = FNS.circ_map(input, on_size, size)