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
        self.hycx = self.HyperComplex(size, orient, scale)
        self.bipo = self.Bipole(size)
        self.targ = self.TargPosn(size)
        self.summ = self.ActvSum(size)

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

    class TargPosn:
        def __init__(self, size):
            self.targ_map = np.zeros((2 * size, 2 * size))
            self.move_cmd = np.zeros(2)
            self.CoM = np.zeros(2)

    class ActvSum:
        def __init__(self, size):
            self.sum_map = np.zeros((2 * size, 2 * size))


    class HyperComplex:
        def __init__(self, size, orient, scale):
            self.pre_map = np.empty((scale, orient), dtype=object)
            self.pos_map = np.empty(scale, dtype=object)
            self.orient_map = np.empty(scale, dtype=object)

            for s in range(scale):
                for k in range(orient):
                    self.pre_map[s, k] = np.zeros((2 * size, 2 * size))

                self.pos_map[s] = np.zeros((2 * size, 2 * size))
                self.orient_map[s] = np.zeros((2 * size, 2 * size), dtype=int)

    class Bipole:
        def __init__(self, size):
            self.pre_map = np.zeros((2 * size, 2 * size))
            self.pos_map = np.zeros((2 * size, 2 * size))
            self.out_map = np.zeros((2 * size, 2 * size))



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


    def TargPosn(self):
        FNS = self.FNS
        size = self.size

        input = self.Corx.cmpx.comb_map[0][2]  # small scale, vertical boundary

        #cent_y, cent_x = nd.center_of_mass(input)

        left_map = FNS.left_map(input, size, 0.2)
        right_map = FNS.right_map(input, size, 0.2)



        left_max = np.max(left_map)
        right_max = np.max(right_map)

        left = FNS.thresh_fn((left_map / left_max) ** 1, 0.9)
        right = FNS.thresh_fn((right_map / right_max) ** 1, 0.9)

        # merge left and right into output w/ format (2 * size, 2 * size)

        output = np.concatenate((left, right), axis=1)
        self.Corx.targ.targ_map = output


        cent_y, cent_x = nd.center_of_mass(output)
        self.Corx.targ.CoM = cent_y, cent_x
        cent_y, cent_x = int(cent_y), int(cent_x)
        nonzero = np.array(np.nonzero(output[cent_y]))
        d_gain = 1  # calibrate distance according to the rack dimension
        a_gain = 1  # calibrate angle according to the rack dimension
        dist = d_gain * (np.max(nonzero) - np.min(nonzero))
        angle = a_gain * (cent_x - size)  # suppose -ve is clockwise rotation and +ve is counterclockwise rotation
        self.Corx.targ.move_cmd = np.array((dist, angle))

    def ActvSum(self):
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

        self.Corx.summ.sum_map = FNS.thresh_fn((comb_rat / comb_max) ** 2, 0.45)

    def Hypercomplex(self):
        FNS = self.FNS
        size = self.size
        num = self.num
        scale = self.scae
        gain = self.gain
        epsilon = 0.1
        mu = 2

        for s in range(scale):
            for k in range(num):
                hycx_map = sum(FNS.hycx_map(self.Corx.cmpx.comb_map[s, m], (s + 1) * gain, k, num, size) for m in range(num))
                self.Corx.hycx.pre_map[s, k] = self.Corx.cmpx.comb_map[s, k] / (epsilon + mu * hycx_map)

        temp = np.zeros((scale, num, 2 * size, 2 * size))

        for s in range(scale):
            for k in range(num):
                temp[s][k] = self.Corx.hycx.pre_map[s, k]
        temp = np.transpose(temp, (0, 2, 3, 1))

        """
        for s in range(scale):
            for j in range(2 * size):
                for i in range(2 * size):
                    x = temp[s][j][i]
                    orient = np.unravel_index(np.argmax(x), x.shape)[0]
                    self.Corx.hycx.orient_map[s][j][i] = orient
                    self.Corx.hycx.pos_map[s][j][i] = temp[s][j][i][orient]
        """

        for s in range(scale):
            self.Corx.hycx.pos_map[s] = FNS.max_orient(temp, scale, num, size)[0][s]
            self.Corx.hycx.orient_map[s] = FNS.max_orient(temp, scale, num, size)[1][s]

    def Bipole(self):
        FNS = self.FNS
        size = self.size
        scale = self.scae
        num = self.num
        gain = self.gain

        hycx_map = self.Corx.hycx.pos_map
        self.Corx.bipo.pre_map = hycx_map[0] * FNS.bipo_map(hycx_map[1], 0, num, size)


        hycx_map = np.zeros((scale, num, 2 * size, 2 * size))
        orient_map = np.zeros((scale, 2 * size, 2 * size), dtype=np.int32)
        for s in range(scale):
            for k in range(num):
                hycx_map[s][k] = self.Corx.hycx.pre_map[s, k]
            orient_map[s] = self.Corx.hycx.orient_map[s]

        self.Corx.bipo.pos_map = self.Corx.hycx.pos_map[1] * FNS.conv_orient(hycx_map, orient_map, 1, gain, num, size)

        self.Corx.bipo.out_map = self.Corx.bipo.pre_map + self.Corx.bipo.pos_map


"""
To-do:

1. convert CORTX into the format of the programming model with C-extension class over retinal position, and array iteration
over scales, orientations, colors

    a. visualize the output at each stage for static circuit using a test class
    
    b. animate the output at final stage for dynamic circuit using a test class
    
    c. think about other type of visualization for the laminar circuit and the artmap circuit
    
    d. in addition to retinal positions (j, i), there are orientations k, scales s, disparities d, speeds m, directions n,
    colors o
    
    e. there are several possible arrangements
        
        i. C-extension over both retinal positions and other features, and C-extension over other features is before
        C-extension over retinal positions, or C-extension over retinal positions is before C-extension over other 
        features, or C-extension over retinal position is before array iteration over other features given the total
        number of features is small;
        
        ii. C-extension over the other features and gpu over retinal positions;
        
        iii. hypercolumns are constructed by using the same arrangement for each layer to link up the corresponding
        chunks
        
        iv. it seems it is not natural to use gpu over retinal position b/c each call for each (j, i) is invoked with
        cython and calling it in the gpu will require calling cython in the gpu and b/c it is called for each epoch,
        the overhead could be huge, and the optimal solution is better use a large number of cores in the cpu or use
        multiprocess in the cpu. Also, to be able to use gpu, the kernel requires an input object and an output object,
        where the input object need be known from a previous computation. This shows the input object need be computed
        in C and eventually everything will have to encoded in C.

    f. organize the code in the same format as in Synth2021 with core, test, learn, pract, where learn is run in TRAXX.py
    files and pract in PANXX.py files
    
    g. it appears the C extension class has to be directly written in C then use the gpu, especially the computations
    need be in real time or feedback is necessary or it is necessary for motion vision

2. convert CORTX into the format of the programming model with gpu over retinal positions, C-extension class over
scales, orientations, disparities, and array iteration over colors

3. do same for FBF, sVISION0

4. convert laminar architecture into the format of the programming model and do so for sVISION1, mVISION1 etc

5. convert ARTMAP into the format of the programming model





"""



"""
# Lessons:
#
# 1. This is a heuristic way to implement the biological model in the correct perspective where there is a collection
# of fixed oriented filters and they are distributed at each pixel position. In particular, every filter is
# different and each is centered at a unique pixel position, and they do not move across an image. Rather, it is the
# input image that varies in time, and if the filters move, it corresponds to an eye movement and change in the
# fixation point. B/c this perspective requires parallel computation and all filters process the input at the same time,
# and b/c complete parallel computation is not easy to implement at thin point, the input image need be specially
# processed is order to make the computations fast enough.
#
# 2. More specifically, the input image is converted into a black and white image where the figure is white and the
# background is black; it is restricted to a 100x50 canvas, and the canvas is overlaid on a bigger 200x100 canvas in
# order to prevent boundary effect (the bigger canvas may be smaller, but it should be bigger than the smaller canvas);
# the figure is extracted by identifying pixel positions where it is white or the pixel value is white; the image is
# processed for the figure, background pixel positions interior to the figure, and background pixel positions exterior
# to the figure that fall into a 5x5 neighborhood of the minimum and maximum width and minimum and maximum height of
# the figure, e.g., if min_height=10, max_height=50, min_width=20, max_width=90, then the 5x5 neighborhood consists of
# the slightly bigger rectangle with dimensions min_height=10-5, max_height=50+5, min_width=20-5, max_width=90+5 that
# encloses the figure, and this is a way to see the effect of the figure on neighboring pixels, where the neighborhood
# is chosen small enough to make the computations efficient, as it is not enough to restrict processing to the figure;
# note that it is not necessary to process the rest of the canvas that is outside the 5x5 neighborhood b/c the bigger
# canvas is in black, and the array that stores the processed image is initialized to be zero.
"""