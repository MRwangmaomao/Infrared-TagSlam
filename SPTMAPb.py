import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage as nd
from FUNCS import FNS
from TESTS import VisTstVar, VisTstFun

class SptVar:
    def __init__(self, size, num, gain, scale):
        self.num = num
        self.size = size
        self.scae = scale
        self.gain = gain

        self.par = self.ParMap(scale, num, size)
        self.ser = self.SerMap(scale, num, size)

    class ParMap:
        def __init__(self, scale, num, size):
            self.all_map = np.empty((scale, num), dtype=object)

            for s in range(scale):
                for k in range(num):
                    self.all_map[s][k] = np.zeros((2 * size, 2 * size))

            self.fin_map = np.zeros((2 * size, 2 * size))

    class SerMap:
        def __init__(self, scale, num, size):
            self.posn_map = np.empty((scale, num), dtype=object)
            self.orie_map = np.empty((scale, 2), dtype=object)
            self.scae_map = np.empty((scale), dtype=object)

            for s in range(scale):
                for k in range(num):
                    self.posn_map[s][k] = np.zeros((2 * size, 2 * size))

                self.orie_map[s][0] = np.zeros((2 * size, 2 * size))
                self.orie_map[s][1] = np.zeros((2 * size, 2 * size))

                self.scae_map[s] = np.zeros((2 * size, 2 * size))

            self.fin_map = np.zeros((2 * size, 2 * size))




class SptFun:
    def __init__(self, SptVar):
        self.Spt = SptVar
        self.size = SptVar.size
        self.num = SptVar.num
        self.scae = SptVar.scae
        self.gain = SptVar.gain
        self.FNS = FNS()
        self.proto_size = (50, 25)  # pixel format is (width, height)
        self.img_size = (200, 100)  # choose canvas size so that even each prototype is rotated 90` the full image will
        # remain in view and no part of the image goes out of view, thus the height of the canvas must be at least the
        # width of the prototype. Moreover, it also needs to account for translation within a center region
        # (50, 50, 100, 100) and scaling in a range btw 0.2 and 1.2. Generate a sequence of spatially transformed
        # images in either position, orientation, or scaling, and test each type of spatial information separately

        self.img_center = (int(self.img_size[0] / 2), int(self.img_size[1] / 2))

    def ParMap(self, input):
        size = self.size
        scale = self.scae
        num = self.num
        gain = self.gain
        FNS = self.FNS

        for s in range(scale):
            for k in range(num):
                self.Spt.par.all_map[s][k] = FNS.cliff_tri_map(input, s + 2, k * 6, gain, size)

        par_map = self.Spt.par.all_map
        max_scale, max_orient, max_height, max_width = np.unravel_index(np.argmax(par_map), par_map.shape)

        return max_scale, max_orient, max_height, max_width



    def ParTran(self, input):
        size = self.size
        gain = self.gain
        FNS = self.FNS
        img_orig = input
        max_scale, max_orient, max_height, max_width = self.ParMap(img_orig)

        cent = FNS.centroid(img_orig, size)
        img_transl = FNS.transl_map(img_orig, cent, size)

        img_orient = FNS.orient_map(img_transl, max_orient, gain, size)

        scale_factor = 12 / (max_scale + 2)
        new_height = int(size * scale_factor)
        new_width = int(size * scale_factor)
        new_size = np.array((new_height, new_width))
        img_scale = FNS.scale_pillw(img_orient, new_size, size)
        #img_scale = FNS.scale_map(img_orient, max_scale, gain, size)

        self.Spt.par.fin_map = img_scale



    def SerTran(self, input):
        size = self.size
        scale = self.scae
        num = self.num
        gain = self.gain
        FNS = self.FNS

        img_orig = input
        cent = FNS.centroid(img_orig, size)
        img_transl = FNS.transl_map(img_orig, cent, size)

        for s in range(scale):
            for k in range(num):
                self.Spt.ser.posn_map[s][k] = FNS.cliff_tri_map(img_transl, (s + 2) * 4, k * 6, gain, size)


        for s in range(scale):
            for k in range(num):
                self.Spt.ser.posn_map[s][k] = self.Spt.ser.posn_map[s][k] - self.Spt.ser.posn_map[s][(k + 15) % 30]

        ser_map1 = self.Spt.ser.posn_map
        max_scale1, max_orient1 = np.unravel_index(np.argmax(ser_map1), ser_map1.shape)


        img_orient = FNS.orient_map(ser_map1, -max_orient1 * 6, gain, size)


        for s in range(scale):
            self.Spt.ser.orie_map[s][0] = FNS.cliff_dou_map(img_orient, s + 2, 0, gain, size)
            self.Spt.ser.orie_map[s][1] = FNS.cliff_dou_map(img_orient, s + 2, 5, gain, size)
            self.Spt.ser.scae_map[s] = self.Spt.ser.orie_map[s][0] - self.Spt.ser.orie_map[s][1]

        ser_map2 = self.Spt.ser.scae_map
        max_scale2 = np.unravel_index(np.argmax(ser_map2), ser_map2.shape)


        scale_factor = 12 / (max_scale2[0] + 2)
        new_height = int(size * scale_factor)
        new_width = int(size * scale_factor)
        new_size = np.array((new_height, new_width))
        img_scale = FNS.scale_pillw(img_orient, new_size, size)
        # img_scale = FNS.scale_map(img_orient, max_scale, gain, size)

        self.Spt.ser.fin_map = img_scale

    def Preproc(self, name):

        # switch figure to white and background to black and keep the image in grayscale
        fn = lambda x: 255 - x
        proto = Image.open(name).convert('L').point(fn)

        # Resize each prototype to a standard size 200x100 pixels.
        proto = proto.resize(self.proto_size)

        # Generate a sequence of spatially transformed images different in position, orientation, and scales

        # For each prototype, generate 3 random scales; for each random scale, generate 3 random orientations;
        # and for each random orientation, generate 3 random positions.

        # Random translations are restricted within a 30x20 center area, so the extents are
        # (img_center[0]-15, img_center[1]-10, img_center[0]+15, img_center[1]+10); random orientations vary from
        # 0` to 180`; and the random scales vary from 0.2 to 1.2.
        #
        # Note that for a prototype of size 50x25 with centroid (25, 13), it may be rotated by 90` so that its width
        # becomes 25 and height becomes 50. When its center is shifted within a 30x20 center area of canvas size
        # 200x100, i.e., the center area has coordinates (100-15, 50-10, 100+15, 50+10)=(85, 40, 115, 60), the image
        # height may extend up to 40-25=15 pixels or down to 60+25=85 pixels, and the image width may extend up to
        # 85-13=62 pixesl or down to 115+13=128 pixels. When it is scaled up by 1.2, the image height may further
        # extend up to 15-10=5 pixels or down to 85+10=95 pixels as 50*1.2=60 with a difference of 10 pixels at most,
        # and similarly, the image width may further extend up to 62-5=57 pixels or down to 95+5=100 pixels as
        # 25*1.2=30 with a difference of 5 pixels at most. Thus a canvas size 200x100 can contain all the spatially
        # transformed images that are different in orientation, position, and scale.

        rand_scale = [1.0 + np.random.random() for i in range(3)]
        rand_orient = [(180 * np.random.random()) for i in range(3)]  # orient in degrees
        rand_posit = [(-30 + 60 * np.random.random(), -20 + 40 * np.random.random()) for i in range(3)]

        scale1 = []
        for i in range(3):
            out = proto.resize((int(rand_scale[i] * self.proto_size[0]), int(rand_scale[i] * self.proto_size[1])))
            scale1.append(out)

        scale2 = []
        for i in range(3):
            bg = Image.new('L', self.img_size, 0)

            arr = np.array(scale1[i])

            cent = nd.center_of_mass(arr)  # keep in mind if image format is (width, height), then array
            # format is (height, width)

            box = (int(self.img_center[0] - cent[1]), int(self.img_center[1] - cent[0]))

            bg.paste(scale1[i], box)

            scale2.append(bg)

        # It is important to note that setting "resample=0" for nearest interpolation is not enough for making visible
        # part of the image that is either translated or rotated out of view; it must be set with "expand=True".
        orient = []
        for i in range(3):
            for j in range(3):
                out = scale2[i].rotate(angle=rand_orient[j])
                orient.append(out)

        posit = []
        for i in range(9):
            for j in range(3):
                out = orient[i].rotate(angle=0, translate=(rand_posit[j][0], rand_posit[j][1]))
                posit.append(out)

        return posit  # there is a total of 27 spatially transformed images for each type

    def Preproc01(self, name):
        # for each input generate 27 spatially transformed versions that vary in position, orientation and scale
        size = self.size
        img_size = np.array((2 * size, 2 * size))
        gain = self.gain
        FNS = self.FNS

        img_orig = Image.open(name)
        fn = lambda x: 255 - x
        img_gray = img_orig.convert('L').point(fn)
        #img_gray = img_orig.convert('L')
        img_resize = img_gray.resize((2 * size, 2 * size))
        img_norm = np.array(img_resize) / 255

        proto = [img_resize] * 3
        rand_scale = [0.5 + np.random.random() for i in range(3)]
        rand_orient = [(180 * np.random.random()) for i in range(3)]  # orient in degrees
        rand_posit = [(-30 + 60 * np.random.random(), -20 + 40 * np.random.random()) for i in range(3)]

        """
        scale1 = []
        for i in range(3):
            out = proto[i].resize((int(rand_scale[i] * size), int(rand_scale[i] * size)))
            scale1.append(out)

        scale2 = []
        for i in range(3):
            img_bg = Image.new('L', (2 * size, 2 * size), 0)
            arr = np.array(scale1[i])
            cent = FNS.centroid(arr, size)
            box = (int(size - cent[1]), int(size - cent[0]))
            img_bg.paste(scale1[i], box)
            scale2.append(img_bg)
        """


        orient = []
        for i in range(3):
            out = FNS.orient_map(img_norm, rand_orient[i], gain, size)
            orient.append(out)

        """
        posit = []
        for i in range(3):
            for j in range(3):
                out = FNS.transl_map(orient[i], *rand_posit[j], size)
                posit.append(out)
        """
        #return img_norm
        return orient


if __name__ == '__main__':
    fig, ax = plt.subplots(2, 4)
    size = 50
    num = 10
    gain = 1
    scale = 10

    SptVar = SptVar(size, num, gain, scale)
    Spt = SptFun(SptVar)

    TstVar = VisTstVar(ax, size)
    Tst = VisTstFun(TstVar)

    car_name = ("MICRO.png", "VAN.png", "BIG_TRUCK.png", "ROADSTER.png")
    pla_name = ("11plane.jpg", "14plane.jpg", "24plane.jpg", "41plane.jpg")


    car_seq = []
    pla_seq = []
    for i in range(4):
        car_seq.append(random.choice(Spt.Preproc(car_name[i])))
        pla_seq.append(random.choice(Spt.Preproc(pla_name[i])))

    data_S = [car_seq[i] for i in range(4)]
    data_L = [pla_seq[i] for i in range(4)]

    data = data_S, data_L
    Tst.Simple(data)

    # flip y-axis upside down
    for i in range(2):
        for j in range(4):
            ax[i, j].invert_yaxis()

    plt.show()


"""
To-do:

1. test the tranlaation, rotation and scaling transformations; generate a sequence of image frames of the same image
that translates the centroid leftward, rightward, upward, downward, or a mix of the movement directions; 


"""