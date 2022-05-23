import os, random
import numpy as np
import cv2 as cv
from PIL import Image
from scipy import ndimage as nd
import matplotlib.pyplot as plt
from FUNCS import FNS
from TESTS import VisTstVar, VisTstFun

class SptVar:
    def __init__(self, size, num, gain, scale):
        self.num = num
        self.size = size
        self.scae = scale
        self.gain = gain


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

        rand_scale = [1.0 + 0 * np.random.random() for i in range(3)]
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


    def Transfm(self, name):

        input = cv.imread(name)
        img_size = input.shape[:2]
        img_cent = np.array((260, 320))  # corresponding image center for the given image size

        img_gray = cv.cvtColor(input, cv.COLOR_BGR2GRAY)
        img_inv = cv.bitwise_not(img_gray)

        # select random rotation
        rand_ang = -40 + (40 - -40) * np.random.random()
        mat_rotate = cv.getRotationMatrix2D(center=(img_cent[1], img_cent[0]), angle=rand_ang, scale=1)
        img_rotate = cv.warpAffine(src=img_inv, M=mat_rotate, dsize=(img_size[1], img_size[0]))

        # select random translation
        rand_x = -30 + (30 - -30) * np.random.random()
        rand_y = -20 + (20 - -20) * np.random.random()
        mat_transl = np.array([[1, 0, rand_x], [0, 1, rand_y]], dtype=np.float32)
        img_transl = cv.warpAffine(src=img_rotate, M=mat_transl, dsize=(img_size[1], img_size[0]))

        return img_transl


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

    set01 = []
    set02 = []
    curr_path = os.getcwd()
    pare_path = os.path.dirname(curr_path)
    dest_path = os.path.join(pare_path, 'Resource')

    for i in range(9):
        file = "SAMPLE0{}".format(i)
        name = os.path.join(dest_path, file + '.png')
        if i < 4:
            set01.append(name)
        if i > 4:
            set02.append(name)


    seq01 = []
    seq02 = []
    for i in range(4):
        seq01.append(Spt.Transfm(set01[i]))
        seq02.append(Spt.Transfm(set02[i]))

    for i in range(4):
        ax[0, i].imshow(seq01[i])
        ax[1, i].imshow(seq02[i])


    # flip y-axis upside down
    #for i in range(2):
    #    for j in range(4):
    #        ax[i, j].invert_yaxis()


    plt.show()


"""
To-do:

1. test the translation, rotation and scaling transformations; generate a sequence of image frames of the same image
that translates the centroid leftward, rightward, upward, downward, or a mix of the movement directions; 


"""