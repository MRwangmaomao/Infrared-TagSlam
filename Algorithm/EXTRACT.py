import numpy as np
import os, copy
import cv2 as cv
from FUNCS import FNS
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

class ExtrVar:
    def __init__(self, name):
        self.name = name
        self.input = 0
        self.img_size = np.array((480, 640))
        self.img_cent = np.array((260, 320))
        self.contour = np.zeros(self.img_size, dtype=np.uint8)
        self.downSide = []
        self.leftSide = []
        self.upSide = []
        self.rightSide = []
        self.centers = []
        self.box = 0
        self.pre_transl = 0
        self.pre_rotate = 0
        self.pos_transl = 0
        self.pos_rotate = 0
        self.cent_order = []


class ExtrFun:
    def __init__(self, ExtrVar):
        self.Ext = ExtrVar
        self.FNS = FNS()

    def CompCents(self):
        name = self.Ext.name
        FNS = self.FNS

        rand_input = FNS.transfm_fn(name)
        self.Ext.input = rand_input[0]
        self.Ext.pre_rotate = rand_input[1]
        self.Ext.pre_transl = rand_input[2]

        ret, thresh = cv.threshold(self.Ext.input, 127, 255, 0)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

        # draw contours of holes
        for i in range(len(contours)):
            if hierarchy[0][i][3] != -1:
                cv.drawContours(self.Ext.contour, contours, i, (255, 0, 0), 1)

        # reset centers list
        self.Ext.centers = []

        # compute centers of holes
        for i in range(len(contours)):
            if hierarchy[0][i][3] != -1:
                M = cv.moments(contours[i])
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    self.Ext.centers.append([cy, cx])
                    self.Ext.cent_order.append(i)


    def CompTransf(self):
        FNS = self.FNS
        img_cent = self.Ext.img_cent
        cents = self.Ext.centers
        order = self.Ext.cent_order


        # ---------------------------------------------------------------------
        # make sure to reset the sides before add in the pts

        # reset leftSide and downSide
        self.Ext.leftSide = []
        self.Ext.downSide = []

        # idea is first find leftSide, then check for the two pts in leftSide their angles w/ the
        # remaining two pts in cents to exhaust all combinations

        # identify leftSide
        leftSide = []
        max_dist = 0
        for pt01 in cents:
            for pt02 in cents:
                dist = np.linalg.norm(np.array(pt01) - np.array(pt02))
                if dist > 0 and dist > max_dist:
                    max_dist = dist
                    leftSide = [pt01, pt02]


        # identify downSide
        def unique_fn(listpt):
            out = []
            for i in range(len(listpt)):
                if listpt[i] not in out:
                    out.append(listpt[i])
            return out

        fringe = copy.copy(cents)
        for pt in unique_fn(leftSide):
            fringe.remove(pt)

        testSide = np.zeros((4), dtype=object)
        testSide[0] = [leftSide[0], fringe[0]]
        testSide[1] = [leftSide[0], fringe[1]]
        testSide[2] = [leftSide[1], fringe[0]]
        testSide[3] = [leftSide[1], fringe[1]]

        referSide = np.zeros((4), dtype=object)
        referSide[0] = [leftSide[0], leftSide[1]]
        referSide[1] = [leftSide[0], leftSide[1]]
        referSide[2] = [leftSide[1], leftSide[0]]
        referSide[3] = [leftSide[1], leftSide[0]]

        downNorm = np.zeros((4))
        downPerp = np.zeros((4))
        for i in range(4):
            downNorm[i] = FNS.norm_fn(testSide[i])
            downPerp[i] = FNS.dot_fn(testSide[i], referSide[i])

        # find pair w/ minimum angle
        index = np.argmin(downPerp)

        self.Ext.leftSide = referSide[index]
        self.Ext.downSide = testSide[index]


       # --------------------------------------------------------------------------------------------------

        leftSide = self.Ext.leftSide
        downSide = self.Ext.downSide

        common = leftSide[0]
        min_dist = downNorm[index]

        # reset upSide and rightSide
        self.Ext.upSide = []
        self.Ext.rightSide = []

        # identify upSide and rightSide
        upSide = []
        rightSide = []

        # downSide is associated w/ rightSide, not upSide
        for pt in downSide:
            dist = np.linalg.norm(np.array(pt) - np.array(common))
            if dist > 0:
                rightSide.append(pt)

        # leftSide is associated w/ upSide, not rightSide
        for pt in leftSide:
            dist = np.linalg.norm(np.array(pt) - np.array(common))
            if dist > 0:
                upSide.append(pt)


        # find pt x st dist(upSide_01, x) = dist(downSide_01, downSide_02) = min_dist, and
        # dist(rightSide_02, x) = dist(leftSide_01, leftSide_02) = max_dist

        def root_fn(x):
            func = []
            # suppose the common pt is the only pt in rightSide and upSide
            func.append((upSide[0][0] - x[0]) ** 2 + (upSide[0][1] - x[1]) ** 2 - min_dist ** 2)
            func.append((rightSide[0][0] - x[0]) ** 2 + (rightSide[0][1] - x[1]) ** 2 - max_dist ** 2)

            return func

        # the root depends on the initial condition, it is better choose near the image center
        root = list(np.int0(fsolve(root_fn, img_cent)))
        self.Ext.upSide.append(root)
        self.Ext.upSide.append(upSide[0])

        self.Ext.rightSide.append(root)
        self.Ext.rightSide.append(rightSide[0])


        # direct computation of box centroid for translation and rotation
        leftSide = np.array(self.Ext.leftSide)
        rightSide = np.array(self.Ext.rightSide)
        leftMid = leftSide[0] + 0.5 * (leftSide[1] - leftSide[0])
        rightMid = rightSide[1] + 0.5 * (rightSide[0] - rightSide[1])

        centroid = np.int0(leftMid + 0.5 * (rightMid - leftMid))

        # --------------------------------------------------------------------------------------------------------
        # draw bounding box; drawing order seems strange, it is criss-crossing from down right, up left, up right,
        # down left
        self.Ext.box = [self.Ext.downSide[1], self.Ext.upSide[1], self.Ext.rightSide[0], self.Ext.leftSide[0]]

        box = np.array(self.Ext.box)
        rev_box = box[:, (1, 0)]
        rect = cv.polylines(self.Ext.contour, [rev_box], True, (255, 0, 0), 1)
        contours, hierarchy = cv.findContours(rect, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

        #cnt = contours[0]
        #M = cv.moments(cnt)
        #cent_y = int(M['m01'] / M['m00'])
        #cent_x = int(M['m10'] / M['m00'])

        #cv.drawContours(self.Ext.contour, cnt, 0, (255, 0, 0), 1)



        # --------------------------------------------------------------------------------------------------------
        # draw polygon formed by the centers; drawing order seems to be from down up
        polygon = [self.Ext.downSide[1], self.Ext.downSide[0], fringe[(index + 1) % 2], self.Ext.leftSide[1]]
        polygon = np.array(polygon)
        rev_poly = polygon[:, (1, 0)]
        poly = cv.polylines(self.Ext.contour, [rev_poly], True, (255, 0, 0), 1)
        contours, hierarchy = cv.findContours(poly, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

        #cnt = contours[0]
        #cv.drawContours(self.Ext.contour, cnt, 0, (255, 0, 0), 1)

        # --------------------------------------------------------------------

        # compute translation
        self.Ext.pos_transl = centroid - img_cent


        # compute rotation; need to translate before compute angle

        upLine = np.array(self.Ext.upSide) - self.Ext.pos_transl
        midpt = upLine[0] + 0.5 * (upLine[1] - upLine[0])

        midpt = midpt - img_cent
        angle = FNS.angle_fn(midpt[0], midpt[1])
        self.Ext.pos_rotate = np.degrees(angle)



    def RestoreImg(self):
        input = self.Ext.input
        img_cent = self.Ext.img_cent
        img_size = self.Ext.img_size
        transl = self.Ext.pos_transl
        rotate = self.Ext.pos_rotate

        mat_transl = np.array([[1, 0, -transl[1]], [0, 1, -transl[0]]], dtype=np.float32)
        pos_transl = cv.warpAffine(src=input, M=mat_transl, dsize=(img_size[1], img_size[0]))

        mat_rotate = cv.getRotationMatrix2D(center=(img_cent[1], img_cent[0]), angle=rotate, scale=1)
        pos_rotate = cv.warpAffine(src=pos_transl, M=mat_rotate, dsize=(img_size[1], img_size[0]))

        return pos_rotate


if __name__ == '__main__':
    fig, ax = plt.subplots(1, 3)
    for i in range(3):
        ax[i].xaxis.set_major_locator(plt.NullLocator())
        ax[i].yaxis.set_major_locator(plt.NullLocator())

    curr_path = os.getcwd()
    pare_path = os.path.dirname(curr_path)
    dest_path = os.path.join(pare_path, 'Resource')

    file = "SAMPLE0{}".format(np.random.randint(0, 10))
    name = os.path.join(dest_path, file + '.png')

    ExtrVar = ExtrVar(name)
    Ext = ExtrFun(ExtrVar)

    img_size = ExtrVar.img_size
    img_cent = ExtrVar.img_cent

    # --------------------------------------------------------------------------------

    Ext.CompCents()
    Ext.CompTransf()

    input = ExtrVar.input
    pre_rotate = ExtrVar.pre_rotate
    pre_transl = ExtrVar.pre_transl
    box = ExtrVar.box
    cents = ExtrVar.centers
    downSide = np.array(ExtrVar.downSide) - ExtrVar.pos_transl
    upSide = np.array(ExtrVar.upSide) - ExtrVar.pos_transl
    pos_rotate = ExtrVar.pos_rotate
    pos_transl = ExtrVar.pos_transl
    contour = ExtrVar.contour
    output = Ext.RestoreImg()
    normz = output / 255

    # -------------------------------------------------------------------------------------------------------------
    ax[0].imshow(input)
    ax[0].set_title("input image")
    ax[1].imshow(contour)
    ax[1].set_title("contour")
    ax[2].imshow(output)
    ax[2].set_title("output image")
    plt.show()
