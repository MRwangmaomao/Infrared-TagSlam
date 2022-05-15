import numpy as np
import random, os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from EXTRACT import CorxVar, CorxFun
from TESTS import VisTstVar, VisTstFun
from FUNCS import FNS

# ---------------------------------------------------------------------------------------------------------------------
# CORTX Module - compute visual percept offline

if __name__ == '__main__':
    size = 150
    orient = 2
    length = 1
    empty = np.zeros((2 * size, 2 * size))

    fig2d, ax2d = plt.subplots(2, 4)

    # ----------------------------------------------------------------------------------------------------------------
    # initialize variables

    car_name = ("MICRO.png", "VAN.png", "BIG_TRUCK.png", "BIG_TRUCK.png")
    pla_name = ("11plane.jpg", "24plane.jpg", "41plane.jpg", "41plane.jpg")

    label = np.random.randint(0, 2)
    #pick = FNS().delta_fn(label, 0) * random.choice(car_name) + FNS().delta_fn(label, 1) * random.choice(pla_name)


    # ----------------------------
    dirPath = os.path.join(os.path.abspath(".") + os.path.sep + '..', 'Resource')
    pick = name = os.path.join(dirPath, "marker_07" + '.png')
    # ----------------------------


    img_load = Image.open(pick)
    img_gray = img_load.convert('L')  # convert image to grayscale

    #fn = lambda x: 255 if x > 140 else 0
    #img_bw = img_load.convert('L').point(fn, mode='1')  # convert figure to white and background to black

    img_resize = img_gray.resize((2 * size, 2 * size))
    img_norm = np.array(img_resize) / 255   # normalize it btw 0 and 1

    CorxVar = CorxVar(img_norm, size, 2 * orient, 1, 2)
    Corx = CorxFun(CorxVar)

    TstVar = VisTstVar(ax2d, size)
    Tst = VisTstFun(TstVar)

    # ----------------------------------------------------------------------------------------------------------------


    Corx.LGN()
    Corx.Simple()
    Corx.Complex()
    Corx.CoordSys()

    """
    lgn_on = CorxVar.lgnn.on_map
    lgn_off = CorxVar.lgnn.off_map
    left_on = CorxVar.simp.left_on
    left_off = CorxVar.simp.left_off
    right_on = CorxVar.simp.right_on
    right_off = CorxVar.simp.right_off
    complex = CorxVar.cmpx.comb_map
    """
    invert_map = CorxVar.coord.sum_map


    #pre_out = sum(complex[0][o] for o in range(2 * orient))
    #CoM = FNS().centroid(pre_out, size)
    pos_out = invert_map

    data_S = [pos_out] * 4
    data_L = [img_norm for j in range(4)]

    data = data_S, data_L
    Tst.Simple(data)

    # flip y-axis upside down
    #for i in range(2):
    #    for j in range(4):
    #        ax2d[i, j].invert_yaxis()


    plt.show()




