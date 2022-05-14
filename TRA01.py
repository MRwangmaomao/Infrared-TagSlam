import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from CORTXb import CorxVar, CorxFun
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
    pick = "marker_08.png"
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
    Corx.ActvSum()
    #Corx.TargPosn()
    #Corx.Hypercomplex()
    #Corx.Bipole()


    lgn_on = CorxVar.lgnn.on_map
    lgn_off = CorxVar.lgnn.off_map
    left_on = CorxVar.simp.left_on
    left_off = CorxVar.simp.left_off
    right_on = CorxVar.simp.right_on
    right_off = CorxVar.simp.right_off
    complex = CorxVar.cmpx.comb_map
    summ = CorxVar.summ.sum_map

    #targ_map = CorxVar.targ.targ_map
    #move_cmd = CorxVar.targ.move_cmd
    #CoM = CorxVar.targ.CoM

    hyperpre = CorxVar.hycx.pre_map
    hyperpos = CorxVar.hycx.pos_map
    orient_map = CorxVar.hycx.orient_map
    bippre = CorxVar.bipo.pre_map
    bippos = CorxVar.bipo.pos_map
    bipole = CorxVar.bipo.out_map


    #data_S = [bippre, bippre, bippos, bippos]
    #data_L = [bipole for j in range(4)]

    #data_S = [np.array([[FNS().cfns.laplac_gradient(q, p, size, size, 3)
    #                     for p in range(2 * size)] for q in range(2 * size)]) for j in range(4)]
    #data_L = [np.array([[FNS().cfns.laplac_gradient(q, p, size, size, 3)
    #                     for p in range(2 * size)] for q in range(2 * size)]) for j in range(4)]
    pre_out = sum(complex[0][o] for o in range(2 * orient))
    pos_out = summ
    CoM = FNS().centroid(pre_out, size)
    data_S = [pos_out] * 4
    #data_L = [FNS().laplac_map(img_norm, 3, size) for j in range(4)]
    data_L = [img_norm for j in range(4)]

    #data_S = [np.array([[FNS().cfns.gauss_gradient(q, p, size, size, -1, 4, j, 4 * orient)
    #for p in range(2 * size)] for q in range(2 * size)]) for j in range(4)]
    #data_L = [np.array([[FNS().cfns.gauss_gradient(q, p, size, size, 1, 4, j, 4 * orient)
    #for p in range(2 * size)] for q in range(2 * size)]) for j in range(4)]
    #data_S = [FNS().thresh_fn(data_S[j] - data_L[j], 0) for j in range(4)]
    #data_L = [FNS().thresh_fn(data_L[j] - data_S[j], 0) for j in range(4)]
    #input = 1/8 * np.ones((2 * size, 2 * size))
    #for j in range(-5, 5):
    #    for i in range(-5, 5):
    #        input[j + size][i + size] = 1/2
    #data_S = [FNS().simp_map(input, 0, 5, j, 4 * orient, size) for j in range(4)]
    #data_L = [FNS().simp_map(input, 0, 10, j, 4 * orient, size) for j in range(4)]
    data = data_S, data_L
    Tst.Simple(data)

    # flip y-axis upside down
    #for i in range(2):
    #    for j in range(4):
    #        ax2d[i, j].invert_yaxis()

    print("CoM is {com}".format(com=CoM))
    plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# sVISION Module - compute visual percept in real-time using C extensions and gpu where images appear alternatively
# btw the two sides and visual processing is assisted by eye-head coordination



