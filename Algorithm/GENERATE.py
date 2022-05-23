import numpy as np
import random, os
import matplotlib.pyplot as plt
import matplotlib.patches as ptc

if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # initialize parameters
    rad = 1
    h_gap = 3
    v_gap = 3

    fig, ax = plt.subplots()
    ax.set_xlim(-5, 30)
    ax.set_ylim(0, 20)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    # -------------------------------------------------------------------------
    # collect fixed positions for annuli that define the coordinate system
    fix_set = [(2 * h_gap, 5 * v_gap), (2 * h_gap, 1 * v_gap), (6 * h_gap, 1 * v_gap)]

    # collect random positions for circles and annuli that define the marker pattern; separate them into boundary set
    # and interior set
    bdry_set = []
    intr_set = []
    for i in range(0, 9):
        if i % 2 == 0:
            if 0 < i and i < 8:
                intr_set.append((i * h_gap, 3 * v_gap))
            else:
                bdry_set.append((i * h_gap, 3 * v_gap))

    for i in range(1, 8):
        if i % 2 == 1:
            if 1 < i and i < 7:
                intr_set.append((i * h_gap, 2 * v_gap))
                intr_set.append((i * h_gap, 4 * v_gap))
            else:
                bdry_set.append((i * h_gap, 2 * v_gap))
                bdry_set.append((i * h_gap, 4 * v_gap))

    for i in range(2, 7):
        if i % 2 == 0:
            # every point is on the boundary
            bdry_set.append((i * h_gap, 1 * v_gap))
            bdry_set.append((i * h_gap, 5 * v_gap))

    bdry_set.remove((2 * h_gap, 5 * v_gap))
    bdry_set.remove((2 * h_gap, 1 * v_gap))
    bdry_set.remove((6 * h_gap, 1 * v_gap))

    # -------------------------------------------------------------------------
    # imprint a hexagon template
    """
    # 1st row, 3 occupied spots over 5 positions
    for i in range(2, 7):
        if i % 2 == 0:
            patch = ptc.Circle((i * h_gap, 5 * v_gap), rad, fc='gray')
            ax.add_patch(patch)

    # 2nd row, 4 occupied spots over 7 positions
    for i in range(1, 8):
        if i % 2 == 1:
            patch = ptc.Circle((i * h_gap, 4 * v_gap), rad, fc='gray')
            ax.add_patch(patch)

    # 3rd row, 5 occupied spots over 9 positions
    for i in range(0, 9):
        if i % 2 == 0:
            patch = ptc.Circle((i * h_gap, 3 * v_gap), rad, fc='gray')
            ax.add_patch(patch)

    # 4th row, 4 occupied spots over 7 positions
    for i in range(1, 8):
        if i % 2 == 1:
            patch = ptc.Circle((i * h_gap, 2 * v_gap), rad, fc='gray')
            ax.add_patch(patch)

    # 5th row, 3 occupied spots over 5 positions
    for i in range(2, 7):
        if i % 2 == 0:
            patch = ptc.Circle((i * h_gap, 1 * v_gap), rad, fc='gray')
            ax.add_patch(patch)
    """
    # -------------------------------------------------------------------------
    # define the coordinate system
    for pt in fix_set:
        patch01 = ptc.Circle(pt, rad, fc="black")
        patch02 = ptc.Circle(pt, rad * 0.3, fc="white")
        ax.add_patch(patch01)
        ax.add_patch(patch02)

    #-------------------------------------------------------------------------
    # define the marker pattern
    count_ann = 0
    count_cir = 0

    # generate remaining annulus
    pt = random.choice(intr_set)
    patch01 = ptc.Circle(pt, rad, fc="black")
    patch02 = ptc.Circle(pt, rad * 0.3, fc="white")
    ax.add_patch(patch01)
    ax.add_patch(patch02)
    count_ann += 1
    intr_set.remove(pt)

    # generate all circles
    while count_cir < 6:
        sel = np.random.randint(0, 2)
        # sel = 0 is bdry_set and sel = 1 is intr_set
        if sel == 0:
            pt = random.choice(bdry_set)
            patch = ptc.Circle(pt, rad, fc="black")
            ax.add_patch(patch)
            count_cir += 1
            bdry_set.remove(pt)
        if sel == 1:
            pt = random.choice(intr_set)
            patch = ptc.Circle(pt, rad, fc="black")
            ax.add_patch(patch)
            count_cir += 1
            intr_set.remove(pt)

    plt.axis('off')
    filePath = os.path.join(os.path.abspath(".") + os.path.sep + '..', 'Resource')
    img_path = os.path.join(filePath, 'SAMPLE00.png')

    plt.show()