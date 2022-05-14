import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as ptc

class MarVar:
    def __init__(self, input):
        self.wid = 11
        self.heg = 5

        self.sample = np.zeros((5, 11), dtype=int)

        self.num_hole = 4

        self.num_full = 6

        self.num_empt = self.wid * self.heg - (self.num_hole + self.num_full)

        self.sample = np.zeros((5, 11), dtype=int)

        count_hole = 0
        count_full = 0
        heg_list = [j for j in range(self.heg)]
        wid_list = [i for i in range(self.wid)]

        while count_hole < 5 and count_full < 7:
            heg = np.random.choice(heg_list)
            wid = np.random.choice(wid_list)
            val = np.random.randint(0, 3)
            if val == 1:
                count_full += 1
            if val == 2:
                count_hole += 1

            self.sample[heg][wid] = val

            heg_list.remove(heg)
            wid_list.remove(wid)






class MarFun:
    def __init__(self, MarVar):
        self.Mar = MarVar


    def genSample(self):
        pass










if __name__ == '__main__':
    rad = 1
    h_gap = 3
    v_gap = 3
    h_len = 5
    v_len = 5

    """
    # generate sample pattern in standard position

    1. it is 3-4-5-4-3 number of points or circles in each row

    2. both horizontal and vertical spacing are 3 or can be varied by a single parameter

    3. height and width are at least 15

    4. an annulus is placed in the 1st position of 1st row, and two annuli are placed in the 1st and 3rd positions
    of 5th row, and the remaining annulus is placed in the interior at any position

    5. when assign the circles and 4th annulus, keep count
    
    6. note only need collect the coordinates of the circles and annuli, where some are fixed and the rest are randomly
    chosen

    """
    fig, ax = plt.subplots()
    ax.set_xlim(-5, 30)
    ax.set_ylim(0, 20)

    # fixed positions for annuli that define the coordinate system
    fix_set = [(2 * h_gap, 15), (2 * h_gap, 3), (6 * h_gap, 3)]

    # random positions for circles and annuli that define the marker pattern; separate them into boundary set and
    # interior set
    bdry_set = []
    intr_set = []
    for i in range(0, 9):
        if i % 2 == 0:
            if 0 < i and i < 8:
                intr_set.append((i * h_gap, 9))
            else:
                bdry_set.append((i * h_gap, 9))

    for i in range(1, 8):
        if i % 2 == 1:
            if 1 < i and i < 7:
                intr_set.append((i * h_gap, 6))
                intr_set.append((i * h_gap, 12))
            else:
                bdry_set.append((i * h_gap, 6))
                bdry_set.append((i * h_gap, 12))

    for i in range(2, 7):
        if i % 2 == 0:
            # every point is on the boundary
            bdry_set.append((i * h_gap, 3))
            bdry_set.append((i * h_gap, 15))

    bdry_set.remove((2 * h_gap, 15))
    bdry_set.remove((2 * h_gap, 3))
    bdry_set.remove((6 * h_gap, 3))

    # 1st row, 3 occupied spots over 5 positions
    for i in range(2, 7):
        if i % 2 == 0:
            patch = ptc.Circle((i * h_gap, 15), rad, fc='yellow')
            ax.add_patch(patch)

    # 2nd row, 4 occupied spots over 7 positions
    for i in range(1, 8):
        if i % 2 == 1:
            patch = ptc.Circle((i * h_gap, 12), rad, fc='yellow')
            ax.add_patch(patch)

    # 3rd row, 5 occupied spots over 9 positions
    for i in range(0, 9):
        if i % 2 == 0:
            patch = ptc.Circle((i * h_gap, 9), rad, fc='yellow')
            ax.add_patch(patch)

    # 4th row, 4 occupied spots over 7 positions
    for i in range(1, 8):
        if i % 2 == 1:
            patch = ptc.Circle((i * h_gap, 6), rad, fc='yellow')
            ax.add_patch(patch)

    # 5th row, 3 occupied spots over 5 positions
    for i in range(2, 7):
        if i % 2 == 0:
            patch = ptc.Circle((i * h_gap, 3), rad, fc='yellow')
            ax.add_patch(patch)


    # define the coordinate system
    for pt in fix_set:
        patch01 = ptc.Circle(pt, rad)
        patch02 = ptc.Circle(pt, rad * 0.3, fc="white")
        ax.add_patch(patch01)
        ax.add_patch(patch02)

    # define the marker pattern
    count_ann = 0
    count_cir = 0

    # generate remaining annulus
    pt = random.choice(intr_set)
    patch01 = ptc.Circle(pt, rad)
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
            patch = ptc.Circle(pt, rad)
            ax.add_patch(patch)
            count_cir += 1
            bdry_set.remove(pt)
        if sel == 1:
            pt = random.choice(intr_set)
            patch = ptc.Circle(pt, rad)
            ax.add_patch(patch)
            count_cir += 1
            intr_set.remove(pt)







    plt.show()

"""
To-do:

1. generate samples and parameterize different size by a single parameter

    a. start w/ a hexagon template that is 3-4-5-4-3 number of positions or points in each row
    
    b. use a grid and fit circle or annulus in each position
    
    c. use the resulting image to generate samples in standard position
    
    d. translate, orientate, and resize to generate samples in arbitrary positions

2. given a sample in an arbitrary orientation, and possibly of arbitrary size that depends on the viewing distance,
identify the embedded coordinate system, imprint w/ a hexagon template, reorientate and resize to standard position
and size, classify the pattern, and retrieve the corresponding id

    a. based on the arbitrary samples, complete each step
    
    b. compute centroid and move to the center of view
    
    c. identify the coordinate system by using the fact that each annulus has a larger sum of boundary activity than
    a circle, compute the orientation of the coordinate system, and reorientate and resize to standard position
    
    d. recognize and retrieve the corresponding id, and b/c each pattern is distinct, each pattern is supposedly
    associated w/ a unique category, but due to noise, it may require learning the unique category 

3. 


"""