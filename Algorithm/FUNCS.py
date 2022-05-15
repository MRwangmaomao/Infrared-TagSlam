import numpy as np
from PIL import Image

from timeit import default_timer as tmt
from CFNS import CyFns as cfns


class RK4:
    # Define the 4th order Runge-Kutta algorithm.

    def rk4(self, y0, dy, step):
        k1 = step * dy
        k2 = step * (dy + 1 / 2 * k1)
        k3 = step * (dy + 1 / 2 * k2)
        k4 = step * (dy + k3)
        y1 = y0 + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return y1


class FNS:
    def __init__(self):
        self.cfns = cfns()

    # list the common functions

    def thresh_fn(self, x, thresh):
        return np.heaviside(x - thresh, 0) * (x - thresh)

    def bound_fn(self, x, thresh):  # bound x within -thresh<= x <= thresh
        upbd = np.heaviside(thresh - x, 0)  # upbd is 0 and 1
        out = x * upbd + thresh * (upbd + 1) % 2  # values larger than thresh are first set at 0 then changed to thresh
        lowbd = np.heaviside(out - -thresh, 0)  # lowbd is 0 and 1
        out = out * lowbd + -thresh * ((lowbd + 1) % 2)  # values smaller than thresh are first set at 0 then
        # changed to -thresh
        return out

    def cutoff_fn(self, x, thresh):  # bound x outside -thresh<= x <= thresh, or in x<= -thresh, x>= thresh
        rightbd = np.heaviside(x - thresh, 0)
        rightout = x * rightbd
        leftbd = np.heaviside(-x - thresh, 0)
        leftout = x * leftbd
        return rightout + leftout

    def delta_fn(self, x, a):
        if np.all(x == a) == True:
            return 1
        else:
            return 0

    def cond_fn(self, x, a):
        out = np.heaviside(a - x, 0) * np.heaviside(x - -a, 0)
        return out

    def index_fn(self, j, i, b, a):
        return 1 - self.delta_fn(j, b) * self.delta_fn(i, a)

    def indic_fn(self, x):
        return np.heaviside(x, 0)

    def sample_fn(self, x, thresh):
        return 1 * self.indic_fn(x - thresh)

    def signal_fn(self, x, thresh):
        return self.thresh_fn(x, thresh) ** 2

    def sigmoid_fn(self, x, offset, power):
        return x ** power / (offset ** power + x ** power)



    def kernel_bound(self, j, i, size):  # restrict to positions at up/down and right/left
        return range(max(j - 1, 0), min(j + 1, size)), \
               range(max(i - 1, 0), min(i + 1, size))


    # be careful that use of LoG kernel requires finding the zero crossings to extract edges and it is not just applying
    # the kernel to the image
    def laplac_map(self, map, s, size):
        out = self.cfns.laplac_map(map, s, size)
        return out

    def circ_map(self, map, s, size):
        out = self.cfns.circ_map(map, s, size)
        return out

    def invert_map(self, map, s, size):
        out = self.cfns.invert_map(map, s, size)
        return out

    def lgnn_map(self, map, s, num, size):
        out = self.cfns.gauss_map(map, 0, s, 0, num, size)
        return out

    def simp_map(self, map, sign, s, k, num, size):
        out = self.cfns.gauss_map(map, sign, s, k, num, size)
        return out


    def hycx_map(self, map, s, k, num, size):
        out = self.cfns.gauss_map(map, 0, s, k, num, size)
        return out


    def max_orient(self, map, scale, num, size):
        out = self.cfns.max_orient(map, scale, num, size)
        return out

    def bipo_map(self, map, k, num, size):
        out = self.cfns.gauss_map(map, 0, 2, k, num, size)
        return out

    def conv_orient(self, map, orient, s, gain, num, size):
        out = self.cfns.conv_orient(map, orient, s, gain, num, size)
        return out

    def img_normz(self, img):
        out = (img / (np.max(img) + 0.01)) * 255
        # convert figure to black background to white
        #out = 255 - out
        return out

    def img_flip(self, img):
        out = (img / (np.max(img) + 0.01)) * 255
        # convert figure to black background to white
        out = 255 - out
        return out

    def std_lgnn(self, map, j, i, scale, gain, num, size):
        out = self.cfns.std_dou_gauss(map, 0, j, i, scale, gain, num, size)
        return out

    def std_simp(self, map, sign, j, i, scale, gain, num, size):
        out = self.cfns.std_dou_gauss(map, sign, j, i, scale, gain, num, size)
        return out

    def std_hycx(self, map, j, i, scale, gain, num, size):
        out = self.cfns.std_tri_gauss(map, 0, j, i, scale, gain, num, size)
        return out

    def std_bipo(self, map, j, i, scale, gain, num, size):
        out = self.cfns.std_tri_exp(map, j, i, scale, gain, num, size)
        return out

    def logpol_map(self, map, size):
        out = self.cfns.logpolar_map(map, size)
        return out

    def left_map(self, map, size, thresh):
        out = self.cfns.bdry_remap(map, size, thresh, 0)
        return out

    def right_map(self, map, size, thresh):
        out = self.cfns.bdry_remap(map, size, thresh, 1)
        return out

    def coarse_map(self, map, period, size):
        out = self.cfns.coarse_map(map, period, size)
        return out

    def comple_map(self, map):
        norm = (map / (np.max(map) + 0.01))
        comple = 1 - norm
        return np.array((norm, comple))


    def centroid(self, map, size):
        out = self.cfns.centroid(map, size)
        return out

    def orient_map(self, map, orient, gain, size):
        out = self.cfns.orient_map(map, orient, gain, size)
        return out

    def transl_map(self, map, cent_y, cent_x, size):
        out = self.cfns.transl_map(map, cent_y, cent_x, size)
        return out

    def scale_map(self, map, scale, gain, size):
        out = self.cfns.scale_map(map, scale, gain, size)
        return out

    def scale_pillw(self, map, new_size, size):
        norm = self.img_normz(map)
        old_size = np.array((2 * size, 2 * size))
        orig = Image.fromarray(norm)
        img_fig = orig.resize(new_size)
        img_bg = Image.new('L', old_size, 0)
        cent = self.centroid(map, size)
        box = np.array([int(size - cent[1]), int(size - cent[0])])
        img_bg.paste(img_fig, box)
        return img_bg

    def cliff_dou_map(self, map, s, k, gain, size):
        out = self.cfns.cliff_dou_map(map, s, k, gain, size)
        return out

    def cliff_tri_map(self, map, s, k, gain, size):
        out = self.cfns.cliff_tri_map(map, s, k, gain, size)
        return out

    def forwd_period(self, t, T, interval):
        if (t // interval) * interval + 0 <= t and t <= (t // interval) * interval + T:
            return 1
        else:
            return 0

    def backw_period(self, t, T, interval):
        if (t // interval + 1) * interval - T <= t and t <= (t // interval + 1) * interval + 0:
            return 1
        else:
            return 0

    def intv_period(self, t, interval):
        lb = (t // interval) * interval
        ub = (t // interval + 1) * interval
        return np.arange(lb, ub, 1)  # lower limit is lb and upper limit is ub - 1

    def add_error(self, z, t, interval):
        range = self.intv_period(t, interval)
        value = [z] * interval
        add = [(x, y) for x, y in zip(range, value)]
        return add

    def test_zero(self, x):
        if np.array_equal(x, np.zeros(x.shape)):
            return 1
        else:
            return 0

    def argmax(self, x, size):
        if self.test_zero(x) != 1:
            out = np.array(np.unravel_index(np.argmax(x), x.shape)) - size  # format is (height, width)
            return out
        else:
            return np.zeros(2)





def pysum(map1, map2, size):
    return np.array(
        [[[np.sum(map1[s] * map2[s][j][i]) for i in range(2 * size)] for j in range(2 * size)] for s in range(2)])


if __name__ == '__main__':
    FNS = FNS()
    size = 20
    num = 2 * size

    map1 = np.random.randint(0, 10, (2, 2 * size, 2 * size), dtype=np.int32)
    map2 = np.random.randint(0, 10, (2, 2 * size, 2 * size, 2 * size, 2 * size), dtype=np.int32)

    start = tmt()
    pyarr = pysum(map1, map2, size)
    py = tmt() - start

    start = tmt()
    cyarr = 0
    for j in range(400):
        for i in range(200):
            cyarr += 1
    cy = tmt() - start


    print(cy, py)
    print('Cython is {}x faster'.format(py / cy))
    print(np.array_equal(pyarr, cyarr))
    # print(pyarr, cyarr)

