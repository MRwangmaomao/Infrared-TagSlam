import numpy as np
cimport cython
from libc.math cimport sqrt, exp, sin, cos, atan, log, abs

# C extension class for FUNCS Module
cdef class CyFns:


    # bound index array
    cdef int retmap_bound(self, int x, int size):
        if x < 0:
            return 0
        if x > size - 1:
            return size - 1
        else:
            return x

   # check at some value
    cdef int delta_fn(self, int x, int a):
        if (x == a) == True:
            return 1
        else:
            return 0

    # check at some index
    cdef int index_fn(self, int j, int i, int b, int a):
        return 1 - self.delta_fn(j, b) * self.delta_fn(i, a)


    cdef double thresh_fn(self, double x, double thresh):
        if x >= thresh:
            return x - thresh
        else:
            return 0

    cdef double heavs_fn(self, double x, double thresh):
        if x >= thresh:
            return 1
        else:
            return 0


    def bdry_remap(self, double[:,:] map, int size, double thresh, int half):
        out = np.zeros((2 * size, size), dtype=np.float64)
        cdef double[:, :] out_view = out
        cdef double[:, :] map_view = map
        cdef Py_ssize_t j, i

        if half == 0:
            for j in range(2 * size):
                for i in range(size):
                    out_view[j][i] = self.thresh_fn(map_view[j][i], thresh) * 1 / (size - i + 1) ** 1

        if half == 1:
            for j in range(2 * size):
                for i in range(size):
                    out_view[j][i] = self.thresh_fn(map_view[j][size + i], thresh) * 1 / (i + 1) ** 1

        return out


    # apply log polar transformation
    @cython.cdivision(True)
    cdef double[:] logpolar_fn(self, int q, int p, int size):
        cdef double y_c, x_c, radius, theta, pi, y_incre, x_incre
        cdef double new_y, new_x
        pi = 3.14159
        y_c = q - size
        x_c = p - size
        y_incre = (log(size) - log(1)) / (2 * size)  # it is log(size) b/c the radius is size, not 2 * size
        x_incre = (2 * pi - 0) / (2 * size)  # make sure the angle is not btw pi and -pi

        radius = sqrt(y_c**2 + x_c**2)
        theta = self.angle_fn(y_c, x_c)

        new_y = 2 * size - log(radius) / y_incre
        new_x = 2 * size - theta / x_incre
        #new_y = log(radius) / y_incre
        #new_x = theta / x_incre

        return np.array((new_y, new_x))

    cdef double angle_fn(self, double y, double x):
        cdef double theta, pi
        pi = 3.14159
        theta = 0

        if x == 0 and y == 0:
            theta = 0

        elif x == 0 and y > 0:
            theta = 0 + pi / 2

        elif x == 0 and y < 0:
            theta = 2 * pi + -pi / 2

        elif x > 0 and y >= 0:
            theta = 0 + atan(y / x)

        elif x > 0 and y <= 0:
            theta = 2 * pi + atan(y / x)

        elif x < 0 and y <= 0:
            theta = pi + atan(y / x)

        elif x < 0 and y >= 0:
            theta = pi + atan(y / x)

        return theta


    def logpolar_map(self, double[:, :] map, int size):
        out = np.zeros((2 * size, 2 * size), dtype=np.float64)
        cdef double[:, :] out_view = out
        cdef double[:, :] map_view = map
        cdef Py_ssize_t j, i
        cdef int new_y, new_x
        for j in range(2 * size):
            for i in range(2 * size):
                new_y = self.retmap_bound(int(self.logpolar_fn(j, i, size)[0]), 2 * size)
                new_x = self.retmap_bound(int(self.logpolar_fn(j, i, size)[1]), 2 * size)
                # if values of transformed points are cumulative, then they need be weighted else they can easily
                # saturate the pixel values, and the weights depend on the resolution and are proportional to
                # average number of points per bin; for most practical purposes, 0.5 is enough, and do not use 1/2
                out_view[new_y][new_x] = 0 * out_view[new_y][new_x] + 1 * map_view[j][i]
        return out

    cdef double coarse_gradient(self, int q, int p, int j, int i, int period):
        cdef double new_y, new_x
        cdef double r, grad
        new_y = (q - j)
        new_x = (p - i)

        r = sqrt((new_y / period) ** 2 + (new_x / period)** 2)
        grad = exp(-1 * r ** 2)

        return grad

    cdef double coarse_norm(self, int j, int i, int period, int size):
        cdef double norm = 0
        cdef Py_ssize_t q, p
        for q in range(max(j - period, 0), min(j + period, size)):
            for p in range(max(i - period, 0), min(i + period, size)):
                norm += self.coarse_gradient(q, p, j, i, period)
        return norm

    @cython.cdivision(True)
    def coarse_map(self, double[:, :] map, int period, int size):
        pre_out = np.zeros((2 * size, 2 * size), dtype=np.float64)
        cdef double[:, :] preout_view = pre_out
        cdef double[:, :] map_view = map
        cdef Py_ssize_t j, i, q, p, new_j, new_i, new_lim
        cdef double accum
        for j in range(2 * size):
            for i in range(2 * size):
                accum = 0
                for q in range(max(j - period, 0), min(j + period, 2 * size)):
                    for p in range(max(i - period, 0), min(i + period, 2 * size)):
                        accum += map_view[q][p] * self.coarse_gradient(q, p, j, i, period)
                preout_view[j][i] = accum / self.coarse_norm(j, i, period, 2 * size)

        new_lim = int(2 * size / period)
        pos_out = np.zeros((new_lim, new_lim), dtype=np.float64)
        cdef double[:, :] posout_view = pos_out
        for new_j in range(new_lim):
            for new_i in range(new_lim):
                posout_view[new_j][new_i] = preout_view[new_j * period][new_i * period]

        return pos_out

    cdef double[:] transl_fn(self, int q, int p, int q0, int p0, int size):
        cdef double new_y, new_x
        new_y = (q - q0) + size
        new_x = (p - p0) + size
        return np.array((new_y, new_x))

    def transl_map(self, double[:, :] map, int y0, int x0, int size):
        out = np.zeros((2 * size, 2 * size), dtype=np.float64)
        cdef double[:, :] out_view = out
        cdef double[:, :] map_view = map
        cdef Py_ssize_t j, i,
        cdef int new_y, new_x
        for j in range(2 * size):
            for i in range(2 * size):
                new_y = self.retmap_bound(int(self.transl_fn(j, i, y0, x0, size)[0]), 2 * size)
                new_x = self.retmap_bound(int(self.transl_fn(j, i, y0, x0, size)[1]), 2 * size)
                out_view[new_y][new_x] = map_view[j][i]
        return out

    cdef double[:] orient_fn(self, int q, int p, int k0, int gain, int size):
        cdef double new_y, new_x, pi, orient
        pi = 3.14159
        orient = pi * (k0 * gain / float(180))
        #new_y = (q - size) * cos(orient) + (p - size) * sin(orient)
        #new_x = (p - size) * cos(orient) - (q - size) * sin(orient)
        new_y = (q - 0) * cos(orient) + (p - 0) * sin(orient)
        new_x = (p - 0) * cos(orient) - (q - 0) * sin(orient)

        return np.array((new_y, new_x))


    def orient_map(self, double[:, :] map, int k0, int gain, int size):
        out = np.zeros((2 * size, 2 * size), dtype=np.float64)
        cdef double[:, :] out_view = out
        cdef double[:, :] map_view = map
        cdef Py_ssize_t j, i,
        cdef int new_y, new_x
        for j in range(2 * size):
            for i in range(2 * size):
                #if map_view[j][i] != 0:
                new_y = self.retmap_bound(int(self.orient_fn(j, i, k0, gain, size)[0]), 2 * size)
                new_x = self.retmap_bound(int(self.orient_fn(j, i, k0, gain, size)[1]), 2 * size)
                out_view[new_y][new_x] = map_view[j][i]
        return out


    cdef double[:] scale_fn(self, int q, int p, int s0, int gain, int size):
        cdef double new_y, new_x
        new_y = q * s0 * gain
        new_x = p * s0 * gain

        return np.array((new_y, new_x))


    def scale_map(self, double[:, :] map, int s0, int gain, int size):
        out = np.zeros((2 * size, 2 * size), dtype=np.float64)
        cdef double[:, :] out_view = out
        cdef double[:, :] map_view = map
        cdef Py_ssize_t j, i,
        cdef int new_y, new_x
        for j in range(2 * size):
            for i in range(2 * size):
                new_y = self.retmap_bound(int(self.scale_fn(j, i, s0, gain, size)[0]), 2 * size)
                new_x = self.retmap_bound(int(self.scale_fn(j, i, s0, gain, size)[1]), 2 * size)
                out_view[new_y][new_x] = map_view[j][i]
        return out



    @cython.cdivision(True)
    cdef double cliff_fn(self, double q, double p):
        cdef double r, out
        r = sqrt(q ** 2 + p ** 2)
        out = self.thresh_fn((1 - r ** 3) * exp(- r ** 2 / (1 + r)), 0)
        return out

    # 2d counter-clockwise oriented/unoriented cliff kernel
    @cython.cdivision(True)
    cdef double cliff_gradient(self, int q, int p, int j, int i, int s, int k, int gain):
        cdef double new_y, new_x, pi, orient
        pi = 3.14159
        orient = pi * float(k * gain) / float(180)
        new_y = (q - j) * cos(orient) + (p - i) * sin(orient)
        new_x = (p - i) * cos(orient) - (q - j) * sin(orient)

        return self.cliff_fn(new_y / (s * 1), new_x / (s * 5))

    # 2d cliff norm
    cdef double cliff_norm(self, int j, int i, int s, int k, int gain, int size):
        cdef double norm = 0
        cdef Py_ssize_t q, p
        for q in range(max(j - s, 0), min(j + s, size)):
            for p in range(max(i - s, 0), min(i + s, size)):
                norm += self.cliff_gradient(q, p, j, i, s, k, gain)

        return norm


    # filter retinal map with cliff kernel; gain is orient gain, not scale gain, where num * gain = 180
    @cython.cdivision(True)
    def cliff_dou_map(self, double[:, :] map, int s, int k, int gain, int size):
        out = np.zeros((2 * size, 2 * size), dtype=np.float64)
        cdef double[:, :] out_view = out
        cdef double[:, :] map_view = map
        cdef Py_ssize_t j, i, q, p
        cdef double accum
        for j in range(2 * size):
            for i in range(2 * size):
                accum = 0
                for q in range(max(j - s, 0), min(j + s, 2 * size)):
                    for p in range(max(i - s, 0), min(i + s, 2 * size)):
                        accum += map_view[q][p] * self.cliff_gradient(q, p, j, i, s, k, gain)
                out_view[j][i] = accum / self.cliff_norm(j, i, s, k, gain, 2 * size)
        return out

    @cython.cdivision(True)
    def cliff_tri_map(self, double[:, :] map, int s, int k, int gain, int size):
        out = np.zeros((2 * size, 2 * size), dtype=np.float64)
        cdef double[:, :] out_view = out
        cdef double[:, :] map_view = map
        cdef Py_ssize_t j, i, q, p, l
        cdef double accum
        for j in range(2 * size):
            for i in range(2 * size):
                accum = 0
                for l in range(0, 180, gain):
                    for q in range(max(j - s, 0), min(j + s, 2 * size)):
                        for p in range(max(i - s, 0), min(i + s, 2 * size)):
                            accum += map_view[q][p] * self.cliff_gradient(q, p, j, i, s, k, gain) * \
                            self.orient_gradient(l, k)
                out_view[j][i] = accum / \
                (self.cliff_norm(j, i, s, k, gain, 2 * size) * self.orient_norm(k, gain))
        return out


    # be careful with radians or degrees
    cdef double orient_gradient(self, int l, int k):
        cdef double r, grad
        r = sqrt(abs(l - k) / 180)
        grad = exp( -1 * r ** 2)
        return grad

    cdef double orient_norm(self, int k, int gain):
        cdef double norm = 0
        cdef Py_ssize_t l
        for l in range(0, 180, gain):
            norm += self.orient_gradient(l, k)
        return norm


    @cython.cdivision(True)
    def centroid(self, double[:, :] map, int size):
        cdef double[:, :] map_view = map
        cdef double norm, height, width
        norm = 0
        height = 0
        width = 0
        cdef Py_ssize_t q, p
        for q in range(2 * size):
            for p in range(2 * size):
                norm += map_view[q][p]
                height += q * map_view[q][p]
                width += p * map_view[q][p]

        return height / (norm + 0.01), width / (norm + 0.01)

    cdef double laplac_gradient(self, int q, int p, int j, int i, int s):
        cdef double grad, new_y, new_x, r, factor
        new_y = q - j
        new_x = p - i
        r = sqrt((new_y / (1.5 * 1)) ** 2 + (new_x / (1.5 * 1)) ** 2)
        grad = exp(-1 * r ** 2)
        factor = new_y ** 2 + new_x ** 2 - 1 * 1.5 ** 2
        return factor * grad

    cdef double laplac_norm(self, int j, int i, int s, int size):
        cdef double norm = 0
        cdef Py_ssize_t q, p
        for q in range(max(j - s, 0), min(j + s, size)):
            for p in range(max(i - s, 0), min(i + s, size)):
                norm += self.thresh_fn(self.laplac_gradient(q, p, j, i, s), 0)
        return norm

    @cython.cdivision(True)
    def laplac_map(self, double[:, :] map, int s, int size):
        out = np.zeros((2 * size, 2 * size), dtype=np.float64)
        cdef double[:, :] out_view = out
        cdef double[:, :] map_view = map
        cdef Py_ssize_t j, i, q, p
        cdef double accum
        for j in range(2 * size):
            for i in range(2 * size):
                accum = 0
                for q in range(max(j - s, 0), min(j + s, 2 * size)):
                    for p in range(max(i - s, 0), min(i + s, 2 * size)):
                        accum += map_view[q][p] * self.thresh_fn(self.laplac_gradient(q, p, j, i, s), 0)
                out_view[j][i] = accum / self.laplac_norm(j, i, s, 2 * size)
        return out

    # 2d counter-clockwise oriented/unoriented gaussian kernel (clockwise rotation is - in new_y and + in new_x)
    # recall to convert it to float, else it is treated as int
    @cython.cdivision(True)
    cdef double gauss_gradient(self, int q, int p, int j, int i, int sign, int s, int k, int num):
        cdef double new_y, new_x, pi, off_y, off_x
        cdef double r, grad, sigma, orient
        pi = 3.14159
        orient = pi * (k / float(num))
        off_y = sign * cos(orient)
        off_x = sign * sin(orient)
        new_y = (q - (j - off_y)) * cos(orient) + (p - (i + off_x)) * sin(orient)
        new_x = (p - (i + off_x)) * cos(orient) - (q - (j - off_y)) * sin(orient)

        r = sqrt((new_y / (s * 1)) ** 2 + (new_x / (s * 5))** 2)
        sigma = s * 5
        grad = exp(-1 * r ** 2)

        return grad

    # 2d gaussian norm
    cdef double gauss_norm(self, int j, int i, int sign, int s, int k, int num, int size):
        cdef double norm = 0
        cdef Py_ssize_t q, p
        for q in range(max(j - s, 0), min(j + s, size)):
            for p in range(max(i - s, 0), min(i + s, size)):
                norm += self.gauss_gradient(q, p, j, i, sign, s, k, num)
        return norm


    # filter retinal map with gaussian kernel
    @cython.cdivision(True)
    def gauss_map(self, double[:, :] map, int sign, int s, int k, int num, int size):
        out = np.zeros((2 * size, 2 * size), dtype=np.float64)
        cdef double[:, :] out_view = out
        cdef double[:, :] map_view = map
        cdef Py_ssize_t j, i, q, p
        cdef double accum
        for j in range(2 * size):
            for i in range(2 * size):
                accum = 0
                for q in range(max(j - s, 0), min(j + s, 2 * size)):
                    for p in range(max(i - s, 0), min(i + s, 2 * size)):
                        accum += map_view[q][p] * self.gauss_gradient(q, p, j, i, sign, s, k, num)
                out_view[j][i] = accum / self.gauss_norm(j, i, sign, s, k, num, 2 * size)
        return out


    @cython.cdivision(True)
    cdef double circ_grad(self, int q, int p, int j, int i, int s):
        cdef double new_y, new_x
        cdef double r, grad
        new_y = (q - j)
        new_x = (p - i)
        r = sqrt((new_y / (s * 1)) ** 2 + (new_x / (s * 1))** 2)
        grad = exp(-1 * r ** 2)

        return grad

    cdef double circ_norm(self, int j, int i, int s, int size):
        cdef double norm = 0
        cdef Py_ssize_t q, p
        for q in range(max(j - s, 0), min(j + s, size)):
            for p in range(max(i - s, 0), min(i + s, size)):
                norm += self.circ_grad(q, p, j, i, s)
        return norm


    @cython.cdivision(True)
    def circ_map(self, double[:, :] map, int s, int size):
        out = np.zeros((2 * size, 2 * size), dtype=np.float64)
        cdef double[:, :] out_view = out
        cdef double[:, :] map_view = map
        cdef Py_ssize_t j, i, q, p
        cdef double accum
        for j in range(2 * size):
            for i in range(2 * size):
                accum = 0
                for q in range(max(j - s, 0), min(j + s, 2 * size)):
                    for p in range(max(i - s, 0), min(i + s, 2 * size)):
                        accum += map_view[q][p] * self.circ_grad(q, p, j, i, s)
                out_view[j][i] = accum / self.circ_norm(j, i, s, 2 * size)
        return out




    # filter retinal map with gaussian kernel; gain is scale gain, not orient gain
    @cython.cdivision(True)
    def std_dou_gauss(self, double[:, :] map, int sign, int j, int i, int scale, int gain, int num, int size):
        out = np.zeros((scale, num), dtype=np.float64)
        cdef double[:, :] out_view = out
        cdef double[:, :] map_view = map
        cdef Py_ssize_t s, k, q, p
        cdef double accum
        for s in range(1, scale + 1):
            for k in range(num):
                accum = 0
                for q in range(max(j - s * gain, 0), min(j + s * gain, 2 * size)):
                    for p in range(max(i - s * gain, 0), min(i + s * gain, 2 * size)):
                        accum += map_view[q][p] * self.gauss_gradient(q, p, j, i, sign, s * gain, k, num)
                out_view[s][k] = accum / self.gauss_norm(j, i, sign, s * gain, k, num, 2 * size)
        return out

    # filter retinal map with gaussian kernel
    @cython.cdivision(True)
    def std_tri_gauss(self, double[:, :] map, int sign, int j, int i, int scale, int gain, int num, int size):
        out = np.zeros((scale, num), dtype=np.float64)
        cdef double[:, :] out_view = out
        cdef double[:, :] map_view = map
        cdef Py_ssize_t s, k, q, p, l
        cdef double accum
        for s in range(1, scale + 1):
            for k in range(num):
                accum = 0
                for q in range(max(j - s * gain, 0), min(j + s * gain, 2 * size)):
                    for p in range(max(i - s * gain, 0), min(i + s * gain, 2 * size)):
                        for l in range(0, 180, int(180 / num)):
                            accum += map_view[q][p] * self.gauss_gradient(q, p, j, i, sign, s * gain, k, num) * \
                            self.orient_gradient(l, k)
                out_view[s][k] = accum / \
                (self.gauss_norm(j, i, sign, s * gain, k, num, 2 * size) * self.orient_norm(k, int(180 / num)))
        return out


    @cython.cdivision(True)
    cdef double bipo_grad(self, int q, int p, int j, int i, int l, int k, int sign, int gain, int num):
        cdef double new_y, new_x, orient, orient_l, orient_k, pi, dist, coll, ang, grad, new_sign
        new_sign = (1 - sign) / float(2)
        pi = 3.14159265
        orient_l = pi * (l / float(num))
        orient_k = new_sign * pi +  (pi * (k / float(num)))  # when sign=1, it is angle, when sign=-1,
        # it is pi + angle, where 0 < angle < pi

        new_y = (q - j) * cos(orient_k) + (p - i) * sin(orient_k)
        new_x = (p - i) * cos(orient_k) - (q - j) * sin(orient_k)
        orient = self.angle_fn(new_y, new_x)
        dist = exp(-(sqrt(new_y ** 2 + new_x ** 2) / gain - 1) ** 2)
        coll = exp(-(new_y / (new_x + 0.01) ** 2) ** 2 / gain)
        ang = abs(cos(orient_l - orient)) * cos(orient_k - orient)
        grad = self.thresh_fn(sign * dist * coll * ang, 0)
        return grad

    cdef double bipo_norm(self, int j, int i, int k, int sign, int gain, int num, int size):
        cdef double norm = 0
        cdef Py_ssize_t q, p, l
        for q in range(max(j - gain, 0), min(j + gain, size)):
            for p in range(max(i - gain, 0), min(i + gain, size)):
                for l in range(0, 180, int(180 / num)):
                    norm += self.bipo_gradient(q, p, j, i, l, k, sign, gain, num)
        return norm

    @cython.cdivision(True)
    def bipo_map(self, double[:, :] map, int sign, int j, int i, int scale, int gain, int num, int size):
        out = np.zeros((scale, num), dtype=np.float64)
        cdef double[:, :] out_view = out
        cdef double[:, :] map_view = map
        cdef Py_ssize_t s, k, q, p, l
        cdef double accum
        for s in range(1, scale + 1):
            for k in range(num):
                accum = 0
                for q in range(max(j - s * gain, 0), min(j + s * gain, 2 * size)):
                    for p in range(max(i - s * gain, 0), min(i + s * gain, 2 * size)):
                        for l in range(0, 180, int(180 / num)):
                            accum += map_view[q][p] * self.bipo_gradient(q, p, j, i, l, k, sign, s * gain, num)
                out_view[s][k] = accum / self.bipo_norm(j, i, k, sign, s * gain, num, 2 * size)
        return out

    cdef double near_gradient(self):
        pass

    cdef double near_norm(self):
        pass

    def near_map(self):
        pass


    def max_orient(self, double[:, :, :, :] map, int scale, int num, int size):
        out_pos = np.zeros((scale, 2 * size, 2 * size), dtype=np.float64)
        out_ore = np.zeros((scale, 2 * size, 2 * size), dtype=np.intc)
        cdef double[:, :, :] outpos_view = out_pos
        cdef int[:, :, :] outore_view = out_ore
        cdef double[:, :, :, :] map_view = map
        cdef Py_ssize_t s, j, i, k
        cdef int orient = 0
        cdef double compare = 0
        for s in range(scale):
            for j in range(2 * size):
                for i in range(2 * size):
                    compare = 0
                    orient = 0
                    for k in range(num):
                        if map_view[s][j][i][k] >= compare:
                            compare = map_view[s][j][i][k]
                            orient = k
                    outpos_view[s][j][i] = compare
                    outore_view[s][j][i] = orient
        return out_pos, out_ore

    @cython.cdivision(True)
    def conv_orient(self, double[:, :, :, :] map, int[:, :, :] orient, int s, int gain, int num, int size):
        out = np.zeros((2 * size, 2 * size), dtype=np.float64)
        cdef double[:, :] out_view = out
        cdef double[:, :, :, :] map_view = map
        cdef int[:, :, :] orient_view = orient
        cdef Py_ssize_t j, i, q, p, k
        cdef double accum
        for j in range(2 * size):
            for i in range(2 * size):
                accum = 0
                k = orient_view[s][j][i]
                for q in range(max(j - s * gain, 0), min(j + s * gain, 2 * size)):
                    for p in range(max(i - s * gain, 0), min(i + s * gain, 2 * size)):
                        accum += map_view[s][k][q][p] * self.gauss_gradient(q, p, j, i, 0, s * gain, k, num)
                out_view[j][i] = accum / self.gauss_norm(j, i, 0, s * gain, k, num, 2 * size)
        return out
