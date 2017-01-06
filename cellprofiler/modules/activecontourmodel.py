# coding=utf-8

"""

Active contour model

"""
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

from future import standard_library
standard_library.install_aliases()
from builtins import *
from builtins import range
from past.utils import old_div
import cellprofiler.module
import cellprofiler.object
import cellprofiler.setting
import numpy
import scipy.ndimage
import skimage.draw
import skimage.morphology
import skimage.filters
import skimage.measure
import skimage.segmentation


class ActiveContourModel(cellprofiler.module.ImageSegmentation):
    module_name = "Active contour model"

    variable_revision_number = 1

    def create_settings(self):
        super(ActiveContourModel, self).create_settings()

        self.iterations = cellprofiler.setting.Integer(
            text="Iterations",
            value=20
        )

        self.alpha = cellprofiler.setting.Float(
            text="Alpha",
            value=0.2
        )

        self.threshold = cellprofiler.setting.Float(
            text="Threshold",
            value=0
        )

    def settings(self):
        __settings__ = super(ActiveContourModel, self).settings()

        return __settings__ + [
            self.iterations,
            self.alpha,
            self.threshold
        ]

    def visible_settings(self):
        __settings__ = super(ActiveContourModel, self).settings()

        return __settings__ + [
            self.iterations,
            self.alpha,
            self.threshold
        ]

    def run(self, workspace):
        x_name = self.x_name.value

        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        x_data = x.pixel_data

        thresholding = skimage.filters.threshold_otsu(x_data)

        thresholding = thresholding * 0.9

        binary = x_data > thresholding

        y_data, phi = chan_vese(x_data, binary, alpha=self.alpha.value, iterations=self.iterations.value, threshold=self.threshold.value)

        y_data = skimage.measure.label(y_data)

        objects = cellprofiler.object.Objects()

        objects.segmented = y_data

        objects.parent_image = x

        workspace.object_set.add_objects(objects, y_name)

        if self.show_window:
            workspace.display_data.x_data = x_data

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = x.dimensions


epsilon = numpy.finfo(numpy.float).eps


def chan_vese(image, mask, iterations, alpha, threshold):
    image = skimage.img_as_float(image)

    # -- Create a signed distance map (SDF) from mask
    phi = bwdist(mask) - bwdist(1 - mask) + mask - 0.5

    # --main loop
    iteration = 0

    stop = False

    previous_mask = mask

    c = 0

    while iteration < iterations and not stop:
        # get the curve's narrow band
        index = numpy.flatnonzero(numpy.logical_and(phi <= 1.2, phi >= -1.2))

        if len(index) > 0:
            interior_points = numpy.flatnonzero(phi <= 0)

            exterior_points = numpy.flatnonzero(phi > 0)

            interior_mean = old_div(numpy.sum(image.flat[interior_points]), (len(interior_points) + epsilon))

            exterior_mean = old_div(numpy.sum(image.flat[exterior_points]), (len(exterior_points) + epsilon))

            force = (image.flat[index] - interior_mean) ** 2 - (image.flat[index] - exterior_mean) ** 2

            curvature = get_curvature(phi, index)

            gradient_descent = old_div(force, numpy.max(numpy.abs(force))) + alpha * curvature

            # -- maintain the CFL condition
            dt = old_div(0.45, (numpy.max(numpy.abs(gradient_descent)) + epsilon))

            # -- evolve the curve
            phi.flat[index] += dt * gradient_descent

            # -- Keep SDF smooth
            phi = sussman(phi, 0.5)

            new_mask = phi <= 0

            c = convergence(previous_mask, new_mask, threshold, c)

            if c <= 5:
                iteration += 1

                previous_mask = new_mask
            else:
                stop = True

        else:
            break

    # -- make mask from SDF
    segmentation = phi <= 0  # -- Get mask from levelset

    return segmentation, phi


def bwdist(a):
    """
    this is an intermediary function, 'a' has only True, False vals,
    so we convert them into 0, 1 values -- in reverse. True is 0,
    False is 1, distance_transform_edt wants it that way.
    """
    return scipy.ndimage.distance_transform_edt(a == 0)


# -- compute curvature along SDF
def get_curvature(phi, index):
    dimz, dimy, dimx = phi.shape
    zyx = numpy.array([numpy.unravel_index(i, phi.shape) for i in index])  # get subscripts
    z = zyx[:, 0]
    y = zyx[:, 1]
    x = zyx[:, 2]

    # -- get subscripts of neighbors
    zm1 = z - 1
    ym1 = y - 1
    xm1 = x - 1
    zp1 = z + 1
    yp1 = y + 1
    xp1 = x + 1

    # -- bounds checking
    zm1[zm1 < 0] = 0
    ym1[ym1 < 0] = 0
    xm1[xm1 < 0] = 0
    zp1[zp1 >= dimz] = dimz - 1
    yp1[yp1 >= dimy] = dimy - 1
    xp1[xp1 >= dimx] = dimx - 1

    # -- get central derivatives of SDF at x,y
    dx = old_div((phi[z, y, xm1] - phi[z, y, xp1]), 2)  # (l-r)/2

    dxx = phi[z, y, xm1] - 2 * phi[z, y, x] + phi[z, y, xp1]  # l-2c+r

    dx2 = dx * dx

    dy = old_div((phi[z, ym1, x] - phi[z, yp1, x]), 2)  # (u-d)/2

    dyy = phi[z, ym1, x] - 2 * phi[z, y, x] + phi[z, yp1, x]  # u-2c+d

    dy2 = dy * dy

    dz = old_div((phi[zm1, y, x] - phi[zp1, y, x]), 2)  # (b-f)/2

    dzz = phi[zm1, y, x] - 2 * phi[z, y, x] + phi[zp1, y, x]  # b-2c+f

    dz2 = dz * dz

    # (ul+dr-ur-dl)/4
    dxy = old_div((phi[z, ym1, xm1] + phi[z, yp1, xp1] - phi[z, ym1, xp1] - phi[z, yp1, xm1]), 4)

    # (lf+rb-rf-lb)/4
    dxz = old_div((phi[zp1, y, xm1] + phi[zm1, y, xp1] - phi[zp1, y, xp1] - phi[zm1, y, xm1]), 4)

    # (uf+db-df-ub)/4
    dyz = old_div((phi[zp1, ym1, x] + phi[zm1, yp1, x] - phi[zp1, yp1, x] - phi[zm1, ym1, x]), 4)

    # -- compute curvature (Kappa)
    curvature = (old_div((dxx * (dy2 + dz2) + dyy * (dx2 + dz2) + dzz * (dx2 + dy2) - 2 * dx * dy * dxy - 2 * dx * dz * dxz - 2 * dy * dz * dyz), (dx2 + dy2 + dz2 + epsilon)))

    return curvature


def mymax(a, b):
    return old_div((a + b + numpy.abs(a - b)), 2)


# -- level set re-initialization by the sussman method
def sussman(D, dt):
    # forward/backward differences
    a = D - shiftr(D)  # backward

    b = shiftl(D) - D  # forward

    c = D - shiftd(D)  # backward

    d = shiftu(D) - D  # forward

    e = D - shiftf(D)  # backward

    f = shiftb(D) - D  # forward

    a_p = a
    a_n = a.copy()  # a+ and a-
    b_p = b
    b_n = b.copy()
    c_p = c
    c_n = c.copy()
    d_p = d
    d_n = d.copy()
    e_p = e
    e_n = e.copy()
    f_p = f
    f_n = f.copy()

    a_p[a < 0] = 0
    a_n[a > 0] = 0
    b_p[b < 0] = 0
    b_n[b > 0] = 0
    c_p[c < 0] = 0
    c_n[c > 0] = 0
    d_p[d < 0] = 0
    d_n[d > 0] = 0

    dD = numpy.zeros(D.shape)
    D_neg_ind = numpy.flatnonzero(D < 0)
    D_pos_ind = numpy.flatnonzero(D > 0)

    dD.flat[D_pos_ind] = numpy.sqrt(mymax(a_p.flat[D_pos_ind] ** 2, b_n.flat[D_pos_ind] ** 2)
                                    + mymax(c_p.flat[D_pos_ind] ** 2, d_n.flat[D_pos_ind] ** 2)
                                    + mymax(e_p.flat[D_pos_ind] ** 2, f_n.flat[D_pos_ind] ** 2)
                                    ) - 1

    dD.flat[D_neg_ind] = numpy.sqrt(mymax(a_n.flat[D_neg_ind] ** 2, b_p.flat[D_neg_ind] ** 2)
                                    + mymax(c_n.flat[D_neg_ind] ** 2, d_p.flat[D_neg_ind] ** 2)
                                    + mymax(e_n.flat[D_neg_ind] ** 2, f_p.flat[D_neg_ind] ** 2)
                                    ) - 1

    D = D - dt * numpy.sign(D) * dD

    return D


# -- whole matrix derivatives
def shiftd(m):
    return m[:, list(range(1, m.shape[1])) + [m.shape[1] - 1], :]


def shiftl(m):
    return m[:, :, list(range(1, m.shape[2])) + [m.shape[2] - 1]]


def shiftr(m):
    return m[:, :, [0] + list(range(0, m.shape[2] - 1))]


def shiftu(m):
    return m[:, [0] + list(range(0, m.shape[1] - 1)), :]


def shiftf(m):
    return m[[0] + list(range(0, m.shape[0] - 1)), :, :]


def shiftb(m):
    return m[list(range(1, m.shape[0])) + [m.shape[0] - 1], :, :]


# Convergence Test
def convergence(p_mask, n_mask, thresh, c):
    if numpy.sum(numpy.abs(p_mask - n_mask)) < thresh:
        c += 1
    else:
        c = 0

    return c

