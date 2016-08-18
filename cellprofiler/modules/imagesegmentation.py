"""

Image segmentation

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.object
import cellprofiler.setting
import skimage.color
import skimage.filters
import skimage.morphology
import skimage.segmentation


class ImageSegmentation(cellprofiler.module.Module):
    module_name = "ImageSegmentation"
    category = "Volumetric"
    variable_revision_number = 1

    def create_settings(self):
        self.input_image_name = cellprofiler.setting.ImageNameSubscriber(
            "Input image name",
            cellprofiler.setting.NONE
        )

        self.object_name = cellprofiler.setting.ObjectNameProvider(
            "Object name",
            ""
        )

        self.method = cellprofiler.setting.Choice(
            "Method",
            [
                "Active contour model",
                "Graph partition",
                "Partial differential equation (PDE)",
                "Region growing"
            ]
        )

        self.active_contour_model_implementation = cellprofiler.setting.Choice(
            "Implementation",
            [
                "Chan-Vese"
            ]
        )

        self.chan_vese_mask = cellprofiler.setting.ImageNameSubscriber(
            "Mask",
            cellprofiler.setting.NONE
        )

        self.chan_vese_iterations = cellprofiler.setting.Integer(
            "Iterations",
            200
        )

        self.graph_partition_implementation = cellprofiler.setting.Choice(
            "Implementation",
            [
                "Random walker algorithm"
            ]
        )

        self.random_walker_algorithm_labels = cellprofiler.setting.ImageNameSubscriber(
            "Labels",
            cellprofiler.setting.NONE
        )

        self.random_walker_algorithm_beta = cellprofiler.setting.Float(
            "Beta",
            130.0
        )

        self.random_walker_algorithm_mode = cellprofiler.setting.Choice(
            "Mode",
            [
                "Brute force",
                "Conjugate gradient",
                "Conjugate gradient with multigrid preconditioner"
            ]
        )

        self.random_walker_algorithm_tolerance = cellprofiler.setting.Float(
            "Tolerance",
            0.001
        )

        self.partial_differential_equation_implementation = cellprofiler.setting.Choice(
            "Implementation",
            [
                "Level set method (LSM)"
            ]
        )

        self.region_growing_implementation = cellprofiler.setting.Choice(
            "Implementation",
            [
                "Simple Linear Iterative Clustering (SLIC)"
            ]
        )

        self.simple_linear_iterative_clustering_segments = cellprofiler.setting.Integer(
            "Segments",
            200
        )

        self.simple_linear_iterative_clustering_compactness = cellprofiler.setting.Float(
            "Compactness",
            10.0
        )

        self.simple_linear_iterative_clustering_iterations = cellprofiler.setting.Integer(
            "Iterations",
            10
        )

        self.simple_linear_iterative_clustering_sigma = cellprofiler.setting.Float(
            "Sigma",
            0
        )

    def settings(self):
        return [
            self.input_image_name,
            self.object_name,
            self.method,
            self.active_contour_model_implementation,
            self.chan_vese_mask,
            self.chan_vese_iterations,
            self.graph_partition_implementation,
            self.random_walker_algorithm_labels,
            self.random_walker_algorithm_beta,
            self.random_walker_algorithm_mode,
            self.random_walker_algorithm_tolerance,
            self.partial_differential_equation_implementation,
            self.region_growing_implementation,
            self.simple_linear_iterative_clustering_segments,
            self.simple_linear_iterative_clustering_compactness,
            self.simple_linear_iterative_clustering_iterations,
            self.simple_linear_iterative_clustering_sigma
        ]

    def visible_settings(self):
        settings = [
            self.input_image_name,
            self.object_name,
            self.method
        ]

        if self.method.value == "Active contour model":
            settings = settings + [
                self.active_contour_model_implementation
            ]

            if self.active_contour_model_implementation == "Chan-Vese":
                settings = settings + [
                    self.chan_vese_mask,
                    self.chan_vese_iterations
                ]

        if self.method.value == "Graph partition":
            settings = settings + [
                self.graph_partition_implementation
            ]

            if self.graph_partition_implementation == "Random walker algorithm":
                settings = settings + [
                    self.random_walker_algorithm_beta,
                    self.random_walker_algorithm_labels,
                    self.random_walker_algorithm_mode,
                    self.random_walker_algorithm_tolerance
                ]

        if self.method.value == "Partial differential equation (PDE)":
            settings = settings + [
                self.partial_differential_equation_implementation
            ]

        if self.method.value == "Region growing":
            settings = settings + [
                self.region_growing_implementation
            ]

            if self.region_growing_implementation == "Simple Linear Iterative Clustering (SLIC)":
                settings = settings + [
                    self.simple_linear_iterative_clustering_segments,
                    self.simple_linear_iterative_clustering_compactness,
                    self.simple_linear_iterative_clustering_iterations,
                    self.simple_linear_iterative_clustering_sigma
                ]

        return settings

    def run(self, workspace):
        name = self.input_image_name.value

        images = workspace.image_set

        image = images.get_image(name)

        data = image.pixel_data

        if self.method.value == "Active contour model":
            if self.active_contour_model_implementation == "Chan-Vese":
                mask = self.chan_vese_mask.value
                mask_image = images.get_image(mask)
                mask_data = mask_image.pixel_data

                iterations = self.chan_vese_iterations.value

                segmentation = chanvese3d(data, mask_data, iterations)

        if self.method.value == "Graph partition":
            if self.graph_partition_implementation == "Random walker algorithm":
                pass

        if self.method.value == "Partial differential equation (PDE)":
            pass

        if self.method.value == "Region growing":
            if self.region_growing_implementation == "Simple Linear Iterative Clustering (SLIC)":
                segments = self.simple_linear_iterative_clustering_segments.value

                compactness = self.simple_linear_iterative_clustering_compactness.value

                iterations = self.simple_linear_iterative_clustering_iterations.value

                sigma = self.simple_linear_iterative_clustering_sigma.value

                segmentation = skimage.segmentation.slic(data, segments, compactness, iterations, sigma, spacing=image.spacing)

                segmentation = skimage.color.label2rgb(segmentation, data, kind="avg")

        output_object = cellprofiler.object.Objects()
        output_object.segmented = segmentation
        workspace.object_set.add_objects(output_object, self.object_name.value)

        if self.show_window:
            workspace.display_data.image = data

            workspace.display_data.segmentation = segmentation

    def display(self, workspace, figure):
        figure.set_subplots((2, 1))

        figure.subplot_imshow_grayscale(
            0,
            0,
            workspace.display_data.image[16],
            ""
        )

        figure.subplot_imshow_grayscale(
            1,
            0,
            workspace.display_data.segmentation[16],
            ""
        )


import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from scipy import weave

eps = np.finfo(np.float).eps


def chanvese3d(I, init_mask, max_its=200, alpha=0.2, thresh=0, color='r', display=False):
    I = I.astype('float')

    # -- Create a signed distance map (SDF) from mask
    phi = mask2phi(init_mask)

    if display:
        plt.ion()
        showCurveAndPhi(I, phi, color)

    # --main loop
    its = 0
    stop = False
    prev_mask = init_mask
    c = 0

    while (its < max_its and not stop):

        # get the curve's narrow band
        idx = np.flatnonzero(np.logical_and(phi <= 1.2, phi >= -1.2))

        if len(idx) > 0:
            # -- intermediate output
            if display:
                if np.mod(its, 10) == 0:
                    # set(ud.txtInfo1,'string',sprintf('iteration: %d',its),'color',[1 1 0]);
                    showCurveAndPhi(I, phi, color)

            else:
                if np.mod(its, 10) == 0:
                    # set(ud.txtInfo1,'string',sprintf('iteration: %d',its),'color',[1 1 0]);
                    # drawnow;
                    pass

            # -- find interior and exterior mean
            upts = np.flatnonzero(phi <= 0)  # interior points
            vpts = np.flatnonzero(phi > 0)  # exterior points
            u = np.sum(I.flat[upts]) / (len(upts) + eps)  # interior mean
            v = np.sum(I.flat[vpts]) / (len(vpts) + eps)  # exterior mean

            F = (I.flat[idx] - u) ** 2 - (I.flat[idx] - v) ** 2  # force from image information
            curvature = get_curvature(phi, idx)  # force from curvature penalty

            dphidt = F / np.max(np.abs(F)) + alpha * curvature  # gradient descent to minimize energy

            # -- maintain the CFL condition
            dt = 0.45 / (np.max(np.abs(dphidt)) + eps)

            # -- evolve the curve
            phi.flat[idx] += dt * dphidt

            # -- Keep SDF smooth
            phi = sussman(phi, 0.5)

            new_mask = phi <= 0
            c = convergence(prev_mask, new_mask, thresh, c)

            if c <= 5:
                its = its + 1
                prev_mask = new_mask
            else:
                stop = True

        else:
            break

    # -- final output
    if display:
        showCurveAndPhi(I, phi, color)
        time.sleep(10)

    # -- make mask from SDF
    seg = phi <= 0  # -- Get mask from levelset

    return seg, phi, its


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# -- AUXILIARY FUNCTIONS ----------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

def bwdist(a):
    """
    this is an intermediary function, 'a' has only True, False vals,
    so we convert them into 0, 1 values -- in reverse. True is 0,
    False is 1, distance_transform_edt wants it that way.
    """
    return nd.distance_transform_edt(a == 0)


import time


# -- Displays the image with curve superimposed
def showCurveAndPhi(I, phi, color):
    # subplot(numRows, numCols, plotNum)
    plt.subplot(321)
    plt.imshow(I[:, :, I.shape[2] / 2], cmap='gray')
    plt.hold(True)
    CS = plt.contour(phi[:, :, I.shape[2] / 2], 0, colors=color)
    plt.hold(False)

    plt.subplot(322)
    plt.imshow(phi[:, :, I.shape[2] / 2])

    plt.subplot(323)
    plt.imshow(I[:, I.shape[1] / 2, :], cmap='gray')
    plt.hold(True)
    CS = plt.contour(phi[:, I.shape[1] / 2, :], 0, colors=color)
    plt.hold(False)

    plt.subplot(324)
    plt.imshow(phi[:, I.shape[1] / 2, :])

    plt.subplot(325)
    plt.imshow(I[I.shape[0] / 2, :, :], cmap='gray')
    plt.hold(True)
    CS = plt.contour(phi[I.shape[0] / 2, :, :], 0, colors=color)
    plt.hold(False)

    plt.subplot(326)
    plt.imshow(phi[I.shape[0] / 2, :, :])

    plt.draw()
    # time.sleep(1)


# -- converts a mask to a SDF
def mask2phi(init_a):
    phi = bwdist(init_a) - bwdist(1 - init_a) + init_a - 0.5
    return phi


# -- compute curvature along SDF
def get_curvature(phi, idx):
    dimz, dimy, dimx = phi.shape
    zyx = np.array([np.unravel_index(i, phi.shape) for i in idx])  # get subscripts
    z = zyx[:, 0]
    y = zyx[:, 1]
    x = zyx[:, 2]

    # -- get subscripts of neighbors
    zm1 = z - 1;
    ym1 = y - 1;
    xm1 = x - 1;
    zp1 = z + 1;
    yp1 = y + 1;
    xp1 = x + 1;

    # -- bounds checking
    zm1[zm1 < 0] = 0;
    ym1[ym1 < 0] = 0;
    xm1[xm1 < 0] = 0;
    zp1[zp1 >= dimz] = dimz - 1;
    yp1[yp1 >= dimy] = dimy - 1;
    xp1[xp1 >= dimx] = dimx - 1;

    # -- get central derivatives of SDF at x,y
    dx = (phi[z, y, xm1] - phi[z, y, xp1]) / 2  # (l-r)/2
    dxx = phi[z, y, xm1] - 2 * phi[z, y, x] + phi[z, y, xp1]  # l-2c+r
    dx2 = dx * dx

    dy = (phi[z, ym1, x] - phi[z, yp1, x]) / 2  # (u-d)/2
    dyy = phi[z, ym1, x] - 2 * phi[z, y, x] + phi[z, yp1, x]  # u-2c+d
    dy2 = dy * dy

    dz = (phi[zm1, y, x] - phi[zp1, y, x]) / 2  # (b-f)/2
    dzz = phi[zm1, y, x] - 2 * phi[z, y, x] + phi[zp1, y, x]  # b-2c+f
    dz2 = dz * dz

    # (ul+dr-ur-dl)/4
    dxy = (phi[z, ym1, xm1] + phi[z, yp1, xp1] - phi[z, ym1, xp1] - phi[z, yp1, xm1]) / 4

    # (lf+rb-rf-lb)/4
    dxz = (phi[zp1, y, xm1] + phi[zm1, y, xp1] - phi[zp1, y, xp1] - phi[zm1, y, xm1]) / 4

    # (uf+db-df-ub)/4
    dyz = (phi[zp1, ym1, x] + phi[zm1, yp1, x] - phi[zp1, yp1, x] - phi[zm1, ym1, x]) / 4

    # -- compute curvature (Kappa)
    curvature = ((dxx * (dy2 + dz2) + dyy * (dx2 + dz2) + dzz * (dx2 + dy2) -
                  2 * dx * dy * dxy - 2 * dx * dz * dxz - 2 * dy * dz * dyz) /
                 (dx2 + dy2 + dz2 + eps))

    return curvature


def mymax(a, b):
    return (a + b + np.abs(a - b)) / 2


# -- level set re-initialization by the sussman method
def sussman(D, dt):
    # forward/backward differences
    a = D - shiftR(D)  # backward
    b = shiftL(D) - D  # forward
    c = D - shiftD(D)  # backward
    d = shiftU(D) - D  # forward
    e = D - shiftF(D)  # backward
    f = shiftB(D) - D  # forward

    a_p = a;
    a_n = a.copy();  # a+ and a-
    b_p = b;
    b_n = b.copy();
    c_p = c;
    c_n = c.copy();
    d_p = d;
    d_n = d.copy();
    e_p = e;
    e_n = e.copy();
    f_p = f;
    f_n = f.copy();

    i_max = D.shape[0] * D.shape[1] * D.shape[2]
    code = """
           for (int i = 0; i < i_max; i++) {
               if ( a_p[i] < 0 ) { a_p[i] = 0; }
               if ( a_n[i] > 0 ) { a_n[i] = 0; }
               if ( b_p[i] < 0 ) { b_p[i] = 0; }
               if ( b_n[i] > 0 ) { b_n[i] = 0; }
               if ( c_p[i] < 0 ) { c_p[i] = 0; }
               if ( c_n[i] > 0 ) { c_n[i] = 0; }
               if ( d_p[i] < 0 ) { d_p[i] = 0; }
               if ( d_n[i] > 0 ) { d_n[i] = 0; }
               if ( e_p[i] < 0 ) { e_p[i] = 0; }
               if ( e_n[i] > 0 ) { e_n[i] = 0; }
               if ( f_p[i] < 0 ) { f_p[i] = 0; }
               if ( f_n[i] > 0 ) { f_n[i] = 0; }
            }
    """
    weave.inline(code,
                 ['i_max',
                  'a_p', 'a_n', 'b_p', 'b_n', 'c_p', 'c_n', 'd_p', 'd_n', 'e_p', 'e_n', 'f_p', 'f_n']
                 )

    dD = np.zeros(D.shape)
    D_neg_ind = np.flatnonzero(D < 0)
    D_pos_ind = np.flatnonzero(D > 0)

    dD.flat[D_pos_ind] = np.sqrt(mymax(a_p.flat[D_pos_ind] ** 2, b_n.flat[D_pos_ind] ** 2)
                                 + mymax(c_p.flat[D_pos_ind] ** 2, d_n.flat[D_pos_ind] ** 2)
                                 + mymax(e_p.flat[D_pos_ind] ** 2, f_n.flat[D_pos_ind] ** 2)
                                 ) - 1

    dD.flat[D_neg_ind] = np.sqrt(mymax(a_n.flat[D_neg_ind] ** 2, b_p.flat[D_neg_ind] ** 2)
                                 + mymax(c_n.flat[D_neg_ind] ** 2, d_p.flat[D_neg_ind] ** 2)
                                 + mymax(e_n.flat[D_neg_ind] ** 2, f_p.flat[D_neg_ind] ** 2)
                                 ) - 1

    D = D - dt * np.sign(D) * dD

    return D


# -- whole matrix derivatives
def shiftD(M):
    shift = M[:, range(1, M.shape[1]) + [M.shape[1] - 1], :]
    return shift


def shiftL(M):
    shift = M[:, :, range(1, M.shape[2]) + [M.shape[2] - 1]]
    return shift


def shiftR(M):
    shift = M[:, :, [0] + range(0, M.shape[2] - 1)]
    return shift


def shiftU(M):
    shift = M[:, [0] + range(0, M.shape[1] - 1), :]
    return shift


def shiftF(M):
    shift = M[[0] + range(0, M.shape[0] - 1), :, :]
    return shift


def shiftB(M):
    shift = M[range(1, M.shape[0]) + [M.shape[0] - 1], :, :]
    return shift


# Convergence Test
def convergence(p_mask, n_mask, thresh, c):
    diff = p_mask - n_mask
    n_diff = np.sum(np.abs(diff))
    if n_diff < thresh:
        c = c + 1
    else:
        c = 0

    return c