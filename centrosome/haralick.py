from __future__ import absolute_import
import numpy as np
import scipy.ndimage as scind
from scipy.linalg import toeplitz

from .cpmorphology import fixup_scipy_ndimage_result as fix
from six.moves import range


def minimum(input, labels, index):
    return fix(scind.minimum(input, labels, index))

def maximum(input, labels, index):
    return fix(scind.maximum(input, labels, index))

def normalized_per_object(image, labels):
    """Normalize the intensities of each object to the [0, 1] range."""
    nobjects = labels.max()
    objects = np.arange(nobjects + 1)
    lmin, lmax = scind.extrema(image, labels, objects)[:2]
    # Divisor is the object's max - min, or 1 if they are the same.
    divisor = np.ones((nobjects + 1,))
    divisor[lmax > lmin] = (lmax - lmin)[lmax > lmin]
    return (image - lmin[labels]) / divisor[labels]

def quantize(image, nlevels):
    """Quantize an image into integers 0, 1, ..., nlevels - 1.

    image   -- a numpy array of type float, range [0, 1]
    nlevels -- an integer
    """
    tmp = np.array(image // (1.0 / nlevels), dtype='i1')
    return tmp.clip(0, nlevels - 1)

def cooccurrence(quantized_image, labels, scale_i=3, scale_j=0):
    """Calculates co-occurrence matrices for all the objects in the image.

    Return an array P of shape (nobjects, nlevels, nlevels) such that
    P[o, :, :] is the cooccurence matrix for object o.

    quantized_image -- a numpy array of integer type
    labels          -- a numpy array of integer type
    scale           -- an integer

    For each object O, the cooccurrence matrix is defined as follows.
    Given a row number I in the matrix, let A be the set of pixels in
    O with gray level I, excluding pixels in the rightmost S
    columns of the image.  Let B be the set of pixels in O that are S
    pixels to the right of a pixel in A.  Row I of the cooccurence
    matrix is the gray-level histogram of the pixels in B.
    """
    labels = labels.astype(int)
    nlevels = quantized_image.max() + 1
    nobjects = labels.max()
    if scale_i < 0:
        scale_i = -scale_i
        scale_j = -scale_j
    if scale_i == 0 and scale_j > 0:
        image_a = quantized_image[:, :-scale_j]
        image_b = quantized_image[:, scale_j:]
        labels_ab = labels_a = labels[:, :-scale_j]
        labels_b = labels[:, scale_j:]
    elif scale_i > 0 and scale_j == 0:
        image_a = quantized_image[:-scale_i, :]
        image_b = quantized_image[scale_i:, :]
        labels_ab = labels_a = labels[:-scale_i, :]
        labels_b = labels[scale_i:, :]
    elif scale_i > 0 and scale_j > 0:
        image_a = quantized_image[:-scale_i, :-scale_j]
        image_b = quantized_image[scale_i:, scale_j:]
        labels_ab = labels_a = labels[:-scale_i, :-scale_j]
        labels_b = labels[scale_i:, scale_j:]
    else:
        # scale_j should be negative
        image_a = quantized_image[:-scale_i, -scale_j:]
        image_b = quantized_image[scale_i:, :scale_j]
        labels_ab = labels_a = labels[:-scale_i, -scale_j:]
        labels_b = labels[scale_i:, :scale_j]
    equilabel = ((labels_a == labels_b) & (labels_a > 0))
    if np.any(equilabel):

        Q = (nlevels*nlevels*(labels_ab[equilabel]-1)+
             nlevels*image_a[equilabel]+image_b[equilabel])
        R = np.bincount(Q)
        if R.size != nobjects*nlevels*nlevels:
            S = np.zeros(nobjects*nlevels*nlevels-R.size)
            R = np.hstack((R, S))
        P = R.reshape(nobjects, nlevels, nlevels)
        pixel_count = fix(scind.sum(equilabel, labels_ab,
                                    np.arange(nobjects, dtype=np.int32)+1))
        pixel_count = np.tile(pixel_count[:,np.newaxis,np.newaxis],
                              (1,nlevels,nlevels))
        return (P.astype(float) / pixel_count.astype(float), nlevels)
    else:
        return np.zeros((nobjects, nlevels, nlevels)), nlevels

class Haralick(object):
    """
    Calculate the Haralick texture features.

    Currently, the implementation uses nevels = 8 different grey
    levels.

    The original reference is: Haralick et al. (1973), Textural
    Features for Image Classification, _IEEE Transaction on Systems
    Man, Cybernetics_, SMC-3(6):610-621.  BEWARE: There are lots of
    erroneous formulas for the Haralick features in the
    literature.  There is also an error in the original paper.
    """
    def __init__(self, image, labels, scale_i, scale_j, nlevels=8, mask=None):
        """
        image   -- 2-D numpy array of 32-bit floating-point numbers.
        labels  -- 2-D numpy array of integers.
        scale   -- an integer.
        nlevels -- an integer
        """
        if mask is not None:
            labels = labels.copy()
            labels[~mask] = 0
        normalized = normalized_per_object(image, labels)
        quantized = quantize(normalized, nlevels)
        self.P,nlevels = cooccurrence(quantized, labels, scale_i, scale_j)

        self.nobjects = labels.max()
        px = self.P.sum(2) # nobjects x nlevels
        py = self.P.sum(1) # nobjects x nlevels
        #
        # Normalize px and py to deal with roundoff errors in sums
        #
        px = px / np.sum(px,1)[:,np.newaxis]
        py = py / np.sum(py,1)[:,np.newaxis]
        self.nlevels = nlevels
        self.levels = np.arange(nlevels)
        self.rlevels = np.tile(self.levels, (self.nobjects, 1))
        self.levels2 = np.arange(2 * nlevels - 1)
        self.rlevels2 = np.tile(self.levels2, (self.nobjects, 1))
        self.mux = ((self.rlevels + 1) * px).sum(1)
        mux = np.tile(self.mux, (nlevels,1)).transpose()
        self.muy = ((self.rlevels + 1) * py).sum(1)
        muy = np.tile(self.muy, (nlevels,1)).transpose()
        self.sigmax = np.sqrt(((self.rlevels + 1 - mux) ** 2 * px).sum(1))
        self.sigmay = np.sqrt(((self.rlevels + 1 - muy) ** 2 * py).sum(1))
        eps = np.finfo(float).eps
        self.hx = -(px * np.log(px + eps)).sum(1)
        self.hy = -(py * np.log(py + eps)).sum(1)
        pxpy = np.array([np.dot(px[i,:,np.newaxis], py[i,np.newaxis])
                         for i in range(self.nobjects)]).reshape(self.P.shape)
        self.hxy1 = -(self.P * np.log(pxpy + eps)).sum(2).sum(1)
        self.hxy2 = -(pxpy * np.log(pxpy + eps)).sum(2).sum(1)
        self.eps = eps

        self.p_xplusy = np.zeros((self.nobjects, 2 * nlevels - 1))
        self.p_xminusy = np.zeros((self.nobjects, nlevels))
        for x in self.levels:
            for y in self.levels:
                self.p_xplusy[:, x + y] += self.P[:, x, y]
                self.p_xminusy[:, np.abs(x - y)] += self.P[:, x, y]

    # The remaining methods are for computing all the Haralick
    # features.  Each methods returns a vector of length nobjects.

    def H1(self):
        "Angular second moment."
        return(self.P ** 2).sum(2).sum(1)

    def H2(self):
        "Contrast."
        return (self.rlevels ** 2 * self.p_xminusy).sum(1)

    def H3(self):
        "Correlation."
        multiplied = np.dot(self.levels[:, np.newaxis] + 1,
                            self.levels[np.newaxis] + 1)
        repeated = np.tile(multiplied[np.newaxis], (self.nobjects, 1, 1))
        summed = (repeated * self.P).sum(2).sum(1)
        h3 = (summed - self.mux * self.muy) / (self.sigmax * self.sigmay)
        h3[np.isinf(h3)] = 0
        return h3

    def H4(self):
        "Sum of squares: variation."
        return self.sigmax ** 2

    def H5(self):
        "Inverse difference moment."
        t = 1 + toeplitz(self.levels) ** 2
        repeated = np.tile(t[np.newaxis], (self.nobjects, 1, 1))
        return (1.0 / repeated * self.P).sum(2).sum(1)

    def H6(self):
        "Sum average."
        if not hasattr(self, '_H6'):
            self._H6 = ((self.rlevels2 + 2) * self.p_xplusy).sum(1)
        return self._H6

    def H7(self):
        "Sum variance (error in Haralick's original paper here)."
        h6 = np.tile(self.H6(), (self.rlevels2.shape[1], 1)).transpose()
        return (((self.rlevels2 + 2) - h6) ** 2 * self.p_xplusy).sum(1)

    def H8(self):
        "Sum entropy."
        return -(self.p_xplusy * np.log(self.p_xplusy + self.eps)).sum(1)

    def H9(self):
        "Entropy."
        if not hasattr(self, '_H9'):
            self._H9 = -(self.P * np.log(self.P + self.eps)).sum(2).sum(1)
        return self._H9

    def H10(self):
        "Difference variance."
        c = (self.rlevels * self.p_xminusy).sum(1)
        c1 = np.tile(c, (self.nlevels,1)).transpose()
        e = self.rlevels - c1
        return (self.p_xminusy * e ** 2).sum(1)

    def H11(self):
        "Difference entropy."
        return -(self.p_xminusy * np.log(self.p_xminusy + self.eps)).sum(1)

    def H12(self):
        "Information measure of correlation 1."
        maxima = np.vstack((self.hx, self.hy)).max(0)
        return (self.H9() - self.hxy1) / maxima

    def H13(self):
        "Information measure of correlation 2."
        # An imaginary result has been encountered once in the Matlab
        # version.  The reason is unclear.
        return np.sqrt(1 - np.exp(-2 * (self.hxy2 - self.H9())))

    # There is a H14, max correlation coefficient, but we don't currently
    # use it.

    def all(self):
        return [self.H1(), self.H2(), self.H3(), self.H4(), self.H5(),
                self.H6(), self.H7(), self.H8(), self.H9(), self.H10(),
                self.H11(), self.H12(), self.H13()]
