from __future__ import absolute_import
import numpy
import scipy.ndimage.filters
from six.moves import range


def kirsch(image):
    convolution_mask = [5, -3, -3, -3, -3, -3, 5, 5]

    derivatives = numpy.zeros(image.shape)

    kernel = numpy.zeros((3, 3), image.dtype)
    kindex = numpy.array([[0,  1, 2],
                          [7, -1, 3],
                          [6,  5, 4]])
    for _ in range(len(convolution_mask)):
        kernel[kindex >= 0] = numpy.array(convolution_mask)[kindex[kindex >= 0]]
        derivatives = numpy.maximum(
            derivatives, scipy.ndimage.filters.convolve(image, kernel))

        convolution_mask = convolution_mask[-1:] + convolution_mask[:-1]

    return derivatives
