import logging
import numpy as np
import math
import sys

from struct import unpack
from zlib import decompress
from StringIO import StringIO
from numpy import fromstring, uint8, uint16
from cPickle import dump, Unpickler

logger = logging.getLogger(__name__)


def crop_image(image, crop_mask, crop_internal=False):
    """Crop an image to the size of the nonzero portion of a crop mask"""
    i_histogram = crop_mask.sum(axis=1)
    i_cumsum = np.cumsum(i_histogram != 0)
    j_histogram = crop_mask.sum(axis=0)
    j_cumsum = np.cumsum(j_histogram != 0)
    if i_cumsum[-1] == 0:
        # The whole image is cropped away
        return np.zeros((0, 0), dtype=image.dtype)
    if crop_internal:
        #
        # Make up sequences of rows and columns to keep
        #
        i_keep = np.argwhere(i_histogram > 0)
        j_keep = np.argwhere(j_histogram > 0)
        #
        # Then slice the array by I, then by J to get what's not blank
        #
        return image[i_keep.flatten(), :][:, j_keep.flatten()].copy()
    else:
        #
        # The first non-blank row and column are where the cumsum is 1
        # The last are at the first where the cumsum is it's max (meaning
        # what came after was all zeros and added nothing)
        #
        i_first = np.argwhere(i_cumsum == 1)[0]
        i_last = np.argwhere(i_cumsum == i_cumsum.max())[0]
        i_end = i_last + 1
        j_first = np.argwhere(j_cumsum == 1)[0]
        j_last = np.argwhere(j_cumsum == j_cumsum.max())[0]
        j_end = j_last + 1
        if image.ndim == 3:
            return image[i_first:i_end, j_first:j_end, :].copy()
        return image[i_first:i_end, j_first:j_end].copy()


def check_consistency(image, mask):
    """Check that the image, mask and labels arrays have the same shape and that the arrays are of the right dtype"""
    assert (image is None) or (len(image.shape) in (2, 3)), "Image must have 2 or 3 dimensions"
    assert (mask is None) or (len(mask.shape) == 2), "Mask must have 2 dimensions"
    assert (image is None) or (mask is None) or (image.shape[:2] == mask.shape), "Image and mask sizes don't match"
    assert (mask is None) or (mask.dtype.type is np.bool_), "Mask must be boolean, was %s" % (repr(mask.dtype.type))


def make_dictionary_key(key):
    '''Make a dictionary into a stable key for another dictionary'''
    return u", ".join([u":".join([unicode(y) for y in x])
                       for x in sorted(key.iteritems())])


def readc01(fname):
    '''Read a Cellomics file into an array

    fname - the name of the file
    '''

    def readint(f):
        return unpack("<l", f.read(4))[0]

    def readshort(f):
        return unpack("<h", f.read(2))[0]

    f = open(fname, "rb")

    # verify it's a c01 format, and skip the first four bytes
    assert readint(f) == 16 << 24

    # decompress
    g = StringIO(decompress(f.read()))

    # skip four bytes
    g.seek(4, 1)

    x = readint(g)
    y = readint(g)

    nplanes = readshort(g)
    nbits = readshort(g)

    compression = readint(g)
    assert compression == 0, "can't read compressed pixel data"

    # skip 4 bytes
    g.seek(4, 1)

    pixelwidth = readint(g)
    pixelheight = readint(g)
    colors = readint(g)
    colors_important = readint(g)

    # skip 12 bytes
    g.seek(12, 1)

    data = fromstring(g.read(), uint16 if nbits == 16 else uint8, x * y)
    return data.reshape(x, y).T
