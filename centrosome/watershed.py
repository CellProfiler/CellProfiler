from __future__ import absolute_import
import skimage.segmentation


def watershed(image, markers, connectivity=None, offset=None, mask=None):
    return skimage.segmentation.watershed(image, markers, connectivity, offset, mask)
