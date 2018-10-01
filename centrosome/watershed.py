from __future__ import absolute_import
import skimage.morphology


def watershed(image, markers, connectivity=None, offset=None, mask=None):
    return skimage.morphology.watershed(image, markers, connectivity, offset, mask)
