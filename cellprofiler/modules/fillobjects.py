# coding=utf-8

"""
FillObjects
===========

**FillObjects** fills holes within all objects in an image

This module works best on integer-labeled images (i.e., the output of **ConvertObjectsToImage**
when the color format is *uint16*).

The output of this module is a labeled image of the same data type as the input.
**FillObjects** can be run *after* any labeling or segmentation module (e.g.,
**ConvertImageToObjects** or **Watershed**). Labels are preserved and, where possible, holes
entirely within the boundary of labeled objects are filled with the surrounding object number.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

"""

import numpy
import skimage.morphology
import scipy.ndimage.morphology

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting


class FillObjects(cellprofiler.module.ImageProcessing):
    category = "Advanced"

    module_name = "FillObjects"

    variable_revision_number = 1

    def create_settings(self):
        super(FillObjects, self).create_settings()

        self.size = cellprofiler.setting.Float(
            text="Size",
            value=1.0,
            doc="Holes smaller than this diameter will be filled."
        )

    def settings(self):
        __settings__ = super(FillObjects, self).settings()

        return __settings__ + [
            self.size
        ]

    def visible_settings(self):
        __settings__ = super(FillObjects, self).visible_settings()

        return __settings__ + [
            self.size
        ]

    def run(self, workspace):
        self.function = lambda image, diameter: remove_objects_and_fill_holes(image, diameter)

        super(FillObjects, self).run(workspace)


def remove_objects_and_fill_holes(image, diameter):
    radius = diameter / 2.0

    if image.ndim == 2 or image.shape[-1] in (3, 4):
        factor = radius ** 2
    else:
        factor = (4.0 / 3.0) * (radius ** 3)

    min_obj_size = numpy.pi * factor

    removed = skimage.morphology.remove_small_objects(image, min_obj_size)

    # Iterate through each label as a mask, fill holes on the mask, and reapply to original image
    for n in numpy.unique(removed):
        if n == 0:
            continue

        filled_mask = scipy.ndimage.morphology.binary_fill_holes(removed == n)
        removed[filled_mask] = n

    return removed
