# coding=utf-8

"""
FillObjects
===========

**FillObjects** fills holes within all objects in an image.

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

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting


class FillObjects(cellprofiler.module.ObjectProcessing):
    category = "Advanced"

    module_name = "FillObjects"

    variable_revision_number = 1

    def create_settings(self):
        super(FillObjects, self).create_settings()

        self.size = cellprofiler.setting.Float(
            text="Minimum hole size",
            value=64.,
            doc="Holes smaller than this diameter will be filled."
        )

        self.slice_wise = cellprofiler.setting.Binary(
            text="Slice wise fill",
            value=False,
            doc="""\
Select "*{YES}*" to fill objects on a per-slice level. 
This will perform the hole filling on each slice of a 
volumetric image, rather than on the image as a whole. 
This may be helpful for removing seed artifacts that 
are the result of segmentation.
**Note**: Slice-wise operations will be considerably slower.
""".format(**{
                "YES": cellprofiler.setting.YES
            })
        )

    def settings(self):
        __settings__ = super(FillObjects, self).settings()

        return __settings__ + [
            self.size,
            self.slice_wise
        ]

    def visible_settings(self):
        __settings__ = super(FillObjects, self).visible_settings()

        return __settings__ + [
            self.size,
            self.slice_wise
        ]

    def run(self, workspace):
        self.function = lambda labels, diameter, slicewise: \
            fill_object_holes(labels, diameter, slicewise)

        super(FillObjects, self).run(workspace)


def _fill_holes(labels, min_obj_size):
    array = labels.copy()
    # Iterate through each label as a mask, fill holes on the mask, and reapply to original image
    for n in numpy.unique(array):
        if n == 0:
            continue

        filled_mask = skimage.morphology.remove_small_holes(array == n, min_obj_size)
        array[filled_mask] = n
    return array


def fill_object_holes(labels, diameter, slicewise):
    radius = diameter / 2.0

    if labels.ndim == 2 or labels.shape[-1] in (3, 4) or slicewise:
        factor = radius ** 2
    else:
        factor = (4.0 / 3.0) * (radius ** 3)

    min_obj_size = numpy.pi * factor

    if not slicewise:
        return _fill_holes(labels, min_obj_size)
    return numpy.array([_fill_holes(x, min_obj_size) for x in labels])

