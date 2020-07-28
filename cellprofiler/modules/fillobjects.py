"""
FillObjects
===========

**FillObjects** fills holes within all objects in an image.

**FillObjects** can be run *after* any labeling or segmentation module (e.g.,
**ConvertImageToObjects** or **Watershed**). Labels are preserved and, where possible, holes
entirely within the boundary of labeled objects are filled with the surrounding object number.

**FillObjects** can also be optionally run on a "per-plane" basis working with volumetric data.
Holes will be filled for each XY plane, rather than on the whole volume.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

"""

import numpy
import skimage.morphology
from cellprofiler_core.module.image_segmentation import ObjectProcessing
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting.text import Float


class FillObjects(ObjectProcessing):
    category = "Advanced"

    module_name = "FillObjects"

    variable_revision_number = 1

    def create_settings(self):
        super(FillObjects, self).create_settings()

        self.size = Float(
            text="Minimum hole size",
            value=64.0,
            doc="Holes smaller than this diameter will be filled.",
        )

        self.planewise = Binary(
            text="Planewise fill",
            value=False,
            doc="""\
Select "*{YES}*" to fill objects on a per-plane level. 
This will perform the hole filling on each plane of a 
volumetric image, rather than on the image as a whole. 
This may be helpful for removing seed artifacts that 
are the result of segmentation.
**Note**: Planewise operations will be considerably slower.
""".format(
                **{"YES": "Yes"}
            ),
        )

    def settings(self):
        __settings__ = super(FillObjects, self).settings()

        return __settings__ + [self.size, self.planewise]

    def visible_settings(self):
        __settings__ = super(FillObjects, self).visible_settings()

        return __settings__ + [self.size, self.planewise]

    def run(self, workspace):
        self.function = lambda labels, diameter, planewise: fill_object_holes(
            labels, diameter, planewise
        )

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


def fill_object_holes(labels, diameter, planewise):
    radius = diameter / 2.0

    if labels.ndim == 2 or labels.shape[-1] in (3, 4) or planewise:
        factor = radius ** 2
    else:
        factor = (4.0 / 3.0) * (radius ** 3)

    min_obj_size = numpy.pi * factor

    # Only operate planewise if image is 3D and planewise requested
    if planewise and labels.ndim != 2 and labels.shape[-1] not in (3, 4):
        return numpy.array([_fill_holes(x, min_obj_size) for x in labels])
    return _fill_holes(labels, min_obj_size)
