"""
FillObjects
===========

**FillObjects** fills holes within all objects in an image.

**FillObjects** can be run *after* any labeling or segmentation module (e.g.,
**ConvertImageToObjects** or **Watershed**). Labels are preserved and, where possible, holes
entirely within the boundary of labeled objects are filled with the surrounding object number.

**FillObjects** can also be optionally run on a "per-plane" basis working with volumetric data.
Holes will be filled for each XY plane, rather than on the whole volume.

Alternatively, objects can be filled on the basis of a convex hull.  
This is the smallest convex polygon that surrounds all pixels in the object.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

"""

import numpy
import skimage.morphology
import skimage.measure
from cellprofiler_core.module.image_segmentation import ObjectProcessing
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.text import Float

MODE_HOLES = "Holes"
MODE_CHULL = "Convex hull"

class FillObjects(ObjectProcessing):
    category = "Advanced"

    module_name = "FillObjects"

    variable_revision_number = 2

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

        self.mode = Choice(
            "Filling method",
            [MODE_HOLES, MODE_CHULL],
            value=MODE_HOLES,
            doc=f"""\
Choose the mode for hole filling.

In {MODE_HOLES} mode, the module will search for and fill holes entirely enclosed by
each object. Size of the holes to be removed can be controlled. 

In {MODE_CHULL} mode, the module will apply the convex hull of each object to fill 
missing pixels. This can be useful when round objects have partial holes that are 
not entirely enclosed.

Note: Convex hulls for each object are applied sequentially and may overlap. This means 
that touching objects may not be perfectly convex if there was a region of overlap. 
"""           
        )

    def settings(self):
        __settings__ = super(FillObjects, self).settings()

        return __settings__ + [self.size, self.planewise, self.mode]

    def visible_settings(self):
        __settings__ = super(FillObjects, self).visible_settings()
        __settings__ += [self.mode]
        if self.mode.value == MODE_HOLES:
            __settings__ += [self.size, self.planewise]
        return __settings__

    def run(self, workspace):
        if self.mode.value == MODE_CHULL:
            self.function = lambda labels, d, p, m: fill_convex_hulls(labels)
        else:
            self.function = lambda labels, diameter, planewise, mode: fill_object_holes(
                labels, diameter, planewise
            )

        super(FillObjects, self).run(workspace)

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 1:
            setting_values.append(MODE_HOLES)
            variable_revision_number = 2
        return setting_values, variable_revision_number


def fill_convex_hulls(labels):
    data = skimage.measure.regionprops(labels)
    output = numpy.zeros_like(labels)
    for prop in data:
        label = prop['label']
        bbox = prop['bbox']
        cmask = prop['convex_image']
        if len(bbox) <= 4:
            output[bbox[0]:bbox[2], bbox[1]:bbox[3]][cmask] = label
        else:
            output[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]: bbox[5]][cmask] = label
    return output


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
