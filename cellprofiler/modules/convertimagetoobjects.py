# coding=utf-8
import skimage
import skimage.measure

import cellprofiler.module
import cellprofiler.setting

HELP_BINARY_IMAGE = """\
This module can also convert a grayscale image to binary before converting it to an object.
Connected components of the binary image are assigned to the same object. This feature is 
useful for identifying objects that can be cleanly distinguished using **Threshold**. 
If you wish to distinguish clumped objects, see **Watershed** or the **Identify** modules.

Note that grayscale images provided as input with this setting will be converted to binary 
images. Pixel intensities below or equal to 50% of the input's full intensity range are 
assigned to the background (i.e., assigned the value 0). Pixel intensities above 50% of 
the input's full intensity range are assigned to the foreground (i.e., assigned the
value 1).
"""

__doc__ = """\
ConvertImageToObjects
=====================

**ConvertImageToObjects** converts an image to objects. This module is useful for importing
a previously segmented or labeled image into CellProfiler, as it will preserve the labels
of an integer-labelled input. 

{HELP_BINARY_IMAGE}

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

""".format(**{
    "HELP_BINARY_IMAGE": HELP_BINARY_IMAGE
})



class ConvertImageToObjects(cellprofiler.module.ImageSegmentation):
    category = "Object Processing"

    module_name = "ConvertImageToObjects"

    variable_revision_number = 1

    def create_settings(self):
        super(ConvertImageToObjects, self).create_settings()

        self.cast_to_bool = cellprofiler.setting.Binary(
            text="Convert to boolean image",
            value=False,
            doc=HELP_BINARY_IMAGE
        )

    def settings(self):
        __settings__ = super(ConvertImageToObjects, self).settings()

        return __settings__ + [
            self.cast_to_bool
        ]

    def visible_settings(self):
        __settings__ = super(ConvertImageToObjects, self).visible_settings()

        return __settings__ + [
            self.cast_to_bool
        ]

    def run(self, workspace):
        self.function = lambda data, cast_to_bool: \
            convert_to_objects(data, cast_to_bool)

        super(ConvertImageToObjects, self).run(workspace)


def convert_to_objects(data, cast_to_bool):
    caster = skimage.img_as_bool if cast_to_bool else skimage.img_as_uint
    return skimage.measure.label(caster(data))
