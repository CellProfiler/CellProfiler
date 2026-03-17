"""
RemoveHoles
===========

**RemoveHoles** fills holes smaller than the specified diameter.

This module works best on binary and integer-labeled images (i.e., the output of
**ConvertObjectsToImage** when the color format is *uint16*). Grayscale and multichannel
image data is converted to binary by setting values below 50% of the data range to 0 and
the other 50% of values to 1.

The output of this module is a binary image, regardless of the input data type. It is
recommended that **RemoveHoles** is run before any labeling or segmentation module (e.g.,
**ConvertImageToObjects** or **Watershed**).

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============
"""

import numpy
import skimage.morphology
from cellprofiler_core.module import ImageProcessing
from cellprofiler_core.setting.text import Float
from cellprofiler_library.modules._removeholes import remove_holes


class RemoveHoles(ImageProcessing):
    category = "Advanced"

    module_name = "RemoveHoles"

    variable_revision_number = 1

    def create_settings(self):
        super(RemoveHoles, self).create_settings()

        self.size = Float(
            text="Size of holes to fill",
            value=1.0,
            doc="Holes smaller than this diameter will be filled. Note that for 3D\
            images this module operates volumetrically so diameters should be given in voxels",
        )

    def settings(self):
        __settings__ = super(RemoveHoles, self).settings()

        return __settings__ + [self.size]

    def visible_settings(self):
        __settings__ = super(RemoveHoles, self).visible_settings()

        return __settings__ + [self.size]

    def run(self, workspace):
        self.function = remove_holes

        super(RemoveHoles, self).run(workspace)

