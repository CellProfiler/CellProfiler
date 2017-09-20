# coding=utf-8

"""
ConvertObjectsToImage
=====================

**ConvertObjectsToImage** converts a binary image to objects. Connected components of the binary image are assigned to
the same object. This module is useful for identifying objects that can be cleanly distinguished using **Threshold**.
If you wish to distinguish clumped objects, see **Watershed** or the **Identify** modules.

Note that grayscale images provided as input to this module will be converted to binary images. Pixel intensities
below or equal to 50% of the input's full intensity range are assigned to the background (i.e., assigned the value 0).
Pixel intensities above 50% of the input's full intensity range are assigned to the foreground (i.e., assigned the
value 1).

**ConvertObjectsToImage** supports 2D and 3D images.
"""

import skimage
import skimage.measure

import cellprofiler.module


class ConvertImageToObjects(cellprofiler.module.ImageSegmentation):
    category = "Advanced"

    module_name = "ConvertImageToObjects"

    variable_revision_number = 1

    def run(self, workspace):
        self.function = lambda(x_data): skimage.measure.label(skimage.img_as_bool(x_data))

        super(ConvertImageToObjects, self).run(workspace)
