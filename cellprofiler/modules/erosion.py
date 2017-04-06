# -*- coding: utf-8 -*-

"""

<strong>Erosion</strong> shrinks shapes in an image.

"""

import numpy
import skimage.morphology

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting


class Erosion(cellprofiler.module.ImageProcessing):
    category = "Mathematical morphology"

    module_name = "Erosion"

    variable_revision_number = 1

    def create_settings(self):
        super(Erosion, self).create_settings()

        self.structuring_element = cellprofiler.setting.StructuringElement()

    def settings(self):
        __settings__ = super(Erosion, self).settings()

        return __settings__ + [
            self.structuring_element
        ]

    def visible_settings(self):
        __settings__ = super(Erosion, self).settings()

        return __settings__ + [
            self.structuring_element
        ]

    def run(self, workspace):
        x = workspace.image_set.get_image(self.x_name.value)

        if x.pixel_data.dtype == numpy.bool:
            self.function = skimage.morphology.binary_erosion
        else:
            self.function = skimage.morphology.erosion

        super(Erosion, self).run(workspace)
