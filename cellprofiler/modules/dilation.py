# -*- coding: utf-8 -*-

"""

<strong>Dilation</strong> expands shapes in an image.

"""

import numpy
import skimage.morphology

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting


class Dilation(cellprofiler.module.ImageProcessing):
    category = "Mathematical morphology"

    module_name = "Dilation"

    variable_revision_number = 1

    def create_settings(self):
        super(Dilation, self).create_settings()

        self.structuring_element = cellprofiler.setting.StructuringElement()

    def settings(self):
        __settings__ = super(Dilation, self).settings()

        return __settings__ + [
            self.structuring_element
        ]

    def visible_settings(self):
        __settings__ = super(Dilation, self).settings()

        return __settings__ + [
            self.structuring_element
        ]

    def run(self, workspace):
        x = workspace.image_set.get_image(self.x_name.value)

        if x.pixel_data.dtype == numpy.bool:
            self.function = skimage.morphology.binary_dilation
        else:
            self.function = skimage.morphology.dilation

        super(Dilation, self).run(workspace)
