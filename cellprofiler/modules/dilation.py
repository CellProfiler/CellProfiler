# -*- coding: utf-8 -*-

"""

<strong>Dilation</strong> expands shapes in an image.

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import skimage.morphology


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
        self.function = skimage.morphology.dilation

        super(Dilation, self).run(workspace)
