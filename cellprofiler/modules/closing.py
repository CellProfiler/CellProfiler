# -*- coding: utf-8 -*-

"""

<strong>Closing</strong> is the erosion of the dilation of an image. It’s used to remove pepper noise.

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import skimage.morphology


class Closing(cellprofiler.module.ImageProcessing):
    category = "Mathematical morphology"

    module_name = "Closing"

    variable_revision_number = 1

    def create_settings(self):
        super(Closing, self).create_settings()

        self.structuring_element = cellprofiler.setting.StructuringElement()

    def settings(self):
        __settings__ = super(Closing, self).settings()

        return __settings__ + [
            self.structuring_element
        ]

    def visible_settings(self):
        __settings__ = super(Closing, self).settings()

        return __settings__ + [
            self.structuring_element
        ]

    def run(self, workspace):
        self.function = skimage.morphology.closing

        super(Closing, self).run(workspace)
