# -*- coding: utf-8 -*-

"""

<strong>Opening</strong> is the dilation of the erosion of an image. It’s used to remove salt noise.

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import skimage.morphology


class Opening(cellprofiler.module.ImageProcessing):
    category = "Mathematical morphology"

    module_name = "Opening"

    variable_revision_number = 1

    def create_settings(self):
        super(Opening, self).create_settings()

        self.structuring_element = cellprofiler.setting.StructuringElement()

    def settings(self):
        __settings__ = super(Opening, self).settings()

        return __settings__ + [
            self.structuring_element
        ]

    def visible_settings(self):
        __settings__ = super(Opening, self).settings()

        return __settings__ + [
            self.structuring_element
        ]

    def run(self, workspace):
        self.function = skimage.morphology.opening

        super(Opening, self).run(workspace)
