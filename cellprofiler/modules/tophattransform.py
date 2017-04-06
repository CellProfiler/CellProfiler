# -*- coding: utf-8 -*-

"""

Top-hat transform

"""

import cellprofiler.module
import cellprofiler.setting
import skimage.morphology


class TopHatTransform(cellprofiler.module.ImageProcessing):
    module_name = "TopHatTransform"

    variable_revision_number = 1

    def create_settings(self):
        super(TopHatTransform, self).create_settings()

        self.operation_name = cellprofiler.setting.Choice(
            choices=[
                "Black top-hat transform",
                "White top-hat transform"
            ],
            text="Operation",
            value="Black top-hat transform"
        )

        self.structuring_element = cellprofiler.setting.StructuringElement()

    def settings(self):
        __settings__ = super(TopHatTransform, self).settings()
        
        return __settings__ + [
            self.structuring_element
        ]

    def visible_settings(self):
        __settings__ = super(TopHatTransform, self).visible_settings()

        return __settings__ + [
            self.operation_name,
            self.structuring_element
        ]

    def run(self, workspace):
        if self.operation_name.value == "Black top-hat transform":
            self.function = skimage.morphology.black_tophat
        else:
            self.function = skimage.morphology.white_tophat

        super(TopHatTransform, self).run(workspace)
