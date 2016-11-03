# -*- coding: utf-8 -*-

"""

Remove holes

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import skimage.morphology


class RemoveHoles(cellprofiler.module.ImageProcessing):
    category = "Mathematical morphology"

    module_name = "Remove holes"

    variable_revision_number = 1

    def create_settings(self):
        super(RemoveHoles, self).create_settings()

        self.size = cellprofiler.setting.Float(
            text="Size",
            value=1.0
        )

    def settings(self):
        __settings__ = super(RemoveHoles, self).settings()

        return __settings__ + [
            self.size
        ]

    def visible_settings(self):
        __settings__ = super(RemoveHoles, self).visible_settings()

        return __settings__ + [
            self.size
        ]

    def run(self, workspace):
        self.function = skimage.morphology.remove_small_holes

        super(RemoveHoles, self).run(workspace)
