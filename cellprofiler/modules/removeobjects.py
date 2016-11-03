# -*- coding: utf-8 -*-

"""

Remove objects

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import skimage.morphology


class RemoveObjects(cellprofiler.module.ImageProcessing):
    category = "Mathematical morphology"

    module_name = "Remove objects"

    variable_revision_number = 1

    def create_settings(self):
        super(RemoveObjects, self).create_settings()

        self.size = cellprofiler.setting.Float(
            text="Size",
            value=1.0
        )

    def settings(self):
        __settings__ = super(RemoveObjects, self).settings()

        return __settings__ + [
            self.size
        ]

    def visible_settings(self):
        __settings__ = super(RemoveObjects, self).visible_settings()

        return __settings__ + [
            self.size
        ]

    def run(self, workspace):
        self.function = skimage.morphology.remove_small_objects

        super(RemoveObjects, self).run(workspace)
