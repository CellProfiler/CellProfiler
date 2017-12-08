# coding=utf-8

"""
DilateObjects
=============

**DilateObjects** removes objects smaller or larger than the specified diameter.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

"""

import cellprofiler.object
import cellprofiler.module
import cellprofiler.setting
import cellprofiler.utilities.morphology


class DilateObjects(cellprofiler.module.ObjectProcessing):
    category = "Advanced"

    module_name = "DilateObjects"

    variable_revision_number = 1

    def create_settings(self):
        super(DilateObjects, self).create_settings()

        self.structuring_element = cellprofiler.setting.StructuringElement(allow_planewise=True)

    def settings(self):
        __settings__ = super(DilateObjects, self).settings()

        return __settings__ + [
            self.structuring_element
        ]

    def visible_settings(self):
        __settings__ = super(DilateObjects, self).visible_settings()

        return __settings__ + [
            self.structuring_element
        ]

    def run(self, workspace):
        self.function = cellprofiler.utilities.morphology.dilation

        super(DilateObjects, self).run(workspace)
