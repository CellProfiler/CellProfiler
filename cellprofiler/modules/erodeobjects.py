# coding=utf-8

"""
ErodeObjects
=============

**ErodeObjects** shrinks objects based on the structuring element provided.
This function is similar to the "Shrink" function of **ExpandOrShrinkObjects**,
with two major distinctions-

1. **ErodeObjects** supports 3D objects, unlike **ExpandOrShrinkObjects**.
2. In **ExpandOrShrinkObjects**, a small object will only ever be shrunk down to a
   single pixel. In this module, an object smaller than the structuring element will
   be removed entirely.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

"""

import cellprofiler_core.image
import cellprofiler_core.module
import cellprofiler_core.setting
import cellprofiler.utilities.morphology
from cellprofiler.modules._help import HELP_FOR_STREL


class ErodeObjects(cellprofiler_core.module.ObjectProcessing):
    category = "Advanced"

    module_name = "ErodeObjects"

    variable_revision_number = 1

    def create_settings(self):
        super(ErodeObjects, self).create_settings()

        self.structuring_element = cellprofiler_core.setting.StructuringElement(
            allow_planewise=True, doc=HELP_FOR_STREL
        )

    def settings(self):
        __settings__ = super(ErodeObjects, self).settings()

        return __settings__ + [self.structuring_element]

    def visible_settings(self):
        __settings__ = super(ErodeObjects, self).settings()

        return __settings__ + [self.structuring_element]

    def run(self, workspace):
        self.function = cellprofiler.utilities.morphology.erosion

        super(ErodeObjects, self).run(workspace)

