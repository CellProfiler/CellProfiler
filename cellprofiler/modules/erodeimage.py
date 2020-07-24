# coding=utf-8

"""
ErodeImage
=======

**ErodeImage** shrinks bright shapes in an image. See `this tutorial`_ for more information.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

.. _this tutorial: https://scikit-image.org/docs/stable/auto_examples/applications/plot_morphology.html#erosion

"""

import cellprofiler_core.image
import cellprofiler_core.module
import cellprofiler_core.setting
import cellprofiler.utilities.morphology
from cellprofiler.modules._help import HELP_FOR_STREL


class ErodeImage(cellprofiler_core.module.ImageProcessing):
    category = "Advanced"

    module_name = "ErodeImage"

    variable_revision_number = 1

    def create_settings(self):
        super(ErodeImage, self).create_settings()

        self.structuring_element = cellprofiler_core.setting.StructuringElement(
            allow_planewise=True, doc=HELP_FOR_STREL
        )

    def settings(self):
        __settings__ = super(ErodeImage, self).settings()

        return __settings__ + [self.structuring_element]

    def visible_settings(self):
        __settings__ = super(ErodeImage, self).settings()

        return __settings__ + [self.structuring_element]

    def run(self, workspace):
        self.function = cellprofiler.utilities.morphology.erosion

        super(ErodeImage, self).run(workspace)
