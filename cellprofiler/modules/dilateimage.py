# coding=utf-8

"""
DilateImage
===========

**DilateImage** expands bright shapes in an image. See `this tutorial`_ for more information.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

.. _this tutorial: http://scikit-image.org/docs/dev/auto_examples/xx_applications/plot_morphology.html#dilation

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import cellprofiler.utilities.morphology
from cellprofiler.modules._help import HELP_FOR_STREL


class DilateImage(cellprofiler.module.ImageProcessing):
    category = "Advanced"

    module_name = "DilateImage"

    variable_revision_number = 1

    def create_settings(self):
        super(DilateImage, self).create_settings()

        self.structuring_element = cellprofiler.setting.StructuringElement(allow_planewise=True,
                                                                           doc=HELP_FOR_STREL)

    def settings(self):
        __settings__ = super(DilateImage, self).settings()

        return __settings__ + [
            self.structuring_element
        ]

    def visible_settings(self):
        __settings__ = super(DilateImage, self).settings()

        return __settings__ + [
            self.structuring_element
        ]

    def run(self, workspace):
        self.function = cellprofiler.utilities.morphology.dilation

        super(DilateImage, self).run(workspace)
