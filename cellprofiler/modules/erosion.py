# -*- coding: utf-8 -*-

"""

<strong>Erosion</strong> shrinks shapes in an image.

"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from future import standard_library
standard_library.install_aliases()
from builtins import *
import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import skimage.morphology


class Erosion(cellprofiler.module.ImageProcessing):
    category = "Mathematical morphology"

    module_name = "Erosion"

    variable_revision_number = 1

    def create_settings(self):
        super(Erosion, self).create_settings()

        self.structuring_element = cellprofiler.setting.StructuringElement()

    def settings(self):
        __settings__ = super(Erosion, self).settings()

        return __settings__ + [
            self.structuring_element
        ]

    def visible_settings(self):
        __settings__ = super(Erosion, self).settings()

        return __settings__ + [
            self.structuring_element
        ]

    def run(self, workspace):
        self.function = skimage.morphology.erosion

        super(Erosion, self).run(workspace)
