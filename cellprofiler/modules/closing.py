# -*- coding: utf-8 -*-

"""

<strong>Closing</strong> is the erosion of the dilation of an image. It’s used to remove pepper noise.

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
