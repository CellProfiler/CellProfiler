# -*- coding: utf-8 -*-

"""

<strong>Opening</strong> is the dilation of the erosion of an image. It’s used to remove salt noise.

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
