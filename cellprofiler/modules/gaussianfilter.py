# -*- coding: utf-8 -*-

"""

Gaussian filter

"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from future import standard_library
standard_library.install_aliases()
from builtins import *
import cellprofiler.module
import cellprofiler.setting
import skimage.filters


class GaussianFilter(cellprofiler.module.ImageProcessing):
    module_name = "GaussianFilter"

    variable_revision_number = 1

    def create_settings(self):
        super(GaussianFilter, self).create_settings()

        self.sigma = cellprofiler.setting.Integer(
            text="Sigma",
            value=1
        )

    def run(self, workspace):
        self.function = skimage.filters.gaussian

        super(GaussianFilter, self).run(workspace)

    def settings(self):
        __settings__ = super(GaussianFilter, self).settings()

        return __settings__ + [
            self.sigma
        ]

    def visible_settings(self):
        __settings__ = super(GaussianFilter, self).visible_settings()

        return __settings__ + [
            self.sigma
        ]
