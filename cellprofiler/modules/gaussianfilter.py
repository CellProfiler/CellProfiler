# -*- coding: utf-8 -*-

"""

Gaussian filter

"""

import cellprofiler.module
import cellprofiler.setting
import skimage.filters


class GaussianFilter(cellprofiler.module.ImageProcessing):
    module_name = "GaussianFilter"

    variable_revision_number = 1

    def run(self, workspace):
        self.function = skimage.filters.gaussian

        super(GaussianFilter, self).run(workspace)
