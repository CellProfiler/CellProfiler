# -*- coding: utf-8 -*-

"""

Distance transform

"""

import cellprofiler.module
import cellprofiler.setting
import scipy.ndimage


class DistanceTransform(cellprofiler.module.ImageProcessing):
    module_name = "DistanceTransform"

    variable_revision_number = 1

    def run(self, workspace):
        self.function = scipy.ndimage.distance_transform_edt

        super(DistanceTransform, self).run(workspace)
