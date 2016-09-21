# -*- coding: utf-8 -*-

"""

<strong>Gamma correction</strong>

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import skimage.exposure


class GammaCorrection(cellprofiler.module.ImageProcessing):
    module_name = "GammaCorrection"

    variable_revision_number = 1

    def create_settings(self):
        super(GammaCorrection, self).create_settings()

        self.gamma = cellprofiler.setting.Float(
            "Gamma",
            1,
            minval=0.0,
            maxval=100.0,
        )

        self.gain = cellprofiler.setting.Float(
            "Gain",
            1,
            minval=1.0,
            maxval=100,
        )

    def settings(self):
        __settings__ = super(GammaCorrection, self).settings()

        return __settings__ + [
            self.gamma,
            self.gain
        ]

    def run(self, workspace):
        self.function = skimage.exposure.adjust_gamma

        super(GammaCorrection, self).run(workspace)
