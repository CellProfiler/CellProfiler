# coding=utf-8

"""
Gamma correction is a non-linear operation used to encode and decode luminance
values in images.
"""

import cellprofiler.module
import cellprofiler.setting
import skimage.exposure


class GammaCorrection(cellprofiler.module.ImageProcessing):
    module_name = "GammaCorrection"

    variable_revision_number = 1

    def create_settings(self):
        super(GammaCorrection, self).create_settings()

        self.gamma = cellprofiler.setting.Float(
            doc="""
            A gamma value < 1 is an encoding gamma, and the process of
            encoding with this compressive power-law non-linearity, gamma
            compression, darkens images; conversely a gamma value > 1 is a
            decoding gamma and the application of the expansive power-law
            non-linearity, gamma expansion, brightens images.
            """,
            maxval=100.0,
            minval=0.0,
            text="Gamma",
            value=1.0
        )

    def settings(self):
        __settings__ = super(GammaCorrection, self).settings()

        return __settings__ + [
            self.gamma
        ]

    def visible_settings(self):
        __settings__ = super(GammaCorrection, self).visible_settings()

        return __settings__ + [
            self.gamma
        ]

    def run(self, workspace):
        self.function = skimage.exposure.adjust_gamma

        super(GammaCorrection, self).run(workspace)
