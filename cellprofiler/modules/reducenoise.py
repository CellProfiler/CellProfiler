# coding=utf-8

"""
ReduceNoise
===========

**ReduceNoise** performs non-local means noise reduction. Instead of only
using a neighborhood of pixels around a central pixel for denoising, such
as in **GaussianFilter**, multiple neighborhoods are pooled together. The
neighborhood pool is determined by scanning the image for regions similar to
the area around the central pixel using a correlation metric and a cutoff value.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============
"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import skimage.restoration
import skimage.util


class ReduceNoise(cellprofiler.module.ImageProcessing):
    category = "Advanced"

    module_name = "ReduceNoise"

    variable_revision_number = 1

    def create_settings(self):
        super(ReduceNoise, self).create_settings()

        self.size = cellprofiler.setting.Integer(
            doc="Size is the size of patches used for noise reduction.",
            text="Size",
            value=7
        )

        self.distance = cellprofiler.setting.Integer(
            doc="Distance is the maximum distance where to search patches used for noise reduction.",
            text="Distance",
            value=11
        )

        self.cutoff_distance = cellprofiler.setting.Float(
            doc="""\
Cut-off distance is the permissiveness in accepting patches. Increasing the cut-off distance increases
the smoothness of the image. Likewise, decreasing the cut-off distance decreases the smoothness of the
image.
            """,
            text="Cut-off distance",
            value=0.1
        )

    def settings(self):
        __settings__ = super(ReduceNoise, self).settings()

        return __settings__ + [
            self.size,
            self.distance,
            self.cutoff_distance
        ]

    def visible_settings(self):
        __settings__ = super(ReduceNoise, self).visible_settings()

        return __settings__ + [
            self.size,
            self.distance,
            self.cutoff_distance
        ]

    def run(self, workspace):
        x_name = self.x_name.value

        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        dimensions = x.dimensions

        x_data = x.pixel_data

        y_data = skimage.restoration.denoise_nl_means(
            fast_mode=True,
            h=self.cutoff_distance.value,
            image=x_data,
            multichannel=x.multichannel,
            patch_distance=self.distance.value,
            patch_size=self.size.value
        )

        y = cellprofiler.image.Image(
            dimensions=dimensions,
            image=y_data,
            parent_image=x
        )

        images.add(y_name, y)

        if self.show_window:
            workspace.display_data.x_data = x_data

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = dimensions
