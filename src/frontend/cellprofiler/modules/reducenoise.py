"""
ReduceNoise
===========

**ReduceNoise** performs non-local means noise reduction. Instead of only
using a neighborhood of pixels around a central pixel for denoising, such
as in **GaussianFilter**, multiple neighborhoods are pooled together. The
neighborhood pool is determined by scanning the image for regions similar to
the area around the central pixel using a correlation metric and a cutoff value.
See `this tutorial <http://scikit-image.org/docs/dev/auto_examples/filters/plot_nonlocal_means.html>`__ for more information.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

"""

from cellprofiler_core.image import Image
from cellprofiler_core.module import ImageProcessing
from cellprofiler_core.setting.text import Integer, Float
from cellprofiler_library.modules import reducenoise


class ReduceNoise(ImageProcessing):
    category = "Advanced"

    module_name = "ReduceNoise"

    variable_revision_number = 1

    def create_settings(self):
        super(ReduceNoise, self).create_settings()

        self.size = Integer(
            text="Size", value=7, doc="Size of the patches to use for noise reduction."
        )

        self.distance = Integer(
            text="Distance",
            value=11,
            doc="Maximal distance in pixels to search for patches to use for denoising.",
        )

        self.cutoff_distance = Float(
            text="Cut-off distance",
            value=0.1,
            doc="""\
The permissiveness in accepting patches. Increasing the cut-off distance increases
the smoothness of the image. Likewise, decreasing the cut-off distance decreases the smoothness of the
image.
            """,
        )

    def settings(self):
        __settings__ = super(ReduceNoise, self).settings()

        return __settings__ + [self.size, self.distance, self.cutoff_distance]

    def visible_settings(self):
        __settings__ = super(ReduceNoise, self).visible_settings()

        return __settings__ + [self.size, self.distance, self.cutoff_distance]

    def run(self, workspace):
        x_name = self.x_name.value

        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        dimensions = x.dimensions

        x_data = x.pixel_data

        y_data = reducenoise(
            image=x_data,
            patch_distance=self.distance.value,
            patch_size=self.size.value,
            cutoff_distance=self.cutoff_distance.value,
            channel_axis=2 if x.multichannel else None,
        )

        y = Image(dimensions=dimensions, image=y_data, parent_image=x)

        images.add(y_name, y)

        if self.show_window:
            workspace.display_data.x_data = x_data

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = dimensions
