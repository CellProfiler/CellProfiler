"""
GaussianFilter
==============

**GaussianFilter** will blur an image and remove noise. Filtering an
image with a Gaussian filter can be helpful if the foreground signal is
noisy or near the noise floor.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============
"""

import numpy
import skimage.filters
from cellprofiler_core.image import Image
from cellprofiler_core.module import ImageProcessing
from cellprofiler_core.setting.text import Integer
from cellprofiler_library.modules import gaussianfilter

class GaussianFilter(ImageProcessing):
    category = "Advanced"

    module_name = "GaussianFilter"

    variable_revision_number = 1

    def create_settings(self):
        super(GaussianFilter, self).create_settings()

        self.sigma = Integer(
            text="Sigma",
            value=1,
            doc="Standard deviation of the kernel to be used for blurring. Larger sigmas induce more blurring.",
        )

    def run(self, workspace):
        x_name = self.x_name.value

        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        dimensions = x.dimensions

        x_data = x.pixel_data

        sigma = numpy.divide(self.sigma.value, x.spacing) #library function

        y_data = gaussianfilter(x_data, sigma=sigma)

        y = Image(dimensions=dimensions, image=y_data, parent_image=x)

        images.add(y_name, y)

        if self.show_window:
            workspace.display_data.x_data = x_data

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = dimensions

    def settings(self):
        __settings__ = super(GaussianFilter, self).settings()

        return __settings__ + [self.sigma]

    def visible_settings(self):
        __settings__ = super(GaussianFilter, self).visible_settings()

        __settings__ += [self.sigma]

        return __settings__
