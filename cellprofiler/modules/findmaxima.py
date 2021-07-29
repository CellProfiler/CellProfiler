"""
FindMaxima
==========

**FindMaxima** isolates local peaks of high intensity from an image.

The returned image will feature single pixels at each position where
a peak of intensity was found in the input image.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============
"""

from skimage.feature import peak_local_max

from cellprofiler_core.image import Image
from cellprofiler_core.module import ImageProcessing
from cellprofiler_core.setting.text import Integer, Float


class FindMaxima(ImageProcessing):
    category = "Advanced"

    module_name = "FindMaxima"

    variable_revision_number = 1

    def create_settings(self):
        super(FindMaxima, self).create_settings()

        self.min_distance = Integer(
            text="Minimum distance between maxima",
            value=5,
            minval=0,
            doc="""Choose the minimum distance between accepted local maxima"""
        )

        self.min_intensity = Float(
            "Specify the minimum intensity required for a pixel to be considered as a peak",
            0,
            minval=0,
            maxval=99,
            doc="""\
Intensity peaks below this threshold value will be excluded. Use this to ensure that your local 
maxima are within objects of interest."""

        )

    def settings(self):
        __settings__ = super(FindMaxima, self).settings()

        return __settings__ + [self.min_distance, self.min_intensity]

    def visible_settings(self):
        __settings__ = super(FindMaxima, self).visible_settings()

        return __settings__ + [self.min_distance, self.min_intensity]

    def run(self, workspace):

        x_name = self.x_name.value

        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        dimensions = x.dimensions

        x_data = x.pixel_data

        y_data = peak_local_max(
            x_data,
            min_distance=self.min_distance.value,
            threshold_abs=None if self.min_intensity.value == 0 else self.min_intensity.value,
            indices=False
        )

        y = Image(dimensions=dimensions, image=y_data, parent_image=x, convert=False)

        images.add(y_name, y)

        if self.show_window:
            workspace.display_data.x_data = x_data

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = dimensions

