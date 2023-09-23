"""
MedianFilter
============

**MedianFilter** reduces salt-and-pepper noise in an image while preserving
borders.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============
"""


from cellprofiler_core.image import Image
from cellprofiler_core.module import ImageProcessing
from cellprofiler_core.setting.text import Integer
from cellprofiler_library.modules import medianfilter


class MedianFilter(ImageProcessing):
    category = "Advanced"

    module_name = "MedianFilter"

    variable_revision_number = 1

    def create_settings(self):
        super(MedianFilter, self).create_settings()

        self.window = Integer(
            text="Window",
            value=3,
            minval=0,
            doc="""\
Dimension in each direction for computing the median filter. Use a window with a small size to
remove noise that's small in size. A larger window will remove larger scales of noise at the
risk of blurring other features.
""",
        )

    def settings(self):
        __settings__ = super(MedianFilter, self).settings()

        return __settings__ + [self.window]

    def visible_settings(self):
        __settings__ = super(MedianFilter, self).visible_settings()

        return __settings__ + [self.window]

    def run(self, workspace):

        x_name = self.x_name.value

        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        dimensions = x.dimensions

        x_data = x.pixel_data

        y_data = medianfilter(x_data, self.window.value, mode="constant")

        y = Image(dimensions=dimensions, image=y_data, parent_image=x, convert=False)

        images.add(y_name, y)

        if self.show_window:
            workspace.display_data.x_data = x_data

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = dimensions

