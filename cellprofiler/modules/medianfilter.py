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

import scipy.signal
from cellprofiler_core.module import ImageProcessing
from cellprofiler_core.setting.text import OddInteger


class MedianFilter(ImageProcessing):
    category = "Advanced"

    module_name = "MedianFilter"

    variable_revision_number = 1

    def create_settings(self):
        super(MedianFilter, self).create_settings()

        self.window = OddInteger(
            text="Window",
            value=3,
            minval=0,
            doc="""\
Dimension in each direction for computing the median filter. Must be odd. Use a window with a small size to
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
        self.function = scipy.signal.medfilt

        super(MedianFilter, self).run(workspace)
