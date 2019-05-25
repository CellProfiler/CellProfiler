# coding=utf-8

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

import cellprofiler.module
import cellprofiler.setting


class MedianFilter(cellprofiler.module.ImageProcessing):
    category = "Advanced"

    module_name = "MedianFilter"

    variable_revision_number = 1

    def create_settings(self):
        super(MedianFilter, self).create_settings()

        self.window = cellprofiler.setting.OddInteger(
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
