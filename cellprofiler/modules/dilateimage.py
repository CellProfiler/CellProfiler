"""
DilateImage
===========

**DilateImage** expands bright shapes in an image. See `this tutorial <https://scikit-image.org/docs/dev/auto_examples/applications/plot_morphology.html#dilation>`__ for more information.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

"""

from cellprofiler_core.module import ImageProcessing
from cellprofiler_core.setting import StructuringElement

import cellprofiler.utilities.morphology
from ._help import HELP_FOR_STREL


class DilateImage(ImageProcessing):
    category = "Advanced"

    module_name = "DilateImage"

    variable_revision_number = 1

    def create_settings(self):
        super(DilateImage, self).create_settings()

        self.structuring_element = StructuringElement(
            allow_planewise=True, doc=HELP_FOR_STREL
        )

    def settings(self):
        __settings__ = super(DilateImage, self).settings()

        return __settings__ + [self.structuring_element]

    def visible_settings(self):
        __settings__ = super(DilateImage, self).settings()

        return __settings__ + [self.structuring_element]

    def run(self, workspace):
        self.function = cellprofiler.utilities.morphology.dilation

        super(DilateImage, self).run(workspace)
