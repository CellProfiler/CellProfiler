"""
MorphologicalSkeleton
=====================

**MorphologicalSkeleton** thins an image into a single-pixel wide skeleton. See `this tutorial <https://scikit-image.org/docs/0.14.x/auto_examples/xx_applications/plot_morphology.html#skeletonize>`__ for more information.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

"""

import skimage.morphology
from cellprofiler_core.image import Image
from cellprofiler_core.module import ImageProcessing
from cellprofiler_library.modules import morphologicalskeleton

class MorphologicalSkeleton(ImageProcessing):
    category = "Advanced"

    module_name = "MorphologicalSkeleton"

    variable_revision_number = 1

    def volumetric(self):
        return True

    def run(self, workspace):
        x_name = self.x_name.value

        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        dimensions = x.dimensions

        x_data = x.pixel_data

        y_data = morphologicalskeleton(x_data, x.volumetric)

        y = Image(dimensions=dimensions, image=y_data, parent_image=x)

        images.add(y_name, y)

        if self.show_window:
            workspace.display_data.x_data = x_data

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = dimensions
