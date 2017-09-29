# coding=utf-8

"""
MorphologicalSkeleton
=====================

**MorphologicalSkeleton** thins an image into a single-pixel wide skeleton.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           NO
============ ============ ===============

"""

import cellprofiler.image
import cellprofiler.module
import skimage.morphology


class MorphologicalSkeleton(cellprofiler.module.ImageProcessing):
    category = "Advanced"

    module_name = "MorphologicalSkeleton"

    variable_revision_number = 1

    def run(self, workspace):
        x_name = self.x_name.value

        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        dimensions = x.dimensions

        x_data = x.pixel_data

        if x.volumetric:
            y_data = skimage.morphology.skeletonize_3d(x_data)
        else:
            y_data = skimage.morphology.skeletonize(x_data)

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
