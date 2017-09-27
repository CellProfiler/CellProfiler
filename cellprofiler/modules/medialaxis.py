# coding=utf-8

"""
MedialAxis
==========

**MedialAxis** computes the medial axis of a binary image. A medial axis is a
grayscale rather than binary morphological skeleton where each pixel’s
intensity corresponds to a distance to a boundary.

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
import numpy
import skimage.color
import skimage.morphology


class MedialAxis(cellprofiler.module.ImageProcessing):
    category = "Advanced"

    module_name = "MedialAxis"

    variable_revision_number = 1

    def run(self, workspace):
        x_name = self.x_name.value

        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        x_data = x.pixel_data

        if x.multichannel:
            x_data = skimage.color.rgb2gray(x_data)

        if x.dimensions == 3:
            y_data = numpy.zeros_like(x_data)

            for z, image in enumerate(x_data):
                y_data[z] = skimage.morphology.medial_axis(image)
        else:
            y_data = skimage.morphology.medial_axis(x_data)

        y = cellprofiler.image.Image(
            dimensions=x.dimensions,
            image=y_data,
            parent_image=x
        )

        images.add(y_name, y)

        if self.show_window:
            workspace.display_data.x_data = x_data

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = x.dimensions
