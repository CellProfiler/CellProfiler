# coding=utf-8

"""
<strong>Label</strong>
"""

import mahotas
import numpy
import scipy.ndimage
import skimage.color
import skimage.feature
import skimage.measure
import skimage.morphology
import skimage.transform
import skimage.segmentation

import cellprofiler.image
import cellprofiler.module
import cellprofiler.object
import cellprofiler.setting


class Label(cellprofiler.module.ImageSegmentation):
    module_name = "Advanced"

    variable_revision_number = 1

    def run(self, workspace):
        x_name = self.x_name.value

        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        dimensions = x.dimensions

        x_data = x.pixel_data

        y_data = skimage.measure.label(skimage.img_as_int(x_data))

        objects = cellprofiler.object.Objects()

        objects.segmented = y_data

        objects.parent_image = x

        workspace.object_set.add_objects(objects, y_name)

        self.add_measurements(workspace)

        if self.show_window:
            workspace.display_data.x_data = x.pixel_data

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = dimensions
