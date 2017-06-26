# -*- coding: utf-8 -*-

"""

Clear objects connected to the label image border.

"""

import skimage.measure
import skimage.segmentation

import cellprofiler.module
import cellprofiler.object
import cellprofiler.setting


class ClearBorder(cellprofiler.module.ImageSegmentation):
    module_name = "ClearBorder"

    variable_revision_number = 1

    def create_settings(self):
        super(ClearBorder, self).create_settings()

        self.buffer_size = cellprofiler.setting.Integer(
            doc="""
            The width of the border examined. By default, only objects that 
            touch the outside of the image are removed.
            """,
            minval=0,
            text="Buffer size",
            value=0
        )

    def settings(self):
        __settings__ = super(ClearBorder, self).settings()

        return __settings__ + [
            self.buffer_size
        ]

    def visible_settings(self):
        __settings__ = super(ClearBorder, self).visible_settings()

        return __settings__ + [
            self.buffer_size
        ]

    def run(self, workspace):
        x_name = self.x_name.value

        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        dimensions = x.dimensions

        x_data = x.pixel_data

        y_data = skimage.segmentation.clear_border(x_data)

        y_data = skimage.measure.label(y_data)

        objects = cellprofiler.object.Objects()

        objects.segmented = y_data

        objects.parent_image = x

        workspace.object_set.add_objects(objects, y_name)

        if self.show_window:
            workspace.display_data.x_data = x.pixel_data

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = dimensions
