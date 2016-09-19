# -*- coding: utf-8 -*-

"""

<strong>Erosion</strong> shrinks shapes in an image.

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import skimage.morphology


class Erosion(cellprofiler.module.Module):
    category = "Mathematical morphology"
    module_name = "Erosion"
    variable_revision_number = 1

    def create_settings(self):
        self.x_name = cellprofiler.setting.ImageNameSubscriber(
            "Input"
        )

        self.y_name = cellprofiler.setting.ImageNameProvider(
            "Output",
            "Erosion"
        )

        self.structuring_element = cellprofiler.setting.StructuringElement()

    def settings(self):
        return [
            self.x_name,
            self.y_name,
            self.structuring_element
        ]

    def visible_settings(self):
        return [
            self.x_name,
            self.y_name,
            self.structuring_element
        ]

    def run(self, workspace):
        x_name = self.x_name.value

        y_name = self.y_name.value

        structuring_element = self.structuring_element.value

        images = workspace.image_set

        x = images.get_image(x_name)

        dimensions = x.dimensions

        x_data = x.pixel_data

        y_data = skimage.morphology.erosion(x_data, structuring_element)

        y = cellprofiler.image.Image(
            image=y_data,
            parent_image=x,
            dimensions=dimensions
        )

        images.add(y_name, y)

        if self.show_window:
            workspace.display_data.x_data = x_data

            workspace.display_data.y_data = y_data
            
            workspace.display_data.dimensions = dimensions

    def display(self, workspace, figure):
        figure.set_subplots((1, 2), dimensions=workspace.display_data.dimensions)

        figure.subplot_imshow(0, 0, workspace.display_data.x_data, dimensions=workspace.display_data.dimensions)

        figure.subplot_imshow(1, 0, workspace.display_data.y_data, dimensions=workspace.display_data.dimensions)
