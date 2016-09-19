# -*- coding: utf-8 -*-

"""

<strong>Top-hat transform</strong>

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import skimage.morphology


class TopHatTransform(cellprofiler.module.Module):
    category = "Mathematical morphology"
    module_name = "Top-hat transform"
    variable_revision_number = 1

    def create_settings(self):
        self.x_name = cellprofiler.setting.ImageNameSubscriber(
            "Input"
        )

        self.y_name = cellprofiler.setting.ImageNameProvider(
            "Output",
            "TopHatTransform"
        )

        self.operation_name = cellprofiler.setting.Choice(
            u"Operation",
            [
                "Black",
                "White"
            ]
        )

        self.structuring_element = cellprofiler.setting.StructuringElement()

    def settings(self):
        return [
            self.x_name,
            self.y_name,
            self.operation_name,
            self.structuring_element
        ]

    def visible_settings(self):
        return [
            self.x_name,
            self.y_name,
            self.operation_name,
            self.structuring_element
        ]

    def run(self, workspace):
        x_name = self.x_name.value

        y_name = self.y_name.value

        operation_name = self.operation_name.value

        structuring_element = self.structuring_element.value

        images = workspace.image_set

        x = images.get_image(x_name)

        dimensions = x.dimensions

        x_data = x.pixel_data

        if operation_name == "Black":
            y_data = skimage.morphology.black_tophat(x_data, structuring_element)
        else:
            y_data = skimage.morphology.white_tophat(x_data, structuring_element)

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
        figure.set_subplots((2, 1), dimensions=workspace.display_data.dimensions)

        figure.subplot_imshow(0, 0, workspace.display_data.x_data, colormap="gray", dimensions=workspace.display_data.dimensions)

        figure.subplot_imshow(1, 0, workspace.display_data.y_data, colormap="gray", dimensions=workspace.display_data.dimensions)
