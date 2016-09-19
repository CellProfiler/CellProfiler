# -*- coding: utf-8 -*-

"""

<strong>Distance transform</strong>

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import scipy.ndimage


class DistanceTransform(cellprofiler.module.Module):
    category = "Image processing"
    module_name = "Distance transform"
    variable_revision_number = 1

    def create_settings(self):
        self.x_name = cellprofiler.setting.ImageNameSubscriber(
            "Input"
        )

        self.y_name = cellprofiler.setting.ImageNameProvider(
            "Output",
            "DistanceTransform"
        )

    def settings(self):
        return [
            self.x_name,
            self.y_name
        ]

    def visible_settings(self):
        return [
            self.x_name,
            self.y_name
        ]

    def run(self, workspace):
        x_name = self.x_name.value

        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        dimensions = x.dimensions

        x_data = x.pixel_data

        y_data = scipy.ndimage.distance_transform_edt(x_data)

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
        layout = (2, 1)

        figure.set_subplots(
            dimensions=workspace.display_data.dimensions,
            subplots=layout
        )

        figure.subplot_imshow(
            dimensions=workspace.display_data.dimensions,
            image=workspace.display_data.x_data,
            x=0,
            y=0
        )

        figure.subplot_imshow(
            dimensions=workspace.display_data.dimensions,
            image=workspace.display_data.y_data,
            x=1,
            y=0
        )
