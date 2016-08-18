"""

Histogram equalization

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import numpy
import skimage.exposure
import skimage.filters
import skimage.morphology


class HistogramEqualization(cellprofiler.module.Module):
    category = "Volumetric"
    module_name = "HistogramEqualization"
    variable_revision_number = 1

    def create_settings(self):
        self.x_name = cellprofiler.setting.ImageNameSubscriber(
            "Input"
        )

        self.y_name = cellprofiler.setting.ImageNameProvider(
            "Output",
            "OutputImage"
        )

        self.structuring_element = cellprofiler.setting.Choice(
            "Structuing element",
            [
                "Ball"
            ]
        )

        self.radius = cellprofiler.setting.Integer(
            "Radius",
            1
        )

    def settings(self):
        return [
            self.x_name,
            self.y_name,
            self.structuring_element,
            self.radius
        ]

    def visible_settings(self):
        return [
            self.x_name,
            self.y_name,
            self.structuring_element,
            self.radius
        ]

    def run(self, workspace):
        x_name = self.x_name.value
        y_name = self.y_name.value

        radius = self.radius.value

        structuring_element = self.structuring_element.value

        images = workspace.image_set

        x = images.get_image(x_name)

        x_data = x.pixel_data

        x_data = skimage.img_as_uint(x_data)

        disk = skimage.morphology.disk(radius)

        y_data = numpy.zeros_like(x_data)

        for plane, image in enumerate(x_data):
            y_data[plane] = skimage.filters.rank.equalize(image, disk)

        y_data = skimage.exposure.rescale_intensity(y_data * 1.0)

        y = cellprofiler.image.Image(
            image=y_data,
            parent_image=x
        )

        images.add(y_name, y)

        if self.show_window:
            workspace.display_data.x_data = x_data
            workspace.display_data.y_data = y_data

    def display(self, workspace, figure):
        dimensions = (2, 1)

        x_data = workspace.display_data.x_data[16]
        y_data = workspace.display_data.y_data[16]

        figure.set_subplots(dimensions)

        figure.subplot_imshow(
            0,
            0,
            x_data,
            colormap="gray"
        )

        figure.subplot_imshow(
            1,
            0,
            y_data,
            colormap="gray"
        )
