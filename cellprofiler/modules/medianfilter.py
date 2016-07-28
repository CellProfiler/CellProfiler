"""

Median filter

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import numpy
import skimage.filters
import skimage.morphology


class MedianFilter(cellprofiler.module.Module):
    category = "Volumetric"
    module_name = "MedianFilter"
    variable_revision_number = 1

    def create_settings(self):
        self.input_image_name = cellprofiler.setting.ImageNameSubscriber(
            "Input"
        )

        self.output_image_name = cellprofiler.setting.ImageNameProvider(
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
            self.input_image_name,
            self.output_image_name,
            self.structuring_element,
            self.radius
        ]

    def visible_settings(self):
        return [
            self.input_image_name,
            self.output_image_name,
            self.structuring_element,
            self.radius
        ]

    def run(self, workspace):
        input_image_name = self.input_image_name.value
        output_image_name = self.output_image_name.value
        radius = self.radius.value
        structuring_element = self.structuring_element.value

        image_set = workspace.image_set

        input_image = image_set.get_image(input_image_name)

        pixels = input_image.pixel_data

        disk = skimage.morphology.disk(radius)

        output_pixels = numpy.zeros_like(pixels)

        for plane, image in enumerate(output_pixels):
            output_pixels[plane] = skimage.filters.rank.median(image, disk)

        output_image = cellprofiler.image.Image(output_pixels, parent_image=input_image)

        image_set.add(output_image_name, output_image)

        if self.show_window:
            workspace.display_data.input_pixels = pixels
            workspace.display_data.output_pixels = output_pixels

    def display(self, workspace, figure):
        dimensions = (2, 1)

        input = workspace.display_data.input_pixels[16]

        output = workspace.display_data.output_pixels[16]

        figure.set_subplots(dimensions)

        figure.subplot_imshow(
            0,
            0,
            input,
            colormap="gray"
        )

        figure.subplot_imshow(
            1,
            0,
            output,
            colormap="gray"
        )
