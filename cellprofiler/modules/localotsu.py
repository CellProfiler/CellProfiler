"""

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import numpy
import skimage.filters
import skimage.morphology


class LocalOtsu(cellprofiler.module.Module):
    module_name = "LocalOtsu"
    category = "Volumetric"
    variable_revision_number = 1

    def create_settings(self):
        self.input_image_name = cellprofiler.setting.ImageNameSubscriber(
            "Input image name:",
            cellprofiler.setting.NONE
        )

        self.output_image_name = cellprofiler.setting.ImageNameProvider(
            "Output image name:",
            cellprofiler.setting.NONE
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
        # structuring_element = self.structuring_element.value

        image_set = workspace.image_set
        input_image = image_set.get_image(input_image_name)
        pixels = input_image.pixel_data

        disk = skimage.morphology.disk(radius)
        local_otsu = [skimage.filters.rank.otsu(image, selem=disk) for image in pixels]
        local_otsu = numpy.asarray(local_otsu)

        output_pixels = pixels >= local_otsu

        output_image = cellprofiler.image.Image(output_pixels, parent_image=input_image)
        image_set.add(output_image_name, output_image)

        if self.show_window:
            workspace.display_data.input_pixels = pixels
            workspace.display_data.output_pixels = output_pixels

    def display(self, workspace, figure):
        figure.set_subplots((2, 1))

        figure.subplot_imshow_grayscale(
            0,
            0,
            workspace.display_data.input_pixels[16],
            title=self.input_image_name.value
        )

        figure.subplot_imshow_grayscale(
            1,
            0,
            workspace.display_data.output_pixels[16],
            title=self.output_image_name.value
        )
