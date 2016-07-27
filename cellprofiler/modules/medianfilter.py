"""

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import numpy
import skimage.filters
import skimage.morphology


class MedianFilter(cellprofiler.module.Module):
    module_name = "MedianFilter"
    category = "Volumetric"
    variable_revision_number = 1

    def create_settings(self):

        self.input_image_name = cellprofiler.setting.ImageNameSubscriber(
                # The text to the left of the edit box
                "Input image name:",
                # HTML help that gets displayed when the user presses the
                # help button to the right of the edit box
                doc="""This is the image that the module operates on. You can
            choose any image that is made available by a prior module.
            <br>
            <b>ImageTemplate</b> will do something to this image.
            """)


        self.output_image_name = cellprofiler.setting.ImageNameProvider(
                "Output image name:",
                # The second parameter holds a suggested name for the image.
                "OutputImage",
                doc="""This is the image resulting from the operation.""")

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

        output_pixels = [skimage.filters.rank.median(image, disk) for image in pixels]

        output_pixels = numpy.asarray(output_pixels)

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
