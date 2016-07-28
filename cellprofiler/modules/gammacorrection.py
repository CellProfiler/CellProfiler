"""

Gamma correction

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import skimage.exposure


class GammaCorrection(cellprofiler.module.Module):
    category = "Volumetric"
    module_name = "GammaCorrection"
    variable_revision_number = 1

    def create_settings(self):
        self.input_image_name = cellprofiler.setting.ImageNameSubscriber(
            "Input"
        )

        self.output_image_name = cellprofiler.setting.ImageNameProvider(
            "Output",
            "OutputImage"
        )

        self.gamma = cellprofiler.setting.Float(
            "Gamma",
            1,
            minval=1.0,
            maxval=100.0,
        )

        self.gain = cellprofiler.setting.Float(
            "Gain",
            1,
            minval=1.0,
            maxval=100,
        )

    def settings(self):
        return [
            self.input_image_name,
            self.output_image_name,
            self.gamma,
            self.gain
        ]

    def visible_settings(self):
        return [
            self.input_image_name,
            self.output_image_name,
            self.gamma,
            self.gain
        ]

    def run(self, workspace):
        input_image_name = self.input_image_name.value

        output_image_name = self.output_image_name.value

        gamma = self.gamma.value

        gain = self.gain.value

        image_set = workspace.image_set

        input_image = image_set.get_image(input_image_name)

        pixels = input_image.pixel_data

        output_pixels = skimage.exposure.adjust_gamma(pixels, gamma=gamma, gain=gain)

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
