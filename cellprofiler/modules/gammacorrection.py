"""

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import skimage.exposure


class GammaCorrection(cellprofiler.module.Module):
    module_name = "GammaCorrection"
    category = "Volumetric"
    variable_revision_number = 1

    def create_settings(self):
        #
        # The ImageNameSubscriber "subscribes" to all ImageNameProviders in
        # prior modules. Modules before yours will put images into CellProfiler.
        # The ImageSubscriber gives your user a list of these images
        # which can then be used as inputs in your module.
        #
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

        self.gamma = cellprofiler.setting.Float(
            "Gamma:",
            1,
            minval=1.0,
            maxval=100.0,
        )

        self.gain = cellprofiler.setting.Float(
            "Gain:",
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
        figure.set_subplots((2, 1))

        figure.subplot_imshow_grayscale(
            0,
            0,
            workspace.display_data.input_pixels[0],
            title=self.input_image_name.value
        )

        figure.subplot_imshow_grayscale(
            1,
            0,
            workspace.display_data.output_pixels[0],
            title=self.output_image_name.value
        )
