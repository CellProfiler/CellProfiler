"""

Noise reduction

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import skimage.restoration
import skimage.util


class NoiseReduction(cellprofiler.module.Module):
    module_name = "NoiseReduction"
    category = "Volumetric"
    variable_revision_number = 1

    def create_settings(self):
        self.input_image_name = cellprofiler.setting.ImageNameSubscriber(
            "Input image name:",
        )

        self.output_image_name = cellprofiler.setting.ImageNameProvider(
            "Output image name:",
            "noise_reduction",
        )

        self.size = cellprofiler.setting.Integer(
            "Size",
            7
        )

        self.distance = cellprofiler.setting.Integer(
            "Distance",
            11
        )

        self.cutoff_distance = cellprofiler.setting.Float(
            "Cut-off distance",
            0.1
        )

    def settings(self):
        return [
            self.input_image_name,
            self.output_image_name,
            self.size,
            self.distance,
            self.cutoff_distance
        ]

    def visible_settings(self):
        return [
            self.input_image_name,
            self.output_image_name,
            self.size,
            self.distance,
            self.cutoff_distance
        ]

    def run(self, workspace):
        input_image_name = self.input_image_name.value
        output_image_name = self.output_image_name.value
        size = self.size.value
        distance = self.distance.value
        cutoff_distance = self.cutoff_distance.value

        image_set = workspace.image_set
        input_image = image_set.get_image(input_image_name)
        pixels = input_image.pixel_data

        output_pixels = skimage.util.apply_parallel(
            function=skimage.restoration.denoise_nl_means,
            array=pixels,
            extra_keywords={
                'patch_size': size,
                'patch_distance': distance,
                'h': cutoff_distance,
                'multichannel': False
            }
        )

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
