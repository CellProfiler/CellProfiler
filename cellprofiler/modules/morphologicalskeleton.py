"""
3D Morphological Skeleton
"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import skimage.exposure
import skimage.morphology
import skimage.util


class MorphologicalSkeleton(cellprofiler.module.Module):
    module_name = "MorphologicalSkeleton"
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

    def settings(self):
        return [
            self.input_image_name,
            self.output_image_name
        ]

    def visible_settings(self):
        return [
            self.input_image_name,
            self.output_image_name
        ]

    def run(self, workspace):
        input_image_name = self.input_image_name.value
        output_image_name = self.output_image_name.value

        image_set = workspace.image_set
        input_image = image_set.get_image(input_image_name)
        pixels = input_image.pixel_data

        pixels = skimage.exposure.rescale_intensity(pixels * 1.0)
        output_pixels = skimage.morphology.skeletonize_3d(pixels)

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
