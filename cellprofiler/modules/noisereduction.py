"""

Noise reduction

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import numpy
import skimage.restoration
import skimage.util


class NoiseReduction(cellprofiler.module.Module):
    category = "Volumetric"
    module_name = "NoiseReduction"
    variable_revision_number = 1

    def create_settings(self):
        self.x_name = cellprofiler.setting.ImageNameSubscriber(
            "Input"
        )

        self.y_name = cellprofiler.setting.ImageNameProvider(
            "Output",
            "OutputImage"
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
            self.x_name,
            self.y_name,
            self.size,
            self.distance,
            self.cutoff_distance
        ]

    def visible_settings(self):
        return [
            self.x_name,
            self.y_name,
            self.size,
            self.distance,
            self.cutoff_distance
        ]

    def run(self, workspace):
        x_name = self.x_name.value
        y_name = self.y_name.value

        size = self.size.value

        distance = self.distance.value

        cutoff_distance = self.cutoff_distance.value

        images = workspace.image_set

        x = images.get_image(x_name)

        x_data = x.pixel_data

        x_data = skimage.img_as_uint(x_data)

        y_data = skimage.restoration.denoise_nl_means(
            image=x_data,
            patch_size=size,
            patch_distance=distance,
            h=cutoff_distance,
            multichannel=False,
            fast_mode=True
        )

        y = cellprofiler.image.Image(
            image=y_data,
            parent_image=x
        )

        images.add(y_name, y)

        if self.show_window:
            workspace.display_data.x_data = x_data
            workspace.display_data.y_data = y_data

    def display(self, workspace, figure):
        figure.gridspec((1, 2), (3, 3))

        figure.add_grid(0, workspace.display_data.x_data)

        figure.add_grid(1, workspace.display_data.y_data)
