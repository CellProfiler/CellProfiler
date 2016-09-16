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
        self.x_name = cellprofiler.setting.ImageNameSubscriber(
            "Input"
        )

        self.y_name = cellprofiler.setting.ImageNameProvider(
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
            self.x_name,
            self.y_name,
            self.gamma,
            self.gain
        ]

    def visible_settings(self):
        return [
            self.x_name,
            self.y_name,
            self.gamma,
            self.gain
        ]

    def run(self, workspace):
        x_name = self.x_name.value
        y_name = self.y_name.value

        gamma = self.gamma.value

        gain = self.gain.value

        images = workspace.image_set

        x = images.get_image(x_name)

        x_data = x.pixel_data

        y_data = skimage.exposure.adjust_gamma(x_data, gamma, gain)

        y = cellprofiler.image.Image(
            image=y_data,
            parent_image=x
        )

        images.add(y_name, y)

        if self.show_window:
            workspace.display_data.x_data = x_data
            workspace.display_data.y_data = y_data

    def display(self, workspace, figure):
        figure.set_grids((1, 2))

        figure.gridshow(0, 0, workspace.display_data.x_data)

        figure.gridshow(0, 1, workspace.display_data.y_data)
