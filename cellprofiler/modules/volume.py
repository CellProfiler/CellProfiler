"""

Volume

"""

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.setting
import os.path
import skimage.io


class Volume(cellprofiler.module.Module):
    category = "Input/output (I/O)"
    module_name = "Volume"
    variable_revision_number = 1

    def is_3d_load_module(self):
        return True

    def create_settings(self):
        self.directory = cellprofiler.setting.DirectoryPath(
            text = "Image directory:"
        )

        self.filename = cellprofiler.setting.FilenameText(
            text = "Image filename:",
            value = cellprofiler.setting.NONE
        )

        self.channel = cellprofiler.setting.Integer(
            "Channel",
            0
        )

        self.name = cellprofiler.setting.ImageNameProvider(
            text = "Name"
        )

        self.x = cellprofiler.setting.Float(
            "x",
            1.0
        )

        self.y = cellprofiler.setting.Float(
            "y",
            1.0
        )

        self.z = cellprofiler.setting.Float(
            "z",
            1.0
        )

    def settings(self):
        return [
            self.directory,
            self.filename,
            self.name,
            self.channel,
            self.x,
            self.y,
            self.z
        ]

    def visible_settings(self):
        return [
            self.directory,
            self.filename,
            self.name,
            self.channel,
            self.x,
            self.y,
            self.z
        ]

    def prepare_run(self, workspace):
        # Pipeline counts image sets from measurements.image_set_count and will raise an error if there are no image
        # sets (which is apparently the same as no measurements).
        workspace.measurements.add_measurement(
            cellprofiler.measurement.IMAGE,
            cellprofiler.measurement.C_PATH_NAME,
            os.path.join(self.directory.get_absolute_path(), self.filename.value),
            image_set_number=1
        )

        return True

    def run(self, workspace):
        path = os.path.join(self.directory.get_absolute_path(), self.filename.value)
        pixels = skimage.io.imread(path)[:, :, :, 1]

        channel = self.channel.value

        name = self.name.value

        x = skimage.io.imread(path)[:, :, :, channel]

        image = cellprofiler.image.Image(
            dimensions=3,
            spacing=(self.z.value, self.x.value, self.y.value)
        )

        image.pixel_data = x

        workspace.image_set.add(name, image)

        if self.show_window:
            workspace.display_data.image = x

    #
    # display lets you use matplotlib to display your results.
    #
    def display(self, workspace, figure):
        figure.set_grids((1, 1))

        figure.gridshow(0, 0, workspace.display_data.image)