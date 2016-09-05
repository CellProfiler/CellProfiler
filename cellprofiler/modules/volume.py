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
            "Directory"
        )

        self.filename = cellprofiler.setting.FilenameText(
            "Filename",
            cellprofiler.setting.NONE
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

        channel = self.channel.value

        name = self.name.value

        data = skimage.io.imread(path)[:, :, :, channel]

        volume = cellprofiler.image.Volume()

        volume.spacing = (self.z.value, self.x.value, self.y.value)

        volume.pixel_data = data

        workspace.image_set.add(name, volume)

        if self.show_window:
            workspace.display_data.data = data

    def display(self, workspace, figure):
        figure.set_subplots((1, 1))

        figure.subplot_imshow_grayscale(0, 0, workspace.display_data.data[0])