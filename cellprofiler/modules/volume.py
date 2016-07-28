"""

Volume

"""

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.setting
import matplotlib.pyplot
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

    def settings(self):
        return [
            self.directory,
            self.filename,
            self.name,
            self.channel
        ]

    def visible_settings(self):
        return [
            self.directory,
            self.filename,
            self.name,
            self.channel
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

        pixels = skimage.io.imread(path)[:, :, :, channel]

        image = cellprofiler.image.Image(
            dimensions=3
        )

        image.pixel_data = pixels

        workspace.image_set.add(name, image)

        if self.show_window:
            workspace.display_data.input_pixels = pixels

    #
    # display lets you use matplotlib to display your results.
    #
    def display(self, workspace, figure):
        image = workspace.display_data.image[15]

        dimensions = (1, 1)

        figure.set_subplots(dimensions)

        figure.subplot_imshow(
            0,
            0,
            image,
            colormap="gray"
        )
