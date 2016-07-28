"""Load3DImage

"""

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.setting

import os.path

import skimage.io


class Load3DImage(cellprofiler.module.Module):
    module_name = "Load3DImage"
    category = "Volumetric"
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

        self.image_name = cellprofiler.setting.ImageNameProvider(
            text = "Image name:"
        )

        ## TODO: Select channel?

    def settings(self):
        return [
            self.directory,
            self.filename,
            self.image_name
        ]

    def visible_settings(self):
        return [
            self.directory,
            self.filename,
            self.image_name
        ]

    def prepare_run(self, workspace):
        # Pipeline counts image sets from measurements.image_set_count and will raise an error
        # if there are no image sets (which is apparently the same as no measurements).
        workspace.measurements.add_measurement(
            cellprofiler.measurement.IMAGE,
            cellprofiler.measurement.C_PATH_NAME,
            os.path.join(self.directory.get_absolute_path(), self.filename.value),
            image_set_number = 1
        )

        return True

    def run(self, workspace):
        path = os.path.join(self.directory.get_absolute_path(), self.filename.value)
        pixels = skimage.io.imread(path)[:, :, :, 1]

        image = cellprofiler.image.Image(dimensionality=3)

        image.pixel_data = pixels

        workspace.image_set.add(self.image_name.value, image)

        if self.show_window:
            workspace.display_data.input_pixels = pixels

    #
    # display lets you use matplotlib to display your results.
    #
    def display(self, workspace, figure):
        figure.set_subplots((1, 1))

        figure.subplot_imshow_grayscale(
            0,
            0,
            workspace.display_data.input_pixels[0],
            title=self.image_name.value
        )
