"""
Save an image to file.

If you want to save objects, use the Object Processing module ConvertObjectsToImage to convert objects to an image
you can save.
"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from future import standard_library
standard_library.install_aliases()
from builtins import *
import cellprofiler.module
import cellprofiler.setting
import os.path
import skimage.io
import time


class Save(cellprofiler.module.Module):
    category = "File Processing"

    module_name = "Save"

    variable_revision_number = 1

    def create_settings(self):
        self.directory = cellprofiler.setting.DirectoryPath(
            "Directory"
        )

        self.image = cellprofiler.setting.ImageNameSubscriber(
            "Image"
        )

        self.bit_depth = cellprofiler.setting.Choice(
            "Color depth",
            choices=[
                "8",
                "16"
            ],
            value="16"
        )

    def settings(self):
        return [
            self.directory,
            self.image,
            self.bit_depth
        ]

    def visible_settings(self):
        return [
            self.image,
            self.directory,
            self.bit_depth
        ]

    def display(self, workspace, figure):
        figure.set_subplots((1, 1))

        figure.subplot_table(0, 0, [[workspace.display_data.filename]])

    def run(self, workspace):
        image_name = self.image.value

        image = workspace.image_set.get_image(image_name)

        filename = os.path.join(
            self.directory.get_absolute_path(),
            "{}_{}.tiff".format(image_name, int(time.time()))
        )

        if self.bit_depth.value == "8":
            y = skimage.img_as_ubyte(image.pixel_data)
        else:
            y = skimage.img_as_uint(image.pixel_data)

        skimage.io.imsave(filename, y)

        if self.show_window:
            workspace.display_data.filename = filename
