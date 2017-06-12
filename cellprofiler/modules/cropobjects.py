# -*- coding: utf-8 -*-

import numpy
import os.path
import skimage.io
import time

import cellprofiler.module
import cellprofiler.setting


__doc__ = """
CropObjects crops objects into masks. Each object is saved as a mask where the object is labeled as “255” and the
background is labeled as “0.” The dimensions of the mask are the same as the parent image. The filename for a mask is
formatted like “{object name}_{label index}_{timestamp}.tiff”
"""


class CropObjects(cellprofiler.module.Module):
    category = "File Processing"

    module_name = "CropObjects"

    variable_revision_number = 1

    def create_settings(self):
        self.objects_name = cellprofiler.setting.ObjectNameSubscriber(
            "Objects",
            doc="Select the objects you want to export per-object crops of."
        )

        self.image_name = cellprofiler.setting.ImageNameSubscriber(
            "Image",
            doc="Select image to crop"
        )

        self.directory = cellprofiler.setting.DirectoryPath(
            "Directory",
            doc="Enter the directory where object crops are saved."
        )

    def display(self, workspace, figure):
        figure.set_subplots((1, 1))

        figure.subplot_table(0, 0, [["\n".join(workspace.display_data.filenames)]])

    def run(self, workspace):
        objects = workspace.object_set.get_objects(self.objects_name.value)
        orig_image = workspace.image_set.get_image(self.image_name.value)

        labels = objects.segmented

        unique_labels = numpy.unique(labels)

        if unique_labels[0] == 0:
            unique_labels = unique_labels[1:]

        filenames = []

        for label in unique_labels:
            cropped_image = numpy.copy(orig_image.get_image())
            top, bot, left, right = self.find_bounds(labels == label)
            cropped_image = cropped_image[top:bot+1, left:right+1]

            filename = os.path.join(
                self.directory.get_absolute_path(),
                "{}_{:04d}_{}.tiff".format(self.objects_name.value, label, int(time.time()))
            )

            skimage.io.imsave(filename, skimage.img_as_ubyte(cropped_image))
            filenames.append(filename)

        if self.show_window:
            workspace.display_data.filenames = filenames

    def settings(self):
        settings = [
            self.objects_name,
            self.image_name,
            self.directory
        ]

        return settings

    def volumetric(self):
        return True

    def find_bounds(self, array):
        top, bot = numpy.where(numpy.max(array, 1) == 1)[0][[0, -1]]
        left, right = numpy.where(numpy.max(array, 0) == 1)[0][[0, -1]]
        return top, bot, left, right
