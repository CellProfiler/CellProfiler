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

        self.directory = cellprofiler.setting.DirectoryPath(
            "Directory",
            doc="Enter the directory where object crops are saved."
        )

    def display(self, workspace, figure):
        figure.set_subplots((1, 1))

        figure.subplot_table(0, 0, [["\n".join(workspace.display_data.filenames)]])

    def run(self, workspace):
        objects = workspace.object_set.get_objects(self.objects_name.value)

        labels = objects.segmented

        unique_labels = numpy.unique(labels)

        if unique_labels[0] == 0:
            unique_labels = unique_labels[1:]

        filenames = []

        for label in unique_labels:
            mask = labels == label

            filename = os.path.join(
                self.directory.get_absolute_path(),
                "{}_{:04d}_{}.tiff".format(self.objects_name.value, label, int(time.time()))
            )

            skimage.io.imsave(filename, skimage.img_as_ubyte(mask))

            filenames.append(filename)

        if self.show_window:
            workspace.display_data.filenames = filenames

    def settings(self):
        settings = [
            self.objects_name,
            self.directory
        ]

        return settings

    def volumetric(self):
        return True
