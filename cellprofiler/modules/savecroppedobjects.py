# -*- coding: utf-8 -*-

"""
SaveCroppedObjects
==================

**SaveCroppedObjects** exports each object as a binary image. Pixels corresponding to an exported object are assigned
the value 255. All other pixels (i.e., background pixels and pixels corresponding to other objects) are assigned the
value 0. The dimensions of each image are the same as the original image.

The filename for an exported image is formatted as "{object name}_{label index}_{timestamp}.tiff", where *object name*
is the name of the exported objects, *label index* is the integer label of the object exported in the image (starting
from 1), and *timestamp* is the time at which the image was saved (this prevents accidentally overwriting a previously
exported image).

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           YES
============ ============ ===============

"""

import numpy
import os.path
import skimage.io
import time

import cellprofiler.module
import cellprofiler.setting


class SaveCroppedObjects(cellprofiler.module.Module):
    category = "File Processing"

    module_name = "SaveCroppedObjects"

    variable_revision_number = 1

    def create_settings(self):
        self.objects_name = cellprofiler.setting.ObjectNameSubscriber(
            "Objects",
            doc="Select the objects you want to export as per-object crops."
        )

        self.directory = cellprofiler.setting.DirectoryPath(
            "Directory",
            doc="Enter the directory where object crops are saved.",
            value=cellprofiler.setting.DEFAULT_OUTPUT_FOLDER_NAME
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
