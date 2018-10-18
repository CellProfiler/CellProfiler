# -*- coding: utf-8 -*-

"""
SaveCroppedObjects
==================

**SaveCroppedObjects** exports each object as a binary image. Pixels corresponding to an exported object are assigned
the value 255. All other pixels (i.e., background pixels and pixels corresponding to other objects) are assigned the
value 0. The dimensions of each image are the same as the original image.

The filename for an exported image is formatted as "{object name}_{label index}.{image_format}", where *object name*
is the name of the exported objects, *label index* is the integer label of the object exported in the image (starting
from 1).

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
import skimage.measure
import time

import cellprofiler.module
import cellprofiler.setting

O_PNG = "png"
O_TIFF = "tiff"
SAVE_PER_OBJECT = "Images"
SAVE_MASK = "Masks"

class SaveCroppedObjects(cellprofiler.module.Module):
    category = "File Processing"

    module_name = "SaveCroppedObjects"

    variable_revision_number = 2

    def create_settings(self):
        self.export_option = cellprofiler.setting.Choice(
            "Do you want to save cropped images or object masks?",
            [
                SAVE_PER_OBJECT,
                SAVE_MASK
            ],
            doc="""\
Choose the way you want the per-object crops to be exported.

The choices are:

-  *{SAVE_PER_OBJECT}*: Save a per-object crop from the original image
   based on the object's bounding box.
-  *{SAVE_MASK}*: Export a per-object mask.""".format(
                SAVE_PER_OBJECT=SAVE_PER_OBJECT,
                SAVE_MASK=SAVE_MASK
            )
        )

        self.objects_name = cellprofiler.setting.ObjectNameSubscriber(
            "Objects",
            doc="Select the objects you want to export as per-object crops."
        )

        self.image_name = cellprofiler.setting.ImageNameSubscriber(
            "Image",
            doc="Select the image to crop"
        )

        self.directory = cellprofiler.setting.DirectoryPath(
            "Directory",
            doc="Enter the directory where object crops are saved.",
            value=cellprofiler.setting.DEFAULT_OUTPUT_FOLDER_NAME
        )

        self.file_format = cellprofiler.setting.Choice(
            "Saved file format",
            [
                O_PNG,
                O_TIFF
            ],
            value=O_TIFF,
            doc="""\
**{O_PNG}** files do not support 3D. **{O_TIFF}** files use zlib compression level 6.""".format(
                O_PNG=O_PNG,
                O_TIFF=O_TIFF
            )
        )

    def display(self, workspace, figure):
        figure.set_subplots((1, 1))

        figure.subplot_table(0, 0, [["\n".join(workspace.display_data.filenames)]])

    def run(self, workspace):
        objects = workspace.object_set.get_objects(self.objects_name.value)

        directory = self.directory.get_absolute_path(workspace.measurements)

        if not os.path.exists(directory):
            os.makedirs(directory)

        labels = objects.segmented

        unique_labels = numpy.unique(labels)

        if unique_labels[0] == 0:
            unique_labels = unique_labels[1:]

        filenames = []

        for label in unique_labels:
            if self.export_option == SAVE_MASK:
                mask = labels == label

            elif self.export_option == SAVE_PER_OBJECT:
                mask_in = labels == label
                images = workspace.image_set
                x = images.get_image(self.image_name.value)
                properties = skimage.measure.regionprops(mask_in.astype(int), intensity_image=x.pixel_data)
                mask = properties[0].intensity_image

            if self.file_format.value == O_PNG:
                filename = os.path.join(
                    directory,
                    "{}_{}.{}".format(self.objects_name.value, label, O_PNG)
                    )

                skimage.io.imsave(filename, skimage.img_as_ubyte(mask))

            elif self.file_format.value == O_TIFF:
                filename = os.path.join(
                    directory,
                    "{}_{}.{}".format(self.objects_name.value, label, O_TIFF)
                    )

                skimage.io.imsave(filename, skimage.img_as_ubyte(mask), compress=6)

            filenames.append(filename)

        if self.show_window:
            workspace.display_data.filenames = filenames

    def settings(self):
        settings = [
            self.objects_name,
            self.directory,
            self.file_format,
            self.export_option,
            self.image_name
        ]

        return settings

    def visible_settings(self):
        result = [
            self.export_option,
            self.objects_name,
            self.directory,
            self.file_format
        ]

        if self.export_option.value == SAVE_PER_OBJECT:
            result += [self.image_name]

        return result

    def volumetric(self):
        return True
