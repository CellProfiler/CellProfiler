# -*- coding: utf-8 -*-

"""
CropObjects exports individual images per object as masks or as a crop from the original image. In the case of masks,
each object is saved as a mask where the object is labeled as “255” and the background is labeled as “0.”
The dimensions of the mask are the same as the parent image.
In the case of crops from the original image, an image is saved for each object based on its bounding box (the dimensions
of the resulting images are the same as the ones of the bounding boxes)
The filename for a crop or mask is formatted like “{object name}_{label index}_{timestamp}.tiff”
"""

import numpy
import os.path
import skimage.io
import skimage.measure
import time

import cellprofiler.module
import cellprofiler.setting

SAVE_PER_OBJECT = "Objects"
SAVE_MASK = "Masks"


class CropObjects(cellprofiler.module.Module):
    category = "File Processing"

    module_name = "CropObjects"

    variable_revision_number = 1

    def create_settings(self):

        self.export_option = cellprofiler.setting.Choice(
            "Export option",
            [
                SAVE_PER_OBJECT,
                SAVE_MASK
            ],
            doc="""
            Choose the way you want the per-object crops to be exported.
            <p>The choices are:<br>
            <ul><li><i>{SAVE_PER_OBJECT}</i>: Save a per-object crop from the original image based on the object's
            bounding box.</li>
            <li><i>{SAVE_MASK}</i>: Export a per-object mask.</li>
            </ul></p>
            """.format(**{
                "SAVE_PER_OBJECT": SAVE_PER_OBJECT,
                "SAVE_MASK": SAVE_MASK
            })
        )

        self.objects_name = cellprofiler.setting.ObjectNameSubscriber(
            "Objects",
            doc="Select the objects you want to export per-object crops of."
        )

        self.image_name = cellprofiler.setting.ImageNameSubscriber(
            "Image",
            doc="Select the image to crop"
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

        if self.export_option == SAVE_MASK:
            for label in unique_labels:
                mask = labels == label

                filename = os.path.join(
                    self.directory.get_absolute_path(),
                    "{}_{:04d}_{}.tiff".format(self.objects_name.value, label, int(time.time()))
                )

                skimage.io.imsave(filename, skimage.img_as_ubyte(mask))

                filenames.append(filename)

        if self.export_option == SAVE_PER_OBJECT:
            orig_image = workspace.image_set.get_image(self.image_name.value)
            obj_regions = skimage.measure.regionprops(labels)
            for obj in obj_regions:
                cropped_image = numpy.copy(orig_image.get_image())
                top, left, bot, right = obj.bbox

                cropped_image = cropped_image[top:bot, left:right]

                filename = os.path.join(
                    self.directory.get_absolute_path(),
                    "{}_{:04d}_{}.tiff".format(self.objects_name.value, obj.label, int(time.time()))
                )

                skimage.io.imsave(filename, skimage.img_as_ubyte(cropped_image))

                filenames.append(filename)

        if self.show_window:
            workspace.display_data.filenames = filenames

    def settings(self):
        settings = [
            self.export_option,
            self.objects_name,
            self.image_name,
            self.directory
        ]

        return settings

    def volumetric(self):
        return True
