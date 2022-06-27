"""
SaveCroppedObjects
==================

**SaveCroppedObjects** exports each object as an individual image. Pixels corresponding to an exported object are
assigned the value from the input image. All other pixels (i.e., background pixels and pixels corresponding to other
objects) are assigned the value 0. The dimensions of each image are the same as the original image. Multi-channel color
images will be represented as 3-channel RGB images when saved with this module (not available in 3D mode).

The filename for an exported image is formatted as "{object name}_{label index}.{image_format}", where *object name*
is the name of the exported objects, *label index* is the integer label of the object exported in the image (starting
from 1).

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============

"""

import os.path

import numpy
import skimage.io
import skimage.measure
from cellprofiler_core.module import Module
from cellprofiler_core.preferences import DEFAULT_OUTPUT_FOLDER_NAME
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting.subscriber import LabelSubscriber, ImageSubscriber, FileImageSubscriber
from cellprofiler_core.setting.text import Directory
from cellprofiler_core.constants.measurement import C_FILE_NAME

O_PNG = "png"
O_TIFF_8 = "8-bit tiff"
O_TIFF_16 = "16-bit tiff"
SAVE_PER_OBJECT = "Images"
SAVE_MASK = "Masks"


class SaveCroppedObjects(Module):
    category = "File Processing"

    module_name = "SaveCroppedObjects"

    variable_revision_number = 2

    def create_settings(self):
        self.export_option = Choice(
            "Do you want to save cropped images or object masks?",
            [SAVE_PER_OBJECT, SAVE_MASK],
            doc="""\
Choose the way you want the per-object crops to be exported.

The choices are:

-  *{SAVE_PER_OBJECT}*: Save a per-object crop from the original image
   based on the object's bounding box.
-  *{SAVE_MASK}*: Export a per-object mask.""".format(
                SAVE_PER_OBJECT=SAVE_PER_OBJECT, SAVE_MASK=SAVE_MASK
            ),
        )

        self.objects_name = LabelSubscriber(
            "Objects", 
            doc="Select the objects to export as per-object crops.",
        )

        self.image_name = ImageSubscriber(
            "Image to crop", 
            doc="Select the image to crop",
        )

        self.directory = Directory(
            "Directory",
            doc="Enter the directory where object crops are saved.",
            value=DEFAULT_OUTPUT_FOLDER_NAME,
        )

        self.file_image_name = FileImageSubscriber(
            "Select image name for file prefix",
            "None",
            doc="""\
Select an image loaded using **NamesAndTypes**. The original filename
will be used as the prefix for the output filename."""
        )

        self.file_format = Choice(
            "Saved file format",
            [O_PNG, O_TIFF_8, O_TIFF_16],
            value=O_TIFF_8,
            doc="""\
**{O_PNG}** files do not support 3D. **{O_TIFF_8}** files use zlib compression level 6.""".format(
                O_PNG=O_PNG, O_TIFF_8=O_TIFF_8, O_TIFF_16=O_TIFF_16
            ),
        )
        self.nested_save = Binary(
            "Save output crops in nested folders?",
            value=False,
            doc="""\
            If *Yes*, the output crops will be saved into a folder named
            after the image from which the crops were derived. 
            """,
        )

    def settings(self):
        settings = [
            self.objects_name,
            self.directory,
            self.file_image_name,
            self.file_format,
            self.export_option,
            self.image_name,
            self.nested_save,
        ]

        return settings

    def visible_settings(self):
        result = [
            self.export_option,
        ]
        if self.export_option.value == SAVE_PER_OBJECT:
            result += [self.image_name]
        result += [
            self.objects_name,
            self.directory,
            self.file_image_name,
            self.nested_save,
            self.file_format,
        ]
        return result

    def display(self, workspace, figure):
        figure.set_subplots((1, 1))

        figure.subplot_table(0, 0, [["\n".join(workspace.display_data.filenames)]])

    def run(self, workspace):
        objects = workspace.object_set.get_objects(self.objects_name.value)

        directory = self.directory.get_absolute_path(workspace.measurements)

        if not os.path.exists(directory):
            os.makedirs(directory)

        input_filename = workspace.measurements.get_current_measurement("Image", self.source_file_name_feature)
        input_filename = os.path.splitext(input_filename)[0]

        if self.nested_save:
            nested_folder = os.path.join(directory, input_filename)
            if not os.path.exists(nested_folder):
                os.makedirs(nested_folder)

        labels = objects.segmented

        unique_labels = numpy.unique(labels)

        if unique_labels[0] == 0:
            unique_labels = unique_labels[1:]

        filenames = []

        if self.export_option == SAVE_PER_OBJECT:
            images = workspace.image_set
            x = images.get_image(self.image_name.value)
            if len(x.pixel_data.shape) == len(labels.shape) + 1 and not x.volumetric:
                # Color 2D image, repeat mask for all channels
                labels = numpy.repeat(
                    labels[:, :, numpy.newaxis], x.pixel_data.shape[-1], axis=2
                )

        for label in unique_labels:
            if self.export_option == SAVE_MASK:
                mask = labels == label

            elif self.export_option == SAVE_PER_OBJECT:
                mask_in = labels == label
                properties = skimage.measure.regionprops(
                    mask_in.astype(int), intensity_image=x.pixel_data
                )
                mask = properties[0].intensity_image

            if self.nested_save.value:
                filename = os.path.join(
                    nested_folder, "{}_{}_{}".format(input_filename, self.objects_name.value, label)
                )

            elif not self.nested_save.value:
                filename = os.path.join(
                    directory, "{}_{}_{}".format(input_filename, self.objects_name.value, label)
                )

            if self.file_format.value == O_PNG:
                save_filename = filename + ".{}".format(O_PNG)

                skimage.io.imsave(
                    save_filename, 
                    skimage.img_as_ubyte(mask), 
                    check_contrast=False
                )

            elif self.file_format.value == O_TIFF_8:
                save_filename = filename +".{}".format("tiff")
                
                skimage.io.imsave(
                    save_filename,
                    skimage.img_as_ubyte(mask),
                    compress=6,
                    check_contrast=False,
                )

            elif self.file_format.value == O_TIFF_16:
                save_filename = filename + ".{}".format("tiff")
                
                skimage.io.imsave(
                    save_filename,
                    skimage.img_as_uint(mask),
                    compress=6,
                    check_contrast=False,
                )

            filenames.append(save_filename)

        if self.show_window:
            workspace.display_data.filenames = filenames

    @property
    def source_file_name_feature(self):
        """The file name measurement for the exemplar disk image"""
        return "_".join((C_FILE_NAME, self.file_image_name.value))

    def volumetric(self):
        return True
