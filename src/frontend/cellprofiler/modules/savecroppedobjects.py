"""
SaveCroppedObjects
==================

**SaveCroppedObjects** exports each object as an individual image. There are two modes to this module
depending on whether the user wants to save cropped **Images** or **Masks**:

* In **Images** mode, the input image is cropped to the bounding box of each object. Pixels 
  corresponding to an exported object are assigned the value from the input image. All other pixels 
  (i.e., background pixels and pixels corresponding to other objects) are assigned the value 0. The 
  dimensions of each output image match the dimensions of the bounding box of each object.

* In **Masks** mode, a binary mask is produced for each object that is the same size as the original 
  image used to generate the objects. The pixels corresponding to an exported object are assigned the 
  value 1 and all other pixels in the image are assigned the value 0. The dimensions of each output 
  image are the same for all objects and match the original image used when generating the objects. 

**Note**: Multi-channel color images will be represented as 3-channel RGB images when saved with this module 
(not available in 3D mode).

The filename for an exported image is formatted in one of two ways. 
By default, when the *Prefix saved crop image name with input image name* option is enabled, the format is
"{input image name}_{object name}_{label index}.{image_format}",
and when disabled the format is, "{object name}_{label index}.{image_format}", 
where *object name* is the name of the exported objects, 
and *label index* is the integer label of the object exported in the image (starting from 1).

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
from cellprofiler_library.modules import savecroppedobjects

O_PNG = "png"
O_TIFF_8 = "8-bit tiff"
O_TIFF_16 = "16-bit tiff"
SAVE_PER_OBJECT = "Images"
SAVE_MASK = "Masks"


class SaveCroppedObjects(Module):
    category = "File Processing"

    module_name = "SaveCroppedObjects"

    variable_revision_number = 3

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

        self.use_filename = Binary(
            "Prefix saved crop image name with input image name?",
            value=True,
            doc="""\
If *Yes*, the filename of the saved cropped object will be prefixed with
the filename of the input image.

For example:

**Input file name**: positive_treatment.tiff


**Output crop file name**: positive_treatment_Nuclei_1.tiff


where "Nuclei" is the object name and "1" is the object number. 
            """,
        )

        self.file_image_name = FileImageSubscriber(
            "Select image name to use as a prefix",
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
after the selected image name prefix. 

If no image name prefix is selected, crops will be saved into 
a folder named after the input objects.
            """,
        )

    def settings(self):
        settings = [
            self.export_option,
            self.objects_name,
            self.directory,
            self.use_filename,
            self.file_image_name,
            self.nested_save,
            self.file_format,
            self.image_name,
        ]

        return settings

    def visible_settings(self):
        result = [
            self.export_option,
            self.objects_name,
            self.directory,
            self.use_filename,
        ]
        if self.use_filename.value:
            result += [self.file_image_name]
        result += [
            self.nested_save,
            self.file_format,
        ]
        if self.export_option.value == SAVE_PER_OBJECT:
            result += [self.image_name]
        return result

    def display(self, workspace, figure):
        figure.set_subplots((1, 1))

        figure.subplot_table(0, 0, [["\n".join(workspace.display_data.filenames)]])

    def run(self, workspace):

        objects = workspace.object_set.get_objects(self.objects_name.value)

        input_objects = objects.segmented

        input_volumetric = objects.volumetric

        directory = self.directory.get_absolute_path(workspace.measurements)

        input_objects_name = self.objects_name.value

        if self.use_filename:
            input_filename = workspace.measurements.get_current_measurement("Image", self.source_file_name_feature)
            input_filename = os.path.splitext(input_filename)[0]
        else:
            input_filename = None


        if self.export_option == SAVE_PER_OBJECT:
            images = workspace.image_set
            x = images.get_image(self.image_name.value).pixel_data
        else:
            x = None

        # Translate GUI string settings to library
        exp_options = {
            "8-bit tiff": "tiff8",
            "16-bit tiff": "tiff16",
            "png": "png"
        }

        filenames = savecroppedobjects(
            input_objects=input_objects,
            save_dir=directory,
            export_as=self.export_option.value,
            input_image=x,
            file_format=exp_options[self.file_format.value],
            nested_save=self.nested_save.value,
            save_names = {"input_filename": input_filename, "input_objects_name": input_objects_name},
            volumetric=input_volumetric
        )

        if self.show_window:
            workspace.display_data.filenames = filenames

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 1:
            # Old order: 
            # [objects_name, directory, file_format]
            # New order:
            # [objects_name, directory, file_format, export_option, image_name]
            setting_values = (
                setting_values[:3] + [SAVE_PER_OBJECT, "Image"]
            )
            variable_revision_number = 2
        
        if variable_revision_number == 2:
            # Older module version, revert to not using file names in output crops
            # Also, reorder setting_values to reflect order of settings in the GUI. 
            # Original order:
            # [objects_name, directory, file_format, export_option, image_name]
            # New order:
            # [export_option, objects_name, directory, use_filename, file_image_name, nested_save, file_format, image_name]
            setting_values = (
                [setting_values[3]] + setting_values[:2] + [False, "None", False] + [setting_values[2]] + [setting_values[4]]
            )
            variable_revision_number = 3
        return setting_values, variable_revision_number

    @property
    def source_file_name_feature(self):
        """The file name measurement for the exemplar disk image"""
        return "_".join((C_FILE_NAME, self.file_image_name.value))

    def volumetric(self):
        return True
