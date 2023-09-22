"""
MaskImage
=========

**MaskImage** hides certain portions of an image (based on previously
identified objects or a binary image) so they are ignored by subsequent
mask-respecting modules in the pipeline.

This module masks an image so you can use the mask downstream in the
pipeline. The masked image is based on the original image and the
masking object or image that is selected. If using a masking image, the
mask is composed of the foreground (white portions); if using a masking
object, the mask is composed of the area within the object. Note that
the image created by this module for further processing downstream is
grayscale. If a binary mask is desired in subsequent modules, use the
**Threshold** module instead of **MaskImage**.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============

See also
^^^^^^^^

See also **Threshold**, **IdentifyPrimaryObjects**, and
**IdentifyObjectsManually**.
"""

import numpy
from cellprofiler_core.image import Image
from cellprofiler_core.module import Module
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.subscriber import LabelSubscriber, ImageSubscriber
from cellprofiler_core.setting.text import ImageName

IO_IMAGE = "Image"
IO_OBJECTS = "Objects"


class MaskImage(Module):
    module_name = "MaskImage"
    category = "Image Processing"
    variable_revision_number = 3

    def create_settings(self):
        """Create the settings here and set the module name (initialization)

        """
        self.source_choice = Choice(
            "Use objects or an image as a mask?",
            [IO_OBJECTS, IO_IMAGE],
            doc="""\
You can mask an image in two ways:

-  *%(IO_OBJECTS)s*: Using objects created by another module (for
   instance **IdentifyPrimaryObjects**). The module will mask out all
   parts of the image that are not within one of the objects (unless you
   invert the mask).
-  *%(IO_IMAGE)s*: Using a binary image as the mask, where black
   portions of the image (false or zero-value pixels) will be masked
   out. If the image is not binary, the module will use all pixels whose
   intensity is greater than 0.5 as the mask’s foreground (white area).
   You can use **Threshold** instead to create a binary image with
   finer control over the intensity choice.
   """
            % globals(),
        )

        self.object_name = LabelSubscriber(
            "Select object for mask",
            "None",
            doc="""\
*(Used only if mask is to be made from objects)*

Select the objects you would like to use to mask the input image.
""",
        )

        self.masking_image_name = ImageSubscriber(
            "Select image for mask",
            "None",
            doc="""\
*(Used only if mask is to be made from an image)*

Select the image that you like to use to mask the input image.
""",
        )

        self.image_name = ImageSubscriber(
            "Select the input image",
            "None",
            doc="Select the image that you want to mask.",
        )

        self.masked_image_name = ImageName(
            "Name the output image",
            "MaskBlue",
            doc="Enter the name for the output masked image.",
        )

        self.invert_mask = Binary(
            "Invert the mask?",
            False,
            doc="""\
This option reverses the foreground/background relationship of the mask.

-  Select "*No*" to produce the mask from the foreground (white
   portion) of the masking image or the area within the masking objects.
-  Select "*Yes*" to instead produce the mask from the *background*
   (black portions) of the masking image or the area *outside* the
   masking objects.
       """
            % globals(),
        )

    def settings(self):
        """Return the settings in the order that they will be saved or loaded

        Note that the settings are also the visible settings in this case, so
              they also control the display order. Implement visible_settings
              for a different display order.
        """
        return [
            self.image_name,
            self.masked_image_name,
            self.source_choice,
            self.object_name,
            self.masking_image_name,
            self.invert_mask,
        ]

    def visible_settings(self):
        """Return the settings as displayed in the user interface"""
        return [
            self.image_name,
            self.masked_image_name,
            self.source_choice,
            self.object_name
            if self.source_choice == IO_OBJECTS
            else self.masking_image_name,
            self.invert_mask,
        ]

    def run(self, workspace):
        image_set = workspace.image_set
        if self.source_choice == IO_OBJECTS:
            objects = workspace.get_objects(self.object_name.value)
            labels = objects.segmented
            if self.invert_mask.value:
                mask = labels == 0
            else:
                mask = labels > 0
        else:
            objects = None
            try:
                mask = image_set.get_image(
                    self.masking_image_name.value, must_be_binary=True
                ).pixel_data
            except ValueError:
                mask = image_set.get_image(
                    self.masking_image_name.value, must_be_grayscale=True
                ).pixel_data
                mask = mask > 0.5
            if self.invert_mask.value:
                mask = mask == 0
        orig_image = image_set.get_image(self.image_name.value)
        if (
            orig_image.multichannel and mask.shape != orig_image.pixel_data.shape[:-1]
        ) or mask.shape != orig_image.pixel_data.shape:
            tmp = numpy.zeros(orig_image.pixel_data.shape[:2], mask.dtype)
            tmp[mask] = True
            mask = tmp
        if orig_image.has_mask:
            mask = numpy.logical_and(mask, orig_image.mask)
        masked_pixels = orig_image.pixel_data.copy()
        masked_pixels[numpy.logical_not(mask)] = 0
        masked_image = Image(
            masked_pixels,
            mask=mask,
            parent_image=orig_image,
            masking_objects=objects,
            dimensions=orig_image.dimensions,
            convert=False
        )

        image_set.add(self.masked_image_name.value, masked_image)

        if self.show_window:
            workspace.display_data.dimensions = orig_image.dimensions
            workspace.display_data.orig_image_pixel_data = orig_image.pixel_data
            workspace.display_data.masked_pixels = masked_pixels
            workspace.display_data.multichannel = orig_image.multichannel

    def display(self, workspace, figure):
        orig_image_pixel_data = workspace.display_data.orig_image_pixel_data
        masked_pixels = workspace.display_data.masked_pixels
        figure.set_subplots((2, 1), dimensions=workspace.display_data.dimensions)
        if workspace.display_data.multichannel:
            figure.subplot_imshow_color(
                0,
                0,
                orig_image_pixel_data,
                "Original image: %s" % self.image_name.value,
            )
            figure.subplot_imshow_color(
                1,
                0,
                masked_pixels,
                "Masked image: %s" % self.masked_image_name.value,
                sharexy=figure.subplot(0, 0),
            )
        else:
            figure.subplot_imshow_grayscale(
                0,
                0,
                orig_image_pixel_data,
                "Original image: %s" % self.image_name.value,
            )
            figure.subplot_imshow_grayscale(
                1,
                0,
                masked_pixels,
                "Masked image: %s" % self.masked_image_name.value,
                sharexy=figure.subplot(0, 0),
            )

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        """Adjust the setting_values to upgrade from a previous version

        """
        if variable_revision_number == 1:
            #
            # Added ability to select an image
            #
            setting_values = setting_values + [
                IO_IMAGE if setting_values[0] == "Image" else IO_OBJECTS,
                "None",
            ]
            variable_revision_number = 2

        if variable_revision_number == 2:
            # Reordering setting values so the settings order and Help makes sense
            setting_values = [
                setting_values[1],  # Input image name
                setting_values[2],  # Output image name
                setting_values[4],  # Image or objects?
                setting_values[0],  # Object used as mask
                setting_values[5],  # Image used as mask
                setting_values[3],
            ]  # Invert image?
            variable_revision_number = 3

        return setting_values, variable_revision_number

    def volumetric(self):
        return True
