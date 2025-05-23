"""
ConvertObjectsToImage
=====================

**ConvertObjectsToImage** converts objects you have identified into
an image.

This module allows you to take previously identified objects and convert
them into an image according to a colormap you select, which can then be saved 
with the **SaveImages** module.

This module does not support overlapping objects, such as those produced by the
UntangleWorms module. Overlapping regions will be lost during saving.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============
"""

from math import pi
import centrosome.cpmorphology
import matplotlib.cm
import numpy
from cellprofiler_core.image import Image
from cellprofiler_core.module import Module
from cellprofiler_core.preferences import get_default_colormap
from cellprofiler_core.setting.choice import Choice, Colormap
from cellprofiler_core.setting.subscriber import LabelSubscriber
from cellprofiler_core.setting.text import ImageName
from cellprofiler_library.modules._convertobjectstoimage import update_pixel_data
from cellprofiler_library.opts.convertobjectstoimage import ImageMode

DEFAULT_COLORMAP = "Default"


class ConvertObjectsToImage(Module):
    module_name = "ConvertObjectsToImage"

    category = "Object Processing"

    variable_revision_number = 1

    def create_settings(self):
        self.object_name = LabelSubscriber(
            "Select the input objects",
            "None",
            doc="Choose the name of the objects you want to convert to an image.",
        )

        self.image_name = ImageName(
            "Name the output image",
            "CellImage",
            doc="Enter the name of the resulting image.",
        )

        self.image_mode = Choice(
            "Select the color format",
            ["Color", "Binary (black & white)", "Grayscale", "uint16"],
            doc="""\
Select which colors the resulting image should use. You have the
following options:

-  *Color:* Allows you to choose a colormap that will produce jumbled
   colors for your objects.
-  *Binary (black & white):* All object pixels will be assigned 1 and
   all background pixels will be assigned 0, creating a binary image.
-  *Grayscale:* Assigns all background pixels to 0 and assigns each object's pixels with a number 
   specific to that object. Object numbers can range from 1 to 255 (the maximum value that you can put
   in an 8-bit integer, use **uint16** if you expect more than 255 objects).
   This creates an image where objects in the top left corner of the image are
   very dark and the colors progress to white toward the bottom right corner of the image.
   Use **SaveImages** to save the resulting image as a .npy file or .tiff file if you want
   to process the label matrix image using another program or in a separate CellProfiler pipeline.
-  *uint16:* Assigns all background pixels to 0 and assigns each object's pixels with a number 
   specific to that object. Object numbers can range from 1 to 65535 (the maximum value that you can put
   in a 16-bit integer). This creates an image where objects in the top left corner of the image are
   very dark and where the colors progress to white toward the bottom right corner of the image
   (though this can usually only be seen in a scientific image viewer since standard image viewers only
   handle 8-bit images). Use **SaveImages** to save the resulting image as a .npy file or
   **16-bit** (not 8-bit!) .tiff file if you want to process the label matrix image using another
   program or in a separate CellProfiler pipeline.

You can choose *Color* with a *Gray* colormap to produce jumbled gray
objects.
            """,
        )

        self.colormap = Colormap(
            "Select the colormap",
            doc="""\
*(Used only if "Color" output image selected)*

Choose the colormap to be used, which affects how the objects are
colored. You can look up your default colormap under *File >
Preferences*.
""",
        )

    def settings(self):
        return [self.object_name, self.image_name, self.image_mode, self.colormap]

    def visible_settings(self):
        settings = [self.object_name, self.image_name, self.image_mode]

        if self.image_mode == "Color":
            settings = settings + [self.colormap]

        return settings
    

    def run(self, workspace):
        objects = workspace.object_set.get_objects(self.object_name.value)
        object_labels = objects.get_labels()
        pixel_data = update_pixel_data(self.image_mode.value, object_labels, objects.shape, self.colormap.value)

        convert = False if self.image_mode.value.casefold() == ImageMode.UINT16 else True
        image = Image(
            pixel_data,
            parent_image=objects.parent_image,
            convert=convert,
            dimensions=len(objects.shape),
        )

        workspace.image_set.add(self.image_name.value, image)

        if self.show_window:
            if image.dimensions == 2:
                workspace.display_data.ijv = objects.ijv
            else:
                workspace.display_data.segmented = objects.segmented

            workspace.display_data.pixel_data = pixel_data

            workspace.display_data.dimensions = image.dimensions

    def display(self, workspace, figure):
        pixel_data = workspace.display_data.pixel_data

        dimensions = workspace.display_data.dimensions

        cmap = None if self.image_mode == "Color" else "gray"

        figure.set_subplots((2, 1), dimensions=dimensions)

        # TODO: volumetric IJV
        if dimensions == 2:
            figure.subplot_imshow_ijv(
                0,
                0,
                workspace.display_data.ijv,
                shape=workspace.display_data.pixel_data.shape[:2],
                title="Original: %s" % self.object_name.value,
            )
        else:
            figure.subplot_imshow_labels(
                0,
                0,
                workspace.display_data.segmented,
                title="Original: %s" % self.object_name.value,
            )

        figure.subplot_imshow(
            1,
            0,
            pixel_data,
            self.image_name.value,
            colormap=cmap,
            sharexy=figure.subplot(0, 0),
        )

    def volumetric(self):
        return True


#
# Backwards compatibility
#
ConvertToImage = ConvertObjectsToImage
