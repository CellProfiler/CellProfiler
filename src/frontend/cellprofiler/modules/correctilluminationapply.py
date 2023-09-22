"""
CorrectIlluminationApply
========================

**CorrectIlluminationApply** applies an illumination function,
usually created by **CorrectIlluminationCalculate**, to an image in
order to correct for uneven illumination/lighting/shading or to
reduce uneven background in images.

This module applies a previously created illumination correction
function, either loaded by the **Images** module, a **Load** module, or
created by **CorrectIlluminationCalculate**. This module corrects each
image in the pipeline using the function specified.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           NO
============ ============ ===============

See also
^^^^^^^^

See also **CorrectIlluminationCalculate**.
"""

import numpy
from cellprofiler_core.image import Image
from cellprofiler_core.module import Module
from cellprofiler_core.setting import Divider, Binary
from cellprofiler_core.setting import SettingsGroup
from cellprofiler_core.setting import ValidationError
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.do_something import DoSomething
from cellprofiler_core.setting.do_something import RemoveSettingButton
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.text import ImageName

DOS_DIVIDE = "Divide"
DOS_SUBTRACT = "Subtract"

######################################
#
# Rescaling choices - deprecated
#
######################################
RE_NONE = "No rescaling"
RE_STRETCH = "Stretch 0 to 1"
RE_MATCH = "Match maximums"

######################################
#
# # of settings per image when writing pipeline
#
######################################

SETTINGS_PER_IMAGE = 4


class CorrectIlluminationApply(Module):
    category = "Image Processing"
    variable_revision_number = 5
    module_name = "CorrectIlluminationApply"

    def create_settings(self):
        """Make settings here (and set the module name)"""
        self.images = []
        self.add_image(can_delete=False)
        self.add_image_button = DoSomething("", "Add another image", self.add_image)
        self.truncate_low = Binary(
            "Set output image values less than 0 equal to 0?", 
            True, 
            doc="""\
Values outside the range 0 to 1 might not be handled well by other
modules. Select *"Yes"* to set negative values to 0, which was previously
done automatically without ability to override.
""" )

        self.truncate_high = Binary(
            "Set output image values greater than 1 equal to 1?", 
            True, 
            doc="""\
Values outside the range 0 to 1 might not be handled well by other
modules. Select *"Yes"* to set values greater than 1 to a maximum
value of 1.
""")

    def add_image(self, can_delete=True):
        """Add an image and its settings to the list of images"""
        image_name = ImageSubscriber(
            "Select the input image", "None", doc="Select the image to be corrected."
        )

        corrected_image_name = ImageName(
            "Name the output image",
            "CorrBlue",
            doc="Enter a name for the corrected image.",
        )

        illum_correct_function_image_name = ImageSubscriber(
            "Select the illumination function",
            "None",
            doc="""\
Select the illumination correction function image that will be used to
carry out the correction. This image is usually produced by another
module or loaded as a .mat or .npy format image using the **Images** module
or a **LoadData** module.

Note that loading .mat format images is deprecated and will be removed in
a future version of CellProfiler. You can export .mat format images as
.npy format images using **SaveImages** to ensure future compatibility.
""",
        )

        divide_or_subtract = Choice(
            "Select how the illumination function is applied",
            [DOS_DIVIDE, DOS_SUBTRACT],
            doc="""\
This choice depends on how the illumination function was calculated and
on your physical model of the way illumination variation affects the
background of images relative to the objects in images; it is also
somewhat empirical.

-  *%(DOS_SUBTRACT)s:* Use this option if the background signal is
   significant relative to the real signal coming from the cells. If you
   created the illumination correction function using
   *Background*, then you will want to choose
   *%(DOS_SUBTRACT)s* here.
-  *%(DOS_DIVIDE)s:* Choose this option if the signal to background
   ratio is high (the cells are stained very strongly). If you created
   the illumination correction function using *Regular*, then
   you will want to choose *%(DOS_DIVIDE)s* here.
"""
            % globals(),
        )

        image_settings = SettingsGroup()
        image_settings.append("image_name", image_name)
        image_settings.append("corrected_image_name", corrected_image_name)
        image_settings.append(
            "illum_correct_function_image_name", illum_correct_function_image_name
        )
        image_settings.append("divide_or_subtract", divide_or_subtract)
        image_settings.append("rescale_option", RE_NONE)

        if can_delete:
            image_settings.append(
                "remover",
                RemoveSettingButton(
                    "", "Remove this image", self.images, image_settings
                ),
            )
        image_settings.append("divider", Divider())
        self.images.append(image_settings)

    def settings(self):
        """Return the settings to be loaded or saved to/from the pipeline

        These are the settings (from cellprofiler_core.settings) that are
        either read from the strings in the pipeline or written out
        to the pipeline. The settings should appear in a consistent
        order so they can be matched to the strings in the pipeline.
        """
        result = []
        for image in self.images:
            result += [
                image.image_name,
                image.corrected_image_name,
                image.illum_correct_function_image_name,
                image.divide_or_subtract,
            ]
        result += [
            self.truncate_low,
            self.truncate_high,
        ]
        return result

    def visible_settings(self):
        """Return the list of displayed settings
        """
        result = []
        for image in self.images:
            result += [
                image.image_name,
                image.corrected_image_name,
                image.illum_correct_function_image_name,
                image.divide_or_subtract,
            ]
            #
            # Get the "remover" button if there is one
            #
            remover = getattr(image, "remover", None)
            if remover is not None:
                result.append(remover)
            result.append(image.divider)
        result.append(self.add_image_button)
        result.append(self.truncate_low)
        result.append(self.truncate_high)
        return result

    def prepare_settings(self, setting_values):
        """Do any sort of adjustment to the settings required for the given values

        setting_values - the values for the settings

        This method allows a module to specialize itself according to
        the number of settings and their value. For instance, a module that
        takes a variable number of images or objects can increase or decrease
        the number of relevant settings so they map correctly to the values.
        """
        #
        # Figure out how many images there are based on the number of setting_values
        #
        assert len(setting_values) % SETTINGS_PER_IMAGE == 2
        image_count = len(setting_values) // SETTINGS_PER_IMAGE
        del self.images[image_count:]
        while len(self.images) < image_count:
            self.add_image()

    def run(self, workspace):
        """Run the module

        workspace    - The workspace contains
            pipeline     - instance of cpp for this run
            image_set    - the images in the image set being processed
            object_set   - the objects (labeled masks) in this image set
            measurements - the measurements for this run
            frame        - the parent frame to whatever frame is created. None means don't draw.
        """
        for image in self.images:
            self.run_image(image, workspace)

    def run_image(self, image, workspace):
        """Perform illumination according to the parameters of one image setting group

        """
        #
        # Get the image names from the settings
        #
        image_name = image.image_name.value
        illum_correct_name = image.illum_correct_function_image_name.value
        corrected_image_name = image.corrected_image_name.value
        #
        # Get images from the image set
        #
        orig_image = workspace.image_set.get_image(image_name)
        illum_function = workspace.image_set.get_image(illum_correct_name)
        illum_function_pixel_data = illum_function.pixel_data
        if orig_image.pixel_data.ndim == 2:
            illum_function = workspace.image_set.get_image(
                illum_correct_name, must_be_grayscale=True
            )
        else:
            if illum_function_pixel_data.ndim == 2:
                illum_function_pixel_data = illum_function_pixel_data[
                    :, :, numpy.newaxis
                ]
        # Throw an error if image and illum data are incompatible
        if orig_image.pixel_data.shape[:2] != illum_function_pixel_data.shape[:2]:
            raise ValueError(
                "This module requires that the image and illumination function have equal dimensions.\n"
                "The %s image and %s illumination function do not (%s vs %s).\n"
                "If they are paired correctly you may want to use the Resize or Crop module to make them the same size."
                % (
                    image_name,
                    illum_correct_name,
                    orig_image.pixel_data.shape,
                    illum_function_pixel_data.shape,
                )
            )
        #
        # Either divide or subtract the illumination image from the original
        #
        if image.divide_or_subtract == DOS_DIVIDE:
            output_pixels = orig_image.pixel_data / illum_function_pixel_data
        elif image.divide_or_subtract == DOS_SUBTRACT:
            output_pixels = orig_image.pixel_data - illum_function_pixel_data
            output_pixels[output_pixels < 0] = 0
        else:
            raise ValueError(
                "Unhandled option for divide or subtract: %s"
                % image.divide_or_subtract.value
            )
        
        #
        # Optionally, clip high and low values
        #
        if self.truncate_low.value:
            output_pixels = numpy.where(output_pixels < 0, 0, output_pixels)
        if self.truncate_high.value:
            output_pixels = numpy.where(output_pixels > 1, 1, output_pixels)
        
        #
        # Save the output image in the image set and have it inherit
        # mask & cropping from the original image.
        #
        output_image = Image(output_pixels, parent_image=orig_image)
        workspace.image_set.add(corrected_image_name, output_image)
        #
        # Save images for display
        #
        if self.show_window:
            if not hasattr(workspace.display_data, "images"):
                workspace.display_data.images = {}
            workspace.display_data.images[image_name] = orig_image.pixel_data
            workspace.display_data.images[corrected_image_name] = output_pixels
            workspace.display_data.images[
                illum_correct_name
            ] = illum_function.pixel_data

    def display(self, workspace, figure):
        """ Display one row of orig / illum / output per image setting group"""
        figure.set_subplots((3, len(self.images)))
        nametemplate = "Illumination function:" if len(self.images) < 3 else "Illum:"
        for j, image in enumerate(self.images):
            image_name = image.image_name.value
            illum_correct_function_image_name = (
                image.illum_correct_function_image_name.value
            )
            corrected_image_name = image.corrected_image_name.value
            orig_image = workspace.display_data.images[image_name]
            illum_image = workspace.display_data.images[
                illum_correct_function_image_name
            ]
            corrected_image = workspace.display_data.images[corrected_image_name]

            def imshow(x, y, image, *args, **kwargs):
                if image.ndim == 2:
                    f = figure.subplot_imshow_grayscale
                else:
                    f = figure.subplot_imshow_color
                return f(x, y, image, *args, **kwargs)

            imshow(
                0,
                j,
                orig_image,
                "Original image: %s" % image_name,
                sharexy=figure.subplot(0, 0),
            )
            title = f"{nametemplate} {illum_correct_function_image_name}, " \
                    f"min={illum_image.min():0.4f}, max={illum_image.max():0.4f}"

            imshow(1, j, illum_image, title, sharexy=figure.subplot(0, 0))
            imshow(
                2,
                j,
                corrected_image,
                "Final image: %s" % corrected_image_name,
                sharexy=figure.subplot(0, 0),
            )

    def validate_module_warnings(self, pipeline):
        """If a CP 1.0 pipeline used a rescaling option other than 'No rescaling', warn the user."""
        for j, image in enumerate(self.images):
            if image.rescale_option != RE_NONE:
                raise ValidationError(
                    (
                        "Your original pipeline used '%s' to rescale the final image, "
                        "but the rescaling option has been removed. Please use "
                        "RescaleIntensity to rescale your output image. Save your "
                        "pipeline to get rid of this warning."
                    )
                    % image.rescale_option,
                    image.divide_or_subtract,
                )

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        """Adjust settings based on revision # of save file

        setting_values - sequence of string values as they appear in the
                         saved pipeline
        variable_revision_number - the variable revision number of the module
                                   at the time of saving
        module_name - the name of the module that did the saving

        returns the updated setting_values, revision # and matlab flag
        """
        if variable_revision_number == 1:
            # Added multiple settings, but, if you only had 1,
            # the order didn't change
            variable_revision_number = 2

        if variable_revision_number == 2:
            # If revision < 2, remove rescaling option; warning user and suggest RescaleIntensity instead.
            # Keep the prior selection around for the validation warning.
            SLOT_RESCALE_OPTION = 4
            SETTINGS_PER_IMAGE_V2 = 5
            rescale_option = setting_values[SLOT_RESCALE_OPTION::SETTINGS_PER_IMAGE_V2]
            for i, image in enumerate(self.images):
                image.rescale_option = rescale_option[i]
            del setting_values[SLOT_RESCALE_OPTION::SETTINGS_PER_IMAGE_V2]

            variable_revision_number = 3
        else:
            # If revision >= 2, initialize rescaling option for validation warning
            for i, image in enumerate(self.images):
                image.rescale_option = RE_NONE

        if variable_revision_number == 3:
            setting_values.append("No")
            variable_revision_number = 4

        if variable_revision_number == 4:
            setting_values = setting_values[:-1]
            setting_values += [True,True]
            variable_revision_number = 5

        return setting_values, variable_revision_number
