"""
RescaleIntensity
================

**RescaleIntensity** changes the intensity range of an image to your
desired specifications.

This module lets you rescale the intensity of the input images by any of
several methods. You should use caution when interpreting intensity and
texture measurements derived from images that have been rescaled because
certain options for this module do not preserve the relative intensities
from image to image.

As this module rescales data it will not attempt to normalize displayed previews
(as this could make it appear that the scaling had done nothing). As a result images rescaled
to large ranges may appear dim after scaling. To normalize values for viewing,
right-click an image and choose an image contrast transform.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============
"""

import numpy
import skimage.exposure
from cellprofiler_core.image import Image
from cellprofiler_core.module import ImageProcessing
from cellprofiler_core.setting import Measurement
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.range import FloatRange
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.text import Float

M_STRETCH = "Stretch each image to use the full intensity range"
M_MANUAL_INPUT_RANGE = "Choose specific values to be reset to the full intensity range"
M_MANUAL_IO_RANGE = "Choose specific values to be reset to a custom range"
M_DIVIDE_BY_IMAGE_MINIMUM = "Divide by the image's minimum"
M_DIVIDE_BY_IMAGE_MAXIMUM = "Divide by the image's maximum"
M_DIVIDE_BY_VALUE = "Divide each image by the same value"
M_DIVIDE_BY_MEASUREMENT = "Divide each image by a previously calculated value"
M_SCALE_BY_IMAGE_MAXIMUM = "Match the image's maximum to another image's maximum"

M_ALL = [
    M_STRETCH,
    M_MANUAL_INPUT_RANGE,
    M_MANUAL_IO_RANGE,
    M_DIVIDE_BY_IMAGE_MINIMUM,
    M_DIVIDE_BY_IMAGE_MAXIMUM,
    M_DIVIDE_BY_VALUE,
    M_DIVIDE_BY_MEASUREMENT,
    M_SCALE_BY_IMAGE_MAXIMUM,
]

R_SCALE = "Scale similarly to others"
R_MASK = "Mask pixels"
R_SET_TO_ZERO = "Set to zero"
R_SET_TO_CUSTOM = "Set to custom value"
R_SET_TO_ONE = "Set to one"

LOW_ALL_IMAGES = "Minimum of all images"
LOW_EACH_IMAGE = "Minimum for each image"
CUSTOM_VALUE = "Custom"
LOW_ALL = [CUSTOM_VALUE, LOW_EACH_IMAGE, LOW_ALL_IMAGES]

HIGH_ALL_IMAGES = "Maximum of all images"
HIGH_EACH_IMAGE = "Maximum for each image"

HIGH_ALL = [CUSTOM_VALUE, HIGH_EACH_IMAGE, HIGH_ALL_IMAGES]


class RescaleIntensity(ImageProcessing):
    module_name = "RescaleIntensity"

    variable_revision_number = 3

    def create_settings(self):
        super(RescaleIntensity, self).create_settings()

        self.rescale_method = Choice(
            "Rescaling method",
            choices=M_ALL,
            doc="""\
There are a number of options for rescaling the input image:

-  *%(M_STRETCH)s:* Find the minimum and maximum values within the
   unmasked part of the image (or the whole image if there is no mask)
   and rescale every pixel so that the minimum has an intensity of zero
   and the maximum has an intensity of one. If performed on color images
   each channel will be considered separately.
-  *%(M_MANUAL_INPUT_RANGE)s:* Pixels are scaled from an original range
   (which you provide) to the range 0 to 1. Options are
   available to handle values outside of the original range.
   To convert 12-bit images saved in 16-bit format to the correct range,
   use the range 0 to 0.0625. The value 0.0625 is equivalent to
   2\ :sup:`12` divided by 2\ :sup:`16`, so it will convert a 16 bit
   image containing only 12 bits of data to the proper range.
-  *%(M_MANUAL_IO_RANGE)s:* Pixels are scaled from their original
   range to the new target range. Options are available to handle values
   outside of the original range.
-  *%(M_DIVIDE_BY_IMAGE_MINIMUM)s:* Divide the intensity value of
   each pixel by the image’s minimum intensity value so that all pixel
   intensities are equal to or greater than 1. The rescaled image can
   serve as an illumination correction function in
   **CorrectIlluminationApply**.
-  *%(M_DIVIDE_BY_IMAGE_MAXIMUM)s:* Divide the intensity value of
   each pixel by the image’s maximum intensity value so that all pixel
   intensities are less than or equal to 1.
-  *%(M_DIVIDE_BY_VALUE)s:* Divide the intensity value of each pixel
   by a value that you choose.
-  *%(M_DIVIDE_BY_MEASUREMENT)s:* The intensity value of each pixel
   is divided by some previously calculated measurement. This
   measurement can be the output of some other module or can be a value
   loaded by the **Metadata** module.
-  *%(M_SCALE_BY_IMAGE_MAXIMUM)s:* Scale an image so that its
   maximum value is the same as the maximum value within the reference
   image."""
            % globals(),
        )

        self.wants_automatic_low = Choice(
            "Method to calculate the minimum intensity",
            LOW_ALL,
            doc="""\
*(Used only if “%(M_MANUAL_IO_RANGE)s” is selected)*

This setting controls how the minimum intensity is determined.

-  *%(CUSTOM_VALUE)s:* Enter the minimum intensity manually below.
-  *%(LOW_EACH_IMAGE)s*: use the lowest intensity in this image as the
   minimum intensity for rescaling
-  *%(LOW_ALL_IMAGES)s*: use the lowest intensity from all images in
   the image group or the experiment if grouping is not being used.
   Note that choosing this option may have undesirable results for a
   large ungrouped experiment split into a number of batches. Each batch
   will open all images from the chosen channel at the start of the run.
   This sort of synchronized action may have a severe impact on your
   network file system.
"""
            % globals(),
        )

        self.wants_automatic_high = Choice(
            "Method to calculate the maximum intensity",
            HIGH_ALL,
            doc="""\
*(Used only if “%(M_MANUAL_IO_RANGE)s” is selected)*

This setting controls how the maximum intensity is determined.

-  *%(CUSTOM_VALUE)s*: Enter the maximum intensity manually below.
-  *%(HIGH_EACH_IMAGE)s*: Use the highest intensity in this image as
   the maximum intensity for rescaling
-  *%(HIGH_ALL_IMAGES)s*: Use the highest intensity from all images in
   the image group or the experiment if grouping is not being used.
   Note that choosing this option may have undesirable results for a
   large ungrouped experiment split into a number of batches. Each batch
   will open all images from the chosen channel at the start of the run.
   This sort of synchronized action may have a severe impact on your
   network file system.
"""
            % globals(),
        )

        self.source_low = Float(
            "Lower intensity limit for the input image",
            0,
            doc="""\
*(Used only if "{RESCALE_METHOD}" is "{M_MANUAL_INPUT_RANGE}" or "{M_MANUAL_IO_RANGE}" and
"{WANTS_AUTOMATIC_LOW}" is "{CUSTOM_VALUE}")*

The value of pixels in the input image that you want to rescale to the minimum pixel
value in the output image. Pixel intensities less than this value in the input image are
also rescaled to the minimum pixel value in the output image.
""".format(
                **{
                    "CUSTOM_VALUE": CUSTOM_VALUE,
                    "M_MANUAL_INPUT_RANGE": M_MANUAL_INPUT_RANGE,
                    "M_MANUAL_IO_RANGE": M_MANUAL_IO_RANGE,
                    "RESCALE_METHOD": self.rescale_method.text,
                    "WANTS_AUTOMATIC_LOW": self.wants_automatic_low.text,
                }
            ),
        )

        self.source_high = Float(
            "Upper intensity limit for the input image",
            1,
            doc="""\
*(Used only if "{RESCALE_METHOD}" is "{M_MANUAL_INPUT_RANGE}" or "{M_MANUAL_IO_RANGE}" and
"{WANTS_AUTOMATIC_HIGH}" is "{CUSTOM_VALUE}")*

The value of pixels in the input image that you want to rescale to the maximum pixel
value in the output image. Pixel intensities less than this value in the input image are
also rescaled to the maximum pixel value in the output image.
""".format(
                **{
                    "CUSTOM_VALUE": CUSTOM_VALUE,
                    "M_MANUAL_INPUT_RANGE": M_MANUAL_INPUT_RANGE,
                    "M_MANUAL_IO_RANGE": M_MANUAL_IO_RANGE,
                    "RESCALE_METHOD": self.rescale_method.text,
                    "WANTS_AUTOMATIC_HIGH": self.wants_automatic_high.text,
                }
            ),
        )

        self.source_scale = FloatRange(
            "Intensity range for the input image",
            (0, 1),
            doc="""\
*(Used only if "{RESCALE_METHOD}" is "{M_MANUAL_INPUT_RANGE}" or "{M_MANUAL_IO_RANGE}" and
"{WANTS_AUTOMATIC_LOW}" is "{CUSTOM_VALUE}" and "{WANTS_AUTOMATIC_HIGH}" is "{CUSTOM_VALUE}")*

Select the range of pixel intensities in the input image to rescale to the range of output
pixel intensities. Pixel intensities outside this range will be clipped to the new minimum
or maximum, respectively.
""".format(
                **{
                    "CUSTOM_VALUE": CUSTOM_VALUE,
                    "M_MANUAL_INPUT_RANGE": M_MANUAL_INPUT_RANGE,
                    "M_MANUAL_IO_RANGE": M_MANUAL_IO_RANGE,
                    "RESCALE_METHOD": self.rescale_method.text,
                    "WANTS_AUTOMATIC_HIGH": self.wants_automatic_high.text,
                    "WANTS_AUTOMATIC_LOW": self.wants_automatic_low.text,
                }
            ),
        )

        self.dest_scale = FloatRange(
            "Intensity range for the output image",
            (0, 1),
            doc="""\
*(Used only if "{RESCALE_METHOD}" is "{M_MANUAL_IO_RANGE}")*

Set the range of pixel intensities in the output image. The minimum pixel intensity of the input
image will be rescaled to the minimum output image intensity. The maximum pixel intensity of the
output image will be rescaled to the maximum output image intensity.
""".format(
                **{
                    "M_MANUAL_IO_RANGE": M_MANUAL_IO_RANGE,
                    "RESCALE_METHOD": self.rescale_method.text,
                }
            ),
        )

        self.matching_image_name = ImageSubscriber(
            "Select image to match in maximum intensity",
            "None",
            doc="""\
*(Used only if “%(M_SCALE_BY_IMAGE_MAXIMUM)s” is selected)*

Select the image whose maximum you want the rescaled image to match.
"""
            % globals(),
        )

        self.divisor_value = Float(
            "Divisor value",
            1,
            minval=numpy.finfo(float).eps,
            doc="""\
*(Used only if “%(M_DIVIDE_BY_VALUE)s” is selected)*

Enter the value to use as the divisor for the final image.
"""
            % globals(),
        )

        self.divisor_measurement = Measurement(
            "Divisor measurement",
            lambda: "Image",
            doc="""\
*(Used only if “%(M_DIVIDE_BY_MEASUREMENT)s” is selected)*

Select the measurement value to use as the divisor for the final image.
"""
            % globals(),
        )

    def settings(self):
        __settings__ = super(RescaleIntensity, self).settings()

        return __settings__ + [
            self.rescale_method,
            self.wants_automatic_low,
            self.wants_automatic_high,
            self.source_low,
            self.source_high,
            self.source_scale,
            self.dest_scale,
            self.matching_image_name,
            self.divisor_value,
            self.divisor_measurement,
        ]

    def visible_settings(self):
        __settings__ = super(RescaleIntensity, self).visible_settings()

        __settings__ += [self.rescale_method]
        if self.rescale_method in (M_MANUAL_INPUT_RANGE, M_MANUAL_IO_RANGE):
            __settings__ += [self.wants_automatic_low]
            if self.wants_automatic_low.value == CUSTOM_VALUE:
                if self.wants_automatic_high != CUSTOM_VALUE:
                    __settings__ += [self.source_low, self.wants_automatic_high]
                else:
                    __settings__ += [self.wants_automatic_high, self.source_scale]
            else:
                __settings__ += [self.wants_automatic_high]
                if self.wants_automatic_high == CUSTOM_VALUE:
                    __settings__ += [self.source_high]
        if self.rescale_method == M_MANUAL_IO_RANGE:
            __settings__ += [self.dest_scale]

        if self.rescale_method == M_SCALE_BY_IMAGE_MAXIMUM:
            __settings__ += [self.matching_image_name]
        elif self.rescale_method == M_DIVIDE_BY_MEASUREMENT:
            __settings__ += [self.divisor_measurement]
        elif self.rescale_method == M_DIVIDE_BY_VALUE:
            __settings__ += [self.divisor_value]
        return __settings__

    def set_automatic_minimum(self, image_set_list, value):
        d = self.get_dictionary(image_set_list)
        d[LOW_ALL_IMAGES] = value

    def get_automatic_minimum(self, image_set_list):
        d = self.get_dictionary(image_set_list)
        return d[LOW_ALL_IMAGES]

    def set_automatic_maximum(self, image_set_list, value):
        d = self.get_dictionary(image_set_list)
        d[HIGH_ALL_IMAGES] = value

    def get_automatic_maximum(self, image_set_list):
        d = self.get_dictionary(image_set_list)
        return d[HIGH_ALL_IMAGES]

    def prepare_group(self, workspace, grouping, image_numbers):
        """Handle initialization per-group

        pipeline - the pipeline being run
        image_set_list - the list of image sets for the whole experiment
        grouping - a dictionary that describes the key for the grouping.
                   For instance, { 'Metadata_Row':'A','Metadata_Column':'01'}
        image_numbers - a sequence of the image numbers within the
                   group (image sets can be retrieved as
                   image_set_list.get_image_set(image_numbers[i]-1)

        We use prepare_group to compute the minimum or maximum values
        among all images in the group for certain values of
        "wants_automatic_[low,high]".
        """
        if (
            self.wants_automatic_high != HIGH_ALL_IMAGES
            and self.wants_automatic_low != LOW_ALL_IMAGES
        ):
            return True

        title = "#%d: RescaleIntensity for %s" % (self.module_num, self.x_name.value)
        message = (
            "RescaleIntensity will process %d images while "
            "preparing for run" % (len(image_numbers))
        )
        min_value = None
        max_value = None
        for w in workspace.pipeline.run_group_with_yield(
            workspace, grouping, image_numbers, self, title, message
        ):
            image_set = w.image_set
            image = image_set.get_image(
                self.x_name.value, must_be_grayscale=True, cache=False
            )
            if self.wants_automatic_high == HIGH_ALL_IMAGES:
                if image.has_mask:
                    vmax = numpy.max(image.pixel_data[image.mask])
                else:
                    vmax = numpy.max(image.pixel_data)
                    max_value = vmax if max_value is None else max(max_value, vmax)

            if self.wants_automatic_low == LOW_ALL_IMAGES:
                if image.has_mask:
                    vmin = numpy.min(image.pixel_data[image.mask])
                else:
                    vmin = numpy.min(image.pixel_data)
                    min_value = vmin if min_value is None else min(min_value, vmin)

        if self.wants_automatic_high == HIGH_ALL_IMAGES:
            self.set_automatic_maximum(workspace.image_set_list, max_value)
        if self.wants_automatic_low == LOW_ALL_IMAGES:
            self.set_automatic_minimum(workspace.image_set_list, min_value)

    def is_aggregation_module(self):
        """We scan through all images in a group in some cases"""
        return (self.wants_automatic_high == HIGH_ALL_IMAGES) or (
            self.wants_automatic_low == LOW_ALL_IMAGES
        )

    def run(self, workspace):
        input_image = workspace.image_set.get_image(self.x_name.value)

        if self.rescale_method == M_STRETCH:
            output_image = self.stretch(input_image)
        elif self.rescale_method == M_MANUAL_INPUT_RANGE:
            output_image = self.manual_input_range(input_image, workspace)
        elif self.rescale_method == M_MANUAL_IO_RANGE:
            output_image = self.manual_io_range(input_image, workspace)
        elif self.rescale_method == M_DIVIDE_BY_IMAGE_MINIMUM:
            output_image = self.divide_by_image_minimum(input_image)
        elif self.rescale_method == M_DIVIDE_BY_IMAGE_MAXIMUM:
            output_image = self.divide_by_image_maximum(input_image)
        elif self.rescale_method == M_DIVIDE_BY_VALUE:
            output_image = self.divide_by_value(input_image)
        elif self.rescale_method == M_DIVIDE_BY_MEASUREMENT:
            output_image = self.divide_by_measurement(workspace, input_image)
        elif self.rescale_method == M_SCALE_BY_IMAGE_MAXIMUM:
            output_image = self.scale_by_image_maximum(workspace, input_image)

        rescaled_image = Image(
            output_image,
            parent_image=input_image,
            convert=False,
            dimensions=input_image.dimensions,
        )

        workspace.image_set.add(self.y_name.value, rescaled_image)

        if self.show_window:
            workspace.display_data.x_data = input_image.pixel_data

            workspace.display_data.y_data = output_image

            workspace.display_data.dimensions = input_image.dimensions

    def display(self, workspace, figure):
        figure.set_subplots((2, 1))

        figure.set_subplots(
            dimensions=workspace.display_data.dimensions, subplots=(2, 1)
        )

        figure.subplot_imshow(
            image=workspace.display_data.x_data,
            title=self.x_name.value,
            normalize=False,
            colormap="gray",
            x=0,
            y=0,
        )

        figure.subplot_imshow(
            image=workspace.display_data.y_data,
            sharexy=figure.subplot(0, 0),
            title=self.y_name.value,
            colormap="gray",
            normalize=False,
            x=1,
            y=0,
        )

    def rescale(self, image, in_range, out_range=(0.0, 1.0)):
        data = 1.0 * image.pixel_data

        rescaled = skimage.exposure.rescale_intensity(
            data, in_range=in_range, out_range=out_range
        )

        return rescaled

    def stretch(self, input_image):
        data = input_image.pixel_data
        mask = input_image.mask

        if input_image.multichannel:
            splitaxis = data.ndim - 1
            singlechannels = numpy.split(data, data.shape[-1], splitaxis)
            newchannels = []
            for channel in singlechannels:
                channel = numpy.squeeze(channel, axis=splitaxis)
                if (masked_channel := channel[mask]).size == 0:
                    in_range = (0, 1)
                else:
                    in_range = (min(masked_channel), max(masked_channel))

                channelholder = Image(channel, convert=False)

                rescaled = self.rescale(channelholder, in_range)
                newchannels.append(rescaled)
            full_rescaled = numpy.stack(newchannels, axis=-1)
            return full_rescaled
        if (masked_data := data[mask]).size == 0:
            in_range = (0, 1)
        else:
            in_range = (min(masked_data), max(masked_data))
        return self.rescale(input_image, in_range)

    def manual_input_range(self, input_image, workspace):
        in_range = self.get_source_range(input_image, workspace)

        return self.rescale(input_image, in_range)

    def manual_io_range(self, input_image, workspace):
        in_range = self.get_source_range(input_image, workspace)

        out_range = (self.dest_scale.min, self.dest_scale.max)

        return self.rescale(input_image, in_range, out_range)

    def divide(self, data, value):
        if value == 0.0:
            raise ZeroDivisionError("Cannot divide pixel intensity by 0.")

        return data / float(value)

    def divide_by_image_minimum(self, input_image):
        data = input_image.pixel_data

        if (masked_data := data[input_image.mask]).size == 0:
            src_min = 0
        else:
            src_min = numpy.min(masked_data)

        return self.divide(data, src_min)

    def divide_by_image_maximum(self, input_image):
        data = input_image.pixel_data

        if (masked_data := data[input_image.mask]).size == 0:
            src_max = 1
        else:
            src_max = numpy.max(masked_data)

        return self.divide(data, src_max)

    def divide_by_value(self, input_image):
        return self.divide(input_image.pixel_data, self.divisor_value.value)

    def divide_by_measurement(self, workspace, input_image):
        m = workspace.measurements

        value = m.get_current_image_measurement(self.divisor_measurement.value)

        return self.divide(input_image.pixel_data, value)

    def scale_by_image_maximum(self, workspace, input_image):
        ###
        # Scale the image by the maximum of another image
        #
        # Find the maximum value within the unmasked region of the input
        # and reference image. Multiply by the reference maximum, divide
        # by the input maximum to scale the input image to the same
        # range as the reference image
        ###
        if (masked_input := input_image.pixel_data[input_image.mask]).size == 0:
            return input_image.pixel_data
        else:
            image_max = numpy.max(masked_input)

        if image_max == 0:
            return input_image.pixel_data

        reference_image = workspace.image_set.get_image(self.matching_image_name.value)

        if (masked_ref := reference_image.pixel_data[reference_image.mask]).size == 0:
            reference_max = 1
        else:
            reference_max = numpy.max(masked_ref)

        return self.divide(input_image.pixel_data * reference_max, image_max)

    def get_source_range(self, input_image, workspace):
        """Get the source range, accounting for automatically computed values"""
        if (
            self.wants_automatic_high == CUSTOM_VALUE
            and self.wants_automatic_low == CUSTOM_VALUE
        ):
            return self.source_scale.min, self.source_scale.max

        if (
            self.wants_automatic_low == LOW_EACH_IMAGE
            or self.wants_automatic_high == HIGH_EACH_IMAGE
        ):
            input_pixels = input_image.pixel_data
            if input_image.has_mask:
                input_pixels = input_pixels[input_image.mask]
                if input_pixels.size == 0:
                    return 0, 1

        if self.wants_automatic_low == LOW_ALL_IMAGES:
            src_min = self.get_automatic_minimum(workspace.image_set_list)
        elif self.wants_automatic_low == LOW_EACH_IMAGE:
            src_min = numpy.min(input_pixels)
        else:
            src_min = self.source_low.value
        if self.wants_automatic_high.value == HIGH_ALL_IMAGES:
            src_max = self.get_automatic_maximum(workspace.image_set_list)
        elif self.wants_automatic_high == HIGH_EACH_IMAGE:
            src_max = numpy.max(input_pixels)
        else:
            src_max = self.source_high.value
        return src_min, src_max

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 1:
            #
            # wants_automatic_low (# 3) and wants_automatic_high (# 4)
            # changed to a choice: yes = each, no = custom
            #
            setting_values = list(setting_values)

            for i, automatic in ((3, LOW_EACH_IMAGE), (4, HIGH_EACH_IMAGE)):
                if setting_values[i] == "Yes":
                    setting_values[i] = automatic
                else:
                    setting_values[i] = CUSTOM_VALUE

            variable_revision_number = 2

        if variable_revision_number == 2:
            #
            # removed settings low_truncation_choice, custom_low_truncation,
            # high_truncation_choice, custom_high_truncation (#9-#12)
            #
            setting_values = setting_values[:9] + setting_values[13:]

            variable_revision_number = 3

        return setting_values, variable_revision_number
