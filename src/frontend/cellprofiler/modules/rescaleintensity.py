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
from cellprofiler_core.image import Image
from cellprofiler_core.module import ImageProcessing
from cellprofiler_core.setting import Measurement
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.range import FloatRange
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.text import Float
from cellprofiler_library.opts.rescaleintensity import RescaleMethod, MinimumIntensityMethod, MaximumIntensityMethod, M_ALL, LOW_ALL, HIGH_ALL
from cellprofiler_library.modules._rescaleintensity import rescale_intensity
from cellprofiler_library.functions.image_processing import divide
# Legacy opts
# R_SCALE = "Scale similarly to others"
# R_MASK = "Mask pixels"
# R_SET_TO_ZERO = "Set to zero"
# R_SET_TO_CUSTOM = "Set to custom value"
# R_SET_TO_ONE = "Set to one"

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

-  *{M_STRETCH}:* Find the minimum and maximum values within the
   unmasked part of the image (or the whole image if there is no mask)
   and rescale every pixel so that the minimum has an intensity of zero
   and the maximum has an intensity of one. If performed on color images
   each channel will be considered separately.
-  *{M_MANUAL_INPUT_RANGE}:* Pixels are scaled from an original range
   (which you provide) to the range 0 to 1. Options are
   available to handle values outside of the original range.
   To convert 12-bit images saved in 16-bit format to the correct range,
   use the range 0 to 0.0625. The value 0.0625 is equivalent to
   2\ :sup:`12` divided by 2\ :sup:`16`, so it will convert a 16 bit
   image containing only 12 bits of data to the proper range.
-  *{M_MANUAL_IO_RANGE}:* Pixels are scaled from their original
   range to the new target range. Options are available to handle values
   outside of the original range.
-  *{M_DIVIDE_BY_IMAGE_MINIMUM}:* Divide the intensity value of
   each pixel by the image’s minimum intensity value so that all pixel
   intensities are equal to or greater than 1. The rescaled image can
   serve as an illumination correction function in
   **CorrectIlluminationApply**.
-  *{M_DIVIDE_BY_IMAGE_MAXIMUM}:* Divide the intensity value of
   each pixel by the image’s maximum intensity value so that all pixel
   intensities are less than or equal to 1.
-  *{M_DIVIDE_BY_VALUE}:* Divide the intensity value of each pixel
   by a value that you choose.
-  *{M_DIVIDE_BY_MEASUREMENT}:* The intensity value of each pixel
   is divided by some previously calculated measurement. This
   measurement can be the output of some other module or can be a value
   loaded by the **Metadata** module.
-  *{M_SCALE_BY_IMAGE_MAXIMUM}:* Scale an image so that its
   maximum value is the same as the maximum value within the reference
   image.""".format(
       **{
           "M_STRETCH": RescaleMethod.STRETCH.value,
           "M_MANUAL_INPUT_RANGE": RescaleMethod.MANUAL_INPUT_RANGE.value,
           "M_MANUAL_IO_RANGE": RescaleMethod.MANUAL_IO_RANGE.value,
           "M_DIVIDE_BY_IMAGE_MINIMUM": RescaleMethod.DIVIDE_BY_IMAGE_MINIMUM.value,
           "M_DIVIDE_BY_IMAGE_MAXIMUM": RescaleMethod.DIVIDE_BY_IMAGE_MAXIMUM.value,
           "M_DIVIDE_BY_VALUE": RescaleMethod.DIVIDE_BY_VALUE.value,
           "M_DIVIDE_BY_MEASUREMENT": RescaleMethod.DIVIDE_BY_MEASUREMENT.value,
           "M_SCALE_BY_IMAGE_MAXIMUM": RescaleMethod.SCALE_BY_IMAGE_MAXIMUM.value,
       }
   ),
        )

        self.wants_automatic_low = Choice(
            "Method to calculate the minimum intensity",
            LOW_ALL,
            doc="""\
*(Used only if “{M_MANUAL_IO_RANGE}” is selected)*

This setting controls how the minimum intensity is determined.

-  *{CUSTOM_VALUE}:* Enter the minimum intensity manually below.
-  *{LOW_EACH_IMAGE}*: use the lowest intensity in this image as the
   minimum intensity for rescaling
-  *{LOW_ALL_IMAGES}*: use the lowest intensity from all images in
   the image group or the experiment if grouping is not being used.
   Note that choosing this option may have undesirable results for a
   large ungrouped experiment split into a number of batches. Each batch
   will open all images from the chosen channel at the start of the run.
   This sort of synchronized action may have a severe impact on your
   network file system.
""".format(
    **{
        "M_MANUAL_IO_RANGE": RescaleMethod.MANUAL_IO_RANGE.value,
        "CUSTOM_VALUE": MinimumIntensityMethod.CUSTOM_VALUE.value,
        "LOW_EACH_IMAGE": MinimumIntensityMethod.EACH_IMAGE.value,
        "LOW_ALL_IMAGES": MinimumIntensityMethod.ALL_IMAGES.value,
    }
),
        )

        self.wants_automatic_high = Choice(
            "Method to calculate the maximum intensity",
            HIGH_ALL,
            doc="""\
*(Used only if “{M_MANUAL_IO_RANGE}” is selected)*

This setting controls how the maximum intensity is determined.

-  *{CUSTOM_VALUE}*: Enter the maximum intensity manually below.
-  *{HIGH_EACH_IMAGE}*: Use the highest intensity in this image as
   the maximum intensity for rescaling
-  *{HIGH_ALL_IMAGES}*: Use the highest intensity from all images in
   the image group or the experiment if grouping is not being used.
   Note that choosing this option may have undesirable results for a
   large ungrouped experiment split into a number of batches. Each batch
   will open all images from the chosen channel at the start of the run.
   This sort of synchronized action may have a severe impact on your
   network file system.
""".format(
    **{
        "M_MANUAL_IO_RANGE": RescaleMethod.MANUAL_IO_RANGE.value,
        "CUSTOM_VALUE": MaximumIntensityMethod.CUSTOM_VALUE.value,
        "HIGH_EACH_IMAGE": MaximumIntensityMethod.EACH_IMAGE.value,
        "HIGH_ALL_IMAGES": MaximumIntensityMethod.ALL_IMAGES.value,
    }
),
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
                    "CUSTOM_VALUE": MinimumIntensityMethod.CUSTOM_VALUE.value,
                    "M_MANUAL_INPUT_RANGE": RescaleMethod.MANUAL_INPUT_RANGE.value,
                    "M_MANUAL_IO_RANGE": RescaleMethod.MANUAL_IO_RANGE.value,
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
                    "CUSTOM_VALUE": MaximumIntensityMethod.CUSTOM_VALUE.value,
                    "M_MANUAL_INPUT_RANGE": RescaleMethod.MANUAL_INPUT_RANGE.value,
                    "M_MANUAL_IO_RANGE": RescaleMethod.MANUAL_IO_RANGE.value,
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
                    "CUSTOM_VALUE": MinimumIntensityMethod.CUSTOM_VALUE.value,
                    "M_MANUAL_INPUT_RANGE": RescaleMethod.MANUAL_INPUT_RANGE.value,
                    "M_MANUAL_IO_RANGE": RescaleMethod.MANUAL_IO_RANGE.value,
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
                    "M_MANUAL_IO_RANGE": RescaleMethod.MANUAL_IO_RANGE.value,
                    "RESCALE_METHOD": self.rescale_method.text,
                }
            ),
        )

        self.matching_image_name = ImageSubscriber(
            "Select image to match in maximum intensity",
            "None",
            doc="""\
*(Used only if “{M_SCALE_BY_IMAGE_MAXIMUM}” is selected)*

Select the image whose maximum you want the rescaled image to match.
""".format(
    **{
        "M_SCALE_BY_IMAGE_MAXIMUM": RescaleMethod.SCALE_BY_IMAGE_MAXIMUM.value,
    }
),
        )

        self.divisor_value = Float(
            "Divisor value",
            1,
            minval=numpy.finfo(float).eps,
            doc="""\
*(Used only if “{M_DIVIDE_BY_VALUE}” is selected)*

Enter the value to use as the divisor for the final image.
""".format(
    **{
        "M_DIVIDE_BY_VALUE": RescaleMethod.DIVIDE_BY_VALUE.value,
    }
),
        )

        self.divisor_measurement = Measurement(
            "Divisor measurement",
            lambda: "Image",
            doc="""\
*(Used only if “{M_DIVIDE_BY_MEASUREMENT}” is selected)*

Select the measurement value to use as the divisor for the final image.
""".format(
    **{
        "M_DIVIDE_BY_MEASUREMENT": RescaleMethod.DIVIDE_BY_MEASUREMENT.value,
    }
),
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
        if self.rescale_method in (RescaleMethod.MANUAL_INPUT_RANGE.value, RescaleMethod.MANUAL_IO_RANGE.value):
            __settings__ += [self.wants_automatic_low]
            if self.wants_automatic_low.value == MinimumIntensityMethod.CUSTOM_VALUE.value:
                if self.wants_automatic_high != MaximumIntensityMethod.CUSTOM_VALUE.value:
                    __settings__ += [self.source_low, self.wants_automatic_high]
                else:
                    __settings__ += [self.wants_automatic_high, self.source_scale]
            else:
                __settings__ += [self.wants_automatic_high]
                if self.wants_automatic_high == MaximumIntensityMethod.CUSTOM_VALUE.value:
                    __settings__ += [self.source_high]
        if self.rescale_method == RescaleMethod.MANUAL_IO_RANGE.value:
            __settings__ += [self.dest_scale]

        if self.rescale_method == RescaleMethod.SCALE_BY_IMAGE_MAXIMUM.value:
            __settings__ += [self.matching_image_name]
        elif self.rescale_method == RescaleMethod.DIVIDE_BY_MEASUREMENT.value:
            __settings__ += [self.divisor_measurement]
        elif self.rescale_method == RescaleMethod.DIVIDE_BY_VALUE.value:
            __settings__ += [self.divisor_value]
        return __settings__

    def set_automatic_minimum(self, image_set_list, value):
        d = self.get_dictionary(image_set_list)
        d[MinimumIntensityMethod.ALL_IMAGES.value] = value

    def get_automatic_minimum(self, image_set_list):
        d = self.get_dictionary(image_set_list)
        return d[MinimumIntensityMethod.ALL_IMAGES.value]

    def set_automatic_maximum(self, image_set_list, value):
        d = self.get_dictionary(image_set_list)
        d[MaximumIntensityMethod.ALL_IMAGES.value] = value

    def get_automatic_maximum(self, image_set_list):
        d = self.get_dictionary(image_set_list)
        return d[MaximumIntensityMethod.ALL_IMAGES.value]

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
            self.wants_automatic_high != MaximumIntensityMethod.ALL_IMAGES.value
            and self.wants_automatic_low != MinimumIntensityMethod.ALL_IMAGES.value
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
            if self.wants_automatic_high == MaximumIntensityMethod.ALL_IMAGES.value:
                if image.has_mask:
                    vmax = numpy.max(image.pixel_data[image.mask])
                else:
                    vmax = numpy.max(image.pixel_data)
                    max_value = vmax if max_value is None else max(max_value, vmax)

            if self.wants_automatic_low == MinimumIntensityMethod.ALL_IMAGES.value:
                if image.has_mask:
                    vmin = numpy.min(image.pixel_data[image.mask])
                else:
                    vmin = numpy.min(image.pixel_data)
                    min_value = vmin if min_value is None else min(min_value, vmin)

        if self.wants_automatic_high == MaximumIntensityMethod.ALL_IMAGES.value:
            self.set_automatic_maximum(workspace.image_set_list, max_value)
        if self.wants_automatic_low == MinimumIntensityMethod.ALL_IMAGES.value:
            self.set_automatic_minimum(workspace.image_set_list, min_value)

    def is_aggregation_module(self):
        """We scan through all images in a group in some cases"""
        return (self.wants_automatic_high == MaximumIntensityMethod.ALL_IMAGES.value) or (
            self.wants_automatic_low == MinimumIntensityMethod.ALL_IMAGES.value
        )

    def run(self, workspace):
        input_image = workspace.image_set.get_image(self.x_name.value)
        in_pixel_data = input_image.pixel_data
        in_mask = input_image.mask
        input_image_has_mask = input_image.has_mask
        in_multichannel = input_image.multichannel
        if self.rescale_method == RescaleMethod.DIVIDE_BY_MEASUREMENT.value:
            output_image = self.divide_by_measurement(workspace, input_image)
        else:
            divisor_value = self.divisor_value.value
            auto_high = self.wants_automatic_high.value
            auto_low = self.wants_automatic_low.value
            source_high = self.source_high.value
            source_low = self.source_low.value
            source_scale_min = self.source_scale.min
            source_scale_max = self.source_scale.max
            shared_dict = self.get_dictionary(workspace.image_set_list)
            reference_image_pixel_data = None
            reference_image_mask = None
            if self.rescale_method == RescaleMethod.SCALE_BY_IMAGE_MAXIMUM.value:
                reference_image = workspace.image_set.get_image(self.matching_image_name.value)
                reference_image_pixel_data = reference_image.pixel_data
                reference_image_mask = reference_image.mask
            output_image = rescale_intensity(
                self.rescale_method.value,
                in_pixel_data,
                in_mask,
                input_image_has_mask,
                in_multichannel,
                divisor_value,
                auto_high,
                auto_low,
                source_high,
                source_low,
                source_scale_min,
                source_scale_max,
                shared_dict,
                self.dest_scale.min,
                self.dest_scale.max,
                reference_image_pixel_data,
                reference_image_mask,
            )

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

    # TODO #5088 update this once measurement format is finalized
    def divide_by_measurement(self, workspace, input_image):
        m = workspace.measurements
        value = m.get_current_image_measurement(self.divisor_measurement.value)
        return divide(input_image.pixel_data, value)

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 1:
            #
            # wants_automatic_low (# 3) and wants_automatic_high (# 4)
            # changed to a choice: yes = each, no = custom
            #
            setting_values = list(setting_values)

            for i, automatic in ((3, MinimumIntensityMethod.EACH_IMAGE.value), (4, MaximumIntensityMethod.EACH_IMAGE.value)):
                if setting_values[i] == "Yes":
                    setting_values[i] = automatic
                else:
                    setting_values[i] = MaximumIntensityMethod.CUSTOM_VALUE.value

            variable_revision_number = 2

        if variable_revision_number == 2:
            #
            # removed settings low_truncation_choice, custom_low_truncation,
            # high_truncation_choice, custom_high_truncation (#9-#12)
            #
            setting_values = setting_values[:9] + setting_values[13:]

            variable_revision_number = 3

        return setting_values, variable_revision_number
