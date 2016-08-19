'''<b>RescaleIntensity</b> changes the intensity range of an image to your
desired specifications.
<hr>
This module lets you rescale the intensity of the input images by any of several
methods. You should use caution when interpreting intensity and texture measurements
derived from images that have been rescaled because certain options for this module
do not preserve the relative intensities from image to image.
'''

import numpy as np
from centrosome.filter import stretch

import cellprofiler.image as cpi
import cellprofiler.module as cpm
import cellprofiler.measurement as cpmeas
import cellprofiler.preferences as cpprefs
import cellprofiler.setting as cps

M_STRETCH = 'Stretch each image to use the full intensity range'
M_MANUAL_INPUT_RANGE = 'Choose specific values to be reset to the full intensity range'
M_MANUAL_IO_RANGE = 'Choose specific values to be reset to a custom range'
M_DIVIDE_BY_IMAGE_MINIMUM = "Divide by the image's minimum"
M_DIVIDE_BY_IMAGE_MAXIMUM = "Divide by the image's maximum"
M_DIVIDE_BY_VALUE = 'Divide each image by the same value'
M_DIVIDE_BY_MEASUREMENT = 'Divide each image by a previously calculated value'
M_SCALE_BY_IMAGE_MAXIMUM = "Match the image's maximum to another image's maximum"
M_CONVERT_TO_8_BIT = 'Convert to 8 bit'

M_ALL = [M_STRETCH, M_MANUAL_INPUT_RANGE, M_MANUAL_IO_RANGE,
         M_DIVIDE_BY_IMAGE_MINIMUM, M_DIVIDE_BY_IMAGE_MAXIMUM,
         M_DIVIDE_BY_VALUE, M_DIVIDE_BY_MEASUREMENT,
         M_SCALE_BY_IMAGE_MAXIMUM, M_CONVERT_TO_8_BIT]

R_SCALE = 'Scale similarly to others'
R_MASK = 'Mask pixels'
R_SET_TO_ZERO = 'Set to zero'
R_SET_TO_CUSTOM = 'Set to custom value'
R_SET_TO_ONE = 'Set to one'

LOW_ALL_IMAGES = 'Minimum of all images'
LOW_EACH_IMAGE = 'Minimum for each image'
CUSTOM_VALUE = 'Custom'
LOW_ALL = [CUSTOM_VALUE, LOW_EACH_IMAGE, LOW_ALL_IMAGES]

HIGH_ALL_IMAGES = 'Maximum of all images'
HIGH_EACH_IMAGE = 'Maximum for each image'

HIGH_ALL = [CUSTOM_VALUE, HIGH_EACH_IMAGE, HIGH_ALL_IMAGES]


class RescaleIntensity(cpm.Module):
    module_name = "RescaleIntensity"
    category = "Image Processing"
    variable_revision_number = 2

    def create_settings(self):
        self.image_name = cps.ImageNameSubscriber(
                "Select the input image", cps.NONE, doc=
                '''Select the image to be rescaled.''')

        self.rescaled_image_name = cps.ImageNameProvider(
                "Name the output image", "RescaledBlue", doc=
                '''Enter the name of output rescaled image.''')

        self.rescale_method = cps.Choice(
                'Rescaling method',
                choices=M_ALL, doc='''
            There are a number of options for rescaling the input image:
            <ul>
            <li><i>%(M_STRETCH)s:</i> Find the minimum and maximum values within the unmasked part of the image
            (or the whole image if there is no mask) and rescale every pixel so that
            the minimum has an intensity of zero and the maximum has an intensity of one.</li>
            <li><i>%(M_MANUAL_INPUT_RANGE)s:</i> Pixels are
            scaled from their user-specified original range to the range 0 to 1.
            Options are available to handle values outside of the original range.<br>
            To convert 12-bit images saved in 16-bit format to the correct range,
            use the range 0 to 0.0625. The value 0.0625 is equivalent
            to 2<sup>12</sup> divided by 2<sup>16</sup>, so it will convert a 16 bit image containing
            only 12 bits of data to the proper range.</li>
            <li><i>%(M_MANUAL_IO_RANGE)s:</i> Pixels are scaled from their original range to
            the new target range. Options are available to handle values outside
            of the original range.</li>
            <li><i>%(M_DIVIDE_BY_IMAGE_MINIMUM)s:</i> Divide the intensity value of each pixel
            by the image's minimum intensity value so that all pixel intensities are equal to or
            greater than 1. The rescaled image can serve as an illumination correction function in
            <b>CorrectIlluminationApply</b>.</li>
            <li><i>%(M_DIVIDE_BY_IMAGE_MAXIMUM)s:</i> Divide the intensity value of each pixel by the
            image's maximum intensity value so that all pixel intensities are less than or equal to 1.</li>
            <li><i>%(M_DIVIDE_BY_VALUE)s:</i> Divide the intensity value of each pixel by the value entered.</li>
            <li><i>%(M_DIVIDE_BY_MEASUREMENT)s:</i> The intensity value of each pixel is divided by some
            previously calculated measurement. This measurement can be the output of some other module
            or can be a value loaded by the <b>Metadata</b> module.</li>
            <li><i>%(M_SCALE_BY_IMAGE_MAXIMUM)s:</i> Scale an image so that its maximum value is the
            same as the maximum value within the reference image.</li>
            <li><i>%(M_CONVERT_TO_8_BIT)s:</i> Images in CellProfiler are normally stored as a floating
            point number in the range of 0 to 1. This option converts these images to class uint8,
            meaning an 8 bit integer in the range of 0 to 255,
            reducing the amount of memory required to store the image. <i>Warning:</i> Most
            CellProfiler modules require the incoming image to be in the standard 0
            to 1 range, so this conversion may cause downstream modules to behave
            in unexpected ways.</li>
            </ul>''' % globals())

        self.wants_automatic_low = cps.Choice(
                'Method to calculate the minimum intensity',
                LOW_ALL, doc="""
            <i>(Used only if "%(M_MANUAL_IO_RANGE)s" is selected)</i><br>
            This setting controls how the minimum intensity is determined.
            <ul>
            <li><i>%(CUSTOM_VALUE)s:</i> Enter the minimum intensity manually below.</li>
            <li><i>%(LOW_EACH_IMAGE)s</i>: use the lowest intensity in this image
            as the minimum intensity for rescaling</li>
            <li><i>%(LOW_ALL_IMAGES)s</i>: use the lowest intensity from all images
            in the image group or the experiment if grouping is not being used.
            <b>Note:</b> Choosing this option may have undesirable results for
            a large ungrouped experiment split into a number of batches. Each batch
            will open all images from the chosen channel at the start of the run.
            This sort of synchronized action may have a severe impact on your
            network file system.</li>
            </ul>
            """ % globals())

        self.wants_automatic_high = cps.Choice(
                'Method to calculate the maximum intensity',
                HIGH_ALL, doc="""
            <i>(Used only if "%(M_MANUAL_IO_RANGE)s" is selected)</i><br>
            This setting controls how the maximum intensity is determined.
            <ul>
            <li><i>%(CUSTOM_VALUE)s</i>: Enter the maximum intensity manually below.</li>
            <li><i>%(HIGH_EACH_IMAGE)s</i>: Use the highest intensity in this image
            as the maximum intensity for rescaling</li>
            <li><i>%(HIGH_ALL_IMAGES)s</i>: Use the highest intensity from all images
            in the image group or the experiment if grouping is not being used.
            <b>Note:</b> Choosing this option may have undesirable results for
            a large ungrouped experiment split into a number of batches. Each batch
            will open all images from the chosen channel at the start of the run.
            This sort of synchronized action may have a severe impact on your
            network file system.</li>
            </ul>
            """ % globals())

        self.source_low = cps.Float('Lower intensity limit for the input image', 0)

        self.source_high = cps.Float('Upper intensity limit for the input image', 1)

        self.source_scale = cps.FloatRange('Intensity range for the input image', (0, 1))

        self.dest_scale = cps.FloatRange('Intensity range for the output image', (0, 1))

        self.low_truncation_choice = cps.Choice(
                'Method to rescale pixels below the lower limit',
                [R_MASK, R_SET_TO_ZERO, R_SET_TO_CUSTOM, R_SCALE], doc='''
            <i>(Used only if "%(M_MANUAL_IO_RANGE)s" is selected)</i><br>
            There are several ways to handle values less than the lower limit of the intensity range:
            <ul>
            <li><i>%(R_MASK)s:</i> Creates a mask for the output image. All pixels below
            the lower limit will be masked out.</li>
            <li><i>%(R_SET_TO_ZERO)s:</i> Sets all pixels below the lower limit to zero.</li>
            <li><i>%(R_SET_TO_CUSTOM)s:</i> Sets all pixels below the lower limit to a custom
            value.</li>
            <li><i>%(R_SCALE)s:</i> Scales pixels with values below the lower limit
            using the same offset and divisor as other pixels. The results
            will be less than zero.</li>
            </ul>''' % globals())

        self.custom_low_truncation = cps.Float(
                "Custom value for pixels below lower limit", 0, doc="""
            <i>(Used only if "%(M_MANUAL_IO_RANGE)s" and "%(R_SET_TO_CUSTOM)s are selected)</i><br>
            enter the custom value to be assigned to pixels with values below the lower limit.""" % globals())

        self.high_truncation_choice = cps.Choice(
                'Method to rescale pixels above the upper limit',
                [R_MASK, R_SET_TO_ONE, R_SET_TO_CUSTOM, R_SCALE], doc="""
            <i>(Used only if "%(M_MANUAL_IO_RANGE)s" is selected)</i><br>
            There are several ways to handle values greater than the upper limit of the intensity range;
            Options are described in the Help for the equivalent lower limit question.""" % globals())

        self.custom_high_truncation = cps.Float(
                "Custom value for pixels above upper limit", 0, doc="""
            <i>(Used only if "%(M_MANUAL_IO_RANGE)s" and "%(R_SET_TO_CUSTOM)s are selected)</i><br>
            Enter the custom value to be assigned to pixels with values above the upper limit.""" % globals())

        self.matching_image_name = cps.ImageNameSubscriber(
                "Select image to match in maximum intensity", cps.NONE, doc="""
            <i>(Used only if "%(M_SCALE_BY_IMAGE_MAXIMUM)s" is selected)</i><br>
            Select the image whose maximum you want the rescaled image to match.""" % globals())

        self.divisor_value = cps.Float(
                "Divisor value",
                1, minval=np.finfo(float).eps, doc="""
            <i>(Used only if "%(M_DIVIDE_BY_VALUE)s" is selected)</i><br>
            Enter the value to use as the divisor for the final image.""" % globals())

        self.divisor_measurement = cps.Measurement(
                "Divisor measurement",
                lambda: cpmeas.IMAGE, doc="""
            <i>(Used only if "%(M_DIVIDE_BY_MEASUREMENT)s" is selected)</i><br>
            Select the measurement value to use as the divisor for the final image.""" % globals())

    def settings(self):
        return [self.image_name, self.rescaled_image_name, self.rescale_method,
                self.wants_automatic_low, self.wants_automatic_high,
                self.source_low, self.source_high,
                self.source_scale, self.dest_scale, self.low_truncation_choice,
                self.custom_low_truncation, self.high_truncation_choice,
                self.custom_high_truncation, self.matching_image_name,
                self.divisor_value, self.divisor_measurement]

    def visible_settings(self):
        result = [self.image_name, self.rescaled_image_name,
                  self.rescale_method]
        if self.rescale_method in (M_MANUAL_INPUT_RANGE, M_MANUAL_IO_RANGE):
            result += [self.wants_automatic_low]
            if self.wants_automatic_low.value == CUSTOM_VALUE:
                if self.wants_automatic_high != CUSTOM_VALUE:
                    result += [self.source_low, self.wants_automatic_high]
                else:
                    result += [self.wants_automatic_high, self.source_scale]
            else:
                result += [self.wants_automatic_high]
                if self.wants_automatic_high == CUSTOM_VALUE:
                    result += [self.source_high]
        if self.rescale_method == M_MANUAL_IO_RANGE:
            result += [self.dest_scale]
        if self.rescale_method in (M_MANUAL_INPUT_RANGE, M_MANUAL_IO_RANGE):
            result += [self.low_truncation_choice]
            if self.low_truncation_choice.value == R_SET_TO_CUSTOM:
                result += [self.custom_low_truncation]
            result += [self.high_truncation_choice]
            if self.high_truncation_choice.value == R_SET_TO_CUSTOM:
                result += [self.custom_high_truncation]

        if self.rescale_method == M_SCALE_BY_IMAGE_MAXIMUM:
            result += [self.matching_image_name]
        elif self.rescale_method == M_DIVIDE_BY_MEASUREMENT:
            result += [self.divisor_measurement]
        elif self.rescale_method == M_DIVIDE_BY_VALUE:
            result += [self.divisor_value]
        return result

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
        '''Handle initialization per-group

        pipeline - the pipeline being run
        image_set_list - the list of image sets for the whole experiment
        grouping - a dictionary that describes the key for the grouping.
                   For instance, { 'Metadata_Row':'A','Metadata_Column':'01'}
        image_numbers - a sequence of the image numbers within the
                   group (image sets can be retreved as
                   image_set_list.get_image_set(image_numbers[i]-1)

        We use prepare_group to compute the minimum or maximum values
        among all images in the group for certain values of
        "wants_automatic_[low,high]".
        '''
        if (self.wants_automatic_high != HIGH_ALL_IMAGES and
                    self.wants_automatic_low != LOW_ALL_IMAGES):
            return True

        title = "#%d: RescaleIntensity for %s" % (
            self.module_num, self.image_name.value)
        message = ("RescaleIntensity will process %d images while "
                   "preparing for run" % (len(image_numbers)))
        min_value = None
        max_value = None
        for w in workspace.pipeline.run_group_with_yield(
                workspace, grouping, image_numbers, self, title, message):
            image_set = w.image_set
            image = image_set.get_image(self.image_name.value,
                                        must_be_grayscale=True,
                                        cache=False)
            if self.wants_automatic_high == HIGH_ALL_IMAGES:
                if image.has_mask:
                    vmax = np.max(image.pixel_data[image.mask])
                else:
                    vmax = np.max(image.pixel_data)
                    max_value = vmax if max_value is None else max(max_value, vmax)

            if self.wants_automatic_low == LOW_ALL_IMAGES:
                if image.has_mask:
                    vmin = np.min(image.pixel_data[image.mask])
                else:
                    vmin = np.min(image.pixel_data)
                    min_value = vmin if min_value is None else min(min_value, vmin)

        if self.wants_automatic_high == HIGH_ALL_IMAGES:
            self.set_automatic_maximum(workspace.image_set_list, max_value)
        if self.wants_automatic_low == LOW_ALL_IMAGES:
            self.set_automatic_minimum(workspace.image_set_list, min_value)

    def is_aggregation_module(self):
        '''We scan through all images in a group in some cases'''
        return ((self.wants_automatic_high == HIGH_ALL_IMAGES) or
                (self.wants_automatic_low == LOW_ALL_IMAGES))

    def run(self, workspace):
        input_image = workspace.image_set.get_image(self.image_name.value)
        output_mask = None
        if self.rescale_method == M_STRETCH:
            output_image = self.stretch(input_image)
        elif self.rescale_method == M_MANUAL_INPUT_RANGE:
            output_image, output_mask = self.manual_input_range(input_image, workspace)
        elif self.rescale_method == M_MANUAL_IO_RANGE:
            output_image, output_mask = self.manual_io_range(input_image, workspace)
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
        elif self.rescale_method == M_CONVERT_TO_8_BIT:
            output_image = self.convert_to_8_bit(input_image)
        if output_mask is not None:
            rescaled_image = cpi.Image(output_image,
                                       mask=output_mask,
                                       parent_image=input_image,
                                       convert=False,
                                       dimensions=input_image.dimensions)
        else:
            rescaled_image = cpi.Image(output_image,
                                       parent_image=input_image,
                                       convert=False,
                                       dimensions=input_image.dimensions)
        workspace.image_set.add(self.rescaled_image_name.value, rescaled_image)
        if self.show_window:
            workspace.display_data.image_data = [input_image.pixel_data,
                                                 rescaled_image.pixel_data]
            workspace.display_data.dimensionality = input_image.dimensions

    def display(self, workspace, figure):
        '''Display the input image and rescaled image'''
        figure.set_subplots((2, 1))

        for image_name, i, j in ((self.image_name, 0, 0),
                                 (self.rescaled_image_name, 1, 0)):
            image_name = image_name.value
            pixel_data = workspace.display_data.image_data[i]
            if workspace.display_data.dimensionality == 3:
                figure.subplot_imshow_grayscale(
                    i,
                    j,
                    pixel_data[0],
                    title=image_name,
                    vmin=0,
                    vmax=1,
                    sharexy=figure.subplot(0, 0)
                )
            elif pixel_data.ndim == 2:
                figure.subplot_imshow_grayscale(i, j, pixel_data,
                                                title=image_name,
                                                vmin=0, vmax=1,
                                                sharexy=figure.subplot(0, 0))
            else:
                figure.subplot_imshow(i, j, pixel_data, title=image_name,
                                      normalize=False,
                                      sharexy=figure.subplot(0, 0))

    def stretch(self, input_image):
        '''Stretch the input image to the range 0:1'''
        if input_image.has_mask:
            return stretch(input_image.pixel_data, input_image.mask)
        else:
            return stretch(input_image.pixel_data)

    def manual_input_range(self, input_image, workspace):
        '''Stretch the input image from the requested range to 0:1'''

        src_min, src_max = self.get_source_range(input_image, workspace)
        rescaled_image = ((input_image.pixel_data - src_min) /
                          (src_max - src_min))
        return self.truncate_values(input_image, rescaled_image, 0, 1)

    def manual_io_range(self, input_image, workspace):
        '''Stretch the input image using manual input and output values'''

        src_min, src_max = self.get_source_range(input_image, workspace)
        rescaled_image = ((input_image.pixel_data - src_min) /
                          (src_max - src_min))
        dest_min = self.dest_scale.min
        dest_max = self.dest_scale.max
        rescaled_image = rescaled_image * (dest_max - dest_min) + dest_min
        return self.truncate_values(input_image,
                                    rescaled_image,
                                    dest_min, dest_max)

    def divide_by_image_minimum(self, input_image):
        '''Divide the image by its minimum to get an illumination correction function'''

        if input_image.has_mask:
            src_min = np.min(input_image.pixel_data[input_image.mask])
        else:
            src_min = np.min(input_image.pixel_data)
        if src_min != 0:
            rescaled_image = input_image.pixel_data / src_min
        return rescaled_image

    def divide_by_image_maximum(self, input_image):
        '''Stretch the input image from 0 to the image maximum'''

        if input_image.has_mask:
            src_max = np.max(input_image.pixel_data[input_image.mask])
        else:
            src_max = np.max(input_image.pixel_data)
        if src_max != 0:
            rescaled_image = input_image.pixel_data / src_max
        return rescaled_image

    def divide_by_value(self, input_image):
        '''Divide the image by a user-specified value'''
        return input_image.pixel_data / self.divisor_value.value

    def divide_by_measurement(self, workspace, input_image):
        '''Divide the image by the value of an image measurement'''
        m = workspace.measurements
        value = m.get_current_image_measurement(self.divisor_measurement.value)
        return input_image.pixel_data / float(value)

    def scale_by_image_maximum(self, workspace, input_image):
        '''Scale the image by the maximum of another image

        Find the maximum value within the unmasked region of the input
        and reference image. Multiply by the reference maximum, divide
        by the input maximum to scale the input image to the same
        range as the reference image
        '''
        reference_image = workspace.image_set.get_image(self.matching_image_name.value)
        reference_pixels = reference_image.pixel_data
        if reference_image.has_mask:
            reference_pixels = reference_pixels[reference_image.mask]
        reference_max = np.max(reference_pixels)
        if input_image.has_mask:
            image_max = np.max(input_image.pixel_data[input_image.mask])
        else:
            image_max = np.max(input_image.pixel_data)
        if image_max == 0:
            return input_image.pixel_data
        return input_image.pixel_data * reference_max / image_max

    def convert_to_8_bit(self, input_image):
        '''Convert the image data to uint8'''
        return (input_image.pixel_data * 255).astype(np.uint8)

    def get_source_range(self, input_image, workspace):
        '''Get the source range, accounting for automatically computed values'''
        if (self.wants_automatic_high == CUSTOM_VALUE and
                    self.wants_automatic_low == CUSTOM_VALUE):
            return self.source_scale.min, self.source_scale.max

        if (self.wants_automatic_low == LOW_EACH_IMAGE or
                    self.wants_automatic_high == HIGH_EACH_IMAGE):
            input_pixels = input_image.pixel_data
            if input_image.has_mask:
                input_pixels = input_pixels[input_image.mask]

        if self.wants_automatic_low == LOW_ALL_IMAGES:
            src_min = self.get_automatic_minimum(workspace.image_set_list)
        elif self.wants_automatic_low == LOW_EACH_IMAGE:
            src_min = np.min(input_pixels)
        else:
            src_min = self.source_low.value
        if self.wants_automatic_high.value == HIGH_ALL_IMAGES:
            src_max = self.get_automatic_maximum(workspace.image_set_list)
        elif self.wants_automatic_high == HIGH_EACH_IMAGE:
            src_max = np.max(input_pixels)
        else:
            src_max = self.source_high.value
        return src_min, src_max

    def truncate_values(self, input_image, rescaled_image, target_min, target_max):
        '''Handle out of range values based on user settings

        input_image - the original input image
        rescaled_image - the pixel data after scaling
        target_min - values below this are out of range
        target_max - values above this are out of range

        returns the truncated pixel data and either a mask or None
        if the user doesn't want to mask out-of-range values
        '''

        if (self.low_truncation_choice == R_MASK or
                    self.high_truncation_choice == R_MASK):
            if input_image.has_mask:
                mask = input_image.mask.copy()
            else:
                mask = np.ones(rescaled_image.shape, bool)
            if self.low_truncation_choice == R_MASK:
                mask[rescaled_image < target_min] = False
            if self.high_truncation_choice == R_MASK:
                mask[rescaled_image > target_max] = False
        else:
            mask = None
        if self.low_truncation_choice == R_SET_TO_ZERO:
            rescaled_image[rescaled_image < target_min] = 0
        elif self.low_truncation_choice == R_SET_TO_CUSTOM:
            rescaled_image[rescaled_image < target_min] = \
                self.custom_low_truncation.value

        if self.high_truncation_choice == R_SET_TO_ONE:
            rescaled_image[rescaled_image > target_max] = 1
        elif self.high_truncation_choice == R_SET_TO_CUSTOM:
            rescaled_image[rescaled_image > target_max] = \
                self.custom_high_truncation.value
        if mask is not None and mask.ndim == 3:
            # Color image -> 3-d mask. Collapse the 3rd dimension
            # so any point is masked if any color fails
            mask = np.all(mask, 2)
        return rescaled_image, mask

    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        if from_matlab and variable_revision_number == 2:
            #
            # Added custom_low_truncation and custom_high_truncation
            #
            setting_values = (setting_values[:7] + ["0", "1"] +
                              setting_values[7:])
            variable_revision_number = 3
        if from_matlab and variable_revision_number == 3:
            #
            # Added load text name at the end
            #
            setting_values = setting_values + [cps.NONE]
            variable_revision_number = 4
        if from_matlab and variable_revision_number == 4:
            new_setting_values = (setting_values[:2] +
                                  [M_STRETCH,  # 2: rescale_method,
                                   cps.NO,  # 3: wants_automatic_low
                                   cps.NO,  # 4: wants_automatic_high
                                   "0",  # 5: source_low
                                   "1",  # 6: source_high
                                   "0,1",  # 7: source_scale
                                   "0,1",  # 8: dest_scale
                                   R_MASK,  # 9: low_truncation_choice
                                   "0",  # 10: custom_low_truncation
                                   R_MASK,  # 11: high_truncation_choice
                                   "1",  # 12: custom_high_truncation
                                   cps.NONE,  # 13: matching_image_name
                                   "1",  # 14: divisor_value
                                   cps.NONE  # 15: divisor_measurement
                                   ])
            code = setting_values[2][0]
            if code.upper() == 'S':
                new_setting_values[2] = M_STRETCH
            elif code.upper() == 'E':
                if setting_values[5] == "0" and setting_values[6] == "1":
                    new_setting_values[2] = M_MANUAL_INPUT_RANGE
                else:
                    new_setting_values[2] = M_MANUAL_IO_RANGE
                if setting_values[3].upper() == "AA":
                    new_setting_values[3] = LOW_ALL_IMAGES
                elif setting_values[3].upper() == "AE":
                    new_setting_values[3] = LOW_EACH_IMAGE
                else:
                    new_setting_values[3] = CUSTOM_VALUE
                    new_setting_values[5] = setting_values[3]
                if setting_values[4].upper() == "AA":
                    new_setting_values[4] = HIGH_ALL_IMAGES
                elif setting_values[4].upper() == "AE":
                    new_setting_values[4] = HIGH_EACH_IMAGE
                else:
                    new_setting_values[4] = CUSTOM_VALUE
                    new_setting_values[6] = setting_values[4]
                if all([x.upper() not in ("AA", "AE")
                        for x in setting_values[3:4]]):
                    # Both are manual, put them in the range variable
                    new_setting_values[7] = ",".join(setting_values[3:5])
                new_setting_values[8] = ",".join(setting_values[5:7])
                new_setting_values[9] = R_SET_TO_CUSTOM
                new_setting_values[10] = setting_values[7]
                new_setting_values[11] = R_SET_TO_CUSTOM
                new_setting_values[12] = setting_values[8]
            elif code.upper() == 'G':
                new_setting_values[2] = M_DIVIDE_BY_IMAGE_MINIMUM
            elif code.upper() == 'M':
                new_setting_values[2] = M_SCALE_BY_IMAGE_MAXIMUM
                new_setting_values[13] = setting_values[9]
            elif code.upper() == 'C':
                new_setting_values[2] = M_CONVERT_TO_8_BIT
            elif code.upper() == 'T':
                new_setting_values[2] = M_DIVIDE_BY_MEASUREMENT
                new_setting_values[15] = setting_values[10]
            setting_values = new_setting_values
            variable_revision_number = 2
            from_matlab = False
        if (not from_matlab) and (variable_revision_number == 1):
            #
            # wants_automatic_low (# 3) and wants_automatic_high (# 4)
            # changed to a choice: yes = each, no = custom
            #
            setting_values = list(setting_values)
            for i, automatic in ((3, LOW_EACH_IMAGE), (4, HIGH_EACH_IMAGE)):
                if setting_values[i] == cps.YES:
                    setting_values[i] = automatic
                else:
                    setting_values[i] = CUSTOM_VALUE
            variable_revision_number = 2
        return setting_values, variable_revision_number, from_matlab
