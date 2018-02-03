# coding=utf-8

"""
ImageMath
=========

**ImageMath** performs simple mathematical operations on image
intensities.

This module can perform addition, subtraction, multiplication, division,
or averaging of two or more image intensities, as well as inversion, log
transform, or scaling by a constant for individual image intensities.

Keep in mind that after the requested operations are carried out, the
final image may have a substantially different range of pixel
intensities than the original. CellProfiler assumes that the image is
scaled from 0 – 1 for object identification and display purposes, so
additional rescaling may be needed. Please see the **RescaleIntensity**
module for more scaling options.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============

See also
^^^^^^^^

See also **Threshold**, **RescaleIntensity**,
**CorrectIlluminationCalculate**.
"""

import inflect
import numpy
import skimage.util

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.setting

from cellprofiler.setting import YES, NO

O_ADD = "Add"
O_SUBTRACT = "Subtract"
O_DIFFERENCE = "Absolute Difference"
O_MULTIPLY = "Multiply"
O_DIVIDE = "Divide"
O_AVERAGE = "Average"
O_MINIMUM = "Minimum"
O_MAXIMUM = "Maximum"
O_INVERT = "Invert"
O_COMPLEMENT = "Complement"
O_LOG_TRANSFORM_LEGACY = "Log transform (legacy)"
O_LOG_TRANSFORM = "Log transform (base 2)"
O_NONE = "None"
# Combine is now obsolete - done by Add now, but we need the string for upgrade_settings
O_COMBINE = "Combine"
O_OR = "Or"
O_AND = "And"
O_NOT = "Not"
O_EQUALS = "Equals"

BINARY_OUTPUT_OPS = [O_AND, O_OR, O_NOT, O_EQUALS]

IM_IMAGE = "Image"
IM_MEASUREMENT = "Measurement"

# The number of settings per image
IMAGE_SETTING_COUNT_1 = 2
IMAGE_SETTING_COUNT = 4

# The number of settings other than for images
FIXED_SETTING_COUNT_1 = 7
FIXED_SETTING_COUNT = 8


class ImageMath(cellprofiler.module.ImageProcessing):
    variable_revision_number = 4

    module_name = "ImageMath"

    def create_settings(self):
        # the list of per image settings (name & scaling factor)
        self.images = []
        # create the first two images (the default number)
        self.add_image(False)
        self.add_image(False)

        # other settings
        self.operation = cellprofiler.setting.Choice(
                "Operation",
                [O_ADD, O_SUBTRACT, O_DIFFERENCE, O_MULTIPLY, O_DIVIDE, O_AVERAGE,
                 O_MINIMUM, O_MAXIMUM, O_INVERT,
                 O_LOG_TRANSFORM, O_LOG_TRANSFORM_LEGACY,
                 O_AND, O_OR, O_NOT, O_EQUALS, O_NONE], doc="""\
Select the operation to perform. Note that if more than two images are
chosen, then operations will be performed sequentially from first to
last, e.g., for “Divide”, (Image1 / Image2) / Image3

-  *%(O_ADD)s:* Adds the first image to the second, and so on.
-  *%(O_SUBTRACT)s:* Subtracts the second image from the first.
-  *%(O_DIFFERENCE)s:* The absolute value of the difference between the
   first and second images.
-  *%(O_MULTIPLY)s:* Multiplies the first image by the second.
-  *%(O_DIVIDE)s:* Divides the first image by the second.
-  *%(O_AVERAGE)s:* Calculates the mean intensity of the images loaded
   in the module. This is equivalent to the Add option divided by the
   number of images loaded by this module. If you would like to average
   all of the images in an entire pipeline, i.e., across cycles, you
   should instead use the **CorrectIlluminationCalculate** module and
   choose the *All* (vs. *Each*) option.
-  *%(O_MINIMUM)s:* Returns the element-wise minimum value at each
   pixel location.
-  *%(O_MAXIMUM)s:* Returns the element-wise maximum value at each
   pixel location.
-  *%(O_INVERT)s:* Subtracts the image intensities from 1. This makes
   the darkest color the brightest and vice-versa. Note that if a
   mask has been applied to the image, the mask will also be inverted.
-  *%(O_LOG_TRANSFORM)s:* Log transforms each pixel’s intensity. The
   actual function is log\ :sub:`2`\ (image + 1), transforming values
   from 0 to 1 into values from 0 to 1.
-  *%(O_LOG_TRANSFORM_LEGACY)s:* Log\ :sub:`2` transform for backwards
   compatibility.
-  *%(O_NONE)s:* This option is useful if you simply want to select some
   of the later options in the module, such as adding, multiplying, or
   exponentiating your image by a constant.

The following are operations that produce binary images. In a binary
image, the foreground has a truth value of “true” (ones) and the background has
a truth value of “false” (zeros). The operations, *%(O_OR)s, %(O_AND)s and
%(O_NOT)s* will convert the input images to binary by changing all zero
values to background (false) and all other values to foreground (true).

-  *%(O_AND)s:* a pixel in the output image is in the foreground only
   if all corresponding pixels in the input images are also in the
   foreground.
-  *%(O_OR)s:* a pixel in the output image is in the foreground if a
   corresponding pixel in any of the input images is also in the
   foreground.
-  *%(O_NOT)s:* the foreground of the input image becomes the
   background of the output image and vice-versa.
-  *%(O_EQUALS)s:* a pixel in the output image is in the foreground if
   the corresponding pixels in the input images have the same value.

Note that *%(O_INVERT)s*, *%(O_LOG_TRANSFORM)s*,
*%(O_LOG_TRANSFORM_LEGACY)s* and *%(O_NONE)s* operate on only a
single image.
""" % globals())
        self.divider_top = cellprofiler.setting.Divider(line=False)

        self.exponent = cellprofiler.setting.Float(
                "Raise the power of the result by", 1, doc="""\
Enter an exponent to raise the result to *after* the chosen operation.""")

        self.after_factor = cellprofiler.setting.Float(
                "Multiply the result by", 1, doc="""\
Enter a factor to multiply the result by *after* the chosen operation.""")

        self.addend = cellprofiler.setting.Float(
                "Add to result", 0, doc="""\
Enter a number to add to the result *after* the chosen operation.""")

        self.truncate_low = cellprofiler.setting.Binary(
                "Set values less than 0 equal to 0?", True, doc="""\
Values outside the range 0 to 1 might not be handled well by other
modules. Select *%(YES)s* to set negative values to 0.
""" % globals())

        self.truncate_high = cellprofiler.setting.Binary(
                "Set values greater than 1 equal to 1?", True, doc="""\
Values outside the range 0 to 1 might not be handled well by other
modules. Select *%(YES)s* to set values greater than 1 to a maximum
value of 1.
""" % globals())

        self.ignore_mask = cellprofiler.setting.Binary(
                "Ignore the image masks?", False, doc="""\
Select *%(YES)s* to set equal to zero all previously masked pixels and
operate on the masked images as if no mask had been applied. Otherwise,
the smallest image mask is applied after image math has been completed.
""" % globals())

        self.output_image_name = cellprofiler.setting.ImageNameProvider(
                "Name the output image", "ImageAfterMath", doc="""\
Enter a name for the resulting image.""")

        self.add_button = cellprofiler.setting.DoSomething("", "Add another image", self.add_image)

        self.divider_bottom = cellprofiler.setting.Divider(line=False)

    def add_image(self, removable=True):
        # The text for these settings will be replaced in renumber_settings()
        group = cellprofiler.setting.SettingsGroup()
        group.removable = removable
        group.append("image_or_measurement", cellprofiler.setting.Choice(
                "Image or measurement?", [IM_IMAGE, IM_MEASUREMENT], doc="""\
You can perform math operations using two images or you can use a
measurement for one of the operands. For instance, to divide the
intensity of one image by another, choose *%(IM_IMAGE)s* for both and
pick the respective images. To divide the intensity of an image by its
median intensity, use **MeasureImageIntensity** prior to this module to
calculate the median intensity, then select *%(IM_MEASUREMENT)s* and
use the median intensity measurement as the denominator.
""" % globals()))

        group.append("image_name", cellprofiler.setting.ImageNameSubscriber("Select the image", "", doc="""\
Select the image that you want to use for this operation."""))

        group.append("measurement", cellprofiler.setting.Measurement(
                "Measurement", lambda: cellprofiler.measurement.IMAGE, "", doc="""\
Select a measurement made on the image. The value of the
measurement is used for the operand for all of the pixels of the
other operand's image."""))

        group.append("factor", cellprofiler.setting.Float("Multiply the image by", 1, doc="""\
Enter the number that you would like to multiply the above image by. This multiplication
is applied before other operations."""))

        if removable:
            group.append(
                "remover",
                cellprofiler.setting.RemoveSettingButton(
                    "",
                    "Remove this image",
                    self.images,
                    group
                )
            )

        group.append("divider", cellprofiler.setting.Divider())
        self.images.append(group)

    def renumber_settings(self):
        inflection = inflect.engine()

        for idx, image in enumerate(self.images):
            image.image_name.text = "Select the %s image" % (inflection.number_to_words(inflection.ordinal(idx + 1)))
            image.factor.text = "Multiply the %s image by" % inflection.number_to_words(inflection.ordinal(idx + 1))

    def settings(self):
        result = [self.operation, self.exponent, self.after_factor, self.addend,
                  self.truncate_low, self.truncate_high, self.ignore_mask,
                  self.output_image_name]
        for image in self.images:
            result += [image.image_or_measurement, image.image_name,
                       image.factor, image.measurement]
        return result

    @property
    def operand_count(self):
        '''# of operands, taking the operation into consideration'''
        if self.operation.value in (O_INVERT, O_LOG_TRANSFORM, O_LOG_TRANSFORM_LEGACY, O_NONE, O_NOT):
            return 1
        return len(self.images)

    def visible_settings(self):
        result = [self.operation, self.output_image_name, self.divider_top]
        self.renumber_settings()
        single_image = self.operand_count == 1
        for index in range(self.operand_count):
            image = self.images[index]
            if single_image:
                result += [image.image_name]
            else:
                result += [image.image_or_measurement]
                if image.image_or_measurement == IM_IMAGE:
                    result += [image.image_name]
                else:
                    result += [image.measurement]
            if self.operation not in BINARY_OUTPUT_OPS:
                result += [image.factor]
            if image.removable:
                result += [image.remover]
            result += [image.divider]

        if single_image:
            result[-1] = self.divider_bottom  # this looks better when there's just one image
        else:
            result += [self.add_button, self.divider_bottom]

        if self.operation not in BINARY_OUTPUT_OPS:
            result += [
                self.exponent, self.after_factor, self.addend,
                self.truncate_low, self.truncate_high]
        result += [self.ignore_mask]
        return result

    def help_settings(self):
        return [
            self.operation,
            self.output_image_name,
            self.images[0].image_or_measurement,
            self.images[0].image_name,
            self.images[0].measurement,
            self.images[0].factor,
            self.exponent,
            self.after_factor,
            self.addend,
            self.truncate_low,
            self.truncate_high,
            self.ignore_mask
        ]

    def prepare_settings(self, setting_values):
        value_count = len(setting_values)
        assert (value_count - FIXED_SETTING_COUNT) % IMAGE_SETTING_COUNT == 0
        image_count = (value_count - FIXED_SETTING_COUNT) / IMAGE_SETTING_COUNT
        # always keep the first two images
        del self.images[2:]
        while len(self.images) < image_count:
            self.add_image()

    def use_logical_operation(self, pixel_data):
        return all([pd.dtype == numpy.bool for pd in pixel_data if not numpy.isscalar(pd)])

    def run(self, workspace):
        image_names = [image.image_name.value for image in self.images if image.image_or_measurement == IM_IMAGE]
        image_factors = [image.factor.value for image in self.images]
        wants_image = [image.image_or_measurement == IM_IMAGE for image in self.images]

        if self.operation.value in [O_INVERT, O_LOG_TRANSFORM, O_LOG_TRANSFORM_LEGACY, O_NOT, O_NONE]:
            # these only operate on the first image
            image_names = image_names[:1]
            image_factors = image_factors[:1]

        images = [workspace.image_set.get_image(x) for x in image_names]
        pixel_data = [image.pixel_data for image in images]
        masks = [image.mask if image.has_mask else None for image in images]

        # Crop all of the images similarly
        smallest = numpy.argmin([numpy.product(pd.shape) for pd in pixel_data])
        smallest_image = images[smallest]
        for i in [x for x in range(len(images)) if x != smallest]:
            pixel_data[i] = smallest_image.crop_image_similarly(pixel_data[i])
            if masks[i] is not None:
                masks[i] = smallest_image.crop_image_similarly(masks[i])

        # weave in the measurements
        idx = 0
        measurements = workspace.measurements
        for i in range(self.operand_count):
            if not wants_image[i]:
                value = measurements.get_current_image_measurement(self.images[i].measurement.value)
                value = numpy.NaN if value is None else float(value)
                pixel_data.insert(i, value)
                masks.insert(i, True)

        # Multiply images by their factors
        for i, image_factor in enumerate(image_factors):
            if image_factor != 1 and self.operation not in BINARY_OUTPUT_OPS:
                pixel_data[i] = pixel_data[i] * image_factors[i]

        output_pixel_data = pixel_data[0]
        output_mask = masks[0]

        opval = self.operation.value
        if opval in [O_ADD, O_SUBTRACT, O_DIFFERENCE, O_MULTIPLY, O_DIVIDE,
                     O_AVERAGE, O_MAXIMUM, O_MINIMUM, O_AND, O_OR, O_EQUALS]:
            # Binary operations
            if opval in (O_ADD, O_AVERAGE):
                op = numpy.add
            elif opval == O_SUBTRACT:
                if self.use_logical_operation(pixel_data):
                    op = numpy.logical_xor
                else:
                    op = numpy.subtract
            elif opval == O_DIFFERENCE:
                if self.use_logical_operation(pixel_data):
                    op = numpy.logical_xor
                else:
                    def op(x, y):
                        return numpy.abs(numpy.subtract(x, y))
            elif opval == O_MULTIPLY:
                if self.use_logical_operation(pixel_data):
                    op = numpy.logical_and
                else:
                    op = numpy.multiply
            elif opval == O_MINIMUM:
                op = numpy.minimum
            elif opval == O_MAXIMUM:
                op = numpy.maximum
            elif opval == O_AND:
                op = numpy.logical_and
            elif opval == O_OR:
                op = numpy.logical_or
            elif opval == O_EQUALS:
                output_pixel_data = numpy.ones(pixel_data[0].shape, bool)
                comparitor = pixel_data[0]
            else:
                op = numpy.divide
            for pd, mask in zip(pixel_data[1:], masks[1:]):
                if not numpy.isscalar(pd) and output_pixel_data.ndim != pd.ndim:
                    if output_pixel_data.ndim == 2:
                        output_pixel_data = output_pixel_data[:, :, numpy.newaxis]
                        if opval == O_EQUALS and not numpy.isscalar(comparitor):
                            comparitor = comparitor[:, :, numpy.newaxis]
                    if pd.ndim == 2:
                        pd = pd[:, :, numpy.newaxis]
                if opval == O_EQUALS:
                    output_pixel_data = output_pixel_data & (comparitor == pd)
                else:
                    output_pixel_data = op(output_pixel_data, pd)
                if self.ignore_mask:
                    continue
                else:
                    if output_mask is None:
                        output_mask = mask
                    elif mask is not None:
                        output_mask = (output_mask & mask)
            if opval == O_AVERAGE:
                if not self.use_logical_operation(pixel_data):
                    output_pixel_data /= sum(image_factors)
        elif opval == O_INVERT:
            output_pixel_data = skimage.util.invert(output_pixel_data)
        elif opval == O_NOT:
            output_pixel_data = numpy.logical_not(output_pixel_data)
        elif opval == O_LOG_TRANSFORM:
            output_pixel_data = numpy.log2(output_pixel_data + 1)
        elif opval == O_LOG_TRANSFORM_LEGACY:
            output_pixel_data = numpy.log2(output_pixel_data)
        elif opval == O_NONE:
            output_pixel_data = output_pixel_data.copy()
        else:
            raise NotImplementedError("The operation %s has not been implemented" % opval)

        # Check to see if there was a measurement & image w/o mask. If so
        # set mask to none
        if numpy.isscalar(output_mask):
            output_mask = None
        if opval not in BINARY_OUTPUT_OPS:
            #
            # Post-processing: exponent, multiply, add
            #
            if self.exponent.value != 1:
                output_pixel_data **= self.exponent.value
            if self.after_factor.value != 1:
                output_pixel_data *= self.after_factor.value
            if self.addend.value != 0:
                output_pixel_data += self.addend.value

            #
            # truncate values
            #
            if self.truncate_low.value:
                output_pixel_data[output_pixel_data < 0] = 0
            if self.truncate_high.value:
                output_pixel_data[output_pixel_data > 1] = 1

        #
        # add the output image to the workspace
        #
        crop_mask = (smallest_image.crop_mask
                     if smallest_image.has_crop_mask else None)
        masking_objects = (smallest_image.masking_objects
                           if smallest_image.has_masking_objects else None)
        output_image = cellprofiler.image.Image(output_pixel_data,
                                                mask=output_mask,
                                                crop_mask=crop_mask,
                                                parent_image=images[0],
                                                masking_objects=masking_objects,
                                                convert=False,
                                                dimensions=images[0].dimensions)
        workspace.image_set.add(self.output_image_name.value, output_image)

        #
        # Display results
        #
        if self.show_window:
            workspace.display_data.pixel_data = [image.pixel_data for image in images] + [output_pixel_data]

            workspace.display_data.display_names = image_names + [self.output_image_name.value]

            workspace.display_data.dimensions = output_image.dimensions

    def display(self, workspace, figure):
        import matplotlib.cm

        pixel_data = workspace.display_data.pixel_data

        display_names = workspace.display_data.display_names

        columns = (len(pixel_data) + 1) / 2

        figure.set_subplots((columns, 2), dimensions=workspace.display_data.dimensions)

        for i in range(len(pixel_data)):
            if pixel_data[i].shape[-1] in (3, 4):
                cmap = None
            elif pixel_data[i].dtype.kind == 'b':
                cmap = matplotlib.cm.binary_r
            else:
                cmap = matplotlib.cm.Greys_r

            figure.subplot_imshow(
                i % columns,
                int(i / columns),
                pixel_data[i],
                title=display_names[i],
                sharexy=figure.subplot(0, 0),
                colormap=cmap
            )

    def validate_module(self, pipeline):
        '''Guarantee that at least one operand is an image'''
        for i in range(self.operand_count):
            op = self.images[i]
            if op.image_or_measurement == IM_IMAGE:
                return
        raise cellprofiler.setting.ValidationError("At least one of the operands must be an image",
                                                   op.image_or_measurement)

    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        if (from_matlab and module_name == 'Subtract' and variable_revision_number == 3):
            subtract_image_name, basic_image_name, resulting_image_name, \
                multiply_factor_1, multiply_factor_2, truncate = setting_values
            setting_values = [O_SUBTRACT,
                              1,  # exponent
                              1,  # post-multiply factor
                              0,  # addend
                              truncate,  # truncate low
                              cellprofiler.setting.NO,  # truncate high
                              resulting_image_name,
                              basic_image_name,
                              multiply_factor_2,
                              subtract_image_name,
                              multiply_factor_1]
            module_name = 'ImageMath'
            from_matlab = False
            variable_revision_number = 1
        if (from_matlab and module_name == 'Combine' and variable_revision_number == 3):
            names_and_weights = [
                (name, weight)
                for name, weight in zip(setting_values[:3],
                                        setting_values[4:])
                if name.lower() != cellprofiler.setting.DO_NOT_USE.lower()]

            multiplier = 1.0 / sum([float(weight)
                                    for name, weight in names_and_weights])
            output_image = setting_values[3]
            setting_values = [O_ADD,  # Operation
                              "1",  # Exponent
                              str(multiplier),  # Post-operation multiplier
                              "0",  # Post-operation offset
                              cellprofiler.setting.NO,  # Truncate low
                              cellprofiler.setting.NO,  # Truncate high
                              output_image]
            for name, weight in names_and_weights:
                setting_values += [name, weight]
            module_name = 'ImageMath'
            variable_revision_number = 1
            from_matlab = False
        if (from_matlab and module_name == 'InvertIntensity' and variable_revision_number == 1):
            image_name, output_image = setting_values
            setting_values = [image_name, cellprofiler.setting.DO_NOT_USE, cellprofiler.setting.DO_NOT_USE,
                              'Invert',
                              1, 1, 1, 1, 1, cellprofiler.setting.NO, cellprofiler.setting.NO, output_image]
            module_name = 'ImageMath'
            variable_revision_number = 2
        if (from_matlab and module_name == 'Multiply' and variable_revision_number == 1):
            image1, image2, output_image = setting_values
            setting_values = [image1, image2, cellprofiler.setting.DO_NOT_USE,
                              'Multiply', 1, 1, 1, 1, 1, cellprofiler.setting.NO, cellprofiler.setting.NO,
                              output_image]
            module_name = 'ImageMath'
            variable_revision_number = 2

        if (from_matlab and variable_revision_number == 1 and module_name == 'ImageMath'):
            image_names = [setting_values[1]]
            input_factors = [float(setting_values[4])]
            operation = setting_values[3]
            factors = []
            # The user could type in a constant for the second image name
            try:
                factors += [float(setting_values[2]) *
                            float(setting_values[5])]
            except ValueError:
                if setting_values[2] != cellprofiler.setting.DO_NOT_USE:
                    image_names += [setting_values[2]]
                    input_factors += [float(setting_values[5])]
            exponent = 1.0
            multiplier = 1.0
            addend = 0
            wants_truncate_low = setting_values[6]
            wants_truncate_high = setting_values[7]
            output_image_name = setting_values[0]
            old_operation = operation
            if operation == O_DIVIDE and len(factors):
                multiplier /= numpy.product(factors)
            elif operation == O_MULTIPLY and len(factors):
                multiplier *= numpy.product(factors)
            elif operation == O_ADD and len(factors):
                addend = numpy.sum(factors)
            elif operation == O_SUBTRACT:
                addend = -numpy.sum(factors)
            setting_values = [operation, exponent, multiplier, addend,
                              wants_truncate_low, wants_truncate_high,
                              output_image_name]
            if operation == O_COMPLEMENT:
                image_names = image_names[:1]
                input_factors = input_factors[:1]
            elif old_operation in (O_ADD, O_SUBTRACT, O_MULTIPLY, O_DIVIDE):
                if len(image_names) < 2:
                    setting_values[0] = O_NONE
                image_names = image_names[:2]
                input_factors = input_factors[:2]
            for image_name, input_factor in zip(image_names, input_factors):
                setting_values += [image_name, input_factor]
            from_matlab = False
            variable_revision_number = 1

        if (from_matlab and variable_revision_number == 2 and module_name == 'ImageMath'):
            image_names = [setting_values[0]]
            input_factors = [float(setting_values[4])]
            operation = setting_values[3]
            factors = []
            for i in range(1, 3 if operation == O_COMBINE else 2):
                # The user could type in a constant for the second or third image name
                try:
                    factors += [float(setting_values[i]) * float(setting_values[i + 4])]
                except ValueError:
                    if setting_values[i] != cellprofiler.setting.DO_NOT_USE:
                        image_names += [setting_values[i]]
                        input_factors += [float(setting_values[i + 4])]

            exponent = float(setting_values[7])
            multiplier = float(setting_values[8])
            addend = 0
            wants_truncate_low = setting_values[9]
            wants_truncate_high = setting_values[10]
            output_image_name = setting_values[11]
            old_operation = operation
            if operation == O_COMBINE:
                addend = numpy.sum(factors)
                operation = O_ADD
            elif operation == O_DIVIDE and len(factors):
                multiplier /= numpy.product(factors)
            elif operation == O_MULTIPLY and len(factors):
                multiplier *= numpy.product(factors)
            elif operation == O_ADD and len(factors):
                addend = numpy.sum(factors)
            elif operation == O_SUBTRACT:
                addend = -numpy.sum(factors)
            setting_values = [operation, exponent, multiplier, addend,
                              wants_truncate_low, wants_truncate_high,
                              output_image_name]
            if operation in (O_INVERT, O_LOG_TRANSFORM):
                image_names = image_names[:1]
                input_factors = input_factors[:1]
            elif old_operation in (O_ADD, O_SUBTRACT, O_MULTIPLY, O_DIVIDE,
                                   O_AVERAGE):
                if len(image_names) < 2:
                    setting_values[0] = O_NONE
                image_names = image_names[:2]
                input_factors = input_factors[:2]
                # Fix for variable_revision_number 2: subtract reversed operands
                if old_operation == O_SUBTRACT:
                    image_names.reverse()
                    input_factors.reverse()
            for image_name, input_factor in zip(image_names, input_factors):
                setting_values += [image_name, input_factor]
            from_matlab = False
            variable_revision_number = 1
        if (not from_matlab) and variable_revision_number == 1:
            # added image_or_measurement and measurement
            new_setting_values = setting_values[:FIXED_SETTING_COUNT_1]
            for i in range(FIXED_SETTING_COUNT_1, len(setting_values),
                           IMAGE_SETTING_COUNT_1):
                new_setting_values += [IM_IMAGE, setting_values[i],
                                       setting_values[i + 1], ""]
            setting_values = new_setting_values
            variable_revision_number = 2
        if (not from_matlab) and variable_revision_number == 2:
            # added the ability to ignore the mask
            new_setting_values = setting_values
            new_setting_values.insert(6, 'No')
            setting_values = new_setting_values
            variable_revision_number = 3
        if (not from_matlab) and variable_revision_number == 3:
            # Log transform -> legacy log transform
            if setting_values[0] == O_LOG_TRANSFORM:
                setting_values = [O_LOG_TRANSFORM_LEGACY] + setting_values[1:]
            variable_revision_number = 4
        return setting_values, variable_revision_number, from_matlab
