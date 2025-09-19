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

import numpy
import skimage.util
from cellprofiler_core.image import Image
from cellprofiler_core.module import ImageProcessing
from cellprofiler_core.setting import (
    Divider,
    Binary,
    SettingsGroup,
    Measurement,
    ValidationError,
)
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.do_something import DoSomething, RemoveSettingButton
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.text import Float, ImageName
from cellprofiler_library.opts.imagemath import Operator, Operand, BINARY_OUTPUT_OPS
from cellprofiler_library.modules._imagemath import image_math
# Operator.ADD = "Add"
# Operator.SUBTRACT = "Subtract"
# Operator.DIFFERENCE = "Absolute Difference"
# Operator.MULTIPLY = "Multiply"
# Operator.DIVIDE = "Divide"
# Operator.AVERAGE = "Average"
# Operator.MINIMUM = "Minimum"
# Operator.MAXIMUM = "Maximum"
# Operator.STDEV = "Standard Deviation"
# Operator.INVERT = "Invert"
# Operator.COMPLEMENT = "Complement"
# Operator.LOG_TRANSFORM_LEGACY = "Log transform (legacy)"
# Operator.LOG_TRANSFORM = "Log transform (base 2)"
# Operator.NONE = "None"
# Combine is now obsolete - done by Add now, but we need the string for upgrade_settings
O_COMBINE = "Combine"
# Operator.OR = "Or"
# Operator.AND = "And"
# Operator.NOT = "Not"
# Operator.EQUALS = "Equals"

# BINARY_OUTPUT_OPS = [Operator.AND, Operator.OR, Operator.NOT, Operator.EQUALS]

# Operand.IMAGE = "Image"
# Operand.MEASUREMENT = "Measurement"

# The number of settings per image
IMAGE_SETTING_COUNT_1 = 2
IMAGE_SETTING_COUNT = 4

# The number of settings other than for images
FIXED_SETTING_COUNT_1 = 8
FIXED_SETTING_COUNT = 9


class ImageMath(ImageProcessing):
    variable_revision_number = 5

    module_name = "ImageMath"

    def create_settings(self):
        # the list of per image settings (name & scaling factor)
        self.images = []
        # create the first two images (the default number)
        self.add_image(False)
        self.add_image(False)

        # other settings
        self.operation = Choice(
            "Operation",
            [
                Operator.ADD.value,
                Operator.SUBTRACT.value,
                Operator.DIFFERENCE.value,
                Operator.MULTIPLY.value,
                Operator.DIVIDE.value,
                Operator.AVERAGE.value,
                Operator.MINIMUM.value,
                Operator.MAXIMUM.value,
                Operator.STDEV.value,
                Operator.INVERT.value,
                Operator.LOG_TRANSFORM.value,
                Operator.LOG_TRANSFORM_LEGACY.value,
                Operator.AND.value,
                Operator.OR.value,
                Operator.NOT.value,
                Operator.EQUALS.value,
                Operator.NONE.value,
            ],
            doc="""\
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
-  *%(O_STDEV)s:* Returns the element-wise standard deviation value at each
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
""".format(
    **{
        "O_ADD": Operator.ADD.value,
        "O_SUBTRACT": Operator.SUBTRACT.value,
        "O_DIFFERENCE": Operator.DIFFERENCE.value,
        "O_MULTIPLY": Operator.MULTIPLY.value,
        "O_DIVIDE": Operator.DIVIDE.value,
        "O_AVERAGE": Operator.AVERAGE.value,
        "O_MINIMUM": Operator.MINIMUM.value,
        "O_MAXIMUM": Operator.MAXIMUM.value,
        "O_STDEV": Operator.STDEV.value,
        "O_INVERT": Operator.INVERT.value,
        "O_LOG_TRANSFORM": Operator.LOG_TRANSFORM.value,
        "O_LOG_TRANSFORM_LEGACY": Operator.LOG_TRANSFORM_LEGACY.value,
        "O_AND": Operator.AND.value,
        "O_OR": Operator.OR.value,
        "O_NOT": Operator.NOT.value,
        "O_EQUALS": Operator.EQUALS.value,
        "O_NONE": Operator.NONE.value,
    }
),
        )
        self.divider_top = Divider(line=False)

        self.exponent = Float(
            "Raise the power of the result by",
            1,
            doc="""\
Enter an exponent to raise the result to *after* the chosen operation.""",
        )

        self.after_factor = Float(
            "Multiply the result by",
            1,
            doc="""\
Enter a factor to multiply the result by *after* the chosen operation.""",
        )

        self.addend = Float(
            "Add to result",
            0,
            doc="""\
Enter a number to add to the result *after* the chosen operation.""",
        )

        self.truncate_low = Binary(
            "Set values less than 0 equal to 0?",
            True,
            doc="""\
Values outside the range 0 to 1 might not be handled well by other
modules. Select *Yes* to set negative values to 0.
"""
            % globals(),
        )

        self.truncate_high = Binary(
            "Set values greater than 1 equal to 1?",
            True,
            doc="""\
Values outside the range 0 to 1 might not be handled well by other
modules. Select *Yes* to set values greater than 1 to a maximum
value of 1.
"""
            % globals(),
        )

        self.replace_nan = Binary(
            "Replace invalid values with 0?",
            True,
            doc="""\
        Certain operations are mathematically invalid (divide by zero, 
        raise a negative number to the power of a fraction, etc.).
        This setting will set pixels with invalid values to zero.
        Disabling this setting will represent these pixels as "nan" 
        ("Not A Number"). "nan" pixels cannot be displayed properly and 
        may cause errors in other modules.
        """
            % globals(),
        )

        self.ignore_mask = Binary(
            "Ignore the image masks?",
            False,
            doc="""\
Select *Yes* to set equal to zero all previously masked pixels and
operate on the masked images as if no mask had been applied. Otherwise,
the smallest image mask is applied after image math has been completed.
"""
            % globals(),
        )

        self.output_image_name = ImageName(
            "Name the output image",
            "ImageAfterMath",
            doc="""\
Enter a name for the resulting image.""",
        )

        self.add_button = DoSomething("", "Add another image", self.add_image)

        self.divider_bottom = Divider(line=False)

    def add_image(self, removable=True):
        # The text for these settings will be replaced in renumber_settings()
        group = SettingsGroup()
        group.removable = removable
        group.append(
            "image_or_measurement",
            Choice(
                "Image or measurement?",
                [Operand.IMAGE.value, Operand.MEASUREMENT.value],
                doc="""\
You can perform math operations using two images or you can use a
measurement for one of the operands. For instance, to divide the
intensity of one image by another, choose *%(IM_IMAGE)s* for both and
pick the respective images. To divide the intensity of an image by its
median intensity, use **MeasureImageIntensity** prior to this module to
calculate the median intensity, then select *%(IM_MEASUREMENT)s* and
use the median intensity measurement as the denominator.
""".format(
    **{
        "IM_IMAGE": Operand.IMAGE.value,
        "IM_MEASUREMENT": Operand.MEASUREMENT.value,
    }
)
            ),
        )

        group.append(
            "image_name",
            ImageSubscriber(
                "Select the image",
                "None",
                doc="""\
Select the image that you want to use for this operation.""",
            ),
        )

        group.append(
            "measurement",
            Measurement(
                "Measurement",
                lambda: "Image",
                "",
                doc="""\
Select a measurement made on the image. The value of the
measurement is used for the operand for all of the pixels of the
other operand's image.""",
            ),
        )

        group.append(
            "factor",
            Float(
                "Multiply the image by",
                1,
                doc="""\
Enter the number that you would like to multiply the above image by. This multiplication
is applied before other operations.""",
            ),
        )

        if removable:
            group.append(
                "remover",
                RemoveSettingButton("", "Remove this image", self.images, group),
            )

        group.append("divider", Divider())
        self.images.append(group)

    def __make_ordinal(self, n):
        '''
        Convert an integer into its ordinal representation::

            make_ordinal(0)   => '0th'
            make_ordinal(3)   => '3rd'
            make_ordinal(122) => '122nd'
            make_ordinal(213) => '213th'
        '''
        n = int(n)
        if 11 <= (n % 100) <= 13:
            suffix = 'th'
        else:
            suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
        return str(n) + suffix

    def renumber_settings(self):
        for idx, image in enumerate(self.images):
            image.image_name.text = "Select the %s image" % (
                self.__make_ordinal(idx + 1)
            )
            image.factor.text = "Multiply the %s image by" % (
                self.__make_ordinal(idx + 1)
            )
    def settings(self):
        result = [
            self.operation,
            self.exponent,
            self.after_factor,
            self.addend,
            self.truncate_low,
            self.truncate_high,
            self.replace_nan,
            self.ignore_mask,
            self.output_image_name,
        ]
        for image in self.images:
            result += [
                image.image_or_measurement,
                image.image_name,
                image.factor,
                image.measurement,
            ]
        return result

    @property
    def operand_count(self):
        """# of operands, taking the operation into consideration"""
        if self.operation.value in (
            Operator.INVERT,
            Operator.LOG_TRANSFORM,
            Operator.LOG_TRANSFORM_LEGACY,
            Operator.NONE,
            Operator.NOT,
        ):
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
                if image.image_or_measurement.value == Operand.IMAGE:
                    result += [image.image_name]
                else:
                    result += [image.measurement]
            if self.operation not in BINARY_OUTPUT_OPS:
                result += [image.factor]
            if image.removable:
                result += [image.remover]
            result += [image.divider]

        if single_image:
            result[
                -1
            ] = self.divider_bottom  # this looks better when there's just one image
        else:
            result += [self.add_button, self.divider_bottom]

        if self.operation not in BINARY_OUTPUT_OPS:
            result += [
                self.exponent,
                self.after_factor,
                self.addend,
                self.truncate_low,
                self.truncate_high,
                self.replace_nan,
            ]
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
            self.replace_nan,
            self.ignore_mask,
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
        return all(
            [pd.dtype == bool for pd in pixel_data if not numpy.isscalar(pd)]
        )

    def run(self, workspace):
        image_names = [
            image.image_name.value
            for image in self.images
            if image.image_or_measurement.value == Operand.IMAGE
        ]
        image_factors = [image.factor.value for image in self.images]
        wants_image = [image.image_or_measurement.value == Operand.IMAGE for image in self.images]

        if self.operation.value in [
            Operator.INVERT,
            Operator.LOG_TRANSFORM,
            Operator.LOG_TRANSFORM_LEGACY,
            Operator.NOT,
            Operator.NONE,
        ]:
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
        measurements = workspace.measurements
        for i in range(self.operand_count):
            if not wants_image[i]:
                value = measurements.get_current_image_measurement(
                    self.images[i].measurement.value
                )
                value = numpy.NaN if value is None else float(value)
                pixel_data.insert(i, value)
                masks.insert(i, True)

        # Multiply images by their factors
        for i, image_factor in enumerate(image_factors):
            if image_factor != 1 and self.operation not in BINARY_OUTPUT_OPS:
                pixel_data[i] = pixel_data[i] * image_factors[i]

        output_pixel_data = pixel_data[0]
        output_mask = masks[0]

        output_pixel_data, output_mask = image_math(
            opval=self.operation.value,
            pixel_data=pixel_data,
            masks=masks,
            output_pixel_data=output_pixel_data,
            output_mask=output_mask,
            ignore_mask=self.ignore_mask.value,
            image_factors=image_factors,
            exponent=self.exponent.value,
            after_factor=self.after_factor.value,
            addend=self.addend.value,
            truncate_low=self.truncate_low.value,
            truncate_high=self.truncate_high.value,
            replace_nan=self.replace_nan.value,
        )
        #
        # add the output image to the workspace
        #
        crop_mask = smallest_image.crop_mask if smallest_image.has_crop_mask else None
        masking_objects = (
            smallest_image.masking_objects
            if smallest_image.has_masking_objects
            else None
        )

        if not self.ignore_mask:
            if type(output_mask) == numpy.ndarray:
                output_pixel_data = output_pixel_data * output_mask
        output_pixel_data, output_mask
        output_image = Image(
            output_pixel_data,
            mask=output_mask,
            crop_mask=crop_mask,
            parent_image=images[0],
            masking_objects=masking_objects,
            convert=False,
            dimensions=images[0].dimensions,
        )
        workspace.image_set.add(self.output_image_name.value, output_image)

        #
        # Display results
        #
        if self.show_window:
            workspace.display_data.pixel_data = [
                image.pixel_data for image in images
            ] + [output_pixel_data]

            workspace.display_data.display_names = image_names + [
                self.output_image_name.value
            ]

            workspace.display_data.dimensions = output_image.dimensions

    def display(self, workspace, figure):
        import matplotlib.cm

        pixel_data = workspace.display_data.pixel_data

        display_names = workspace.display_data.display_names

        columns = (len(pixel_data) + 1) // 2

        figure.set_subplots((columns, 2), dimensions=workspace.display_data.dimensions)

        for i in range(len(pixel_data)):
            if pixel_data[i].shape[-1] in (3, 4):
                cmap = None
            elif pixel_data[i].dtype.kind == "b":
                cmap = matplotlib.cm.binary_r
            else:
                cmap = matplotlib.cm.Greys_r

            figure.subplot_imshow(
                i % columns,
                int(i / columns),
                pixel_data[i],
                title=display_names[i],
                sharexy=figure.subplot(0, 0),
                colormap=cmap,
            )

    def validate_module(self, pipeline):
        """Guarantee that at least one operand is an image"""
        for i in range(self.operand_count):
            op = self.images[i]
            if op.image_or_measurement.value == Operand.IMAGE:
                return
        raise ValidationError(
            "At least one of the operands must be an image", op.image_or_measurement
        )

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 1:
            # added image_or_measurement and measurement
            new_setting_values = setting_values[:FIXED_SETTING_COUNT_1]
            for i in range(
                FIXED_SETTING_COUNT_1, len(setting_values), IMAGE_SETTING_COUNT_1
            ):
                new_setting_values += [
                    Operand.IMAGE,
                    setting_values[i],
                    setting_values[i + 1],
                    "",
                ]
            setting_values = new_setting_values
            variable_revision_number = 2
        if variable_revision_number == 2:
            # added the ability to ignore the mask
            new_setting_values = setting_values
            new_setting_values.insert(6, "No")
            setting_values = new_setting_values
            variable_revision_number = 3
        if variable_revision_number == 3:
            # Log transform -> legacy log transform
            if setting_values[0] == Operator.LOG_TRANSFORM:
                setting_values = [Operator.LOG_TRANSFORM_LEGACY] + setting_values[1:]
            variable_revision_number = 4
        if variable_revision_number == 4:
            # Add NaN handling
            new_setting_values = setting_values
            new_setting_values.insert(6, "Yes")
            setting_values = new_setting_values
            variable_revision_number = 5
        return setting_values, variable_revision_number
