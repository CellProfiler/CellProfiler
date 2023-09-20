"""
Resize
======

**Resize** resizes images (changes their resolution).

This module is compatible with 2D and 3D/volumetric images. 

Images are resized (made smaller or larger) based on your input. You can
resize an image by applying a resizing factor or by specifying the
desired dimensions, in pixels. You can also select which interpolation
method to use.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============

See also
^^^^^^^^

See also **Crop**.
"""

import logging

import numpy
import skimage.transform
from cellprofiler_core.image import Image
from cellprofiler_core.module import ImageProcessing
from cellprofiler_core.setting import Divider, HiddenCount, SettingsGroup, Binary
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.do_something import DoSomething, RemoveSettingButton
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.text import Float, Integer, ImageName

LOGGER = logging.getLogger(__name__)

R_BY_FACTOR = "Resize by a fraction or multiple of the original size"
R_TO_SIZE = "Resize by specifying desired final dimensions"
R_ALL = [R_BY_FACTOR, R_TO_SIZE]

C_IMAGE = "Image"
C_MANUAL = "Manual"
C_ALL = [C_MANUAL, C_IMAGE]

I_NEAREST_NEIGHBOR = "Nearest Neighbor"
I_BILINEAR = "Bilinear"
I_BICUBIC = "Bicubic"

I_ALL = [I_NEAREST_NEIGHBOR, I_BILINEAR, I_BICUBIC]

S_ADDITIONAL_IMAGE_COUNT = 12


class Resize (ImageProcessing):
    variable_revision_number = 5

    module_name = "Resize"

    def create_settings(self):
        super(Resize, self).create_settings()

        self.size_method = Choice(
            "Resizing method",
            R_ALL,
            doc="""\
The following options are available:

-  *Resize by a fraction or multiple of the original size:* Enter a single value which specifies the scaling.
-  *Resize by specifying desired final dimensions:* Enter the new height and width of the resized image, in units of pixels.""",
        )

        self.resizing_factor_x = Float(
            "X Resizing factor",
            0.25,
            minval=0,
            doc="""\
*(Used only if resizing by a fraction or multiple of the original size)*

Numbers less than one (that is, fractions) will shrink the image;
numbers greater than one (that is, multiples) will enlarge the image.""",
        )

        self.resizing_factor_y= Float(
            "Y Resizing factor",
            0.25,
            minval=0,
            doc="""\
*(Used only if resizing by a fraction or multiple of the original size)*

Numbers less than one (that is, fractions) will shrink the image;
numbers greater than one (that is, multiples) will enlarge the image.""",
        )

        self.resizing_factor_z= Float(
            "Z Resizing factor",
            0.25,
            minval=0,
            doc="""\
*(Used only if resizing by a fraction or multiple of the original size)*

Numbers less than one (that is, fractions) will shrink the image;
numbers greater than one (that is, multiples) will enlarge the image.""",
        )

        self.use_manual_or_image = Choice(
            "Method to specify the dimensions",
            C_ALL,
            doc="""\
*(Used only if resizing by specifying the dimensions)*

You have two options on how to resize your image:

-  *{C_MANUAL}:* Specify the height and width of the output image.
-  *{C_IMAGE}:* Specify an image and the input image will be resized to the same dimensions.
            """.format(
                **{"C_IMAGE": C_IMAGE, "C_MANUAL": C_MANUAL}
            ),
        )

        self.specific_width = Integer(
            "Width (x) of the final image",
            100,
            minval=1,
            doc="""\
*(Used only if resizing by specifying desired final dimensions)*

Enter the desired width of the final image, in pixels.""",
        )

        self.specific_height = Integer(
            "Height (y) of the final image",
            100,
            minval=1,
            doc="""\
*(Used only if resizing by specifying desired final dimensions)*

Enter the desired height of the final image, in pixels.""",
        )

        self.specific_planes = Integer(
            "# of planes (z) in the final image",
            10,
            minval=1,
            doc="""\
*(Used only if resizing by specifying desired final dimensions)*

Enter the desired number of planes in the final image.""",
        )

        self.specific_image = ImageSubscriber(
            "Select the image with the desired dimensions",
            "None",
            doc="""\
*(Used only if resizing by specifying desired final dimensions using an image)*

The input image will be resized to the dimensions of the specified image.""",
        )

        self.interpolation = Choice(
            "Interpolation method",
            I_ALL,
            doc="""\
-  *Nearest Neighbor:* Each output pixel is given the intensity of the
   nearest corresponding pixel in the input image.
-  *Bilinear:* Each output pixel is given the intensity of the weighted
   average of the 2x2 neighborhood at the corresponding position in the
   input image.
-  *Bicubic:* Each output pixel is given the intensity of the weighted
   average of the 4x4 neighborhood at the corresponding position in the
   input image.""",
        )

        self.separator = Divider(line=False)

        self.additional_images = []

        self.additional_image_count = HiddenCount(
            self.additional_images, "Additional image count"
        )

        self.add_button = DoSomething("", "Add another image", self.add_image)

    def add_image(self, can_remove=True):
        group = SettingsGroup()

        if can_remove:
            group.append("divider", Divider(line=False))

        group.append(
            "input_image_name",
            ImageSubscriber(
                "Select the additional image?",
                "None",
                doc="""\
What is the name of the additional image to resize? This image will be
resized with the same settings as the first image.""",
            ),
        )

        group.append(
            "output_image_name",
            ImageName(
                "Name the output image",
                "ResizedBlue",
                doc="What is the name of the additional resized image?",
            ),
        )

        if can_remove:
            group.append(
                "remover",
                RemoveSettingButton(
                    "", "Remove above image", self.additional_images, group
                ),
            )

        self.additional_images.append(group)

    def settings(self):
        settings = super(Resize, self).settings()

        settings += [
            self.size_method,
            self.resizing_factor_x,
            self.resizing_factor_y,
            self.resizing_factor_z,
            self.specific_width,
            self.specific_height,
            self.specific_planes,
            self.interpolation,
            self.use_manual_or_image,
            self.specific_image,
            self.additional_image_count,
        ]

        for additional in self.additional_images:
            settings += [additional.input_image_name, additional.output_image_name]

        return settings

    def help_settings(self):
        return super(Resize, self).help_settings() + [
            self.size_method,
            self.resizing_factor_x,
            self.resizing_factor_y,
            self.resizing_factor_z,
            self.use_manual_or_image,
            self.specific_image,
            self.specific_width,
            self.specific_height,
            self.specific_planes,
            self.interpolation,
        ]

    def visible_settings(self):
        visible_settings = super(Resize, self).visible_settings()

        visible_settings += [self.size_method]

        if self.size_method == R_BY_FACTOR:
            visible_settings += [self.resizing_factor_x, self.resizing_factor_y, self.resizing_factor_z,]
        elif self.size_method == R_TO_SIZE:
            visible_settings += [self.use_manual_or_image]

            if self.use_manual_or_image == C_IMAGE:
                visible_settings += [self.specific_image]
            elif self.use_manual_or_image == C_MANUAL:
                visible_settings += [self.specific_width, self.specific_height, self.specific_planes]
        else:
            raise ValueError(
                "Unsupported size method: {}".format(self.size_method.value)
            )

        visible_settings += [self.interpolation]

        for additional in self.additional_images:
            visible_settings += additional.visible_settings()

        visible_settings += [self.add_button]

        return visible_settings

    def prepare_settings(self, setting_values):
        try:
            additional_image_setting_count = int(
                setting_values[S_ADDITIONAL_IMAGE_COUNT]
            )

            if len(self.additional_images) > additional_image_setting_count:
                del self.additional_images[additional_image_setting_count:]
            else:
                for i in range(
                    len(self.additional_images), additional_image_setting_count
                ):
                    self.add_image()
        except ValueError:
            LOGGER.warning(
                'Additional image setting count was "%s" which is not an integer.',
                setting_values[S_ADDITIONAL_IMAGE_COUNT],
                exc_info=True,
            )

            pass

    def run(self, workspace):
        self.apply_resize(workspace, self.x_name.value, self.y_name.value)

        for additional in self.additional_images:
            self.apply_resize(
                workspace,
                additional.input_image_name.value,
                additional.output_image_name.value,
            )

    def resized_shape(self, image, workspace):
        image_pixels = image.pixel_data

        shape = numpy.array(image_pixels.shape).astype(float)


        if self.size_method.value == R_BY_FACTOR:
            factor_x = self.resizing_factor_x.value

            factor_y = self.resizing_factor_y.value

            if image.volumetric:
                factor_z = self.resizing_factor_z.value
                height, width = shape[1:3]
                planes = shape [0]
                planes = numpy.round(planes * factor_z)
            else:
                height, width = shape[:2]

            height = numpy.round(height * factor_y)

            width = numpy.round(width * factor_x)

        else:
            if self.use_manual_or_image.value == C_MANUAL:
                height = self.specific_height.value
                width = self.specific_width.value
                if image.volumetric:
                    planes = self.specific_planes.value
            else:
                other_image = workspace.image_set.get_image(self.specific_image.value)

                if image.volumetric:
                    planes, height, width = other_image.pixel_data.shape[:3]
                else:
                    height, width = other_image.pixel_data.shape[:2]

        new_shape = []

        if image.volumetric:
            new_shape += [planes]

        new_shape += [height, width]

        if image.multichannel:
            new_shape += [shape[-1]]

        return numpy.asarray(new_shape)

    def spline_order(self):
        if self.interpolation.value == I_NEAREST_NEIGHBOR:
            return 0

        if self.interpolation.value == I_BILINEAR:
            return 1

        return 3

    def apply_resize(self, workspace, input_image_name, output_image_name):
        image = workspace.image_set.get_image(input_image_name)

        image_pixels = image.pixel_data

        new_shape = self.resized_shape(image, workspace)

        order = self.spline_order()

        if image.volumetric and image.multichannel:
            output_pixels = numpy.zeros(new_shape.astype(int), dtype=image_pixels.dtype)

            for idx in range(int(new_shape[-1])):
                output_pixels[:, :, :, idx] = skimage.transform.resize(
                    image_pixels[:, :, :, idx],
                    new_shape[:-1],
                    order=order,
                    mode="symmetric",
                )
        else:
            output_pixels = skimage.transform.resize(
                image_pixels, new_shape, order=order, mode="symmetric"
            )

        if image.multichannel and len(new_shape) > image.dimensions:
            new_shape = new_shape[:-1]

        mask = skimage.transform.resize(image.mask, new_shape, order=0, mode="constant")

        mask = skimage.img_as_bool(mask)

        if image.has_crop_mask:
            cropping = skimage.transform.resize(
                image.crop_mask, new_shape, order=0, mode="constant"
            )

            cropping = skimage.img_as_bool(cropping)
        else:
            cropping = None

        output_image = Image(
            output_pixels,
            parent_image=image,
            mask=mask,
            crop_mask=cropping,
            dimensions=image.dimensions,
        )

        workspace.image_set.add(output_image_name, output_image)

        if self.show_window:
            if hasattr(workspace.display_data, "input_images"):
                workspace.display_data.multichannel += [image.multichannel]
                workspace.display_data.input_images += [image.pixel_data]
                workspace.display_data.output_images += [output_image.pixel_data]
                workspace.display_data.input_image_names += [input_image_name]
                workspace.display_data.output_image_names += [output_image_name]
            else:
                workspace.display_data.dimensions = image.dimensions
                workspace.display_data.multichannel = [image.multichannel]
                workspace.display_data.input_images = [image.pixel_data]
                workspace.display_data.output_images = [output_image.pixel_data]
                workspace.display_data.input_image_names = [input_image_name]
                workspace.display_data.output_image_names = [output_image_name]

    def display(self, workspace, figure):
        """Display the resized images

        workspace - the workspace being run
        statistics - a list of lists:
            0: index of this statistic
            1: input image name of image being aligned
            2: output image name of image being aligned
        """
        dimensions = workspace.display_data.dimensions
        multichannel = workspace.display_data.multichannel
        input_images = workspace.display_data.input_images
        output_images = workspace.display_data.output_images
        input_image_names = workspace.display_data.input_image_names
        output_image_names = workspace.display_data.output_image_names

        figure.set_subplots((2, len(input_images)), dimensions=dimensions)

        for (
            i,
            (
                input_image_pixels,
                output_image_pixels,
                input_image_name,
                output_image_name,
                multichannel,
            ),
        ) in enumerate(
            zip(
                input_images,
                output_images,
                input_image_names,
                output_image_names,
                multichannel,
            )
        ):
            if multichannel:
                figure.subplot_imshow_color(
                    0, i, input_image_pixels, title=input_image_name, volumetric=dimensions==3, normalize=None,
                )

                figure.subplot_imshow_color(
                    1, i, output_image_pixels, title=output_image_name, volumetric=dimensions==3, normalize=None,
                )
            else:
                figure.subplot_imshow_bw(
                    0, i, input_image_pixels, title=input_image_name,
                )

                figure.subplot_imshow_bw(
                    1, i, output_image_pixels, title=output_image_name,
                )

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 1:
            if setting_values[2] == "Resize by a factor of the original size":
                setting_values[2] = R_BY_FACTOR
            if setting_values[2] == "Resize to a size in pixels":
                setting_values[2] = R_TO_SIZE
            variable_revision_number = 2

        if variable_revision_number == 2:
            # Add additional images to be resized similarly, but if you only had 1,
            # the order didn't change
            setting_values = setting_values + ["0"]
            variable_revision_number = 3

        if variable_revision_number == 3:
            # Add resizing to another image size
            setting_values = (
                setting_values[:7] + [C_MANUAL, "None"] + setting_values[7:]
            )
            variable_revision_number = 4
        
        if variable_revision_number == 4:
            #Add X, Y and Z resizing factor 
            setting_values = (
                setting_values[:3] + [setting_values[3], setting_values[3], 1] + setting_values[4:6] + ["10"] + setting_values[6:]
            )
            variable_revision_number = 5

        return setting_values, variable_revision_number
