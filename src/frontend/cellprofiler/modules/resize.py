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
from cellprofiler_core.image import Image
from cellprofiler_core.module import ImageProcessing
from cellprofiler_core.setting import Divider, HiddenCount, SettingsGroup
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.do_something import DoSomething, RemoveSettingButton
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.text import Float, Integer, ImageName

from cellprofiler_library.modules._resize import resize_image
from cellprofiler_library.opts.resize import DimensionMethod, InterpolationMethod, ResizingMethod

LOGGER = logging.getLogger(__name__)

S_ADDITIONAL_IMAGE_COUNT = 12


class Resize(ImageProcessing):
    variable_revision_number = 5

    module_name = "Resize"

    def create_settings(self):
        super(Resize, self).create_settings()

        self.size_method = Choice(
            "Resizing method",
            [rm.value for rm in ResizingMethod],
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
            [dm.value for dm in DimensionMethod],
            doc="""\
*(Used only if resizing by specifying the dimensions)*

You have two options on how to resize your image:

-  *{C_MANUAL}:* Specify the height and width of the output image.
-  *{C_IMAGE}:* Specify an image and the input image will be resized to the same dimensions.
            """.format(
                **{"C_IMAGE": DimensionMethod.IMAGE.value, "C_MANUAL": DimensionMethod.MANUAL.value}
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
            [it.value for it in InterpolationMethod],
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

        if self.size_method.value == ResizingMethod.BY_FACTOR:
            visible_settings += [self.resizing_factor_x, self.resizing_factor_y, self.resizing_factor_z,]
        elif self.size_method.value == ResizingMethod.TO_SIZE:
            visible_settings += [self.use_manual_or_image]

            if self.use_manual_or_image.value == DimensionMethod.IMAGE:
                visible_settings += [self.specific_image]
            elif self.use_manual_or_image.value == DimensionMethod.MANUAL:
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
        # Extract reference image shape if needed
        reference_image_shape = None
        if self.size_method.value == ResizingMethod.TO_SIZE and self.use_manual_or_image.value == DimensionMethod.IMAGE:
            reference_image = workspace.image_set.get_image(self.specific_image.value)
            reference_image_shape = reference_image.pixel_data.shape
        
        # Process main image
        image = workspace.image_set.get_image(self.x_name.value)
        
        # Extract raw data from Image object
        pixel_data = image.pixel_data
        mask = image.mask
        volumetric = image.volumetric
        multichannel = image.multichannel
        dimensions = image.dimensions
        has_crop_mask = image.has_crop_mask
        crop_mask = image.crop_mask
        
        # Call library dispatcher with raw data
        output_pixels, output_mask, output_crop_mask = resize_image(
            pixel_data, mask,
            dimensions, crop_mask,
            self.size_method.value, self.resizing_factor_x.value, self.resizing_factor_y.value,
            self.resizing_factor_z.value, self.use_manual_or_image.value,
            self.specific_width.value, self.specific_height.value, self.specific_planes.value,
            reference_image_shape, self.interpolation.value
        )
        
        # Reconstruct Image object from raw data
        output_image = Image(
            output_pixels,
            parent_image=image,
            mask=output_mask,
            crop_mask=output_crop_mask,
            dimensions=dimensions,
        )
        
        # Add output image to workspace
        workspace.image_set.add(self.y_name.value, output_image)
        
        # Update display data if needed
        if self.show_window:
            if hasattr(workspace.display_data, "input_images"):
                workspace.display_data.multichannel += [multichannel]
                workspace.display_data.input_images += [pixel_data]
                workspace.display_data.output_images += [output_pixels]
                workspace.display_data.input_image_names += [self.x_name.value]
                workspace.display_data.output_image_names += [self.y_name.value]
            else:
                workspace.display_data.dimensions = dimensions
                workspace.display_data.multichannel = [multichannel]
                workspace.display_data.input_images = [pixel_data]
                workspace.display_data.output_images = [output_pixels]
                workspace.display_data.input_image_names = [self.x_name.value]
                workspace.display_data.output_image_names = [self.y_name.value]

        # Process additional images
        for additional in self.additional_images:
            # Extract raw data from Image object
            image = workspace.image_set.get_image(additional.input_image_name.value)
            pixel_data = image.pixel_data
            mask = image.mask
            dimensions = image.dimensions
            crop_mask = image.crop_mask
            
            # Call library dispatcher with raw data
            output_pixels, output_mask, output_crop_mask = resize_image(
                pixel_data, mask,
                dimensions, crop_mask,
                self.size_method.value, self.resizing_factor_x.value, self.resizing_factor_y.value,
                self.resizing_factor_z.value, self.use_manual_or_image.value,
                self.specific_width.value, self.specific_height.value, self.specific_planes.value,
                reference_image_shape, self.interpolation.value
            )
            
            # Reconstruct Image object from raw data
            output_image = Image(
                output_pixels,
                parent_image=image,
                mask=output_mask,
                crop_mask=output_crop_mask,
                dimensions=dimensions,
            )
            
            # Add output image to workspace
            workspace.image_set.add(additional.output_image_name.value, output_image)
            
            # Update display data if needed
            if self.show_window:
                workspace.display_data.multichannel += [multichannel]
                workspace.display_data.input_images += [pixel_data]
                workspace.display_data.output_images += [output_pixels]
                workspace.display_data.input_image_names += [additional.input_image_name.value]
                workspace.display_data.output_image_names += [additional.output_image_name.value]

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
                setting_values[2] = ResizingMethod.BY_FACTOR
            if setting_values[2] == "Resize to a size in pixels":
                setting_values[2] = ResizingMethod.TO_SIZE
            variable_revision_number = 2

        if variable_revision_number == 2:
            # Add additional images to be resized similarly, but if you only had 1,
            # the order didn't change
            setting_values = setting_values + ["0"]
            variable_revision_number = 3

        if variable_revision_number == 3:
            # Add resizing to another image size
            setting_values = (
                setting_values[:7] + [DimensionMethod.MANUAL, "None"] + setting_values[7:]
            )
            variable_revision_number = 4
        
        if variable_revision_number == 4:
            #Add X, Y and Z resizing factor 
            setting_values = (
                setting_values[:3] + [setting_values[3], setting_values[3], 1] + setting_values[4:6] + ["10"] + setting_values[6:]
            )
            variable_revision_number = 5

        return setting_values, variable_revision_number
