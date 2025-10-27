"""
GrayToColor
===========

**GrayToColor** takes grayscale images and produces a color image
from them.

This module takes grayscale images as input and assigns them to colors
in a red, green, blue (RGB) image or a cyan, magenta, yellow, black
(CMYK) image. Each color’s brightness can be adjusted independently by
using relative weights.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           NO
============ ============ ===============

See also
^^^^^^^^

See also **ColorToGray** and **InvertForPrinting**.
"""

import numpy
from typing import List
from cellprofiler_core.image import Image
from cellprofiler_core.module import Module
from cellprofiler_core.setting import Color, Binary
from cellprofiler_core.setting import HiddenCount
from cellprofiler_core.setting import SettingsGroup
from cellprofiler_core.setting import ValidationError
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.do_something import DoSomething, RemoveSettingButton
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.text import ImageName, Float
from cellprofiler_library.modules._graytocolor import gray_to_rgb, gray_to_stacked_color, gray_to_composite_color, gray_to_color
from cellprofiler_library.types import Image2DGrayscale
from cellprofiler_library.opts.graytocolor import Scheme

OFF_RED_IMAGE_NAME = 0
OFF_GREEN_IMAGE_NAME = 1
OFF_BLUE_IMAGE_NAME = 2
OFF_RGB_IMAGE_NAME = 3
OFF_RED_ADJUSTMENT_FACTOR = 4
OFF_GREEN_ADJUSTMENT_FACTOR = 5
OFF_BLUE_ADJUSTMENT_FACTOR = 6

OFF_STACK_CHANNELS_V2 = 16
OFF_STACK_CHANNEL_COUNT_V3 = 16
OFF_STACK_CHANNEL_COUNT = 17

# Scheme.RGB = "RGB"
# Scheme.CMYK = "CMYK"
# Scheme.STACK = "Stack"
# Scheme.COMPOSITE = "Composite"
LEAVE_THIS_BLACK = "Leave this black"

DEFAULT_COLORS = [
    "#%02x%02x%02x" % color
    for color in (
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (128, 128, 0),
        (128, 0, 128),
        (0, 128, 128),
    )
]


class GrayToColor(Module):
    module_name = "GrayToColor"
    variable_revision_number = 4
    category = "Image Processing"

    def create_settings(self):
        self.scheme_choice = Choice(
            "Select a color scheme",
            [Scheme.RGB, Scheme.CMYK, Scheme.STACK, Scheme.COMPOSITE],
            doc="""\
This module can use one of two color schemes to combine images:

-  *{SCHEME_RGB}*: Each input image determines the intensity of one
   of the color channels: red, green, and blue.
-  *{SCHEME_CMYK}*: Three of the input images are combined to
   determine the colors (cyan, magenta, and yellow) and a fourth is used
   only for brightness. The cyan image adds equally to the green and
   blue intensities. The magenta image adds equally to the red and blue
   intensities. The yellow image adds equally to the red and green
   intensities.
-  *{SCHEME_STACK}*: The channels are stacked in the order listed,
   from top to bottom. An arbitrary number of channels is allowed.

   For example, you could create a 5-channel image by providing
   5 grayscale images. The first grayscale image you provide will fill
   the first channel, the second grayscale image you provide will fill
   the second channel, and so on.
-  *{SCHEME_COMPOSITE}*: A color is assigned to each grayscale image.
   Each grayscale image is converted to color by multiplying the
   intensity by the color and the resulting color images are added
   together. An arbitrary number of channels can be composited into a
   single color image.
""".format(
                **{"SCHEME_RGB": Scheme.RGB, "SCHEME_CMYK": Scheme.CMYK, "SCHEME_STACK": Scheme.STACK, "SCHEME_COMPOSITE": Scheme.COMPOSITE}
            )
        )

        self.wants_rescale = Binary(
            "Rescale intensity",
            True,
            doc="""\
Choose whether to rescale each channel individually to 
the range of 0-1. This prevents clipping of channels with intensity 
above 1 and can help to balance the brightness of the different channels. 
This option also ensures that channels occupy the full intensity range 
available, which is useful for displaying images in other software.

This rescaling is applied before any multiplication factors set in this 
module's options. Using a multiplication factor >1 would therefore result 
in clipping.""",
        )

        # # # # # # # # # # # # # # # #
        #
        # RGB settings
        #
        # # # # # # # # # # # # # # # #
        self.red_image_name = ImageSubscriber(
            "Select the image to be colored red",
            can_be_blank=True,
            blank_text=LEAVE_THIS_BLACK,
            doc="""\
*(Used only if "{SCHEME_RGB}" is selected as the color scheme)*

Select the input image to be displayed in red.
""".format(
                **{"SCHEME_RGB": Scheme.RGB}
            )
        )

        self.green_image_name = ImageSubscriber(
            "Select the image to be colored green",
            can_be_blank=True,
            blank_text=LEAVE_THIS_BLACK,
            doc="""\
*(Used only if "{SCHEME_RGB}" is selected as the color scheme)*

Select the input image to be displayed in green.
""".format(
                **{"SCHEME_RGB": Scheme.RGB}
            )
        )

        self.blue_image_name = ImageSubscriber(
            "Select the image to be colored blue",
            can_be_blank=True,
            blank_text=LEAVE_THIS_BLACK,
            doc="""\
*(Used only if "{SCHEME_RGB}" is selected as the color scheme)*

Select the input image to be displayed in blue.
""".format(
                **{"SCHEME_RGB": Scheme.RGB}
            )
        )

        self.rgb_image_name = ImageName(
            "Name the output image",
            "ColorImage",
            doc="""Enter a name for the resulting image.""",
        )

        self.red_adjustment_factor = Float(
            "Relative weight for the red image",
            value=1,
            minval=0,
            doc="""\
*(Used only if "{SCHEME_RGB}" is selected as the color scheme)*

Enter the relative weight for the red image. If all relative weights are
equal, all three colors contribute equally in the final image. To weight
colors relative to each other, increase or decrease the relative
weights.
""".format(
                **{"SCHEME_RGB": Scheme.RGB}
            )
        )

        self.green_adjustment_factor = Float(
            "Relative weight for the green image",
            value=1,
            minval=0,
            doc="""\
*(Used only if "{SCHEME_RGB}" is selected as the color scheme)*

Enter the relative weight for the green image. If all relative weights
are equal, all three colors contribute equally in the final image. To
weight colors relative to each other, increase or decrease the relative
weights.
""".format(
                **{"SCHEME_RGB": Scheme.RGB}
            )
        )

        self.blue_adjustment_factor = Float(
            "Relative weight for the blue image",
            value=1,
            minval=0,
            doc="""\
*(Used only if "{SCHEME_RGB}" is selected as the color scheme)*

Enter the relative weight for the blue image. If all relative weights
are equal, all three colors contribute equally in the final image. To
weight colors relative to each other, increase or decrease the relative
weights.
""".format(
                **{"SCHEME_RGB": Scheme.RGB}
            )
        )
        # # # # # # # # # # # # # #
        #
        # CYMK settings
        #
        # # # # # # # # # # # # # #
        self.cyan_image_name = ImageSubscriber(
            "Select the image to be colored cyan",
            can_be_blank=True,
            blank_text=LEAVE_THIS_BLACK,
            doc="""\
*(Used only if "{SCHEME_CMYK}" is selected as the color scheme)*

Select the input image to be displayed in cyan.
""".format(
                **{"SCHEME_CMYK": Scheme.CMYK}
            )
        )

        self.magenta_image_name = ImageSubscriber(
            "Select the image to be colored magenta",
            can_be_blank=True,
            blank_text=LEAVE_THIS_BLACK,
            doc="""\
*(Used only if "{SCHEME_CMYK}" is selected as the color scheme)*

Select the input image to be displayed in magenta.
""".format(
                **{"SCHEME_CMYK": Scheme.CMYK}
            )
        )

        self.yellow_image_name = ImageSubscriber(
            "Select the image to be colored yellow",
            can_be_blank=True,
            blank_text=LEAVE_THIS_BLACK,
            doc="""\
*(Used only if "{SCHEME_CMYK}" is selected as the color scheme)*

Select the input image to be displayed in yellow.
""".format(
                **{"SCHEME_CMYK": Scheme.CMYK}
            )
        )

        self.gray_image_name = ImageSubscriber(
            "Select the image that determines brightness",
            can_be_blank=True,
            blank_text=LEAVE_THIS_BLACK,
            doc="""\
*(Used only if "{SCHEME_CMYK}" is selected as the color scheme)*

Select the input image that will determine each pixel's brightness.
""".format(
                **{"SCHEME_CMYK": Scheme.CMYK}
            )
        )

        self.cyan_adjustment_factor = Float(
            "Relative weight for the cyan image",
            value=1,
            minval=0,
            doc="""\
*(Used only if "{SCHEME_CMYK}" is selected as the color scheme)*

Enter the relative weight for the cyan image. If all relative weights
are equal, all colors contribute equally in the final image. To weight
colors relative to each other, increase or decrease the relative
weights.
""".format(
                **{"SCHEME_CMYK": Scheme.CMYK}
            )
        )

        self.magenta_adjustment_factor = Float(
            "Relative weight for the magenta image",
            value=1,
            minval=0,
            doc="""\
*(Used only if "{SCHEME_CMYK}" is selected as the color scheme)*

Enter the relative weight for the magenta image. If all relative weights
are equal, all colors contribute equally in the final image. To weight
colors relative to each other, increase or decrease the relative
weights.
""".format(
                **{"SCHEME_CMYK": Scheme.CMYK}
            )
        )

        self.yellow_adjustment_factor = Float(
            "Relative weight for the yellow image",
            value=1,
            minval=0,
            doc="""\
*(Used only if "{SCHEME_CMYK}" is selected as the color scheme)*

Enter the relative weight for the yellow image. If all relative weights
are equal, all colors contribute equally in the final image. To weight
colors relative to each other, increase or decrease the relative
weights.
""".format(
                **{"SCHEME_CMYK": Scheme.CMYK}
            )
        )

        self.gray_adjustment_factor = Float(
            "Relative weight for the brightness image",
            value=1,
            minval=0,
            doc="""\
*(Used only if "{SCHEME_CMYK}" is selected as the color scheme)*

Enter the relative weight for the brightness image. If all relative
weights are equal, all colors contribute equally in the final image. To
weight colors relative to each other, increase or decrease the relative
weights.
""".format(
                **{"SCHEME_CMYK": Scheme.CMYK}
            )
        )

        # # # # # # # # # # # # # #
        #
        # Stack settings
        #
        # # # # # # # # # # # # # #

        self.stack_channels = []
        self.stack_channel_count = HiddenCount(self.stack_channels)
        self.add_stack_channel_cb(can_remove=False)
        self.add_stack_channel = DoSomething(
            "Add another channel",
            "Add another channel",
            self.add_stack_channel_cb,
            doc="""\
    Press this button to add another image to the stack.
    """,
        )

    def add_stack_channel_cb(self, can_remove=True):
        group = SettingsGroup()
        default_color = DEFAULT_COLORS[len(self.stack_channels) % len(DEFAULT_COLORS)]
        group.append(
            "image_name",
            ImageSubscriber(
                "Image name",
                "None",
                doc="""\
*(Used only if "{SCHEME_STACK}" or "{SCHEME_COMPOSITE}" is chosen)*

Select the input image to add to the stacked image.
""".format(
                **{"SCHEME_STACK": Scheme.STACK, "SCHEME_COMPOSITE": Scheme.COMPOSITE}
            )
            ),
        )
        group.append(
            "color",
            Color(
                "Color",
                default_color,
                doc="""\
*(Used only if "{SCHEME_COMPOSITE}" is chosen)*

The color to be assigned to the above image.
""".format(
                **{"SCHEME_COMPOSITE": Scheme.COMPOSITE}
            )
            ),
        )
        group.append(
            "weight",
            Float(
                "Weight",
                1.0,
                minval=0.5 / 255,
                doc="""\
*(Used only if "{SCHEME_COMPOSITE}" is chosen)*

The weighting of the above image relative to the others. The image’s
pixel values are multiplied by this weight before assigning the color.
""".format(
                **{"SCHEME_COMPOSITE": Scheme.COMPOSITE}
            )
            ),
        )

        if can_remove:
            group.append(
                "remover",
                RemoveSettingButton(
                    "", "Remove this image", self.stack_channels, group
                ),
            )
        self.stack_channels.append(group)

    @property
    def color_scheme_settings(self):
        if self.scheme_choice == Scheme.RGB:
            return [
                ColorSchemeSettings(
                    self.red_image_name, self.red_adjustment_factor, 1, 0, 0
                ),
                ColorSchemeSettings(
                    self.green_image_name, self.green_adjustment_factor, 0, 1, 0
                ),
                ColorSchemeSettings(
                    self.blue_image_name, self.blue_adjustment_factor, 0, 0, 1
                ),
            ]
        elif self.scheme_choice == Scheme.CMYK:
            return [
                ColorSchemeSettings(
                    self.cyan_image_name, self.cyan_adjustment_factor, 0, 0.5, 0.5
                ),
                ColorSchemeSettings(
                    self.magenta_image_name, self.magenta_adjustment_factor, 0.5, 0, 0.5
                ),
                ColorSchemeSettings(
                    self.yellow_image_name, self.yellow_adjustment_factor, 0.5, 0.5, 0
                ),
                ColorSchemeSettings(
                    self.gray_image_name,
                    self.gray_adjustment_factor,
                    1.0 / 3.0,
                    1.0 / 3.0,
                    1.0 / 3.0,
                ),
            ]
        else:
            return []

    def settings(self):
        result = [
            self.scheme_choice,
            self.wants_rescale,
            self.red_image_name,
            self.green_image_name,
            self.blue_image_name,
            self.rgb_image_name,
            self.red_adjustment_factor,
            self.green_adjustment_factor,
            self.blue_adjustment_factor,
            self.cyan_image_name,
            self.magenta_image_name,
            self.yellow_image_name,
            self.gray_image_name,
            self.cyan_adjustment_factor,
            self.magenta_adjustment_factor,
            self.yellow_adjustment_factor,
            self.gray_adjustment_factor,
            self.stack_channel_count,
        ]
        for stack_channel in self.stack_channels:
            result += [
                stack_channel.image_name,
                stack_channel.color,
                stack_channel.weight,
            ]
        return result

    def prepare_settings(self, setting_values):
        try:
            num_stack_images = int(setting_values[OFF_STACK_CHANNEL_COUNT])
        except ValueError:
            num_stack_images = 1
        del self.stack_channels[num_stack_images:]
        while len(self.stack_channels) < num_stack_images:
            self.add_stack_channel_cb()

    def visible_settings(self):
        result = [self.scheme_choice]
        result += [
            color_scheme_setting.image_name
            for color_scheme_setting in self.color_scheme_settings
        ]
        result += [self.rgb_image_name]
        if self.scheme_choice != Scheme.STACK:
            result += [self.wants_rescale]
        for color_scheme_setting in self.color_scheme_settings:
            if not color_scheme_setting.image_name.is_blank:
                result.append(color_scheme_setting.adjustment_factor)
        if self.scheme_choice in (Scheme.STACK, Scheme.COMPOSITE):
            for sc_group in self.stack_channels:
                result.append(sc_group.image_name)
                if self.scheme_choice == Scheme.COMPOSITE:
                    result.append(sc_group.color)
                    result.append(sc_group.weight)
                if hasattr(sc_group, "remover"):
                    result.append(sc_group.remover)
            result += [self.add_stack_channel]
        return result

    def validate_module(self, pipeline):
        """Make sure that the module's settings are consistent

        We need at least one image name to be filled in
        """
        if self.scheme_choice not in (Scheme.STACK, Scheme.COMPOSITE):
            if all(
                [
                    color_scheme_setting.image_name.is_blank
                    for color_scheme_setting in self.color_scheme_settings
                ]
            ):
                raise ValidationError(
                    "At least one of the images must not be blank",
                    self.color_scheme_settings[0].image_name,
                )

    def run(self, workspace):
        parent_image = None
        parent_image_name = None
        imgset = workspace.image_set
        rgb_pixel_data = None
        input_image_names = []
        channel_names = []
        channelstack =  self.scheme_choice == Scheme.STACK
        if self.scheme_choice.value not in (Scheme.STACK, Scheme.COMPOSITE):
            if self.scheme_choice.value == Scheme.RGB:
                image_arr: List[Image2DGrayscale] = [
                    None if self.red_image_name.value == LEAVE_THIS_BLACK else imgset.get_image(self.red_image_name.value, must_be_grayscale=True).pixel_data,
                    None if self.green_image_name.value == LEAVE_THIS_BLACK else imgset.get_image(self.green_image_name.value, must_be_grayscale=True).pixel_data,
                    None if self.blue_image_name.value == LEAVE_THIS_BLACK else imgset.get_image(self.blue_image_name.value, must_be_grayscale=True).pixel_data,
                ]
                adjustment_factor_array = [
                    self.red_adjustment_factor.value,
                    self.green_adjustment_factor.value,
                    self.blue_adjustment_factor.value,
                ]
                intensities = [
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                ]
                rgb_pixel_data = gray_to_color(
                # rgb_pixel_data = gray_to_rgb(
                    pixel_data_arr = image_arr, 
                    scheme = self.scheme_choice.value,
                    adjustment_factor_array = adjustment_factor_array, 
                    intensities = intensities, 
                    wants_rescale = self.wants_rescale.value
                    )
                # TODO: is it okay to use the first image as the parent image? I think that's what the original code is doing.
                non_blank_image_names = [i for i in [self.red_image_name.value, self.green_image_name.value, self.blue_image_name.value] if i != LEAVE_THIS_BLACK]
                assert len(non_blank_image_names) != 0, "At least one of the images must not be blank"
                parent_image = imgset.get_image(non_blank_image_names[0], must_be_grayscale=True)
            elif self.scheme_choice.value == Scheme.CMYK:
                image_arr: List[Image2DGrayscale] = [
                    None if self.cyan_image_name.value == LEAVE_THIS_BLACK else imgset.get_image(self.cyan_image_name.value, must_be_grayscale=True).pixel_data,
                    None if self.magenta_image_name.value == LEAVE_THIS_BLACK else imgset.get_image(self.magenta_image_name.value, must_be_grayscale=True).pixel_data,
                    None if self.yellow_image_name.value == LEAVE_THIS_BLACK else imgset.get_image(self.yellow_image_name.value, must_be_grayscale=True).pixel_data,
                    None if self.gray_image_name.value == LEAVE_THIS_BLACK else imgset.get_image(self.gray_image_name.value, must_be_grayscale=True).pixel_data,
                ]
                adjustment_factor_array = [
                    self.cyan_adjustment_factor.value,
                    self.magenta_adjustment_factor.value,
                    self.yellow_adjustment_factor.value,
                    self.gray_adjustment_factor.value,
                ]
                intensities = [
                    (0, 0.5, 0.5),
                    (0.5, 0, 0.5),
                    (0.5, 0.5, 0),
                    (1.0/3.0, 1.0/3.0, 1.0/3.0),
                ]

                rgb_pixel_data = gray_to_color(
                    pixel_data_arr = image_arr, 
                    scheme=self.scheme_choice.value,
                    adjustment_factor_array = adjustment_factor_array, 
                    intensities = intensities, 
                    wants_rescale = self.wants_rescale.value
                    )
                # TODO: is it okay to use the first image as the parent image? I think that's what the original code is doing.
                non_blank_image_names = [i for i in [self.cyan_image_name, self.magenta_image_name, self.yellow_image_name, self.gray_image_name] if i != LEAVE_THIS_BLACK]
                assert len(non_blank_image_names) != 0, "At least one of the images must not be blank"
                parent_image = imgset.get_image(non_blank_image_names[0], must_be_grayscale=True)
            else:
                raise ValueError(f"Unimplemented scheme?: {self.scheme_choice}")
        else:
            input_image_names = [sc.image_name.value for sc in self.stack_channels]
            channel_names = input_image_names
            source_channels: List[Image2DGrayscale] = [
                imgset.get_image(name, must_be_grayscale=True).pixel_data
                for name in input_image_names
            ]
            if self.scheme_choice.value == Scheme.STACK:
                rgb_pixel_data = gray_to_stacked_color(source_channels)
                parent_image = imgset.get_image(input_image_names[0])
            elif self.scheme_choice.value == Scheme.COMPOSITE:
                parent_image = imgset.get_image(input_image_names[0])
                color_array = []
                weight_array = []
                for sc in self.stack_channels:
                    color_array += [sc.color.to_rgb()]
                    weight_array += [sc.weight.value]
                rgb_pixel_data = gray_to_color(
                    pixel_data_arr = source_channels,
                    scheme=self.scheme_choice.value,
                    color_array = color_array,
                    weight_array = weight_array,
                    wants_rescale =self.wants_rescale.value,
                
            )
            else:
                raise ValueError(f"Unimplemented scheme: {self.scheme_choice}")
            
        # TODO move this to library/module
        if self.scheme_choice.value != Scheme.STACK and self.wants_rescale.value:
            # If we rescaled, clip values that went out of range after multiplication
            rgb_pixel_data[rgb_pixel_data > 1] = 1

        ##############
        # Save image #
        ##############
        rgb_image = Image(rgb_pixel_data, parent_image=parent_image, channelstack=channelstack)
        rgb_image.channel_names = channel_names
        imgset.add(self.rgb_image_name.value, rgb_image)

        ##################
        # Display images #
        ##################
        if self.show_window:
            workspace.display_data.input_image_names = input_image_names
            workspace.display_data.rgb_pixel_data = rgb_pixel_data
            workspace.display_data.images = [
                imgset.get_image(name, must_be_grayscale=True).pixel_data
                for name in input_image_names
            ]

    def display(self, workspace, figure):
        input_image_names = workspace.display_data.input_image_names
        images = workspace.display_data.images
        nsubplots = len(input_image_names)

        if self.scheme_choice == Scheme.CMYK:
            subplots = (3, 2)
            subplot_indices = ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0))
            color_subplot = (2, 1)
        elif self.scheme_choice == Scheme.RGB:
            subplots = (2, 2)
            subplot_indices = ((0, 0), (0, 1), (1, 0))
            color_subplot = (1, 1)
        else:
            subplots = (min(nsubplots + 1, 4), int(nsubplots / 4) + 1)
            subplot_indices = [(i % 4, int(i / 4)) for i in range(nsubplots)]
            color_subplot = (nsubplots % 4, int(nsubplots / 4))
        figure.set_subplots(subplots)
        for i, (input_image_name, image_pixel_data) in enumerate(
            zip(input_image_names, images)
        ):
            x, y = subplot_indices[i]
            figure.subplot_imshow_grayscale(
                x,
                y,
                image_pixel_data,
                title=input_image_name,
                sharexy=figure.subplot(0, 0),
            )
            figure.subplot(x, y).set_visible(True)
        for x, y in subplot_indices[len(input_image_names) :]:
            figure.subplot(x, y).set_visible(False)
        figure.subplot_imshow(
            color_subplot[0],
            color_subplot[1],
            workspace.display_data.rgb_pixel_data[:, :, :3],
            title=self.rgb_image_name.value,
            sharexy=figure.subplot(0, 0),
            normalize=False,
        )

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 1:
            #
            # Was RGB-only. Convert values to CYMK-style
            #
            setting_values = [Scheme.CMYK.value] + setting_values + ["None"] * 4 + [1] * 4
            variable_revision_number = 2
        if variable_revision_number == 2:
            #
            # Added composite mode
            #
            n_stacked = len(setting_values) - OFF_STACK_CHANNELS_V2
            new_setting_values = list(setting_values[:OFF_STACK_CHANNELS_V2])
            new_setting_values.append(str(n_stacked))
            for i, image_name in enumerate(setting_values[OFF_STACK_CHANNELS_V2:]):
                new_setting_values += [
                    image_name,
                    DEFAULT_COLORS[i % len(DEFAULT_COLORS)],
                    "1.0",
                ]
            setting_values = new_setting_values
            variable_revision_number = 3
        if variable_revision_number == 3:
            setting_values.insert(1, "No")
            variable_revision_number = 4
        return setting_values, variable_revision_number


class ColorSchemeSettings(object):
    """Collect all of the details for one color in one place"""

    def __init__(
        self,
        image_name_setting,
        adjustment_setting,
        red_intensity,
        green_intensity,
        blue_intensity,
    ):
        """Initialize with settings and multipliers

        image_name_setting - names the image to use for the color
        adjustment_setting - weights the image
        red_intensity - indicates how much it contributes to the red channel
        green_intensity - indicates how much it contributes to the green channel
        blue_intensity - indicates how much it contributes to the blue channel
        """
        self.image_name = image_name_setting
        self.adjustment_factor = adjustment_setting
        self.red_intensity = red_intensity
        self.green_intensity = green_intensity
        self.blue_intensity = blue_intensity

    @property
    def intensities(self):
        """The intensities in RGB order as a numpy array"""
        return numpy.array(
            (self.red_intensity, self.green_intensity, self.blue_intensity)
        )
