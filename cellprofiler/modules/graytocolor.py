# coding=utf-8

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

============ ============
Supports 2D? Supports 3D?
============ ============
YES          NO
============ ============

See also
^^^^^^^^

See also **ColorToGray** and **InvertForPrinting**.
"""

import numpy as np

import cellprofiler.image as cpi
import cellprofiler.module as cpm
import cellprofiler.setting as cps

OFF_RED_IMAGE_NAME = 0
OFF_GREEN_IMAGE_NAME = 1
OFF_BLUE_IMAGE_NAME = 2
OFF_RGB_IMAGE_NAME = 3
OFF_RED_ADJUSTMENT_FACTOR = 4
OFF_GREEN_ADJUSTMENT_FACTOR = 5
OFF_BLUE_ADJUSTMENT_FACTOR = 6

OFF_STACK_CHANNELS_V2 = 16
OFF_STACK_CHANNEL_COUNT_V3 = 16
OFF_STACK_CHANNEL_COUNT = 16

SCHEME_RGB = "RGB"
SCHEME_CMYK = "CMYK"
SCHEME_STACK = "Stack"
SCHEME_COMPOSITE = "Composite"
LEAVE_THIS_BLACK = "Leave this black"

DEFAULT_COLORS = [
    "#%02x%02x%02x" % color for color in
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (128, 128, 0), (128, 0, 128), (0, 128, 128)]


class GrayToColor(cpm.Module):
    module_name = 'GrayToColor'
    variable_revision_number = 3
    category = "Image Processing"

    def create_settings(self):
        self.scheme_choice = cps.Choice(
                "Select a color scheme",
                [SCHEME_RGB, SCHEME_CMYK, SCHEME_STACK, SCHEME_COMPOSITE],
                doc="""\
This module can use one of two color schemes to combine images:

-  *%(SCHEME_RGB)s*: Each input image determines the intensity of one
   of the color channels: red, green, and blue.
-  *%(SCHEME_CMYK)s*: Three of the input images are combined to
   determine the colors (cyan, magenta, and yellow) and a fourth is used
   only for brightness. The cyan image adds equally to the green and
   blue intensities. The magenta image adds equally to the red and blue
   intensities. The yellow image adds equally to the red and green
   intensities.
-  *%(SCHEME_STACK)s*: The channels are stacked in the order listed,
   from top to bottom. An arbitrary number of channels is allowed.

   For example, you could create a 5-channel image by providing
   5 grayscale images. The first grayscale image you provide will fill 
   the first channel, the second grayscale image you provide will fill
   the second channel, and so on.
-  *%(SCHEME_COMPOSITE)s*: A color is assigned to each grayscale image.
   Each grayscale image is converted to color by multiplying the
   intensity by the color and the resulting color images are added
   together. An arbitrary number of channels can be composited into a
   single color image.
""" % globals())

        # # # # # # # # # # # # # # # #
        #
        # RGB settings
        #
        # # # # # # # # # # # # # # # #
        self.red_image_name = cps.ImageNameSubscriber(
                "Select the image to be colored red",
                can_be_blank=True, blank_text=LEAVE_THIS_BLACK,
                doc="""\
*(Used only if "%(SCHEME_RGB)s" is selected as the color scheme)*

Select the input image to be displayed in red.
""" % globals())

        self.green_image_name = cps.ImageNameSubscriber(
                "Select the image to be colored green",
                can_be_blank=True, blank_text=LEAVE_THIS_BLACK,
                doc="""\
*(Used only if "%(SCHEME_RGB)s" is selected as the color scheme)*

Select the input image to be displayed in green.
""" % globals())

        self.blue_image_name = cps.ImageNameSubscriber(
                "Select the image to be colored blue",
                can_be_blank=True, blank_text=LEAVE_THIS_BLACK,
                doc="""\
*(Used only if "%(SCHEME_RGB)s" is selected as the color scheme)*

Select the input image to be displayed in blue.
""" % globals())

        self.rgb_image_name = cps.ImageNameProvider(
                "Name the output image", "ColorImage", doc="""Enter a name for the resulting image.""")

        self.red_adjustment_factor = cps.Float(
                "Relative weight for the red image",
                value=1, minval=0, doc='''\
*(Used only if "%(SCHEME_RGB)s" is selected as the color scheme)*

Enter the relative weight for the red image. If all relative weights are
equal, all three colors contribute equally in the final image. To weight
colors relative to each other, increase or decrease the relative
weights.
''' % globals())

        self.green_adjustment_factor = cps.Float(
                "Relative weight for the green image",
                value=1, minval=0, doc='''\
*(Used only if "%(SCHEME_RGB)s" is selected as the color scheme)*

Enter the relative weight for the green image. If all relative weights
are equal, all three colors contribute equally in the final image. To
weight colors relative to each other, increase or decrease the relative
weights.
''' % globals())

        self.blue_adjustment_factor = cps.Float(
                "Relative weight for the blue image",
                value=1, minval=0, doc='''\
*(Used only if "%(SCHEME_RGB)s" is selected as the color scheme)*

Enter the relative weight for the blue image. If all relative weights
are equal, all three colors contribute equally in the final image. To
weight colors relative to each other, increase or decrease the relative
weights.
''' % globals())
        # # # # # # # # # # # # # #
        #
        # CYMK settings
        #
        # # # # # # # # # # # # # #
        self.cyan_image_name = cps.ImageNameSubscriber(
                "Select the image to be colored cyan", can_be_blank=True,
                blank_text=LEAVE_THIS_BLACK,
                doc="""\
*(Used only if "%(SCHEME_CMYK)s" is selected as the color scheme)*

Select the input image to be displayed in cyan.
""" % globals())

        self.magenta_image_name = cps.ImageNameSubscriber(
                "Select the image to be colored magenta", can_be_blank=True,
                blank_text=LEAVE_THIS_BLACK,
                doc="""\
*(Used only if "%(SCHEME_CMYK)s" is selected as the color scheme)*

Select the input image to be displayed in magenta.
""" % globals())

        self.yellow_image_name = cps.ImageNameSubscriber(
                "Select the image to be colored yellow", can_be_blank=True,
                blank_text=LEAVE_THIS_BLACK,
                doc="""\
*(Used only if "%(SCHEME_CMYK)s" is selected as the color scheme)*

Select the input image to be displayed in yellow.
""" % globals())

        self.gray_image_name = cps.ImageNameSubscriber(
                "Select the image that determines brightness", can_be_blank=True,
                blank_text=LEAVE_THIS_BLACK,
                doc="""\
*(Used only if "%(SCHEME_CMYK)s" is selected as the color scheme)*

Select the input image that will determine each pixel's brightness.
""" % globals())

        self.cyan_adjustment_factor = cps.Float(
                "Relative weight for the cyan image",
                value=1, minval=0, doc='''\
*(Used only if "%(SCHEME_CMYK)s" is selected as the color scheme)*

Enter the relative weight for the cyan image. If all relative weights
are equal, all colors contribute equally in the final image. To weight
colors relative to each other, increase or decrease the relative
weights.
''' % globals())

        self.magenta_adjustment_factor = cps.Float(
                "Relative weight for the magenta image",
                value=1, minval=0, doc='''\
*(Used only if "%(SCHEME_CMYK)s" is selected as the color scheme)*

Enter the relative weight for the magenta image. If all relative weights
are equal, all colors contribute equally in the final image. To weight
colors relative to each other, increase or decrease the relative
weights.
''' % globals())

        self.yellow_adjustment_factor = cps.Float(
                "Relative weight for the yellow image",
                value=1, minval=0, doc='''\
*(Used only if "%(SCHEME_CMYK)s" is selected as the color scheme)*

Enter the relative weight for the yellow image. If all relative weights
are equal, all colors contribute equally in the final image. To weight
colors relative to each other, increase or decrease the relative
weights.
''' % globals())

        self.gray_adjustment_factor = cps.Float(
                "Relative weight for the brightness image",
                value=1, minval=0, doc='''\
*(Used only if "%(SCHEME_CMYK)s" is selected as the color scheme)*

Enter the relative weight for the brightness image. If all relative
weights are equal, all colors contribute equally in the final image. To
weight colors relative to each other, increase or decrease the relative
weights.
''' % globals())

        # # # # # # # # # # # # # #
        #
        # Stack settings
        #
        # # # # # # # # # # # # # #

        self.stack_channels = []
        self.stack_channel_count = cps.HiddenCount(self.stack_channels)
        self.add_stack_channel_cb(can_remove=False)
        self.add_stack_channel = cps.DoSomething("Add another channel", "Add another channel", self.add_stack_channel_cb,
        doc="""\
    Press this button to add another image to the stack.
    """)

    def add_stack_channel_cb(self, can_remove=True):
        group = cps.SettingsGroup()
        default_color = DEFAULT_COLORS[
            len(self.stack_channels) % len(DEFAULT_COLORS)]
        group.append("image_name", cps.ImageNameSubscriber(
                "Image name", cps.NONE,
                doc='''\
*(Used only if "%(SCHEME_STACK)s" or "%(SCHEME_COMPOSITE)s" is chosen)*

Select the input image to add to the stacked image.
''' % globals()))
        group.append("color", cps.Color(
                "Color", default_color,
                doc='''\
*(Used only if "%(SCHEME_COMPOSITE)s" is chosen)*

The color to be assigned to the above image.
''' % globals()))
        group.append("weight", cps.Float(
                "Weight", 1.0, minval=.5 / 255,
                doc='''\
*(Used only if "%(SCHEME_COMPOSITE)s" is chosen)*

The weighting of the above image relative to the others. The image’s
pixel values are multiplied by this weight before assigning the color.
''' % globals()))

        if can_remove:
            group.append("remover", cps.RemoveSettingButton("", "Remove this image", self.stack_channels, group))
        self.stack_channels.append(group)

    @property
    def color_scheme_settings(self):
        if self.scheme_choice == SCHEME_RGB:
            return [ColorSchemeSettings(self.red_image_name,
                                        self.red_adjustment_factor, 1, 0, 0),
                    ColorSchemeSettings(self.green_image_name,
                                        self.green_adjustment_factor, 0, 1, 0),
                    ColorSchemeSettings(self.blue_image_name,
                                        self.blue_adjustment_factor, 0, 0, 1)]
        elif self.scheme_choice == SCHEME_CMYK:
            return [ColorSchemeSettings(self.cyan_image_name,
                                        self.cyan_adjustment_factor, 0, .5, .5),
                    ColorSchemeSettings(self.magenta_image_name,
                                        self.magenta_adjustment_factor, .5, .5, 0),
                    ColorSchemeSettings(self.yellow_image_name,
                                        self.yellow_adjustment_factor, .5, 0, .5),
                    ColorSchemeSettings(self.gray_image_name,
                                        self.gray_adjustment_factor,
                                        1. / 3., 1. / 3., 1. / 3.)]
        else:
            return []

    def settings(self):
        result = [self.scheme_choice,
                  self.red_image_name, self.green_image_name, self.blue_image_name,
                  self.rgb_image_name, self.red_adjustment_factor,
                  self.green_adjustment_factor, self.blue_adjustment_factor,
                  self.cyan_image_name, self.magenta_image_name,
                  self.yellow_image_name, self.gray_image_name,
                  self.cyan_adjustment_factor, self.magenta_adjustment_factor,
                  self.yellow_adjustment_factor, self.gray_adjustment_factor,
                  self.stack_channel_count]
        for stack_channel in self.stack_channels:
            result += [stack_channel.image_name, stack_channel.color,
                       stack_channel.weight]
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
        result += [color_scheme_setting.image_name
                   for color_scheme_setting in self.color_scheme_settings]
        result += [self.rgb_image_name]
        for color_scheme_setting in self.color_scheme_settings:
            if not color_scheme_setting.image_name.is_blank:
                result.append(color_scheme_setting.adjustment_factor)
        if self.scheme_choice in (SCHEME_STACK, SCHEME_COMPOSITE):
            for sc_group in self.stack_channels:
                result.append(sc_group.image_name)
                if self.scheme_choice == SCHEME_COMPOSITE:
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
        if self.scheme_choice not in (SCHEME_STACK, SCHEME_COMPOSITE):
            if all([color_scheme_setting.image_name.is_blank
                    for color_scheme_setting in self.color_scheme_settings]):
                raise cps.ValidationError(
                        "At least one of the images must not be blank",
                        self.color_scheme_settings[0].image_name)

    def run(self, workspace):
        parent_image = None
        parent_image_name = None
        imgset = workspace.image_set
        rgb_pixel_data = None
        input_image_names = []
        channel_names = []
        if self.scheme_choice not in (SCHEME_STACK, SCHEME_COMPOSITE):
            for color_scheme_setting in self.color_scheme_settings:
                if color_scheme_setting.image_name.is_blank:
                    channel_names.append("Blank")
                    continue
                image_name = color_scheme_setting.image_name.value
                input_image_names.append(image_name)
                channel_names.append(image_name)
                image = imgset.get_image(image_name,
                                         must_be_grayscale=True)
                multiplier = (color_scheme_setting.intensities *
                              color_scheme_setting.adjustment_factor.value)
                pixel_data = image.pixel_data
                if parent_image is not None:
                    if parent_image.pixel_data.shape != pixel_data.shape:
                        raise ValueError("The %s image and %s image have different sizes (%s vs %s)" %
                                         (parent_image_name,
                                          color_scheme_setting.image_name.value,
                                          parent_image.pixel_data.shape,
                                          image.pixel_data.shape))
                    rgb_pixel_data += np.dstack([pixel_data] * 3) * multiplier
                else:
                    parent_image = image
                    parent_image_name = color_scheme_setting.image_name.value
                    rgb_pixel_data = np.dstack([pixel_data] * 3) * multiplier
        else:
            input_image_names = [sc.image_name.value for sc in self.stack_channels]
            channel_names = input_image_names
            source_channels = [imgset.get_image(name, must_be_grayscale=True).pixel_data
                               for name in input_image_names]
            parent_image = imgset.get_image(input_image_names[0])
            for idx, pd in enumerate(source_channels):
                if pd.shape != source_channels[0].shape:
                    raise ValueError("The %s image and %s image have different sizes (%s vs %s)" %
                                     (self.stack_channels[0].image_name.value,
                                      self.stack_channels[idx].image_name.value,
                                      source_channels[0].shape,
                                      pd.pixel_data.shape))
            if self.scheme_choice == SCHEME_STACK:
                rgb_pixel_data = np.dstack(source_channels)
            else:
                colors = []
                for sc in self.stack_channels:
                    color_tuple = sc.color.to_rgb()
                    color = sc.weight.value * np.array(color_tuple).astype(
                            parent_image.pixel_data.dtype) / 255
                    colors.append(color[np.newaxis, np.newaxis, :])
                rgb_pixel_data = \
                    parent_image.pixel_data[:, :, np.newaxis] * colors[0]
                for image, color in zip(source_channels[1:], colors[1:]):
                    rgb_pixel_data = rgb_pixel_data + \
                                     image[:, :, np.newaxis] * color

        ##############
        # Save image #
        ##############
        rgb_image = cpi.Image(rgb_pixel_data, parent_image=parent_image)
        rgb_image.channel_names = channel_names
        imgset.add(self.rgb_image_name.value, rgb_image)

        ##################
        # Display images #
        ##################
        if self.show_window:
            workspace.display_data.input_image_names = input_image_names
            workspace.display_data.rgb_pixel_data = rgb_pixel_data
            workspace.display_data.images = \
                [imgset.get_image(name, must_be_grayscale=True).pixel_data
                 for name in input_image_names]

    def display(self, workspace, figure):
        input_image_names = workspace.display_data.input_image_names
        images = workspace.display_data.images
        nsubplots = len(input_image_names)

        if self.scheme_choice == SCHEME_CMYK:
            subplots = (3, 2)
            subplot_indices = ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0))
            color_subplot = (2, 1)
        elif self.scheme_choice == SCHEME_RGB:
            subplots = (2, 2)
            subplot_indices = ((0, 0), (0, 1), (1, 0))
            color_subplot = (1, 1)
        else:
            subplots = (min(nsubplots + 1, 4), int(nsubplots / 4) + 1)
            subplot_indices = [(i % 4, int(i / 4)) for i in range(nsubplots)]
            color_subplot = (nsubplots % 4, int(nsubplots / 4))
        figure.set_subplots(subplots)
        for i, (input_image_name, image_pixel_data) in \
                enumerate(zip(input_image_names, images)):
            x, y = subplot_indices[i]
            figure.subplot_imshow_grayscale(x, y, image_pixel_data,
                                            title=input_image_name,
                                            sharexy=figure.subplot(0, 0))
            figure.subplot(x, y).set_visible(True)
        for x, y in subplot_indices[len(input_image_names):]:
            figure.subplot(x, y).set_visible(False)
        figure.subplot_imshow(color_subplot[0], color_subplot[1],
                              workspace.display_data.rgb_pixel_data[:, :, :3],
                              title=self.rgb_image_name.value,
                              sharexy=figure.subplot(0, 0))

    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        if from_matlab and variable_revision_number == 1:
            # Blue and red were switched: it was BGR
            temp = list(setting_values)
            temp[OFF_RED_IMAGE_NAME] = setting_values[OFF_BLUE_IMAGE_NAME]
            temp[OFF_BLUE_IMAGE_NAME] = setting_values[OFF_RED_IMAGE_NAME]
            temp[OFF_RED_ADJUSTMENT_FACTOR] = setting_values[OFF_BLUE_ADJUSTMENT_FACTOR]
            temp[OFF_BLUE_ADJUSTMENT_FACTOR] = setting_values[OFF_RED_ADJUSTMENT_FACTOR]
            setting_values = temp
            variable_revision_number = 2
        if from_matlab and variable_revision_number == 2:
            from_matlab = False
            variable_revision_number = 1
        if from_matlab and variable_revision_number == 3:
            image_names = [LEAVE_THIS_BLACK if value == cps.DO_NOT_USE
                           else value
                           for value in setting_values[:4]]
            rgb_image_name = setting_values[4]
            adjustment_factors = setting_values[5:]
            if image_names[3] == LEAVE_THIS_BLACK:
                #
                # RGB color scheme
                #
                setting_values = (
                    [SCHEME_RGB] + image_names[:3] + [rgb_image_name] +
                    adjustment_factors[:3] + [cps.NONE] * 4 + [1] * 4)
            else:
                #
                # CYMK color scheme
                #
                setting_values = (
                    [SCHEME_CMYK] + [cps.NONE] * 3 + [rgb_image_name] +
                    [1] * 3 + image_names + adjustment_factors)
            from_matlab = False
            variable_revision_number = 2
        if (not from_matlab) and variable_revision_number == 1:
            #
            # Was RGB-only. Convert values to CYMK-style
            #
            setting_values = (
                [SCHEME_CMYK] + setting_values +
                [cps.NONE] * 4 + [1] * 4)
            variable_revision_number = 2
        if (not from_matlab) and variable_revision_number == 2:
            #
            # Added composite mode
            #
            n_stacked = len(setting_values) - OFF_STACK_CHANNELS_V2
            new_setting_values = list(setting_values[:OFF_STACK_CHANNELS_V2])
            new_setting_values.append(str(n_stacked))
            for i, image_name in enumerate(
                    setting_values[OFF_STACK_CHANNELS_V2:]):
                new_setting_values += \
                    [image_name, DEFAULT_COLORS[i % len(DEFAULT_COLORS)], "1.0"]
            setting_values = new_setting_values
            variable_revision_number = 3
        return setting_values, variable_revision_number, from_matlab


class ColorSchemeSettings(object):
    '''Collect all of the details for one color in one place'''

    def __init__(self, image_name_setting, adjustment_setting,
                 red_intensity, green_intensity, blue_intensity):
        '''Initialize with settings and multipliers

        image_name_setting - names the image to use for the color
        adjustment_setting - weights the image
        red_intensity - indicates how much it contributes to the red channel
        green_intensity - indicates how much it contributes to the green channel
        blue_intensity - indicates how much it contributes to the blue channel
        '''
        self.image_name = image_name_setting
        self.adjustment_factor = adjustment_setting
        self.red_intensity = red_intensity
        self.green_intensity = green_intensity
        self.blue_intensity = blue_intensity

    @property
    def intensities(self):
        '''The intensities in RGB order as a numpy array'''
        return np.array((self.red_intensity,
                         self.green_intensity,
                         self.blue_intensity))
