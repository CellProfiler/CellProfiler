# coding=utf-8

"""
ColorToGray
===========

**ColortoGray** converts an image with multiple color channels to one or more
grayscale images.

This module converts color and channel-stacked
images to grayscale. All channels can be merged into one grayscale image
(*Combine*), or each channel can be extracted into a separate grayscale image
(*Split*). If you use *Combine*, the relative weights you provide allow
adjusting the contribution of the colors relative to each other.
Note that all **Identify** modules require grayscale images.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           NO
============ ============ ===============

See also
^^^^^^^^

See also **GrayToColor**.
"""

import re

import matplotlib.colors
import numpy as np

import cellprofiler.image  as cpi
import cellprofiler.module as cpm
import cellprofiler.setting as cps

from cellprofiler.setting import YES, NO

COMBINE = "Combine"
SPLIT = "Split"

CH_RGB = "RGB"
CH_HSV = "HSV"
CH_CHANNELS = "Channels"

SLOT_CHANNEL_COUNT = 19
SLOT_FIXED_COUNT = 20
SLOTS_PER_CHANNEL = 3
SLOT_CHANNEL_CHOICE = 0


class ColorToGray(cpm.Module):
    module_name = "ColorToGray"
    variable_revision_number = 3
    category = "Image Processing"

    def create_settings(self):
        self.image_name = cps.ImageNameSubscriber(
                "Select the input image", cps.NONE, doc="""Select the multichannel image you want to convert to grayscale.""")

        self.combine_or_split = cps.Choice(
                "Conversion method",
                [COMBINE, SPLIT], doc='''\
How do you want to convert the color image?

-  *%(SPLIT)s:* Splits the channels of a color
   image (e.g., red, green, blue) into separate grayscale images.
-  *%(COMBINE)s:* Converts a color image to a grayscale image by
   combining channels together (e.g., red, green, blue).''' % globals())

        self.rgb_or_channels = cps.Choice(
                "Image type", [CH_RGB, CH_HSV, CH_CHANNELS], doc="""\
This setting provides three options to choose from:

-  *%(CH_RGB)s:* The RGB (red, green, blue) color space is the typical
   model in which color images are stored. Choosing this option will
   split the image into red, green, and blue component images.
-  *%(CH_HSV)s:* The HSV (hue, saturation, value) color space is based
   on color characteristics such as tint, shade, and tone.
   Choosing this option will split the image into the hue,
   saturation, and value component images.
-  *%(CH_CHANNELS)s:* Many images contain color channels other than RGB
   or HSV. For instance, GIF and PNG formats can have an alpha
   channel that encodes transparency. TIF formats can have an arbitrary
   number of channels which represent pixel measurements made by
   different detectors, filters or lighting conditions. This setting
   allows you to handle a more complex model for images that
   have more than three channels.""" % globals())

        # The following settings are used for the combine option
        self.grayscale_name = cps.ImageNameProvider(
                "Name the output image", "OrigGray", doc="""\
*(Used only when combining channels)*

Enter a name for the resulting grayscale image.""")

        self.red_contribution = cps.Float(
                "Relative weight of the red channel",
                1, 0, doc='''\
*(Used only when combining channels)*

Relative weights: If all relative weights are equal, all three colors
contribute equally in the final image. To weight colors relative to each
other, increase or decrease the relative weights.''')

        self.green_contribution = cps.Float(
                "Relative weight of the green channel",
                1, 0, doc='''\
*(Used only when combining channels)*

Relative weights: If all relative weights are equal, all three colors
contribute equally in the final image. To weight colors relative to each
other, increase or decrease the relative weights.''')

        self.blue_contribution = cps.Float(
                "Relative weight of the blue channel",
                1, 0, doc='''\
*(Used only when combining channels)*

Relative weights: If all relative weights are equal, all three colors
contribute equally in the final image. To weight colors relative to each
other, increase or decrease the relative weights.''')

        # The following settings are used for the split RGB option
        self.use_red = cps.Binary('Convert red to gray?', True, doc="""\
*(Used only when splitting RGB images)*

Select *"%(YES)s"* to extract the red channel to grayscale. Otherwise, the
red channel will be ignored.
""" % globals())

        self.red_name = cps.ImageNameProvider('Name the output image', "OrigRed", doc="""\
*(Used only when splitting RGB images)*

Enter a name for the resulting grayscale image coming from the red channel.""")

        self.use_green = cps.Binary('Convert green to gray?', True, doc="""\
*(Used only when splitting RGB images)*

Select *"%(YES)s"* to extract the green channel to grayscale. Otherwise, the
green channel will be ignored.
""" % globals())

        self.green_name = cps.ImageNameProvider('Name the output image', "OrigGreen", doc="""\
*(Used only when splitting RGB images)*

Enter a name for the resulting grayscale image coming from the green channel.""")

        self.use_blue = cps.Binary('Convert blue to gray?', True, doc="""\
*(Used only when splitting RGB images)*

Select *"%(YES)s"* to extract the blue channel to grayscale. Otherwise, the
blue channel will be ignored.
""" % globals())

        self.blue_name = cps.ImageNameProvider('Name the output image', "OrigBlue", doc="""\
*(Used only when splitting RGB images)*

Enter a name for the resulting grayscale image coming from the blue channel.""")

        # The following settings are used for the split HSV ption
        self.use_hue = cps.Binary('Convert hue to gray?', True, doc="""\
*(Used only when splitting HSV images)*

Select *"%(YES)s"* to extract the hue to grayscale. Otherwise, the hue
will be ignored.
""" % globals())

        self.hue_name = cps.ImageNameProvider('Name the output image', "OrigHue", doc="""\
*(Used only when splitting HSV images)*

Enter a name for the resulting grayscale image coming from the hue.""")

        self.use_saturation = cps.Binary('Convert saturation to gray?', True, doc="""\
*(Used only when splitting HSV images)*

Select *"%(YES)s"* to extract the saturation to grayscale. Otherwise, the
saturation will be ignored.
""" % globals())

        self.saturation_name = cps.ImageNameProvider('Name the output image', "OrigSaturation", doc="""\
*(Used only when splitting HSV images)*

Enter a name for the resulting grayscale image coming from the saturation.""")

        self.use_value = cps.Binary('Convert value to gray?', True, doc="""\
*(Used only when splitting HSV images)*

Select *"%(YES)s"* to extract the value to grayscale. Otherwise, the
value will be ignored.
""" % globals())

        self.value_name = cps.ImageNameProvider('Name the output image', "OrigValue", doc="""\
*(Used only when splitting HSV images)*

Enter a name for the resulting grayscale image coming from the value.""")

        # The alternative model:
        self.channels = []
        self.add_channel(False)
        self.channel_button = cps.DoSomething(
                "", "Add another channel", self.add_channel)

        self.channel_count = cps.HiddenCount(self.channels, "Channel count")

    channel_names = (["Red: 1", "Green: 2", "Blue: 3", "Alpha: 4"] +
                     [str(x) for x in range(5, 20)])

    def add_channel(self, can_remove=True):
        '''Add another channel to the channels list'''
        group = cps.SettingsGroup()
        group.can_remove = can_remove
        group.append("channel_choice", cps.Choice(
                "Channel number", self.channel_names,
                self.channel_names[len(self.channels) % len(self.channel_names)], doc="""\
*(Used only when splitting images)*

This setting chooses a channel to be processed. For example, *Red: 1*
is the first
channel in a .TIF or the red channel in a traditional image file.
*Green: 2* and *Blue: 3* are the second and third channels of a TIF or
the green and blue channels in other formats. *Alpha: 4* is the
transparency channel for image formats that support transparency and is
channel # 4 for a .TIF file. **ColorToGray** will fail to process an
image if you select a channel that is not supported by that image, for
example, “5” for a three-channel .PNG file."""))

        group.append("contribution", cps.Float(
                "Relative weight of the channel", 1, 0, doc='''\
*(Used only when combining channels)*

Relative weights: If all relative weights are equal, all three colors
contribute equally in the final image. To weight colors relative to each
other, increase or decrease the relative weights.'''))

        group.append("image_name", cps.ImageNameProvider(
                "Image name", value="Channel%d" % (len(self.channels) + 1), doc="""\
*(Used only when splitting images)*

Select the name of the output grayscale image."""))

        if group.can_remove:
            group.append("remover", cps.RemoveSettingButton(
                    "", "Remove this channel", self.channels, group))
        self.channels.append(group)

    def visible_settings(self):
        """Return either the "combine" or the "split" settings"""
        vv = [self.image_name, self.combine_or_split]
        if self.should_combine():
            vv += [self.grayscale_name, self.rgb_or_channels]
            if self.rgb_or_channels in (CH_RGB, CH_HSV):
                vv.extend([self.red_contribution,
                           self.green_contribution, self.blue_contribution])
            else:
                for channel in self.channels:
                    vv += [channel.channel_choice, channel.contribution]
                    if channel.can_remove:
                        vv += [channel.remover]
                vv += [self.channel_button]
        else:
            vv += [self.rgb_or_channels]
            if self.rgb_or_channels == CH_RGB:
                for v_use, v_name in ((self.use_red, self.red_name),
                                      (self.use_green, self.green_name),
                                      (self.use_blue, self.blue_name)):
                    vv.append(v_use)
                    if v_use.value:
                        vv.append(v_name)
            elif self.rgb_or_channels == CH_HSV:
                for v_use, v_name in ((self.use_hue, self.hue_name),
                                      (self.use_saturation, self.saturation_name),
                                      (self.use_value, self.value_name)):
                    vv.append(v_use)
                    if v_use.value:
                        vv.append(v_name)
            else:
                for channel in self.channels:
                    vv += [channel.channel_choice, channel.image_name]
                    if channel.can_remove:
                        vv += [channel.remover]
                vv += [self.channel_button]
        return vv

    def settings(self):
        """Return all of the settings in a consistent order"""
        return [self.image_name, self.combine_or_split,
                self.rgb_or_channels,
                self.grayscale_name, self.red_contribution,
                self.green_contribution, self.blue_contribution,
                self.use_red, self.red_name,
                self.use_green, self.green_name,
                self.use_blue, self.blue_name,
                self.use_hue, self.hue_name,
                self.use_saturation, self.saturation_name,
                self.use_value, self.value_name,
                self.channel_count
                ] + sum([[channel.channel_choice, channel.contribution,
                          channel.image_name] for channel in self.channels],
                        [])

    def should_combine(self):
        """True if we are supposed to combine RGB to gray"""
        return self.combine_or_split == COMBINE

    def should_split(self):
        """True if we are supposed to split each color into an image"""
        return self.combine_or_split == SPLIT

    def validate_module(self, pipeline):
        """Test to see if the module is in a valid state to run

        Throw a ValidationError exception with an explanation if a module is not valid.
        Make sure that we output at least one image if split
        """
        if self.should_split():
            if (self.rgb_or_channels == CH_RGB) and not any(
                    [self.use_red.value, self.use_blue.value, self.use_green.value]):
                raise cps.ValidationError("You must output at least one of the color images when in split mode",
                                          self.use_red)
            if (self.rgb_or_channels == CH_HSV) and not any(
                    [self.use_hue.value, self.use_saturation.value, self.use_value.value]):
                raise cps.ValidationError("You must output at least one of the color images when in split mode",
                                          self.use_hue)

    def channels_and_contributions(self):
        """Return tuples of channel indexes and their relative contributions

        Used when combining channels to find the channels to combine
        """
        if self.rgb_or_channels in (CH_RGB, CH_HSV):
            return [(i, contribution.value) for i, contribution in enumerate(
                    (self.red_contribution, self.green_contribution,
                     self.blue_contribution))]

        return [(self.channel_names.index(channel.channel_choice),
                 channel.contribution.value) for channel in self.channels]

    @staticmethod
    def get_channel_idx_from_choice(choice):
        '''Convert one of the channel choice strings to a channel index

        choice - one of the strings from channel_choices or similar
                 (string ending in a one-based index)
        returns the zero-based index of the channel.
        '''
        return int(re.search("[0-9]+$", choice).group()) - 1

    def channels_and_image_names(self):
        """Return tuples of channel indexes and the image names for output"""
        if self.rgb_or_channels == CH_RGB:
            rgb = ((self.use_red.value, self.red_name.value, "Red"),
                   (self.use_green.value, self.green_name.value, "Green"),
                   (self.use_blue.value, self.blue_name.value, "Blue"))
            return [(i, name, title) for i, (use_it, name, title)
                    in enumerate(rgb) if use_it]

        if self.rgb_or_channels == CH_HSV:
            hsv = ((self.use_hue.value, self.hue_name.value, "Hue"),
                   (self.use_saturation.value, self.saturation_name.value, "Saturation"),
                   (self.use_value.value, self.value_name.value, "Value"))
            return [(i, name, title) for i, (use_it, name, title)
                    in enumerate(hsv) if use_it]

        result = []
        for channel in self.channels:
            choice = channel.channel_choice.value
            channel_idx = self.get_channel_idx_from_choice(choice)
            result.append((channel_idx, channel.image_name.value,
                           channel.channel_choice.value))
        return result

    def run(self, workspace):
        """Run the module

        pipeline     - instance of CellProfiler.Pipeline for this run
        workspace    - the workspace contains:
            image_set    - the images in the image set being processed
            object_set   - the objects (labeled masks) in this image set
            measurements - the measurements for this run
            frame        - display within this frame (or None to not display)
        """
        image = workspace.image_set.get_image(self.image_name.value,
                                              must_be_color=True)
        if self.should_combine():
            self.run_combine(workspace, image)
        else:
            self.run_split(workspace, image)

    def display(self, workspace, figure):
        if self.should_combine():
            self.display_combine(workspace, figure)
        else:
            self.display_split(workspace, figure)

    def run_combine(self, workspace, image):
        """Combine images to make a grayscale one
        """
        input_image = image.pixel_data
        channels, contributions = zip(*self.channels_and_contributions())
        denominator = sum(contributions)
        channels = np.array(channels, int)
        contributions = np.array(contributions) / denominator

        output_image = np.sum(input_image[:, :, channels] *
                              contributions[np.newaxis, np.newaxis, :], 2)
        image = cpi.Image(output_image, parent_image=image)
        workspace.image_set.add(self.grayscale_name.value, image)

        workspace.display_data.input_image = input_image
        workspace.display_data.output_image = output_image

    def display_combine(self, workspace, figure):
        import matplotlib.cm

        input_image = workspace.display_data.input_image
        output_image = workspace.display_data.output_image
        figure.set_subplots((1, 2))
        figure.subplot_imshow(0, 0, input_image,
                              title="Original image: %s" % self.image_name)
        figure.subplot_imshow(0, 1, output_image,
                              title="Grayscale image: %s" % self.grayscale_name,
                              colormap=matplotlib.cm.Greys_r,
                              sharexy=figure.subplot(0, 0))

    def run_split(self, workspace, image):
        """Split image into individual components
        """
        input_image = image.pixel_data
        disp_collection = []
        if self.rgb_or_channels in (CH_RGB, CH_CHANNELS):
            for index, name, title in self.channels_and_image_names():
                output_image = input_image[:, :, index]
                workspace.image_set.add(name, cpi.Image(output_image, parent_image=image))
                disp_collection.append([output_image, name])
        elif self.rgb_or_channels == CH_HSV:
            output_image = matplotlib.colors.rgb_to_hsv(input_image)
            for index, name, title in self.channels_and_image_names():
                workspace.image_set.add(name, cpi.Image(output_image[:, :, index], parent_image=image))
                disp_collection.append([output_image[:, :, index], name])

        workspace.display_data.input_image = input_image
        workspace.display_data.disp_collection = disp_collection

    def display_split(self, workspace, figure):
        import matplotlib.cm

        input_image = workspace.display_data.input_image
        disp_collection = workspace.display_data.disp_collection
        ndisp = len(disp_collection)
        ncols = int(np.ceil((ndisp+1)**0.5))
        subplots = (ncols, (ndisp/ncols)+1)
        figure.set_subplots(subplots)
        figure.subplot_imshow(0, 0, input_image,
                              title="Original image")

        for eachplot in range(ndisp):
             placenum = eachplot +1
             figure.subplot_imshow(placenum%ncols, placenum/ncols, disp_collection[eachplot][0],
                                   title="%s" % (disp_collection[eachplot][1]),
                                   colormap=matplotlib.cm.Greys_r,
                                   sharexy=figure.subplot(0, 0))

    def prepare_settings(self, setting_values):
        '''Prepare the module to receive the settings

        setting_values - one string per setting to be initialized

        Adjust the number of channels to match the number indicated in
        the settings.
        '''
        del self.channels[1:]
        nchannels = int(setting_values[SLOT_CHANNEL_COUNT])
        while len(self.channels) < nchannels:
            self.add_channel()

    def upgrade_settings(self,
                         setting_values,
                         variable_revision_number,
                         module_name,
                         from_matlab):
        if from_matlab and variable_revision_number == 1:
            new_setting_values = [setting_values[0],  # image name
                                  setting_values[1],  # combine or split
                                  # blank slot for text: "Combine options"
                                  setting_values[3],  # grayscale name
                                  setting_values[4],  # red contribution
                                  setting_values[5],  # green contribution
                                  setting_values[6]  # blue contribution
                                  # blank slot for text: "Split options"
                                  ]
            for i in range(3):
                vv = setting_values[i + 8]
                use_it = ((vv == cps.DO_NOT_USE or vv == "N") and cps.NO) or cps.YES
                new_setting_values.append(use_it)
                new_setting_values.append(vv)
            setting_values = new_setting_values
            module_name = self.module_class()
            variable_revision_number = 1
            from_matlab = False

        if not from_matlab and variable_revision_number == 1:
            #
            # Added rgb_or_channels at position # 2, added channel count
            # at end.
            #
            setting_values = (
                setting_values[:2] + [CH_RGB] + setting_values[2:] +
                ["1", "Red: 1", "1", "Channel1"])
            variable_revision_number = 2

        if not from_matlab and variable_revision_number == 2:
            #
            # Added HSV settings
            #
            setting_values = (setting_values[:13] +
                              [cps.YES, "OrigHue", cps.YES, "OrigSaturation", cps.YES, "OrigValue"] +
                              setting_values[13:])
            variable_revision_number = 3

        #
        # Standardize the channel choices
        #
        setting_values = list(setting_values)
        nchannels = int(setting_values[SLOT_CHANNEL_COUNT])
        for i in range(nchannels):
            idx = SLOT_FIXED_COUNT + SLOT_CHANNEL_CHOICE + i * SLOTS_PER_CHANNEL
            channel_idx = self.get_channel_idx_from_choice(setting_values[idx])
            setting_values[idx] = self.channel_names[channel_idx]

        return setting_values, variable_revision_number, from_matlab
