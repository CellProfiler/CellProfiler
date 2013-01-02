'''
<b> Color to Gray</b> converts an image with three color channels to one <i>or</i> three grayscale images
<hr>

This module converts RGB (Red, Green, Blue) color images to grayscale. All channels
can be merged into one grayscale image (<i>Combine</i>), or each channel 
can be extracted into a separate grayscale image (<i>Split</i>). If you use <i>Combine</i>, 
the relative weights will adjust the contribution of the colors relative to each other.<br>
<br>
<i>Note:</i>All <b>Identify</b> modules require grayscale images.
<p>See also <b>GrayToColor</b>.
'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2013 Broad Institute
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org

import numpy as np
import re

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.cpimage  as cpi

COMBINE = "Combine"
SPLIT = "Split"

CH_RGB = "RGB"
CH_CHANNELS = "Channels"

SLOT_CHANNEL_COUNT = 13
SLOT_FIXED_COUNT = 14
SLOTS_PER_CHANNEL = 3
SLOT_CHANNEL_CHOICE = 0

class ColorToGray(cpm.CPModule):

    module_name = "ColorToGray"
    variable_revision_number = 2
    category = "Image Processing"
    
    def create_settings(self):
        self.image_name = cps.ImageNameSubscriber("Select the input image",
                                             "None")
        self.combine_or_split = cps.Choice("Conversion method",
                                           [COMBINE,SPLIT],doc='''
                                           How do you want to convert the color image? 
                                           <ul>
                                           <li><i>Split:</i> Splits the three channels
                                           (red, green, blue) of a color image into three separate grayscale images. </li>
                                           <li><i>Combine:</i> Converts a color image to a grayscale 
                                           image by combining the three channels (red, green, blue) together.</li>
                                           </ul>''')
        self.rgb_or_channels = cps.Choice(
            "Image type:", [CH_RGB, CH_CHANNELS],
            doc = """Many images contain color channels other than red, green
            and blue. For instance, GIF and PNG formats can have an alpha
            channel that encodes transparency. TIF formats can have an arbitrary
            number of channels which represent pixel measurements made by
            different detectors, filters or lighting conditions. This setting
            chooses between a simple RGB model of the image (the typical way
            images are stored) and a more complex model involving more
            than three chnnels. Choose <i>%(CH_RGB)s</i> for the RGB model or
            <i>%(CH_CHANNELS)s</i> for the channels model.""" % globals())
        
        # The following settings are used for the combine option
        self.grayscale_name = cps.ImageNameProvider("Name the output image",
                                               "OrigGray")
        self.red_contribution = cps.Float("Relative weight of the red channel",
                                          1,0,doc='''<i>(Used only when combining channels)</i><br>
                                          Relative weights: If all relative weights are equal, all three 
                                          colors contribute equally in the final image. To weight colors relative 
                                          to each other, increase or decrease the relative weights.''')
        
        self.green_contribution = cps.Float("Relative weight of the green channel",
                                            1,0,doc='''<i>(Used only when combining channels)</i><br>
                                            Relative weights: If all relative weights are equal, all three 
                                            colors contribute equally in the final image. To weight colors relative 
                                            to each other, increase or decrease the relative weights.''')
        
        self.blue_contribution = cps.Float("Relative weight of the blue channel",
                                           1,0,doc='''<i>(Used only when combining channels)</i><br>
                                           Relative weights: If all relative weights are equal, all three 
                                           colors contribute equally in the final image. To weight colors relative 
                                           to each other, increase or decrease the relative weights.''')
        
        # The following settings are used for the split option
        self.use_red = cps.Binary('Convert red to gray?',True)
        self.red_name = cps.ImageNameProvider('Name the output image',
                                         "OrigRed")
        
        self.use_green = cps.Binary('Convert green to gray?',True)
        self.green_name = cps.ImageNameProvider('Name the output image',
                                         "OrigGreen")
        
        self.use_blue = cps.Binary('Convert blue to gray?',True)
        self.blue_name = cps.ImageNameProvider('Name the output image',
                                         "OrigBlue")
        
        # The alternative model:
        self.channels = []
        self.add_channel(False)
        self.channel_button = cps.DoSomething(
            "Add another channel", "Add", self.add_channel)
        self.channel_count = cps.HiddenCount(self.channels, "Channel count")
        
    channel_names = (["Red: 1", "Green: 2", "Blue: 3", "Alpha: 4"] + 
                     [str(x) for x in range(5,20)])
    
    def add_channel(self, can_remove = True):
        '''Add another channel to the channels list'''
        group = cps.SettingsGroup()
        group.can_remove = can_remove
        group.append("channel_choice", cps.Choice(
            "Channel number:", self.channel_names,
            self.channel_names[len(self.channels) % len(self.channel_names)],
            doc = """This setting chooses a channel to be processed.
            <i>Red: 1</i> is the first channel in a .TIF or the red channel
            in a traditional image file. <i>Green: 2</i> and <i>Blue: 3</i>
            are the second and third channels of a TIF or the green and blue
            channels in other formats. <i>Alpha: 4</i> is the transparency
            channel for image formats that support transparency and is
            channel # 4 for a .TIF file.
            
            <b>ColorToGray</b> will fail to process an image if you select
            a channel that is not supported by that image, for example, "5"
            for a .PNG file"""))
        group.append("contribution", cps.Float(
            "Relative weight of the channel",
            1,0,doc='''<i>(Used only when combining channels)</i><br>
            Relative weights: If all relative weights are equal, all three 
            colors contribute equally in the final image. To weight colors relative 
            to each other, increase or decrease the relative weights.'''))
        group.append("image_name", cps.ImageNameProvider(
            "Image name:", value="Channel%d" % (len(self.channels)+1),
            doc = """This is the name of the grayscale image that holds
            the image data from the chosen channel"""))
        if group.can_remove:
            group.append("remover", cps.RemoveSettingButton(
                "Remove this channel", "Remove", self.channels, group))
        self.channels.append(group)


    def visible_settings(self):
        """Return either the "combine" or the "split" settings"""
        vv = [self.image_name, self.combine_or_split]
        if self.should_combine():
            vv += [self.grayscale_name, self.rgb_or_channels]
            if self.rgb_or_channels == CH_RGB:
                vv.extend([self.red_contribution,
                           self.green_contribution, self.blue_contribution])
            else:
                for channel in self.channels:
                    vv += [channel.channel_choice, channel.contribution]
                    if channel.can_remove:
                        vv += [channel.remover]
                vv += [self.channel_button]
        else:
            vv += [ self.rgb_or_channels ]
            if self.rgb_or_channels == CH_RGB:
                for v_use,v_name in ((self.use_red  ,self.red_name),
                                     (self.use_green,self.green_name),
                                     (self.use_blue ,self.blue_name)):
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
                self.channel_count
                ] + sum([ [channel.channel_choice, channel.contribution,
                           channel.image_name] for channel in self.channels],
                        [])
    
    def should_combine(self):
        """True if we are supposed to combine RGB to gray"""
        return self.combine_or_split == COMBINE
    
    def should_split(self):
        """True if we are supposed to split each color into an image"""
        return self.combine_or_split == SPLIT
    
    def validate_module(self,pipeline):
        """Test to see if the module is in a valid state to run
        
        Throw a ValidationError exception with an explanation if a module is not valid.
        Make sure that we output at least one image if split
        """
        if (self.should_split() and (self.rgb_or_channels == CH_RGB) and
            not any([self.use_red.value, self.use_blue.value, self.use_green.value])):
            raise cps.ValidationError("You must output at least one of the color images when in split mode",
                                      self.use_red)
    
    def channels_and_contributions(self):
        """Return tuples of channel indexes and their relative contributions
        
        Used when combining channels to find the channels to combine
        """
        if self.rgb_or_channels == CH_RGB:
            return [ (i, contribution.value) for i,contribution in enumerate(
                (self.red_contribution, self.green_contribution, 
                 self.blue_contribution))]
        
        return [ (self.channel_names.index(channel.channel_choice),
                  channel.contribution.value) for channel in self.channels ]
    
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
            return [ (i, name, title) for i, (use_it, name, title) 
                     in enumerate(rgb) if use_it ]
        
        result = []
        for channel in self.channels:
            choice = channel.channel_choice.value
            channel_idx = self.get_channel_idx_from_choice(choice)
            result.append((channel_idx, channel.image_name.value,
                           channel.channel_choice.value))
        return result
        
    def run(self,workspace):
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
        input_image  = image.pixel_data
        channels, contributions = zip(*self.channels_and_contributions())
        denominator = sum(contributions)
        channels = np.array(channels, int)
        contributions = np.array(contributions)  / denominator

        output_image = np.sum(input_image[:, :, channels] * 
                              contributions[np.newaxis, np.newaxis, :], 2)
        image = cpi.Image(output_image,parent_image=image)
        workspace.image_set.add(self.grayscale_name.value,image)

        workspace.display_data.input_image = input_image
        workspace.display_data.output_image = output_image
       
    
    def display_combine(self, workspace, figure):
        import matplotlib.cm

        input_image = workspace.display_data.input_image
        output_image = workspace.display_data.output_image
        figure.set_subplots((1, 2))
        figure.subplot_imshow(0, 0, input_image, 
                              title = "Original image: %s"%(self.image_name))
        figure.subplot_imshow(0, 1, output_image,
                              title = "Grayscale image: %s"%(self.grayscale_name),
                              colormap = matplotlib.cm.Greys_r, 
                              sharexy = figure.subplot(0,0))
        
    def run_split(self, workspace, image):
        """Split image into individual components
        """
        input_image  = image.pixel_data
        disp_collection = []
        for index, name, title in self.channels_and_image_names():
            output_image = input_image[:,:,index]
            image = cpi.Image(output_image,parent_image=image)
            workspace.image_set.add(name, image)
            disp_collection.append([output_image, title])

        workspace.display_data.input_image = input_image
        workspace.display_data.disp_collection = disp_collection
    
    def display_split(self, workspace, figure):
        import matplotlib.cm

        input_image = workspace.display_data.input_image
        disp_collection = workspace.display_data.disp_collection
        ndisp = len(disp_collection)
        if ndisp == 1:
            subplots = (1,2)
        else:
            subplots = (2,2)
        figure.set_subplots(subplots)
        figure.subplot_imshow(0, 0, input_image,
                              title = "Original image")
        
        if ndisp == 1:
            layout = [(0,1)]
        elif ndisp == 2:
            layout = [ (1,0),(0,1)]
        else:
            layout = [(1,0),(0,1),(1,1)]
        for xy, disp in zip(layout, disp_collection):
            figure.subplot_imshow(xy[0], xy[1], disp[0],
                                  title = "%s image"%(disp[1]),
                                  colormap = matplotlib.cm.Greys_r, 
                                  sharexy = figure.subplot(0,0))
        
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
            new_setting_values = [ setting_values[0],  # image name
                                    setting_values[1],  # combine or split
                                                         # blank slot for text: "Combine options"
                                    setting_values[3],  # grayscale name
                                    setting_values[4],  # red contribution
                                    setting_values[5],  # green contribution
                                    setting_values[6]   # blue contribution
                                                         # blank slot for text: "Split options"
                                    ]
            for i in range(3):
                vv = setting_values[i+8]
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
                setting_values[:2] + [ CH_RGB ] + setting_values[2:] +
                [ "1", "Red: 1", "1", "Channel1"])
            variable_revision_number = 2
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
        
