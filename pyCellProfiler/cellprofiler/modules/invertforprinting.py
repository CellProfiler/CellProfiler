'''InvertForPrinting - Invert fluorescent images into brightfield-looking images for printing.
<hr>
This module turns a single or multi-channel immunofluorescent-stained image
into an image that resembles a brightfield image stained with similarly-
colored stains, which generally prints better.
    
You have the option of combining up to three grayscale images (representing
the red, green and blue channels of a color image) or of operating on
a single color image. The module can produce either three grayscale
images or one color image on output.

If you want to invert the grayscale intensities of an image, use <b>ImageMath</b>
'''
#CellProfiler is distributed under the GNU General Public License.
#See the accompanying file LICENSE for details.
#
#Developed by the Broad Institute
#Copyright 2003-2009
#
#Please see the AUTHORS file for credits.
#
#Website: http://www.cellprofiler.org
__version__="$Revision$"

import numpy as np

import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps

CC_GRAYSCALE = "Grayscale"
CC_COLOR = "Color"
CC_ALL = [CC_COLOR, CC_GRAYSCALE]
class InvertForPrinting(cpm.CPModule):
   
    category = 'Image Processing'
    variable_revision_number = 1
    
    def create_settings(self):
        # Input settings
        self.module_name = "InvertForPrinting"
        self.input_color_choice = cps.Choice(
            "Do you want to combine grayscale images or load a single color image?",
            CC_ALL)
        self.wants_red_input = cps.Binary(
            "Do you want to load an image for the red channel?",
            True)
        self.red_input_image = cps.ImageNameSubscriber(
            "What did you call the red image?",
            "None")
        self.wants_green_input = cps.Binary(
            "Do you want to load an image for the green channel?",
            True)
        self.green_input_image = cps.ImageNameSubscriber(
            "What did you call the green image?", "None")
        self.wants_blue_input = cps.Binary(
            "Do you want to load an image for the blue channel?", True)
        self.blue_input_image = cps.ImageNameSubscriber(
            "What did you call the blue image?", "None")
        self.color_input_image = cps.ImageNameSubscriber(
            "What did you call the color image?", "None")
        
        # output settings
        self.output_color_choice = cps.Choice(
            "Do you want to produce several grayscale images or one color image?",
            CC_ALL)
        self.wants_red_output = cps.Binary(
            "Do you want to produce an image for the red channel?",
            True)
        self.red_output_image = cps.ImageNameProvider(
            "What do you want to call the red image?",
            "InvertedRed")
        self.wants_green_output = cps.Binary(
            "Do you want to produce an image for the green channel?",
            True)
        self.green_output_image = cps.ImageNameProvider(
            "What do you want to call the green image?", "InvertedGreen")
        self.wants_blue_output = cps.Binary(
            "Do you want to produce an image for the blue channel?", True)
        self.blue_output_image = cps.ImageNameProvider(
            "What do you want to call the blue image?", "InvertedBlue")
        self.color_output_image = cps.ImageNameProvider(
            "What do you want to call the inverted color image?",
            "InvertedColor")
        
    def settings(self):
        '''Return the settings as saved in the pipeline'''
        return [self.input_color_choice, 
                self.wants_red_input, self.red_input_image,
                self.wants_green_input, self.green_input_image,
                self.wants_blue_input, self.blue_input_image,
                self.color_input_image,
                self.output_color_choice,
                self.wants_red_output, self.red_output_image,
                self.wants_green_output, self.green_output_image,
                self.wants_blue_output, self.blue_output_image,
                self.color_output_image]
    
    def backwards_compatibilize(self, setting_values, variable_revision_number,
                                module_name, from_matlab):
        if from_matlab and variable_revision_number == 1:
            setting_values = [
                CC_GRAYSCALE,                # input_color_choice
                setting_values[0] != 'None', # wants_red_input
                setting_values[0],           # red_input_image
                setting_values[1] != 'None',
                setting_values[1],
                setting_values[2] != 'None',
                setting_values[2],
                'None',                      # color
                CC_GRAYSCALE,                # output_color_choice
                setting_values[3] != 'None',
                setting_values[3],
                setting_values[4] != 'None',
                setting_values[4],
                setting_values[5] != 'None',
                setting_values[5],
                'InvertedColor']
            from_matlab = False
            variable_revision_number = 1
            
        return setting_values, variable_revision_number, from_matlab
        
    def visible_settings(self):
        '''Return the settings as displayed in the UI'''
        result = [self.input_color_choice]
        if self.input_color_choice == CC_GRAYSCALE:
            for wants_input, input_image in \
                ((self.wants_red_input, self.red_input_image),
                 (self.wants_green_input, self.green_input_image),
                 (self.wants_blue_input, self.blue_input_image)):
                result += [wants_input]
                if wants_input.value:
                    result += [input_image]
        else:
            result += [self.color_input_image]
        result += [self.output_color_choice]
        if self.output_color_choice == CC_GRAYSCALE:
            for wants_output, output_image in \
                ((self.wants_red_output, self.red_output_image),
                 (self.wants_green_output, self.green_output_image),
                 (self.wants_blue_output, self.blue_output_image)):
                result += [wants_output]
                if wants_output.value:
                    result += [output_image]
        else:
            result += [self.color_output_image]
        return result
    
    def validate_module(self, pipeline):
        '''Make sure the user has at least one of the grayscale boxes checked'''
        if (self.input_color_choice == CC_GRAYSCALE and
            (not self.wants_red_input.value) and
            (not self.wants_green_input.value) and
            (not self.wants_blue_input.value)):
            raise cps.ValidationError("You must supply at least one grayscale input",
                                      self.wants_red_input)
        
    def run(self, workspace):
        image_set = workspace.image_set
        assert isinstance(image_set, cpi.ImageSet)
        shape = None
        if self.input_color_choice == CC_GRAYSCALE:
            if self.wants_red_input.value:
                red_image = image_set.get_image(
                    self.red_input_image.value,
                    must_be_grayscale=True).pixel_data
                shape = red_image.shape
            else:
                red_image = 0
            if self.wants_green_input.value:
                green_image = image_set.get_image(
                    self.green_input_image.value,
                    must_be_grayscale=True).pixel_data
                shape = green_image.shape
            else:
                green_image = 0
            if self.wants_blue_input.value:
                blue_image = image_set.get_image(
                    self.blue_input_image.value,
                    must_be_grayscale=True).pixel_data
                shape = blue_image.shape
            else:
                blue_image = 0
            color_image = np.zeros((shape[0],shape[1],3))
            color_image[:,:,0] = red_image
            color_image[:,:,1] = green_image
            color_image[:,:,2] = blue_image
            red_image = color_image[:,:,0]
            green_image = color_image[:,:,1]
            blue_image = color_image[:,:,2]
        elif self.input_color_choice == CC_COLOR:
            color_image = image_set.get_image(
                self.color_input_image.value,
                must_be_color=True).pixel_data
            red_image = color_image[:,:,0]
            green_image = color_image[:,:,1]
            blue_image = color_image[:,:,2]
        else:
            raise ValueError("Unimplemented color choice: %s" %
                             self.input_color_choice.value)
        inverted_red = (1 - green_image) * (1 - blue_image)
        inverted_green = (1 - red_image) * (1 - blue_image)
        inverted_blue = (1 - red_image) * (1 - green_image)
        inverted_color = np.dstack((inverted_red, inverted_green, inverted_blue))
        if self.output_color_choice == CC_GRAYSCALE:
            for wants_output, output_image_name, output_image in \
                ((self.wants_red_output, self.red_output_image, inverted_red),
                 (self.wants_green_output, self.green_output_image, inverted_green),
                 (self.wants_blue_output, self.blue_output_image, inverted_blue)):
                if wants_output.value:
                    image = cpi.Image(output_image)
                    image_set.add(output_image_name.value, image)
        elif self.output_color_choice == CC_COLOR:
            image = cpi.Image(inverted_color)
            image_set.add(self.color_output_image.value, image)
        else:
            raise ValueError("Unimplemented color choice: %s" %
                             self.output_color_choice.value)
        #
        # display
        #
        if workspace.frame is not None:
            figure = workspace.create_or_find_figure(subplots=(2,1))
            figure.subplot_imshow_color(0,0,color_image, "Original image")
            figure.subplot_imshow_color(1,0,inverted_color, "Color-inverted image")
