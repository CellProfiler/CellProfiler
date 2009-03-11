"""graytocolor.py - the GrayToColor module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import numpy as np

import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps

OFF_RED_IMAGE_NAME = 0
OFF_GREEN_IMAGE_NAME = 1
OFF_BLUE_IMAGE_NAME = 2
OFF_RGB_IMAGE_NAME = 3
OFF_RED_ADJUSTMENT_FACTOR = 4
OFF_GREEN_ADJUSTMENT_FACTOR = 5
OFF_BLUE_ADJUSTMENT_FACTOR = 6

class GrayToColor(cpm.CPModule):
    """SHORT DESCRIPTION:
Takes 1 to 3 images and assigns them to colors in a final red, green,
blue (RGB) image. Each color's brightness can be adjusted independently.
*************************************************************************

This module takes up to three grayscale images as inputs, and produces a
new color (RGB) image which results from assigning each of the input
images the colors red, green, and blue in the color image, respectively.
In addition, each color's intensity can be adjusted independently by
using adjustment factors (see below).

Settings:

Choose the input images: You must select at least one image which you
would like to use to create the color image. Also, all images must be the
same size, since they will combined pixel by pixel.

Adjustment factors: Leaving the adjustment factors set to 1 will balance
all three colors equally in the final image, and they will use the same
range of intensities as each individual incoming image. Using factors
less than 1 will decrease the intensity of that color in the final image,
and values greater than 1 will increase it. Setting the adjustment factor
to zero will cause that color to be entirely blank.

See also ColorToGray.
    """
    variable_revision_number = 1
    category = "Image Processing"
    def create_settings(self):
        self.module_name = 'GrayToColor'
        self.red_image_name = cps.ImageNameSubscriber("What did you call the image to be colored red?",
                                                      can_be_blank = True,
                                                      blank_text = "Leave this black")
        self.green_image_name = cps.ImageNameSubscriber("What did you call the image to be colored green?",
                                                        can_be_blank = True,
                                                        blank_text = "Leave this black")
        self.blue_image_name = cps.ImageNameSubscriber("What did you call the image to be colored blue?",
                                                       can_be_blank = True,
                                                       blank_text = "Leave this black")
        self.rgb_image_name = cps.ImageNameProvider("What do you want to call the resulting image?",
                                                    "ColorImage")
        self.red_adjustment_factor = cps.Float("Enter the adjustment factor for the red image:",
                                               value=1,
                                               minval=0)
        self.green_adjustment_factor = cps.Float("Enter the adjustment factor for the green image:",
                                                 value=1,
                                                 minval=0)
        self.blue_adjustment_factor = cps.Float("Enter the adjustment factor for the blue image:",
                                                value=1,
                                                minval=0)
    
    def settings(self):
        return [self.red_image_name, self.green_image_name, self.blue_image_name,
                self.rgb_image_name, self.red_adjustment_factor, 
                self.green_adjustment_factor, self.blue_adjustment_factor]
    
    def backwards_compatibilize(self,setting_values,variable_revision_number,
                                module_name,from_matlab):
        if from_matlab and variable_revision_number==1:
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
        return setting_values, variable_revision_number, from_matlab
    
    def visible_settings(self):
        result = [self.red_image_name, self.green_image_name, 
                  self.blue_image_name, self.rgb_image_name]
        if not self.red_image_name.is_blank:
            result.append(self.red_adjustment_factor)
        if not self.green_image_name.is_blank:
            result.append(self.green_adjustment_factor)
        if not self.blue_image_name.is_blank:
            result.append(self.blue_adjustment_factor)
        return result
    
    def validate_module(self,pipeline):
        """Make sure that the module's settings are consistent
        
        We need at least one image name to be filled in
        """
        if (self.red_image_name.is_blank and
            self.green_image_name.is_blank and
            self.blue_image_name.is_blank):
            raise cps.ValidationError("At least one of the images must not be blank",\
                                      self.red_image_name)
    def run(self,workspace):
        assert not (self.red_image_name.is_blank and
                    self.green_image_name.is_blank and
                    self.blue_image_name.is_blank),\
                    "At least one of the images must not be blank"
        parent_image = None
        parent_image_name = None
        imgset = workspace.image_set
        if not self.red_image_name.is_blank:
            red_image = imgset.get_image(self.red_image_name.value,
                                         must_be_grayscale=True)
            red_pixel_data = (red_image.pixel_data *
                              self.red_adjustment_factor.value)
            parent_image = red_image
            parent_image_name = self.red_image_name.value
        if not self.green_image_name.is_blank:
            green_image = imgset.get_image(self.green_image_name.value,
                                           must_be_grayscale=True)
            green_pixel_data = (green_image.pixel_data *
                                self.green_adjustment_factor.value) 
            if parent_image != None:
                if (parent_image.pixel_data.shape != green_pixel_data.shape):
                    raise ValueError("The %s image and %s image have different sizes (%s vs %s)"%
                                     (parent_image_name, 
                                      self.green_image_name.value,
                                      parent_image.pixel_data.shape,
                                      green_pixel_data.shape))
            else:
                parent_image = green_image
                parent_image_name = self.green_image_name.value
        if not self.blue_image_name.is_blank: 
            blue_image = imgset.get_image(self.blue_image_name.value,
                                          must_be_grayscale=True)
            blue_pixel_data = (blue_image.pixel_data *
                               self.blue_adjustment_factor.value)
            if parent_image != None:
                if (parent_image.pixel_data.shape != blue_pixel_data.shape):
                    raise ValueError("The %s image and %s image have different sizes (%s vs %s)"%
                                     (parent_image_name, 
                                      self.blue_image_name.value,
                                      parent_image.pixel_data.shape,
                                      blue_pixel_data.shape))
            else:
                parent_image = blue_image
                parent_image_name = self.blue_image_name.value
        if parent_image != None:
            if self.red_image_name.is_blank:
                red_pixel_data = np.zeros(parent_image.pixel_data.shape)
            if self.green_image_name.is_blank:
                green_pixel_data = np.zeros(parent_image.pixel_data.shape)
            if self.blue_image_name.is_blank:
                blue_pixel_data = np.zeros(parent_image.pixel_data.shape)
        
        rgb_pixel_data = np.dstack((red_pixel_data,
                                    green_pixel_data,
                                    blue_pixel_data))
        rgb_pixel_data.shape = (red_pixel_data.shape[0],
                                red_pixel_data.shape[1],3)
        ###############
        # Draw images #
        ###############
        if workspace.frame != None:
            title = "Gray to color #%d"%(self.module_num)
            my_frame = workspace.create_or_find_figure(title,(2,2))
            if self.red_image_name.is_blank:
                my_frame.subplot(0,0).visible = False
            else:
                my_frame.subplot(0,0).visible = True
                my_frame.subplot_imshow_grayscale(0,0,red_pixel_data,
                                                  title=self.red_image_name.value)
            if self.green_image_name.is_blank:
                my_frame.subplot(1,0).visible = False
            else:
                my_frame.subplot(1,0).visible = True
                my_frame.subplot_imshow_grayscale(1,0,green_pixel_data,
                                                  title=self.green_image_name.value)
            if self.blue_image_name.is_blank:
                my_frame.subplot(0,1).visible = False
            else:
                my_frame.subplot(0,1).visible = True
                my_frame.subplot_imshow_grayscale(0,1,blue_pixel_data,
                                                  title=self.blue_image_name.value)
            my_frame.subplot_imshow(1,1,rgb_pixel_data,
                                    title=self.rgb_image_name.value)
        ##############
        # Save image #
        ##############
        rgb_image = cpi.Image(rgb_pixel_data, parent_image = parent_image)
        imgset.add(self.rgb_image_name.value, rgb_image)
