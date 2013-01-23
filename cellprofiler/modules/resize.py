"""<b>Resize</b> resizes images (changes their resolution)
<hr>

Images are resized (made smaller or larger) based on user input. You
can resize an image by applying a resizing factor or by specifying the 
desired dimensions, in pixels. You can also select which interpolation
method to use. 
"""
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2013 Broad Institute
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org
__version__="$Revision$"

import logging
import numpy as np
import traceback
from scipy.ndimage import affine_transform
logger = logging.getLogger(__name__)

import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.settings as cps

R_BY_FACTOR = "Resize by a fraction or multiple of the original size"
R_TO_SIZE = "Resize by specifying desired final dimensions"
R_ALL = [R_BY_FACTOR, R_TO_SIZE]

C_IMAGE = "Image"
C_MANUAL = "Manual"
C_ALL = [C_MANUAL, C_IMAGE]

I_NEAREST_NEIGHBOR = 'Nearest Neighbor'
I_BILINEAR = 'Bilinear'
I_BICUBIC = 'Bicubic'

I_ALL = [I_NEAREST_NEIGHBOR, I_BILINEAR, I_BICUBIC]
'''The index of the additional image count setting'''
S_ADDITIONAL_IMAGE_COUNT = 9

class Resize(cpm.CPModule):

    category = "Image Processing"
    variable_revision_number = 4
    module_name = "Resize"
    
    def create_settings(self):
        self.image_name = cps.ImageNameSubscriber("Select the input image",
                                                  "None", doc = '''What did you call the image to be resized?''')

        self.resized_image_name = cps.ImageNameProvider("Name the output image",
                                                        "ResizedBlue", doc = '''What do you want to call the resized image?''')

        self.size_method = cps.Choice("Select resizing method",
                                      R_ALL, doc = """How do you want to resize the image? 
                                      <ul><li><i>Resize by a fraction or multiple of the original size:</i> 
                                      Enter a single value which specifies the scaling. </li>
                                      <li><i>Resize by specifying desired final dimensions:</i></li> 
                                      Enter the new height and width of the resized image.</ul>""")

        self.resizing_factor = cps.Float("Resizing factor",
                                         .25, minval=0, doc = '''
                                         <i>(Used only if resizing by a fraction or multiple of the original size)</i><br>
                                         Numbers less than one (that is, fractions) will shrink the image; 
                                         numbers greater than one (that is, multiples) will enlarge the image.''')

        self.use_manual_or_image = cps.Choice("How do you want to specify the dimensions?",C_ALL, doc = """
                                        <i>(Used only if resizing by specifying the dimensions)</i><br>
                                        You have two options on how to resize your image:
                                        <ul>
                                        <li><i>Manual:</i> Specify the height and width of the output image.</li>
                                        <li><i>Image:</i> Specify an image and the input image will be resized
                                        to the same dimensions.</li>
                                        </ul>""")
        
        self.specific_width = cps.Integer("Width of the final image, in pixels", 100, minval=1, doc = '''
                                         <i>(Used only if resizing by specifying desired final dimensions)</i><br>
                                         Enter the desired width of the final image.''')

        self.specific_height = cps.Integer("Height of the final image, in pixels", 100, minval=1, doc = '''
                                         <i>(Used only if resizing by specifying desired final dimensions)</i><br>
                                         Enter the desired height of the final image.''')
        
        self.specific_image = cps.ImageNameSubscriber("Select the image with the desired dimensions", "None", doc = """"
                                        <i>(Used only if resizing by specifying desired final dimensions using an image)</i><br>
                                        The input image will be resized to the dimensions of the specified image.""")

        self.interpolation = cps.Choice("Interpolation method",
                                        I_ALL, doc = '''<ul><li><i>Nearest Neighbor:</i> Each output pixel is given the intensity of the nearest
                                        corresponding pixel in the input image.</li>
                                        <li><i>Bilinear:</i> Each output pixel is given the intensity of the weighted average
                                        of the 2x2 neighborhood at the corresponding position in the input image.</li>
                                        <li><i>Bicubic:</i> Each output pixel is given the intensity of the weighted average
                                        of the 4x4 neighborhood at the corresponding position in the input image.</li>
                                        </ul>''')
        
        self.separator = cps.Divider(line=False)
        
        self.additional_images = []
        
        self.additional_image_count = cps.HiddenCount(
            self.additional_images, "Additional image count")
        
        self.add_button = cps.DoSomething("", "Add another image",
                                          self.add_image)
        
    def add_image(self, can_remove = True):
        '''Add an image + associated questions and buttons'''
        group = cps.SettingsGroup()
        if can_remove:
            group.append("divider", cps.Divider(line=False))
        
        group.append("input_image_name", 
                     cps.ImageNameSubscriber("Select the additional image?",
                                            "None",doc="""
                                            What is the name of the additional image to resize? This image will be
                                            resized with the same settings as the first image."""))
        group.append("output_image_name",
                     cps.ImageNameProvider("Name the output image",
                                            "ResizedBlue",doc="""
                                            What is the name of the additional resized image?"""))
        if can_remove:
            group.append("remover", cps.RemoveSettingButton("", "Remove above image", self.additional_images, group))
        self.additional_images.append(group)

    def settings(self):
        result = [self.image_name, self.resized_image_name, self.size_method,
                self.resizing_factor, self.specific_width, 
                self.specific_height, self.interpolation,
                self.use_manual_or_image, self.specific_image,
                self.additional_image_count]
        
        for additional in self.additional_images:
            result += [additional.input_image_name, additional.output_image_name]
            
        return result

    def visible_settings(self):
        result = [self.image_name, self.resized_image_name, self.size_method]
        if self.size_method == R_BY_FACTOR:
            result.append(self.resizing_factor)
        elif self.size_method == R_TO_SIZE:
            result += [self.use_manual_or_image]
            if self.use_manual_or_image == C_IMAGE:
                result += [self.specific_image]
            elif self.use_manual_or_image == C_MANUAL:
                result += [self.specific_width, self.specific_height]
        else:
            raise ValueError("Unsupported size method: %s" % 
                             self.size_method.value)
        result += [self.interpolation]
        
        for additional in self.additional_images:
            result += additional.visible_settings()
        result += [self.add_button]
        return result
    
    def prepare_settings(self, setting_values):
        '''Create the correct number of additional images'''
        try:
            additional_image_setting_count = \
                int(setting_values[S_ADDITIONAL_IMAGE_COUNT])
            if len(self.additional_images) > additional_image_setting_count:
                del self.additional_images[additional_image_setting_count:]
            else:
                for i in range(len(self.additional_images),
                               additional_image_setting_count):
                    self.add_image()
        except ValueError:
            logger.warning(
                'Additional image setting count was "%s" '
                'which is not an integer.',
                setting_values[S_ADDITIONAL_IMAGE_COUNT], exc_info=True)
            pass

    def run(self, workspace):
        self.apply_resize(workspace, self.image_name.value, self.resized_image_name.value)
        statistics = [[0,
                       self.image_name.value, 
                       self.resized_image_name.value]]
        
        for additional in self.additional_images:
            self.apply_resize(workspace, additional.input_image_name.value, additional.output_image_name.value)
            statistics += [[len(statistics),
                            additional.input_image_name.value,
                            additional.output_image_name.value]]
        # Save data for display
        workspace.display_data.statistics = statistics
    
    def apply_resize(self, workspace, input_image_name, output_image_name):
        image = workspace.image_set.get_image(input_image_name)
        image_pixels = image.pixel_data
        if self.size_method == R_BY_FACTOR:
            factor = self.resizing_factor.value
            shape = (np.array(image_pixels.shape[:2])*factor+.5).astype(int)
        elif self.size_method == R_TO_SIZE:
            if self.use_manual_or_image == C_MANUAL:
                shape = np.array([self.specific_height.value,
                                  self.specific_width.value])
            elif self.use_manual_or_image == C_IMAGE:
                shape = np.array(workspace.image_set.get_image(self.specific_image.value).pixel_data.shape).astype(int)
        #
        # Little bit of wierdness here. The input pixels are numbered 0 to
        # shape-1 and so are the output pixels. Therefore the affine transform
        # is the ratio of the two shapes-1
        #
        ratio = ((np.array(image_pixels.shape[:2]).astype(float)-1) /
                 (shape.astype(float)-1))
        transform = np.array([[ratio[0], 0],[0,ratio[1]]])
        if self.interpolation not in I_ALL:
            raise NotImplementedError("Unsupported interpolation method: %s" %
                                      self.interpolation.value)
        order = (0 if self.interpolation == I_NEAREST_NEIGHBOR
                 else 1 if self.interpolation == I_BILINEAR
                 else 2)
        if image_pixels.ndim == 3:
            output_pixels = np.zeros((shape[0],shape[1],image_pixels.shape[2]), 
                                     image_pixels.dtype)
            for i in range(image_pixels.shape[2]):
                affine_transform(image_pixels[:,:,i], transform,
                                 output_shape = tuple(shape),
                                 output = output_pixels[:,:,i],
                                 order = order)
        else:
            output_pixels = affine_transform(image_pixels, transform,
                                             output_shape = shape,
                                             order = order)
        output_image = cpi.Image(output_pixels)
        workspace.image_set.add(output_image_name, output_image) 
        
    def is_interactive(self):
        return False
    
    def display(self, workspace):
        '''Display the resized images
        
        workspace - the workspace being run
        statistics - a list of lists:
            0: index of this statistic
            1: input image name of image being aligned
            2: output image name of image being aligned
        '''
        statistics = workspace.display_data.statistics
        figure = workspace.create_or_find_figure(title="Resize, image cycle #%d"%(
                workspace.measurements.image_set_number),subplots=(2,len(statistics)))
        for i, input_name, output_name in statistics:
            input_image = workspace.image_set.get_image(input_name)
            input_image_pixels = input_image.pixel_data
            
            output_image = workspace.image_set.get_image(output_name)
            output_image_pixels = output_image.pixel_data
            
            if input_image_pixels.ndim == 2:
                figure.subplot_imshow_bw(0,i,input_image_pixels,
                                         title=input_name)
                figure.subplot_imshow_bw(1,i,output_image_pixels,
                                         title=output_name)
            else:
                figure.subplot_imshow(0, i, input_image_pixels, 
                                      title=input_name)
                figure.subplot_imshow(1, i, output_image_pixels,
                                      title=output_name)
                
    def upgrade_settings(self, setting_values, variable_revision_number, 
                         module_name, from_matlab):
        if from_matlab and variable_revision_number == 1:
            width, height = setting_values[3].split(',')
            size_method = R_BY_FACTOR if setting_values[2] != "1" else R_TO_SIZE
            setting_values = [ setting_values[0], #image name
                              setting_values[1],  #resized image name
                              size_method,
                              setting_values[2], # resizing factor
                              width,
                              height,
                              setting_values[4]] #interpolation method
            from_matlab = False
            variable_revision_number = 1
            
        if (not from_matlab) and variable_revision_number == 1:
            if setting_values[2] == "Resize by a factor of the original size":
                setting_values[2] = R_BY_FACTOR
            if setting_values[2] == "Resize to a size in pixels":
                setting_values[2] = R_TO_SIZE
            variable_revision_number = 2
            
        if (not from_matlab) and variable_revision_number == 2:
            # Add additional images to be resized similarly, but if you only had 1,
            # the order didn't change
            setting_values = setting_values + ["0"]
            variable_revision_number = 3
            
        if (not from_matlab) and variable_revision_number == 3:
            # Add resizing to another image size
            setting_values = setting_values[:7] + [C_MANUAL, "None"] + setting_values[7:]
            variable_revision_number = 4

        return setting_values, variable_revision_number, from_matlab
        
