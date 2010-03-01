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
# Developed by the Broad Institute
# Copyright 2003-2010
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org
__version__="$Revision$"

import numpy as np
from scipy.ndimage import affine_transform

import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.settings as cps

R_BY_FACTOR = "Resize by a fraction or multiple of the original size"
R_TO_SIZE = "Resize by specifying desired final dimensions"
R_ALL = [R_BY_FACTOR, R_TO_SIZE]

I_NEAREST_NEIGHBOR = 'Nearest Neighbor'
I_BILINEAR = 'Bilinear'
I_BICUBIC = 'Bicubic'

I_ALL = [I_NEAREST_NEIGHBOR, I_BILINEAR, I_BICUBIC]


class Resize(cpm.CPModule):

    category = "Image Processing"
    variable_revision_number = 2
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

        self.specific_width = cps.Integer("Width of the final image, in pixels", 100, minval=1, doc = '''
                                         <i>(Used only if resizing by a fraction or multiple of the original size)</i><br>
                                         Enter the desired width of the final image.''')

        self.specific_height = cps.Integer("Height of the final image, in pixels", 100, minval=1, doc = '''
                                         <i>(Used only if resizing by a fraction or multiple of the original size)</i><br>
                                         Enter the desired height of the final image.''')

        self.interpolation = cps.Choice("Interpolation method",
                                        I_ALL, doc = '''<ul><li><i>Nearest Neighbor:</i> Each output pixel is given the intensity of the nearest
                                        corresponding pixel in the input image.</li>
                                        <li><i>Bilinear:</i> Each output pixel is given the intensity of the weighted average
                                        of the 2x2 neighborhood at the corresponding position in the input image.</li>
                                        <li><i>Bicubic:</i> Each output pixel is given the intensity of the weighted average
                                        of the 4x4 neighborhood at the corresponding position in the input image.</li>
                                        </ul>''')

    def settings(self):
        return [self.image_name, self.resized_image_name, self.size_method,
                self.resizing_factor, self.specific_width, 
                self.specific_height, self.interpolation]

    def visible_settings(self):
        result = [self.image_name, self.resized_image_name, self.size_method]
        if self.size_method == R_BY_FACTOR:
            result.append(self.resizing_factor)
        elif self.size_method == R_TO_SIZE:
            result += [self.specific_width, self.specific_height]
        else:
            raise ValueError("Unsupported size method: %s" % 
                             self.size_method.value)
        result += [self.interpolation]
        return result

    def run(self, workspace):
        image = workspace.image_set.get_image(self.image_name.value)
        image_pixels = image.pixel_data
        if self.size_method == R_BY_FACTOR:
            factor = self.resizing_factor.value
            shape = (np.array(image_pixels.shape[:2])*factor+.5).astype(int)
        elif self.size_method == R_TO_SIZE:
            shape = np.array([self.specific_height.value,
                              self.specific_width.value])
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
        workspace.image_set.add(self.resized_image_name.value,
                                output_image) 
        if workspace.frame is not None:
            figure = workspace.create_or_find_figure(subplots=(2,1))
            if image_pixels.ndim == 2:
                figure.subplot_imshow_bw(0,0,image_pixels,
                                         title=self.image_name.value)
                figure.subplot_imshow_bw(1,0,output_pixels,
                                         title=self.resized_image_name.value)
            else:
                figure.subplot_imshow(0, 0, image_pixels, 
                                      title=self.image_name.value)
                figure.subplot_imshow(1, 0, output_pixels,
                                      title=self.resized_image_name.value)
                
                
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
            
        return setting_values, variable_revision_number, from_matlab
        
