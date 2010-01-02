'''
<b>Morph</b> performs low-level morphological operations on binary or grayscale images
<hr>
This module performs a series of morphological operations on a binary
image or grayscale image, resulting in an image of the same type.
    
The following operations are supported:<br>
<table border="1">
<tr><td><b>Operation</b></td><td><b>Description</b></td>
<td><b>Input image supported</b></td></tr>
<tr>
<td><i>Bothat</i></td>
<td>Bottom-hat filter: A bottom-hat filter enhances black spots in a white background. 
It subtracts the morphological "close" of the image from the image.</td>
<td>Binary, grayscale</td>
</tr>
<tr><td><i>Branchpoints</i></td>
<td>Remove all pixels except those that are the branchpoints of a skeleton.
This operation should be applied to an image after skeletonizing. It leaves
only those pixels that are at the intersection of branches.<br>
<table>
<tr><td><table border="1">
<tr><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>    
<tr><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td></tr>    
<tr><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td></tr>  
<tr><td>0</td><td>1</td><td>0</td><td>1</td><td>0</td></tr>      
<tr><td>1</td><td>0</td><td>0</td><td>0</td><td>1</td></tr>    
</table></td>
<td>&rarr;</td>
<td><table border="1">
<tr><td>?</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>    
<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>    
<tr><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td></tr>  
<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>      
<tr><td>?</td><td>0</td><td>0</td><td>0</td><td>?</td></tr>    
</table></td></tr>
</table></td>
<td>Binary
</td></tr>
<tr>
<td><i>Bridge</i></td>
<td>Set a pixel to one if it has two non-zero neighbors that are on opposite 
sides of this pixel:<br>
<table> 
<tr>
<td><table border="1">
<tr><td>1</td><td>0</td><td>0</td></tr>    
<tr><td>0</td><td>0</td><td>0</td></tr>  
<tr><td>0</td><td>0</td><td>1</td></tr>      
</table></td>
<td>&rarr;</td>
<td><table border="1">
<tr><td>1</td><td>0</td><td>0</td></tr>
<tr><td>0</td><td>1</td><td>0</td></tr>
<tr><td>0</td><td>0</td><td>1</td></tr>
</table></td>
</tr>
</table>
</td>
<td>Binary</td>
</tr>
<tr>
<td><i>Clean</i></td>
<td>Remove isolated pixels:<br>
<table> 
<tr>
<td><table border=1>
<tr><td>0</td><td>0</td><td>0</td></tr>    
<tr><td>0</td><td>0</td><td>0</td></tr>  
<tr><td>0</td><td>0</td><td>0</td></tr>      
</table></td>
<td>&rarr;</td>
<td><table border=1>
<tr><td>0</td><td>0</td><td>0</td></tr>
<tr><td>0</td><td>1</td><td>0</td></tr>
<tr><td>0</td><td>0</td><td>0</td></tr>
</table></td>
</tr>
</table>
</td>
<td>Binary</td>
</tr>
<tr>
<td><i>Close</i></td>
<td>Performs a dilation followed by an erosion. The effect is to
fill holes and join nearby objects.
</td>
<td>Binary, grayscale</td>
</tr>
<tr>
<td><i>Diag</i></td>
<td>Fill in pixels whose neighbors are diagnonally connected to 4-connect 
pixels that are 8-connected:<br>
<table>
<tr>
<td><table border=1>
<tr><td>0</td><td>1</td></tr>
<tr><td>1</td><td>0</td></tr>
</table></td>
<td>&rarr;</td>
<td><table border=1>
<tr><td>1</td><td>1</td></tr>
<tr><td>1</td><td>1</td></tr>
</table></td>
<td>&nbsp;,&nbsp;</td>
<td><table border=1>
<tr><td>0</td><td>1</td></tr>
<tr><td>1</td><td>1</td></tr>
</table></td>
<td>&rarr;</td>
<td><table border=1>
<tr><td>1</td><td>1</td></tr>
<tr><td>1</td><td>1</td></tr>
</table></td>
</tr>
</table>
</td>
<td>Binary</td>
</tr>
<tr>
<td><i>Dilate</i></td>
<td>For binary, any 0 pixel is replaced by 1 if any of its neighbors is 1. 
For grayscale, each pixel is replaced by the maximum of its neighbors and itself.</td>
<td>Binary, grayscale</td>
</tr>
<tr>
<td><i>Distance</i></td>
<td>Compute the distance transform of a binary image. The distance of each
foreground pixel is computed to the nearest background pixel. The resulting
image is then scaled so that the largest distance is 1.</td>
<td>Binary</td>
</tr>
<tr>        
<td><i>Erode</i></td>
<td>For binary, any 1 pixel is replaced by 0 if any of its neighbors is 0. 
For grayscale, each pixel is replaced by the minimum of its neighbors and itself.</td>
<td>Binary, grayscale</td>
</tr>
<tr>
<td>Endpoints</td>
<td>Remove all pixels except the ones that are at the end of a skeleton:<br>
<table>
<tr>
<td><table border=1>
<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>    
<tr><td>0</td><td>1</td><td>0</td><td>1</td><td>0</td></tr>    
<tr><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td></tr>  
<tr><td>0</td><td>1</td><td>0</td><td>1</td><td>0</td></tr>      
<tr><td>0</td><td>1</td><td>0</td><td>0</td><td>1</td></tr>    
</table></td>
<td>&rarr;</td>
<td><table border="1">
<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>    
<tr><td>0</td><td>1</td><td>0</td><td>1</td><td>0</td></tr>    
<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>  
<tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>      
<tr><td>0</td><td>?</td><td>0</td><td>0</td><td>?</td></tr>    
</table></td>
</tr>
</table>
</td>
<td>Binary</td>
</tr>
<tr>
<td>Fill</td>
<td>Set a pixel to 1 if all of its neighbors are 1:<br>
<table>
<tr>
<td><table border=1>
<tr><td>1</td><td>1</td><td>1</td></tr>
<tr><td>1</td><td>0</td><td>1</td></tr>
<tr><td>1</td><td>1</td><td>1</td></tr>
</table></td>
<td>&rarr;</td>
<td><table border=1>
<tr><td>1</td><td>1</td><td>1</td></tr>
<tr><td>1</td><td>1</td><td>1</td></tr>
<tr><td>1</td><td>1</td><td>1</td></tr>
</table></td>
</tr>
</table>
</td>
<td>Binary</td>
</tr>
<tr>
<td><i>Hbreak</i></td>
<td>Removes pixels that form vertical bridges between horizontal lines:<br>
<table>
<tr>
<td><table border=1>
<tr><td>1</td><td>1</td><td>1</td></tr>
<tr><td>0</td><td>1</td><td>0</td></tr>
<tr><td>1</td><td>1</td><td>1</td></tr>
</table></td>
<td>&rarr;</td>
<td><table border=1>
<tr><td>1</td><td>1</td><td>1</td></tr>
<tr><td>0</td><td>0</td><td>0</td></tr>
<tr><td>1</td><td>1</td><td>1</td></tr>
</table></td>
</tr>
</table>
</td>
<td>Binary</td>
</tr>
<tr>
<td><i>Majority</i></td>
<td>Each pixel takes on the value of the majority that surround it 
(keep pixel value to break ties):<br>
<table>
<tr>
<td><table border=1>
<tr><td>1</td><td>1</td><td>1</td></tr>
<tr><td>1</td><td>0</td><td>1</td></tr>
<tr><td>0</td><td>0</td><td>0</td></tr>
</table></td>
<td>&rarr;</td>
<td><table border=1>
<tr><td>1</td><td>1</td><td>1</td></tr>
<tr><td>1</td><td>1</td><td>1</td></tr>
<tr><td>0</td><td>0</td><td>0</td></tr>
</table></td>
</tr>
</table>
</td>
<td>Binary</td>
</tr>
<tr>
<td><i>Open</i></td><td>Performs an erosion followed by a dilation. The effect is to
break bridges between objects and remove single pixels.</td>
<td>Binary, grayscale</td>
</tr>
<tr>
<td><i>Remove</i></td>
<td>Removes pixels that are otherwise surrounded by others (4 connected). 
The effect is to be left with the perimeter of a solid object:<br>
<table>
<tr>
<td><table border=1>
<tr><td>0</td><td>1</td><td>0</td></tr>
<tr><td>1</td><td>1</td><td>1</td></tr>
<tr><td>0</td><td>1</td><td>0</td></tr>
</table></td>
<td>&rarr;</td>
<td><table border=1>
<tr><td>0</td><td>1</td><td>0</td></tr>
<tr><td>1</td><td>0</td><td>1</td></tr>
<tr><td>0</td><td>1</td><td>0</td></tr>
</table></td>
</tr>
</table>
</td>
<td>Binary</td>
</tr>
<tr>
<td><i>Shrink</i></td>
<td>Perform a thinning operation that erodes unless that operation
would change the image's Euler number. This means that blobs are reduced to single 
points and blobs with holes are reduced to rings if shrunken indefinitely.</td>
<td>Binary</td>
</tr>
<tr>
<td><i>Skel</i></td>
<td>Perform a skeletonizing operation (medial axis transform). Preserves 
the points at the edges of objects but erodes everything else to lines that connect those edges.
See <a href="http://homepages.inf.ed.ac.uk/rbf/HIPR2/skeleton.htm">here</a> for a description.</td>
<td>Binary</td>
</tr>
<tr>
<td><i>Spur</i></td>
<td>Remove spur pixels. These are pixels that are only diagonally
connected to other pixels and connected in only one direction:<br>
<table>
<tr>
<td><table border=1>
<tr><td>0</td><td>0</td><td>0</td><td>0</td></tr>
<tr><td>0</td><td>1</td><td>0</td><td>0</td></tr>
<tr><td>0</td><td>0</td><td>1</td><td>0</td></tr>
<tr><td>1</td><td>1</td><td>1</td><td>1</td></tr>
</table></td>
<td>&rarr;</td>
<td><table border=1>
<tr><td>0</td><td>0</td><td>0</td><td>0</td></tr>
<tr><td>0</td><td>0</td><td>0</td><td>0</td></tr>
<tr><td>0</td><td>0</td><td>1</td><td>0</td></tr>
<tr><td>1</td><td>1</td><td>1</td><td>1</td></tr>
</table></td>
</tr>
</table>
</td>
<td>Binary</td>
</tr>
<tr>
<td><i>Thicken</i></td>
<td>Dilate the exteriors of objects where that dilation does not
8-connect the object with another. The image is labeled and the labeled objects are filled. 
Unlabeled points adjacent to uniquely labeled points change from background to foreground.</td>
<td>Binary</td>
</tr>
<tr>
<td><i>Thin</i></td>
<td>Thin lines preserving the Euler number using the thinning algorithm # 1 described in 
<i>Guo, "Parallel Thinning with Two Subiteration Algorithms", Communications of the ACM, Vol 32 #3</i>
page 359. The result generally preserves the lines in an image while eroding their thickness.</td>
<td>Binary</td>
</tr>
<tr>
<td><i>Tophat</i></td>
<td>Subtract the morphological opening of the image from the image. This enhances white spots 
in a black background.</td>
<td>Binary, grayscale</td>
</tr>
<tr>
  <td><i>Vbreak</i></td>
  <td>Removes pixels that form horizontal bridges between vertical lines:<br>
    <table>
      <tr>
        <td><table border=1>
          <tr><td>1</td><td>0</td><td>1</td></tr>
          <tr><td>1</td><td>1</td><td>1</td></tr>
          <tr><td>1</td><td>0</td><td>1</td></tr>
        </table></td>
        <td>&rarr;</td>
        <td><table border=1>
          <tr><td>1</td><td>0</td><td>1</td></tr>
          <tr><td>1</td><td>0</td><td>1</td></tr>
          <tr><td>1</td><td>0</td><td>1</td></tr>
        </table></td>
      </tr>
    </table></td>
  <td>Binary</td>
</tr>
</table>
<p>The algorithms minimize the interference of masked pixels; for instance,
the dilate operation will only consider unmasked pixels in the neighborhood
of a pixel when determining the maximum within that neighborhood.</p>
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

__version__ = "$Revision$"

import numpy as np
import scipy.ndimage as scind
import sys

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.cpimage as cpi
import cellprofiler.cpmath.cpmorphology as morph

F_BOTHAT = 'bothat'
F_BRANCHPOINTS = 'branchpoints'
F_BRIDGE = 'bridge'
F_CLEAN  = 'clean'
F_CLOSE  = 'close'
F_DIAG   = 'diag'
F_DILATE = 'dilate'
F_DISTANCE = 'distance'
F_ENDPOINTS = 'endpoints'
F_ERODE  = 'erode'
F_FILL   = 'fill'
F_HBREAK = 'hbreak'
F_LIFE   = 'life'
F_MAJORITY = 'majority'
F_OPEN   = 'open'
F_REMOVE = 'remove'
F_SHRINK = 'shrink'
F_SKEL   = 'skel'
F_SPUR   = 'spur'
F_THICKEN = 'thicken'
F_THIN   = 'thin'
F_TOPHAT = 'tophat'
F_VBREAK = 'vbreak'
F_ALL = [F_BOTHAT, F_BRANCHPOINTS, F_BRIDGE, F_CLEAN, F_CLOSE, F_DIAG, 
         F_DILATE, F_DISTANCE, F_ENDPOINTS, F_ERODE,
         F_FILL, F_HBREAK, F_LIFE, F_MAJORITY, F_OPEN, F_REMOVE, F_SHRINK, 
         F_SKEL, F_SPUR, F_THICKEN, F_THIN, F_TOPHAT, F_VBREAK]

R_ONCE = 'Once'
R_FOREVER = 'Forever'
R_CUSTOM = 'Custom'
R_ALL = [R_ONCE, R_FOREVER, R_CUSTOM]

class Morph(cpm.CPModule):

    module_name = "Morph"
    category="Image Processing"
    variable_revision_number = 1
    
    def create_settings(self):
        self.image_name = cps.ImageNameSubscriber("Select input image:","None",doc="""
            What image do you want to morph?
            This is the input image to the module. A grayscale image can be
            converted to binary using the <b>ApplyThreshold</b> module. Objects can be
            converted to binary using the <b>ConvertToImage module</b>.""")
        
        self.output_image_name = cps.ImageNameProvider("Name the output image:","MorphBlue",doc="""
        What do you want to call the resulting image?
            This is the output of the module. It will be of the same type as the
            input image.""")
        
        self.add_button = cps.DoSomething("", "Add operation",
                                          self.add_function,doc="""                                    
            Press this button to add an operation that will be applied to the
            image resulting from the previous transformation. The module repeats
            the previous transformation the number of times indicated by the
            instructions before applying the operation added by this button.""")
        self.functions = []
        self.add_function()
    
    def add_function(self):
        group = MorphSettingsGroup()
        group.append("function", cps.Choice("Select operation to perform:",
                                           F_ALL, F_OPEN,doc="""
                                           What operation do you want to perform?
                    This is one of the functions listed in the module Help."""))
        group.append("repeats_choice", cps.Choice("Repeat operation:",
                                                  R_ALL,doc="""
                    This setting controls the number of times that the same operation is applied
                    successively to the image.
                    <ul>
                    <li><i>Once:</i> Perform one transformation on the image</li>
                    <li><i>Forever:</i> Perform the transformation on the image until successive
                    transformations yield the same image.</li>
                    <li><i>Custom:</i> Perform the transformation a custom number of times.</li>
                    </ul>"""))
        group.append("custom_repeats", cps.Integer("Custom # of repeats",2,1))
        group.append("remove", cps.RemoveSettingButton("", "Remove above operation", self.functions, group))
        group.append("divider", cps.Divider(line=False))
        self.functions.append(group)

    def prepare_settings(self, setting_values):
        '''Adjust the # of functions to match the # of setting values'''
        assert (len(setting_values)-2)%3 == 0
        function_count = (len(setting_values)-2) / 3
        del self.functions[function_count:]
        while len(self.functions) < function_count:
            self.add_function()

    def settings(self):
        '''Return the settings as saved in the pipeline file'''
        result = [self.image_name, self.output_image_name]
        for function in self.functions:
            result += [function.function, function.repeats_choice, function.custom_repeats]
        return result
    
    def visible_settings(self):
        '''Return the settings as displayed to the user'''
        result = [self.image_name, self.output_image_name]
        for function in self.functions:
            result += [function.function, function.repeats_choice]
            if function.repeats_choice == R_CUSTOM:
                result += [function.custom_repeats]
            result += [function.divider]
        result += [self.add_button]
        return result

    def run(self, workspace):
        image = workspace.image_set.get_image(self.image_name.value)
        if image.has_mask:
            mask = image.mask
        else:
            mask = None
        pixel_data = image.pixel_data
        for function in self.functions:
            count = function.repeat_count
            
            pixel_data = self.run_function(function.function.value,
                                           pixel_data, mask, count)
        new_image = cpi.Image(pixel_data, parent_image = image) 
        workspace.image_set.add(self.output_image_name.value, new_image)
        if not workspace.frame is None:
            figure = workspace.create_or_find_figure(subplots=(2,1))
            if pixel_data.dtype.kind == 'b':
                figure.subplot_imshow_bw(0,0,image.pixel_data,
                                         'Original image: %s'%
                                         self.image_name.value)
                figure.subplot_imshow_bw(1,0,pixel_data,
                                         self.output_image_name.value)
            else:
                figure.subplot_imshow_grayscale(0,0,image.pixel_data,
                                                'Original image: %s'%
                                                self.image_name.value)
                figure.subplot_imshow_grayscale(1,0,pixel_data,
                                                self.output_image_name.value)
    
    def run_function(self, function_name, pixel_data, mask, count):
        '''Apply the function once to the image, returning the result'''
        is_binary =  pixel_data.dtype.kind == 'b'
        if (function_name in (F_BRANCHPOINTS, F_BRIDGE, F_CLEAN, F_DIAG, 
                              F_DISTANCE, F_ENDPOINTS, F_FILL,
                              F_HBREAK, F_LIFE, F_MAJORITY, F_REMOVE, F_SHRINK,
                              F_SKEL, F_SPUR, F_THICKEN, F_THIN, F_VBREAK) 
            and not is_binary):
            # Apply a very crude threshold to the image for binary algorithms
            sys.stderr.write("Warning: converting image to binary for %s\n"%
                             function_name)
            pixel_data = pixel_data != 0

        if (function_name in (F_BRANCHPOINTS, F_BRIDGE, F_CLEAN, F_DIAG, 
                              F_DISTANCE, F_ENDPOINTS, F_FILL,
                              F_HBREAK, F_LIFE, F_MAJORITY, F_REMOVE, F_SHRINK,
                              F_SKEL, F_SPUR, F_THICKEN, F_THIN, F_VBREAK) or
            (is_binary and
             function_name in (F_CLOSE, F_DILATE, F_ERODE, F_OPEN))):
            # All of these have an iterations argument or it makes no
            # sense to iterate
            if function_name == F_BRANCHPOINTS:
                return morph.branchpoints(pixel_data, mask)
            elif function_name == F_BRIDGE:
                return morph.bridge(pixel_data, mask, count)
            elif function_name == F_CLEAN:
                return morph.clean(pixel_data, mask, count)
            elif function_name == F_CLOSE:
                if mask is None:
                    return scind.binary_closing(pixel_data,
                                                np.ones((3,3),bool),
                                                iterations = count)
                else:
                    return (scind.binary_closing(pixel_data & mask, 
                                                 np.ones((3,3),bool),
                                                 iterations = count) |
                            (pixel_data & ~ mask))
                
            elif function_name == F_DIAG:
                return morph.diag(pixel_data, mask, count)
            elif function_name == F_DILATE:
                return scind.binary_dilation(pixel_data, 
                                             np.ones((3,3),bool),
                                             iterations=count,
                                             mask=mask)
            elif function_name == F_DISTANCE:
                image = scind.distance_transform_edt(pixel_data)
                img_max = np.max(image)
                if img_max > 0:
                    image = image / img_max
                return image
            elif function_name == F_ENDPOINTS:
                return morph.endpoints(pixel_data, mask)
            elif function_name == F_ERODE:
                return scind.binary_erosion(pixel_data, np.ones((3,3),bool),
                                            iterations = count,
                                            mask = mask)
            elif function_name == F_FILL:
                return morph.fill(pixel_data, mask, count)
            elif function_name == F_HBREAK:
                return morph.hbreak(pixel_data, mask, count)
            elif function_name == F_LIFE:
                return morph.life(pixel_data, count)
            elif function_name == F_MAJORITY:
                return morph.majority(pixel_data, mask, count)
            elif function_name == F_OPEN:
                if mask is None:
                    return scind.binary_opening(pixel_data,
                                                np.ones((3,3),bool),
                                                iterations = count)
                else:
                    return (scind.binary_opening(pixel_data & mask, 
                                                 np.ones((3,3),bool),
                                                 iterations = count) |
                            (pixel_data & ~ mask))
            elif function_name == F_REMOVE:
                return morph.remove(pixel_data, mask, count)
            elif function_name == F_SHRINK:
                return morph.binary_shrink(pixel_data, count)
            elif function_name == F_SKEL:
                return morph.skeletonize(pixel_data, mask)
            elif function_name == F_SPUR:
                return morph.spur(pixel_data, mask, count)
            elif function_name == F_THICKEN:
                return morph.thicken(pixel_data, mask, count)
            elif function_name == F_THIN:
                return morph.thin(pixel_data, mask, count)
            elif function_name == F_VBREAK:
                return morph.vbreak(pixel_data, mask)
            else:
                raise NotImplementedError("Unimplemented morphological function: %s" %
                                          function_name)
        else:
            for i in range(count):
                if function_name == F_BOTHAT:
                    new_pixel_data = morph.black_tophat(pixel_data, mask=mask)
                elif function_name == F_CLOSE:
                                                         
                    new_pixel_data = morph.closing(pixel_data, mask==mask)
                elif function_name == F_DILATE:
                    new_pixel_data = morph.grey_dilation(pixel_data, mask=mask)
                elif function_name == F_ERODE:
                    new_pixel_data = morph.grey_erosion(pixel_data, mask=mask)
                elif function_name == F_OPEN:
                    new_pixel_data = morph.opening(pixel_data, mask==mask)
                elif function_name == F_TOPHAT:
                    new_pixel_data = morph.white_tophat(pixel_data, mask=mask)
                else:
                    raise NotImplementedError("Unimplemented morphological function: %s" %
                                              function_name)
                if np.all(new_pixel_data == pixel_data):
                    break;
                pixel_data = new_pixel_data
            return pixel_data
    
    def upgrade_settings(self, setting_values, 
                         variable_revision_number, module_name, 
                         from_matlab):
        '''Adjust the setting_values of previous revisions to match this one'''
        if from_matlab and variable_revision_number in (1,2):
            # Settings:
            # image name
            # resulting image name
            # (function, count) repeated 6 times
            new_setting_values = [setting_values[0], setting_values[1]]
            for i in range(6):
                if setting_values[i*2+2] != cps.DO_NOT_USE:
                    new_setting_values.append(setting_values[i*2+2])
                    if (setting_values[i*2+3].isdigit() and  
                        int(setting_values[i*2+3])== 1):
                        new_setting_values += [R_ONCE, "1"]
                    elif setting_values[i*2+3].lower() == "inf":
                        new_setting_values += [R_FOREVER,"2"]
                    else:
                        new_setting_values += [R_CUSTOM, setting_values[i*2+3]]
            setting_values = new_setting_values
            from_matlab = False
            variable_revision_number = 1
        return setting_values, variable_revision_number, from_matlab
                        


class MorphSettingsGroup(cps.SettingsGroup):
    @property
    def repeat_count(self):
        ''# of times to repeat'''
        if self.repeats_choice == R_ONCE:
            return 1
        elif self.repeats_choice == R_FOREVER:
            return 10000
        elif self.repeats_choice == R_CUSTOM:
            return self.custom_repeats.value
        else:
            raise ValueError("Unsupported repeat choice: %s"%
                             self.repeats_choice.value)

        '''The thresholding algorithm to run'''
        return self.threshold_method.value.split(' ')[0]
