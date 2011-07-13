'''
<b>Morph</b> performs low-level morphological operations on binary or grayscale images
<hr>

This module performs a series of morphological operations on a binary
image or grayscale image, resulting in an image of the same type. Many require some image processing knowledge to understand how best to use these morphological filters in order to achieve the desired result. Note that the algorithms minimize the interference of masked pixels; for instance,
the dilate operation will only consider unmasked pixels in the neighborhood
of a pixel when determining the maximum within that neighborhood.
<br>
<br>
The following operations are available:<br><br>
<table border="1">
<tr><td><b>Operation</b></td><td><b>Description</b></td>
<td><b>Input image type allowed</b></td></tr>
<tr>
<td><i>Bothat</i></td>
<td>Bottom-hat filter: A bottom-hat filter enhances black spots in a white background. 
It subtracts the morphological <i>Close</i> of the image from the image. See below for a description of <i>Close</i>.</td>
<td>Binary, grayscale</td>
</tr>
<tr><td><i>Branchpoints</i></td>
<td>Removes all pixels except those that are the branchpoints of a skeleton.
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
<td>Sets a pixel to 1 if it has two non-zero neighbors that are on opposite 
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
<td>Removes isolated pixels:<br>
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
<td><i>Convex hull</i></td>
<td>Finds the convex hull of a binary image. The convex hull is the smallest
convex polygon that fits around all foreground pixels of the image: it is
the shape that a rubber band would take if stretched around the foreground
pixels. The convex hull can be used to regularize the boundary of a large,
single object in an image, for instance, the edge of a well.</td>
<td>Binary</td></tr>
<tr>
<td><i>Diag</i></td>
<td>Fills in pixels whose neighbors are diagnonally connected to 4-connect 
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
<td>For binary, replaces any 0 pixel by 1 if any of its neighbors is 1. 
For grayscale, each pixel is replaced by the maximum of its neighbors and itself.</td>
<td>Binary, grayscale</td>
</tr>
<tr>
<td><i>Distance</i></td>
<td>Computes the distance transform of a binary image. The distance of each
foreground pixel is computed to the nearest background pixel. The resulting
image is then scaled so that the largest distance is 1.</td>
<td>Binary</td>
</tr>
<tr>        
<td><i>Erode</i></td>
<td>For binary, replaces any 1 pixel by 0 if any of its neighbors is 0. 
For grayscale, each pixel is replaced by the minimum of its neighbors and itself.</td>
<td>Binary, grayscale</td>
</tr>
<tr>
<td><i>Endpoints</i></td>
<td>Removes all pixels except the ones that are at the end of a skeleton:<br>
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
<td><i>Fill</i></td>
<td>Sets a pixel to 1 if all of its neighbors are 1:<br>
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
<td><i>Fill small holes</i></td>
<td>Sets background pixels surrounded by foreground pixels to 1.<br>
This operation fills in small holes in a binary image. You can set the
maximum area of a hole in order to restrict the operation to holes of
a given size or smaller.
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
<td><i>Invert</i></td>
<td>For a binary image, transforms background to foreground and vice-versa.
For a grayscale image, invert its intensity.
</td><td>Binary, Grayscale</td>
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
<td><i>Life</i></td><td>Applies the interaction rules from the 
<a href="http://en.wikipedia.org/wiki/Conway%27s_Game_of_Life">Game of Life</a>, 
an example of a cellular automaton.</td>
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
The effect is to leave the perimeter of a solid object:<br>
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
<td>Performs a thinning operation that erodes unless that operation
would change the image's Euler number. This means that blobs are reduced to single 
points and blobs with holes are reduced to rings if shrunken indefinitely.</td>
<td>Binary</td>
</tr>
<tr>
<td><i>Skel</i></td>
<td>Performs a skeletonizing operation (medial axis transform). Preserves 
the points at the edges of objects but erodes everything else to lines that connect those edges.
See <a href="http://homepages.inf.ed.ac.uk/rbf/HIPR2/skeleton.htm">here</a> for a description.</td>
<td>Binary</td>
</tr>
<tr>
<td><i>Spur</i></td>
<td>Removes spur pixels. These are pixels that are connected only diagonally
to other pixels and connected in only one direction:<br>
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
<td>Dilates the exteriors of objects where that dilation does not
8-connect the object with another. The image is labeled and the labeled objects are filled. 
Unlabeled points adjacent to uniquely labeled points change from background to foreground.</td>
<td>Binary</td>
</tr>
<tr>
<td><i>Thin</i></td>
<td>Thin lines preserving the Euler number using the thinning algorithm # 1 described in 
Guo, "Parallel Thinning with Two Subiteration Algorithms", <i>Communications of the ACM,</i> Vol 32 #3, 
page 359. The result generally preserves the lines in an image while eroding their thickness.</td>
<td>Binary</td>
</tr>
<tr>
<td><i>Tophat</i></td>
<td>Subtracts the morphological opening of the image from the image. This enhances white spots 
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
'''

# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Developed by the Broad Institute
# Copyright 2003-2011
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org

__version__ = "$Revision$"

import logging
import numpy as np
import scipy.ndimage as scind
import sys

logger = logging.getLogger(__name__)

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.cpimage as cpi
import cellprofiler.cpmath.cpmorphology as morph

F_BOTHAT = 'bothat'
F_BRANCHPOINTS = 'branchpoints'
F_BRIDGE = 'bridge'
F_CLEAN  = 'clean'
F_CLOSE  = 'close'
F_CONVEX_HULL = 'convex hull'
F_DIAG   = 'diag'
F_DILATE = 'dilate'
F_DISTANCE = 'distance'
F_ENDPOINTS = 'endpoints'
F_ERODE  = 'erode'
F_FILL   = 'fill'
F_FILL_SMALL = 'fill small holes'
F_HBREAK = 'hbreak'
F_INVERT = 'invert'
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
F_ALL = [F_BOTHAT, F_BRANCHPOINTS, F_BRIDGE, F_CLEAN, F_CLOSE, F_CONVEX_HULL,
         F_DIAG, F_DILATE, F_DISTANCE, F_ENDPOINTS, F_ERODE, F_FILL, 
         F_FILL_SMALL, F_HBREAK, F_INVERT, F_LIFE, F_MAJORITY, F_OPEN, F_REMOVE, 
         F_SHRINK, F_SKEL, F_SPUR, F_THICKEN, F_THIN, F_TOPHAT, F_VBREAK]

R_ONCE = 'Once'
R_FOREVER = 'Forever'
R_CUSTOM = 'Custom'
R_ALL = [R_ONCE, R_FOREVER, R_CUSTOM]

FUNCTION_SETTING_COUNT_V1 = 3
FUNCTION_SETTING_COUNT_V2 = 4
FUNCTION_SETTING_COUNT = 4

class Morph(cpm.CPModule):

    module_name = "Morph"
    category="Image Processing"
    variable_revision_number = 2
    
    def create_settings(self):
        self.image_name = cps.ImageNameSubscriber("Select the input image","None",doc="""
            What image do you want to morph?
            This is the input image to the module. A grayscale image can be
            converted to binary using the <b>ApplyThreshold</b> module. Objects can be
            converted to binary using the <b>ConvertToImage</b> module.""")
        
        self.output_image_name = cps.ImageNameProvider("Name the output image","MorphBlue",doc="""
        What do you want to call the resulting image?
            The output of the module. It will be of the same type as the
            input image.""")
        
        self.add_button = cps.DoSomething("", "Add another operation",
                                          self.add_function,doc="""                                    
            Press this button to add an operation that will be applied to the
            image resulting from the previous operation(s). The module repeats
            the previous operation the number of times you select before applying the operation added by this button.""")
        self.functions = []
        self.add_function(can_remove = False)
    
    CUSTOM_REPEATS_TEXT = "Repetition number"
    CUSTOM_REPEATS_DOC = "<i>(Used only if Custom selected)</i><br>Enter the number of times to repeat the operation"
    def add_function(self, can_remove = True):
        group = MorphSettingsGroup()
        if can_remove:
            group.append("divider", cps.Divider(line=False))
        group.append("function", cps.Choice("Select the operation to perform",
                                           F_ALL, F_OPEN,doc="""
                                           What operation do you want to perform?
                    Choose one of the operations described in this module's help."""))
        group.append("repeats_choice", cps.Choice("Number of times to repeat operation",
                                                  R_ALL,doc="""
                    This setting controls the number of times that the same operation is applied
                    successively to the image.
                    <ul>
                    <li><i>Once:</i> Perform the operation once on the image.</li>
                    <li><i>Forever:</i> Perform the operation on the image until successive
                    iterations yield the same image.</li>
                    <li><i>Custom:</i> Perform the operation a custom number of times.</li>
                    </ul>"""))
        group.append("custom_repeats", cps.Integer(self.CUSTOM_REPEATS_TEXT,2,1,
                     doc=self.CUSTOM_REPEATS_DOC))
        group.append("scale", cps.Float(
            "Scale",3, minval=3,
            doc="""Morphological open, close, erode and dialate are performed
            with structuring elements which determine the diameter of the
            circle enclosing the pixels to consider when applying the operation.
            This setting controls the diameter of the structuring element."""))
                                        
        if can_remove:
            group.append("remove", cps.RemoveSettingButton("", "Remove this operation", self.functions, group))
        self.functions.append(group)

    def prepare_settings(self, setting_values):
        '''Adjust the # of functions to match the # of setting values'''
        assert (len(setting_values)-2) % FUNCTION_SETTING_COUNT == 0
        function_count = (len(setting_values)-2) / FUNCTION_SETTING_COUNT
        del self.functions[function_count:]
        while len(self.functions) < function_count:
            self.add_function()

    def settings(self):
        '''Return the settings as saved in the pipeline file'''
        result = [self.image_name, self.output_image_name]
        for function in self.functions:
            result += [function.function, function.repeats_choice, 
                       function.custom_repeats, function.scale]
        return result
    
    def visible_settings(self):
        '''Return the settings as displayed to the user'''
        result = [self.image_name, self.output_image_name]
        for function in self.functions:
            temp = function.visible_settings()
            if function.function == F_FILL_SMALL:
                temp.remove(function.repeats_choice)
                function.custom_repeats.text = "Maximum hole area"
                function.custom_repeats.doc = """Fill in all holes that have
                this many pixels or fewer."""
            elif function.repeats_choice != R_CUSTOM:
                temp.remove(function.custom_repeats)
            else:
                function.custom_repeats.text = self.CUSTOM_REPEATS_TEXT
                function.custom_repeats.doc = self.CUSTOM_REPEATS_DOC
            if function.function not in (F_CLOSE, F_OPEN, F_ERODE, F_DILATE, 
                                         F_TOPHAT, F_BOTHAT):
                temp.remove(function.scale)
            result += temp
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
                                           pixel_data, mask, count,
                                           function.scale.value)
        new_image = cpi.Image(pixel_data, parent_image = image) 
        workspace.image_set.add(self.output_image_name.value, new_image)
        if not workspace.frame is None:
            figure = workspace.create_or_find_figure(title="Morph, image cycle #%d"%(
                workspace.measurements.image_set_number),subplots=(2,1))
            if pixel_data.dtype.kind == 'b':
                figure.subplot_imshow_bw(0,0,image.pixel_data,
                                         'Original image: %s'%
                                         self.image_name.value)
                figure.subplot_imshow_bw(1,0,pixel_data,
                                         self.output_image_name.value,
                                         sharex = figure.subplot(0,0),
                                         sharey = figure.subplot(0,0))
            else:
                figure.subplot_imshow_grayscale(0,0,image.pixel_data,
                                                'Original image: %s'%
                                                self.image_name.value)
                figure.subplot_imshow_grayscale(1,0,pixel_data,
                                                self.output_image_name.value,
                                                sharex = figure.subplot(0,0),
                                                sharey = figure.subplot(0,0))
    
    def run_function(self, function_name, pixel_data, mask, count, scale):
        '''Apply the function once to the image, returning the result'''
        is_binary =  pixel_data.dtype.kind == 'b'
        strel = morph.strel_disk(scale / 2.0)
        if (function_name in (F_BRANCHPOINTS, F_BRIDGE, F_CLEAN, F_DIAG, 
                              F_CONVEX_HULL, F_DISTANCE, F_ENDPOINTS, F_FILL,
                              F_FILL_SMALL, F_HBREAK, F_LIFE, F_MAJORITY, 
                              F_REMOVE, F_SHRINK, F_SKEL, F_SPUR, F_THICKEN, 
                              F_THIN, F_VBREAK) 
            and not is_binary):
            # Apply a very crude threshold to the image for binary algorithms
            logger.warning("Warning: converting image to binary for %s\n"%
                           function_name)
            pixel_data = pixel_data != 0

        if (function_name in (F_BRANCHPOINTS, F_BRIDGE, F_CLEAN, F_DIAG, 
                              F_CONVEX_HULL, F_DISTANCE, F_ENDPOINTS, F_FILL,
                              F_FILL_SMALL,
                              F_HBREAK, F_INVERT, F_LIFE, F_MAJORITY, F_REMOVE,
                              F_SHRINK,
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
                                                strel,
                                                iterations = count)
                else:
                    return (scind.binary_closing(pixel_data & mask, 
                                                 strel,
                                                 iterations = count) |
                            (pixel_data & ~ mask))
            elif function_name == F_CONVEX_HULL:
                if mask is None:
                    return morph.convex_hull_image(pixel_data)
                else:
                    return morph.convex_hull_image(pixel_data & mask)
            elif function_name == F_DIAG:
                return morph.diag(pixel_data, mask, count)
            elif function_name == F_DILATE:
                return scind.binary_dilation(pixel_data, 
                                             strel,
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
                return scind.binary_erosion(pixel_data, strel,
                                            iterations = count,
                                            mask = mask)
            elif function_name == F_FILL:
                return morph.fill(pixel_data, mask, count)
            elif function_name == F_FILL_SMALL:
                def small_fn(area, foreground):
                    return (not foreground) and (area <= count)
                return morph.fill_labeled_holes(pixel_data, mask, small_fn)
            elif function_name == F_HBREAK:
                return morph.hbreak(pixel_data, mask, count)
            elif function_name == F_INVERT:
                if is_binary:
                    if mask is None:
                        return ~ pixel_data
                    result = pixel_data.copy()
                    result[mask] = ~result[mask]
                    return result
                elif mask is None:
                    return 1-pixel_data
                else:
                    result = pixel_data.copy()
                    result[mask]  = 1-result[mask]
                    return result
            elif function_name == F_LIFE:
                return morph.life(pixel_data, count)
            elif function_name == F_MAJORITY:
                return morph.majority(pixel_data, mask, count)
            elif function_name == F_OPEN:
                if mask is None:
                    return scind.binary_opening(pixel_data,
                                                strel,
                                                iterations = count)
                else:
                    return (scind.binary_opening(pixel_data & mask, 
                                                 strel,
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
                    new_pixel_data = morph.black_tophat(pixel_data, mask=mask,
                                                        footprint=strel)
                elif function_name == F_CLOSE:
                                                         
                    new_pixel_data = morph.closing(pixel_data, mask=mask,
                                                   footprint=strel)
                elif function_name == F_DILATE:
                    new_pixel_data = morph.grey_dilation(pixel_data, mask=mask,
                                                         footprint=strel)
                elif function_name == F_ERODE:
                    new_pixel_data = morph.grey_erosion(pixel_data, mask=mask,
                                                        footprint=strel)
                elif function_name == F_OPEN:
                    new_pixel_data = morph.opening(pixel_data, mask=mask,
                                                   footprint=strel)
                elif function_name == F_TOPHAT:
                    new_pixel_data = morph.white_tophat(pixel_data, mask=mask,
                                                        footprint=strel)
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
        if ((not from_matlab) and module_name == 'ImageConvexHull' and
            variable_revision_number == 1):
            #
            # Convert ImageConvexHull into an invert operation and
            # a convex hull operation
            #
            image_name, output_name = setting_values
            setting_values = [image_name, output_name, 
                              F_INVERT, R_ONCE, "1",
                              F_CONVEX_HULL, R_ONCE, "1"]
            module_name = self.module_name
            variable_revision_number = 1
        if (not from_matlab) and variable_revision_number == 1:
            new_setting_values = setting_values[:2]
            for i in range(2, len(setting_values), FUNCTION_SETTING_COUNT_V1):
                new_setting_values += setting_values[i:i+FUNCTION_SETTING_COUNT_V1]
                new_setting_values += [ "3" ]
            setting_values = new_setting_values
            variable_revision_number = 2
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
