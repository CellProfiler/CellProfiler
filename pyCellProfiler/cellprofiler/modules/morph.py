'''morph - morphological operations performed on images

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version__ = "$Revision$"

import numpy as np
import scipy.ndimage as scind
import sys
import uuid

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.cpimage as cpi
import cellprofiler.cpmath.cpmorphology as morph

F_BOTHAT = 'bothat'
F_BRIDGE = 'bridge'
F_CLEAN  = 'clean'
F_CLOSE  = 'close'
F_DIAG   = 'diag'
F_DILATE = 'dilate'
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
F_ALL = [F_BOTHAT, F_BRIDGE, F_CLEAN, F_CLOSE, F_DIAG, F_DILATE, F_ERODE,
         F_FILL, F_HBREAK, F_LIFE, F_MAJORITY, F_OPEN, F_REMOVE, F_SHRINK, 
         F_SKEL, F_SPUR, F_THICKEN, F_THIN, F_TOPHAT, F_VBREAK]

R_ONCE = 'Once'
R_FOREVER = 'Forever'
R_CUSTOM = 'Custom'
R_ALL = [R_ONCE, R_FOREVER, R_CUSTOM]

class Morph(cpm.CPModule):
    '''SHORT DESCRIPTION:
    Perform low-level morphological operations on binary or grayscale images
    ************************************************************************
    This module performs a series of morphological operations on a binary
    image, resulting in another binary image or on a grayscale image,
    resulting in another grayscale image.
    
    The following operations are supported:
    
    bothat - Bottom-hat filter: A bottom-hat filter enhances black spots
             in a white background. It subtracts the morphological "close"
             of the image from the image.
             Supported for grayscale and binary images
    bridge - Set a pixel to one if it has two non-zero neighbors that are
             on opposite sides of this pixel:
             1  0  0     1  0  0
             0  0? 0  -> 0  1  0 
             0  0  1     0  0  1
             Supported for binary images
    clean -  Remove isolated pixels
             
             0  0  0     0   0   0
             0  1  0  -> 0   0   0
             0  0  0     0   0   0
             Supported for binary.
    
    close -  Performs a dilation followed by an erosion. The effect is to
             fill holes and join nearby objects.
             Supported for binary and grayscale images.
    
    diag -   Fill in pixels whose neighbors are diagnonally connected to
             to 4-connect pixels that are 8-connected.
             0  1    1  1    0  1     0  1
             1  0 -> 1  1    1  1  -> 1  1
             Supported for binary images
    
    dilate - For binary, any 0 pixel is replaced by 1 if any of its neighbors
             is 1. For grayscale, each pixel is replaced by the maximum
             of its neighbors and itself.
             
    erode  - For binary, any 1 pixel is replaced by 0 if any of its neighbors
             is 0. For grayscale, each pixel is replaced by the minimum of
             its neighbors and itself.
    
    fill   - Set a pixel to 1 if all of its neighbors are 1
              
             1  1  1    1  1  1
             1  0  1 -> 1  1  1 
             1  1  1    1  1  1
             Supported for binary images.
    
    hbreak - Removes pixels that form vertical bridges between horizontal
             lines:
             1  1  1    1  1  1
             0  1  0 -> 0  0  0  (only this pattern)
             1  1  1    1  1  1
             Supported for binary images
    
    majority Each pixel takes on the value of the majority that surround it
             (keep pixel value to break ties):
             1  1  1    1  1  1
             1  0  1 -> 1  1  1
             0  0  0    0  0  0
             Supported for binary images
    
    open   - Performs an erosion followed by a dilation. The effect is to
             break bridges between objects and remove single pixels.
             Supported for binary and grayscale images.
    
    remove - Removes pixels that are otherwise surrounded by others
             (4 connected). The effect is to be left with the perimeter of
             a solid object:
             0  1  0    0  1  0
             1  1  1 -> 1  0  1
             0  1  0    0  1  0
             Supported for binary images
    
    shrink - Perform a thinning operation that erodes unless that operation
             would change the image's Euler number. This means that blobs
             are reduced to single points and blobs with holes are reduced
             to rings if shrunken indefinitely.
             Supported for binary images
    
    skel   - Perform a skeletonizing operation (medial axis transform). Skel
             preserves the points at the edges of objects but erodes everything
             else to lines that connect those edges.
             See http://homepages.inf.ed.ac.uk/rbf/HIPR2/skeleton.htm
             for a description.
             Supported for binary images
    
    spur   - Remove spur pixels. These are pixels that are only diagonally
             connected to other pixels and connected in only one direction.
             For instance:
             
             0 0 0 0    0 0 0 0
             0 1 0 0 -> 0 0 0 0
             0 0 1 0    0 0 1 0
             1 1 1 1    1 1 1 1
             Supported for binary images
    
    thicken - Dilate the exteriors of objects where that dilation does not
             8-connect the object with another. The image is labeled and
             the labeled objects are filled. Unlabeled points adjacent to
             uniquely labeled points change from background to foreground.
             Supported for binary images
    
    thin -   Thin lines preserving the Euler number using the thinning
             algorithm # 1 described in Guo, "Parallel Thinning with Two
             Subiteration Algorithms", Communications of the ACM, Vol 32 #3
             page 359. The result generally preserves the lines in an image
             while eroding their thickness.
             Supported for binary images
    
    tophat - Subtract the morphological opening of the image from the image.
             This enhances white spots in a black background.
             Supported for binary and grayscale images.
    
    vbreak - Removes pixels that form horizontal bridges between vertical
             lines:
             1  0  1    1  0  1
             1  1  1 -> 1  0  1
             1  0  1    1  0  1
             Supported for binary images
    
    The algorithms minimize the interference of masked pixels; for instance,
    the dilate operation will only consider unmasked pixels in the neighborhood
    of a pixel when determining the maximum within that neighborhood.
    
    Settings:
    What image do you want to morph?
    This is the input image to the module. A grayscale image can be
    converted to binary using the ApplyThreshold module. Objects can be
    converted to binary using the ConvertToImage module.
    
    What do you want to call the resulting image?
    This is the output of the module. It will be of the same type as the
    input image.
    
    What function do you want to perform?
    This is one of the functions above.
    
    How many times do you want to repeat the function?
    Once - perform one transformation on the image
    Forever - perform the transformation on the image until successive
              transformations yield the same image.
    Custom - perform the transformation a custom number of times.
    
    Add another function:
    Press this button to add a function that will be applied to the
    image resulting from the previous transformation. The module repeats
    the previous transformation the number of times indicated by the
    instructions before applying the function added by this button.
    
    Remove the above function:
    Press this button to remove a function from the list.
    '''

    category="Image Processing"
    variable_revision_number = 1
    
    def create_settings(self):
        self.module_name = "Morph"
        self.image_name = cps.ImageNameSubscriber("What image do you want to morph?","None")
        self.output_image_name = cps.ImageNameProvider("What do you want to call the resulting image?","MorphBlue")
        self.add_button = cps.DoSomething("Add another function:","Add",
                                          self.add_function)
        self.functions = []
        self.add_function()
    
    def add_function(self):
        class Function:
            '''Represents the variables needed to run a function'''
            def __init__(self, functions):
                self.key = uuid.uuid4()
                self.function = cps.Choice("What function do you want to perform?",
                                           F_ALL, F_OPEN)
                self.repeats_choice = cps.Choice("How many times do you want to repeat the function?",
                                                 R_ALL)
                self.custom_repeats = cps.Integer("Custom # of repeats",2,1)
                def remove(functions = functions, key = self.key):
                    '''Remove this function from the function list'''
                    index = [x.key for x in functions].index(key)
                    del functions[index]
                
                self.remove = remove    
                self.remove_button = cps.DoSomething("Remove the above function:",
                                                     "Remove",
                                                     self.remove)
            
            def settings(self):
                '''The settings to be saved in the pipeline'''
                return [self.function, self.repeats_choice, 
                        self.custom_repeats]
            
            def visible_settings(self):
                '''The settings to be shown to the user'''
                result = [self.function, self.repeats_choice]
                if self.repeats_choice == R_CUSTOM:
                    result += [self.custom_repeats]
                result += [self.remove_button]
                return result
            
            def get_repeat_count(self):
                '''# of times to repeat'''
                if self.repeats_choice == R_ONCE:
                    return 1
                elif self.repeats_choice == R_FOREVER:
                    return 10000
                elif self.repeats_choice == R_CUSTOM:
                    return self.custom_repeats.value
                else:
                    raise ValueError("Unsupported repeat choice: %s"%
                                     self.repeats_choice.value)
            repeat_count = property(get_repeat_count)
            
        function = Function(self.functions)
        self.functions.append(function)

    def prepare_to_set_values(self, setting_values):
        '''Adjust the # of functions to match the # of setting values'''
        assert (len(setting_values)-2)%3 == 0
        function_count = (len(setting_values)-2) / 3
        while len(self.functions) > function_count:
            self.functions[-1].remove()
        while len(self.functions) < function_count:
            self.add_function()

    def backwards_compatibilize(self, setting_values, 
                                variable_revision_number, module_name, 
                                from_matlab):
        '''Adjust the setting_values of previous revisions to match this one'''
        if from_matlab and variable_revision_number == 1:
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
                        
    def settings(self):
        '''Return the settings as saved in the pipeline file'''
        result = [self.image_name, self.output_image_name]
        for function in self.functions:
            result += function.settings()
        return result
    
    def visible_settings(self):
        '''Return the settings as displayed to the user'''
        result = [self.image_name, self.output_image_name]
        for function in self.functions:
            result += function.visible_settings()
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
            
            for i in range(count):
                new_pixel_data = self.run_function(function.function.value,
                                                   pixel_data, mask)
                if i != count-1 and np.all(new_pixel_data == pixel_data):
                    break
                pixel_data = new_pixel_data
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
    
    def run_function(self, function_name, pixel_data, mask):
        '''Apply the function once to the image, returning the result'''
        is_binary =  pixel_data.dtype.kind == 'b'
        if (function_name in (F_BRIDGE, F_CLEAN, F_DIAG, F_FILL,
                              F_HBREAK, F_LIFE, F_MAJORITY, F_SHRINK,
                              F_SKEL, F_SPUR, F_THICKEN, F_THIN, F_VBREAK) and
            not is_binary):
            # Apply a very crude threshold to the image for binary algorithms
            sys.stderr.write("Warning: converting image to binary for %s\n"%
                             function_name)
            pixel_data = pixel_data != 0

        if function_name == F_BOTHAT:
            return morph.black_tophat(pixel_data, mask=mask)
        elif function_name == F_BRIDGE:
            return morph.bridge(pixel_data, mask)
        elif function_name == F_CLEAN:
            return morph.clean(pixel_data, mask)
        elif function_name == F_CLOSE:
            if is_binary:
                if mask is None:
                    return scind.binary_closing(pixel_data,np.ones((3,3),bool))
                else:
                    return (scind.binary_closing(pixel_data & mask, 
                                                 np.ones((3,3),bool)) |
                            (pixel_data & ~ mask))
                                                 
            else:
                return morph.closing(pixel_data, mask==mask)
        elif function_name == F_DIAG:
            return morph.diag(pixel_data, mask)
        elif function_name == F_DILATE:
            if is_binary:
                return scind.binary_dilation(pixel_data, 
                                             np.ones((3,3),bool),1,mask)
            else:
                return morph.grey_dilation(pixel_data, mask=mask)
        elif function_name == F_ERODE:
            if is_binary:
                return scind.binary_erosion(pixel_data, np.ones((3,3),bool),
                                            1,mask)
            else:
                return morph.grey_erosion(pixel_data, mask=mask)
        elif function_name == F_FILL:
            return morph.fill(pixel_data, mask)
        elif function_name == F_HBREAK:
            return morph.hbreak(pixel_data, mask)
        elif function_name == F_LIFE:
            return morph.life(pixel_data)
        elif function_name == F_MAJORITY:
            return morph.majority(pixel_data, mask)
        elif function_name == F_OPEN:
            if is_binary:
                if mask is None:
                    return scind.binary_opening(pixel_data,np.ones((3,3),bool))
                else:
                    return (scind.binary_opening(pixel_data & mask, 
                                                 np.ones((3,3),bool)) |
                            (pixel_data & ~ mask))
                                                 
            else:
                return morph.opening(pixel_data, mask==mask)
        elif function_name == F_REMOVE:
            return morph.remove(pixel_data, mask)
        elif function_name == F_SHRINK:
            return morph.binary_shrink(pixel_data, 1)
        elif function_name == F_SKEL:
            return morph.skeletonize(pixel_data, mask)
        elif function_name == F_SPUR:
            return morph.spur(pixel_data, mask)
        elif function_name == F_THICKEN:
            return morph.thicken(pixel_data, mask)
        elif function_name == F_THIN:
            return morph.thin(pixel_data, mask)
        elif function_name == F_TOPHAT:
            return morph.white_tophat(pixel_data, mask=mask)
        elif function_name == F_VBREAK:
            return morph.vbreak(pixel_data, mask)
        else:
            raise NotImplementedError("Unimplemented morphological function: %s" %
                                      function_name)
