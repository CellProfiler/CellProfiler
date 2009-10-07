'''imagemath.py - the ImageMath module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version__="$Revision$"

import numpy as np
from contrib.english import ordinal
import uuid

import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps

O_ADD = "Add"
O_SUBTRACT = "Subtract"
O_MULTIPLY = "Multiply"
O_DIVIDE = "Divide"
O_INVERT = "Invert"
O_LOG_TRANSFORM = "Log transform (base 2)"
O_AVERAGE = "Average"
O_NONE = "None"
'''Combine is now obsolete - done by Add now'''
O_COMBINE = "Combine"
O_ALL = [O_ADD, O_SUBTRACT, O_MULTIPLY, O_DIVIDE, O_INVERT, O_LOG_TRANSFORM,
         O_AVERAGE, O_NONE]

'''The number of settings aside from the image settings'''
FIXED_SETTING_COUNT = 7

'''The number of settings per image'''
IMAGE_SETTING_COUNT = 2

class ImageMath(cpm.CPModule):
    '''SHORT DESCRIPTION:
Performs simple mathematical operations on image intensities.
*************************************************************************

Operation:

Average in the ImageMath module is the numerical average of the 
images loaded in the module.  If you would like to average many images 
(all of the images in an entire pipeline), please use the 
CorrectIllumination_Calculate module and chose the option 
"(For 'All' mode only) What do you want to call the averaged image (prior 
to dilation or smoothing)? (This is an image produced during the 
calculations - it is typically not needed for downstream modules)"
This will be an average over all images.

Invert subtracts the image intensities from 1. This makes the darkest
color the lightest and vice-versa.

Multiply factors:
The final image may have a substantially different range of pixel
intensities than the originals, so each image can be multiplied by a 
factor prior to the operation. This factor can be any real number.

Do you want values in the image to be set to zero/one?:
Values outside the range of 0 to 1 might not be handled well by other
modules. Here, you have the option of setting negative values to 0.
For other options (e.g. setting values over 1 to equal 1), see the
Rescale Intensity module.

See also SubtractBackground, RescaleIntensity.
'''
    category = "Image Processing"
    variable_revision_number = 1
    def create_settings(self):
        self.module_name = "ImageMath"
        self.images = []
        self.add_image(False)
        self.add_image(False)
        self.operation = cps.Choice("What operation would you like performed?",
                                    O_ALL)
        self.exponent = cps.Float("Enter an exponent to raise the the result to *after* the chosen operation:", 1)
        self.factor = cps.Float("Enter a factor to multiply the result by *after* the chosen operation:", 1)
        self.addend = cps.Float("Enter a number to add to the result *after* the chosen operation:", 0)
        self.truncate_low = cps.Binary("Do you want negative values in the image to be set to zero?", True)
        self.truncate_high = cps.Binary("Do you want values greater than one to be set to one?", True)
        self.output_image_name = cps.ImageNameProvider("What do you want to call the resulting image?", "ImageAfterMath")
        self.add_button = cps.DoSomething("Add another image","Add image",
                                          self.add_image)
    
    def add_image(self, can_remove=True):
        class IMImage(object):
            '''The settings needed for an image'''
            def __init__(self, images, can_remove):
                self.key = uuid.uuid4()
                self.divider = cps.Divider()
                def remove_image(images = images):
                    index = [x.key for x in images].index(self.key)
                    del images[index]
                    for i,image in zip(range(len(images)), images):
                        image.image_name.text = self.subscriber_question(i+1)
                        image.factor.text = self.factor_question(i+1)
                
                self.__image_name = cps.ImageNameSubscriber(self.subscriber_question(len(images)+1),
                                                          "None")
                self.__factor = cps.Float(self.factor_question(len(images)+1), 1)
                self.can_remove = can_remove
                if can_remove:
                    self.remove_button = cps.DoSomething("Remove the above image",
                                                         "Remove",
                                                         remove_image)
            def subscriber_question(self, place):
                '''The text accompanying an image name'''
                return "Choose %s image:"%ordinal(place)
            
            def factor_question(self, place):
                '''the text accompanying the multiplicative factor'''
                return "Enter a factor to multiply the %s image by (before other operations):"%ordinal(place)
            
            def settings(self):
                return [self.image_name, self.factor]
            
            def visible_settings(self):
                if can_remove:
                    return [self.image_name, self.factor, self.remove_button,
                            self.divider]
                else:
                    return [self.image_name, self.factor, self.divider]
            
            @property
            def image_name(self):
                '''The name of the image to retrieve'''
                return self.__image_name
            
            @property
            def factor(self):
                '''Multiply the image by this much before processing'''
                return self.__factor
        self.images.append(IMImage(self.images,can_remove))

    def prepare_to_set_values(self, setting_values):
        value_count = len(setting_values)
        assert (value_count - FIXED_SETTING_COUNT) % IMAGE_SETTING_COUNT == 0
        image_count = (value_count - FIXED_SETTING_COUNT) / IMAGE_SETTING_COUNT
        while len(self.images) > image_count:
            del self.images[-1]
        
        while len(self.images) < image_count:
            self.add_image()

    def settings(self):
        result = [self.operation, self.exponent, self.factor, self.addend,
                  self.truncate_low, self.truncate_high, 
                  self.output_image_name]
        for image in self.images:
            result += image.settings()
        return result

    def backwards_compatibilize(self, setting_values, variable_revision_number, 
                                module_name, from_matlab):
        if from_matlab and variable_revision_number == 2:
            image_names = [setting_values[0]]
            input_factors = [float(setting_values[4])]
            operation = setting_values[3]
            factors = []
            for i in range(1,3 if operation == O_COMBINE else 2):
                # The user could type in a constant for the second or third image name
                try:
                    factors += [float(setting_values[i]) *
                                float(setting_values[i+4])]
                               
                except:
                    if setting_values[i] != cps.DO_NOT_USE:
                        image_names += [setting_values[i]]  
                        input_factors += [float(setting_values[i+4])]

            exponent = float(setting_values[7])
            multiplier = float(setting_values[8])
            addend = 0
            wants_truncate_low = setting_values[9]
            wants_truncate_high = setting_values[10]
            output_image_name = setting_values[11]
            old_operation = operation
            if operation == O_COMBINE:
                addend = np.sum(factors)
                operation = O_ADD
            elif operation == O_DIVIDE and len(factors):
                multiplier /= np.product(factors)
            elif operation == O_MULTIPLY and len(factors):
                multiplier *= np.product(factors)
            elif operation == O_ADD and len(factors):
                addend = np.sum(factors)
            elif operation == O_SUBTRACT:
                addend = -np.sum(factors)
            setting_values = [ operation, exponent, multiplier, addend,
                              wants_truncate_low, wants_truncate_high,
                              output_image_name]
            if operation in (O_INVERT, O_LOG_TRANSFORM):
                image_names = image_names[:1]
                input_factors = input_factors[:1]
            elif old_operation in (O_ADD, O_SUBTRACT, O_MULTIPLY, O_DIVIDE,
                                   O_AVERAGE):
                if len(image_names) < 2:
                    setting_values[0] = O_NONE
                image_names = image_names[:2]
                input_factors = input_factors[:2]
            for image_name, input_factor in zip(image_names, input_factors):
                setting_values += [image_name, input_factor]
            from_matlab = False
            variable_revision_number = 1
        return setting_values, variable_revision_number, from_matlab
     
    def visible_settings(self):
        result = self.images[0].visible_settings()
        result += [self.operation]
        if self.operation not in (O_INVERT, O_LOG_TRANSFORM, O_NONE):
            for image in self.images[1:]:
                result += image.visible_settings()
            result += [self.add_button]
        result += [self.output_image_name, self.exponent, self.factor, 
                   self.addend, self.truncate_low, self.truncate_high]
        return result

    def run(self, workspace):
        if self.operation in (O_INVERT, O_LOG_TRANSFORM, O_NONE):
            im_images = self.images[:1]
        else:
            im_images = self.images
        
        images = [workspace.image_set.get_image(x.image_name.value)
                  for x in im_images]
        pixel_data = [image.pixel_data for image in images]
        masks = [image.mask if image.has_mask else None
                 for image in images]
        #
        # Crop all of the images similarly
        #
        smallest = np.argmin([np.product(pd.shape) for pd in pixel_data])
        smallest_image = images[smallest]
        for i in [x for x in range(len(images)) if x != smallest]:
            pixel_data[i] = smallest_image.crop_image_similarly(pixel_data[i])
            if masks[i] is not None:
                masks[i] = smallest_image.crop_image_similarly(masks[i])
        #
        # Multiply them by their factors
        #
        image_factors = [im_image.factor.value for im_image in im_images]
        for i in range(len(im_images)):
            pixel_data[i] *= image_factors[i]
        
        output_pixel_data = pixel_data[0]
        output_mask = masks[0]
        if self.operation in (O_ADD, O_SUBTRACT, O_MULTIPLY, O_DIVIDE,
                              O_AVERAGE):
            # Binary operations
            if self.operation in (O_ADD, O_AVERAGE):
                operation = np.add
            elif self.operation == O_SUBTRACT:
                operation = np.subtract
            elif self.operation == O_MULTIPLY:
                operation = np.multiply
            else:
                operation = np.divide
            for pd, mask in zip(pixel_data[1:], masks[1:]):
                output_pixel_data = operation(output_pixel_data, pd)
                if output_mask is None:
                    output_mask = mask
                elif mask is not None:
                    output_mask = (output_mask & mask)
            if self.operation == O_AVERAGE:
                output_pixel_data /= sum(image_factors)
        elif self.operation == O_INVERT:
            output_pixel_data = 1 - output_pixel_data 
        elif self.operation == O_LOG_TRANSFORM:
            output_pixel_data = np.log2(output_pixel_data)
        elif self.operation == O_NONE:
            pass
        else:
            raise NotImplementedException("The operation %s has not been implemented"%self.operation.value)
        #
        # Post-processing: exponent, multiply, add
        #
        if self.exponent.value != 1:
            output_pixel_data **= self.exponent.value
        if self.factor.value != 1:
            output_pixel_data *= self.factor.value
        if self.addend.value != 0:
            output_pixel_data += self.addend.value
        #
        # truncate values
        #
        if self.truncate_low.value:
            output_pixel_data[output_pixel_data < 0] = 0
        if self.truncate_high.value:
            output_pixel_data[output_pixel_data > 1] = 1
        #
        # add the image
        #
        crop_mask = (smallest_image.crop_mask 
                     if smallest_image.has_crop_mask else None)
        masking_objects = (smallest_image.masking_objects 
                           if smallest_image.has_masking_objects else None)
        output_image = cpi.Image(output_pixel_data,
                                 mask = output_mask,
                                 crop_mask = crop_mask, 
                                 parent_image = images[0],
                                 masking_objects = masking_objects)
        workspace.image_set.add(self.output_image_name.value, output_image)
        if workspace.frame is not None:
            display_pixel_data = [image.pixel_data for image in images]
            display_name = [im.image_name.value for im in im_images]
            display_pixel_data.append(output_pixel_data)
            display_name.append(self.output_image_name.value)
            columns = (len(display_pixel_data) +1)/2
            figure = workspace.create_or_find_figure(subplots=(columns, 2))
            for i in range(len(display_pixel_data)):
                if display_pixel_data[i].ndim == 3:
                    figure.subplot_imshow_color(i%columns, int(i/columns),
                                                display_pixel_data[i],
                                                title=display_name[i])
                else:
                    figure.subplot_imshow_bw(i%columns, int(i/columns),
                                             display_pixel_data[i],
                                             title=display_name[i])
                    