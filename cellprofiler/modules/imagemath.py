'''<b>Image Math</b> performs
 simple mathematical operations on image intensities
<hr>

This module can perform addition, subtraction, multiplication, division, or averaging
of two or more image intensities, as well as inversion, log transform, or scaling by 
a constant for individual image intensities.

<i>Multiply factors:</i> The final image may have a substantially different range of pixel
intensities than the originals, so each image can be multiplied by a 
factor prior to the operation. This factor can be any real number.  
See the <b>RescaleIntensity</b> module for more scaling options.
<br>
<br>
See also <b>ApplyThreshold</b>, <b>RescaleIntensity</b>, <b>CorrectIllumination_Calculate</b>.
'''
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
from contrib.english import ordinal

import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.measurements as cpmeas

O_ADD = "Add"
O_SUBTRACT = "Subtract"
O_DIFFERENCE = "Absolute Difference"
O_MULTIPLY = "Multiply"
O_DIVIDE = "Divide"
O_AVERAGE = "Average"
O_MAXIMUM = "Maximum"
O_INVERT = "Invert"
O_COMPLEMENT = "Complement"
O_LOG_TRANSFORM = "Log transform (base 2)"
O_NONE = "None"
# Combine is now obsolete - done by Add now, but we need the string for upgrade_settings
O_COMBINE = "Combine"

IM_IMAGE = "Image"
IM_MEASUREMENT = "Measurement"

# The number of settings per image
IMAGE_SETTING_COUNT_1 = 2
IMAGE_SETTING_COUNT = 4

# The number of settings other than for images
FIXED_SETTING_COUNT_1 = 7
FIXED_SETTING_COUNT = 8


class ImageMath(cpm.CPModule):
    
    category = "Image Processing"
    variable_revision_number = 3
    module_name = "ImageMath"

    def create_settings(self):
        # the list of per image settings (name & scaling factor)
        self.images = []
        # create the first two images (the default number)
        self.add_image(False)
        self.add_image(False)

        # other settings
        self.operation = cps.Choice("Operation", 
                                    [O_ADD, O_SUBTRACT, O_DIFFERENCE, O_MULTIPLY, O_DIVIDE, O_AVERAGE, O_MAXIMUM, O_INVERT, O_LOG_TRANSFORM, O_NONE], doc=
            """What operation would you like performed?
                        
            <p><i>Note:</i> If more than 2 images are chosen, then operations will be 
            performed sequentially from first to last, e.g., for "Divide", (Image1 / Image2) / Image3
            
            <ul><li><i>Add</i> adds the first image to the second, and so on.

            <li><i>Subtract</i> subtracts the second image from the first.
            
            <li><i>Multiply</i> multiplies the first image by the second.

            <li><i>Divide </i> divides the first image by the second.
            
            <li><i>Average</i> calculates the mean intensity of the images loaded in the module.  
            This is equivalent to the Add option divided by the number of images loaded 
            by this module.  If you would like to average all of the images in 
            an entire pipeline, i.e., across cycles, you should instead use the <b>CorrectIlluminationCalculate</b> module 
            and choose the <i>All</i> (vs. <i>Each</i>) option.</li>
            
            <li><i>Maximum </i> returns the element-wise maximum value at each pixel location.            
            
            <li><i>Invert</i> subtracts the image intensities from 1. This makes the darkest
            color the brightest and vice-versa.</li>

            <li><i>Log transform (base 2)</i> log transforms each pixel's intensity. </li>

            <li><i>None</i> is useful if you simply want to select some of the later options in the module, such as adding,
            multiplying, or exponentiating your image by a constant.</li>
            
            </ul> <p>Note that <i>Invert</i>, <i>Log Transform</i>, and <i>None</i> operate on only a single image.
            """)
        self.divider_top = cps.Divider(line=False)
        
        self.exponent = cps.Float("Raise the power of the result by", 1, doc="""Enter an exponent to raise the result to *after* the chosen operation""")
        
        self.after_factor = cps.Float("Multiply the result by", 1, doc="""Enter a factor to multiply the result by *after* the chosen operation""")
        
        self.addend = cps.Float("Add to result", 0, doc ="""Enter a number to add to the result *after* the chosen operation""")
        
        self.truncate_low = cps.Binary("Set values less than 0 equal to 0?", True, doc="""
            Do you want negative values to be set to 0?
            Values outside the range 0 to 1 might not be handled well by other modules. 
            Here you have the option of setting negative values to 0.""")
        
        self.truncate_high = cps.Binary("Set values greater than 1 equal to 1?", True, doc ="""
            Do you want values greater than one to be set to 1?
            Values outside the range 0 to 1 might not be handled well by other modules. 
            Here you have the option of setting values greater than 1 to a maximum value of 1.""")
        
        self.ignore_mask = cps.Binary("Ignore the image masks?", False, doc = """Usually, the smallest mask of all image operands is applied after image math has been completed. Choosing to ignore the masks will set equal to zero all previously masked pixels and operate on the masked images as if no mask had been applied.""")
        self.output_image_name = cps.ImageNameProvider("Name the output image", "ImageAfterMath", doc="""What do you want to call the resulting image?""")
        
        self.add_button = cps.DoSomething("", "Add another image", self.add_image)
        
        self.divider_bottom = cps.Divider(line=False)
    
    def add_image(self, removable=True):
        # The text for these settings will be replaced in renumber_settings()
        group = cps.SettingsGroup()
        group.removable = removable
        group.append("image_or_measurement", cps.Choice(
            "Image or measurement?", [IM_IMAGE, IM_MEASUREMENT],
            doc="""You can perform math operations using two images or you
            can use a measurement for one of the operands. For instance,
            to divide the intensity of one image by another, choose Image
            for both and pick the respective images. To divide the intensity
            of an image by its median intensity, use <b>MeasureImageIntensity</b>
            prior to this module to calculate the median intensity, then
            select "Measurement" and use the median intensity measurement as
            the denominator"""))
        group.append("image_name", cps.ImageNameSubscriber("", "",doc="""Which image do you want to use for this operation?"""))
        group.append("measurement", cps.Measurement(
            "Measurement", lambda : cpmeas.IMAGE,"",
            doc="""This is a measurement made on the image. The value of the
            measurement is used for the operand for all of the pixels of the
            other operand's image."""))
        group.append("factor", cps.Float("", 1,doc="""By what number would you like to multiply the above image? This multiplication
                is applied before other operations."""))
        if removable:
            group.append("remover", cps.RemoveSettingButton("", "Remove this image", self.images, group))
        group.append("divider", cps.Divider())
        self.images.append(group)
        
    def renumber_settings(self):
        for idx, image in enumerate(self.images):
            image.image_name.text = "Select the %s image"%(ordinal(idx + 1))
            image.factor.text = "Multiply the %s image by"%ordinal(idx + 1)

    def settings(self):
        result = [self.operation, self.exponent, self.after_factor, self.addend,
                  self.truncate_low, self.truncate_high, self.ignore_mask,
                  self.output_image_name]
        for image in self.images:
            result += [image.image_or_measurement, image.image_name, 
                       image.factor, image.measurement]
        return result

    @property
    def operand_count(self):
        '''# of operands, taking the operation into consideration'''
        if self.operation.value in (O_INVERT, O_LOG_TRANSFORM, O_NONE):
            return 1
        return len(self.images)
    
    def visible_settings(self):
        result = [self.operation, self.output_image_name, self.divider_top]
        self.renumber_settings()
        single_image = self.operand_count == 1
        for index in range(self.operand_count):
            image = self.images[index]
            if single_image:
                result += [image.image_name, image.factor]
            else:
                result += [image.image_or_measurement,
                           image.image_name 
                           if image.image_or_measurement == IM_IMAGE
                           else image.measurement, image.factor]
            if image.removable:
                result += [image.remover]
            result += [image.divider]

        if single_image:
            result[-1] = self.divider_bottom # this looks better when there's just one image
        else:
            result += [self.add_button, self.divider_bottom]

        result += [self.exponent, self.after_factor, 
                   self.addend, self.truncate_low, self.truncate_high, self.ignore_mask]
        return result

    def help_settings(self):
        result = [self.operation, self.output_image_name, ]
        for image in self.images:
            result += [image.image_or_measurement, image.image_name, 
                       image.measurement, image.factor]
        result += [self.exponent, self.after_factor, self.addend,
                  self.truncate_low, self.truncate_high, self.ignore_mask]
        return result
    
    def prepare_settings(self, setting_values):
        value_count = len(setting_values)
        assert (value_count - FIXED_SETTING_COUNT) % IMAGE_SETTING_COUNT == 0
        image_count = (value_count - FIXED_SETTING_COUNT) / IMAGE_SETTING_COUNT
        # always keep the first two images
        del self.images[2:]
        while len(self.images) < image_count:
            self.add_image()


    def run(self, workspace):
        image_names = [image.image_name.value for image in self.images
                       if image.image_or_measurement == IM_IMAGE]
        image_factors = [image.factor.value for image in self.images]
        wants_image = [image.image_or_measurement == IM_IMAGE
                       for image in self.images]
        if self.operation.value in (O_INVERT, O_LOG_TRANSFORM, O_NONE):
            # these only operate on the first image
            image_names = image_names[:1]
            image_factors = image_factors[:1]

        images = [workspace.image_set.get_image(x)
                  for x in image_names]
        pixel_data = [image.pixel_data for image in images]
        masks = [image.mask if image.has_mask else None for image in images]
        #
        # Crop all of the images similarly
        #
        smallest = np.argmin([np.product(pd.shape) for pd in pixel_data])
        smallest_image = images[smallest]
        for i in [x for x in range(len(images)) if x != smallest]:
            pixel_data[i] = smallest_image.crop_image_similarly(pixel_data[i])
            if masks[i] is not None:
                masks[i] = smallest_image.crop_image_similarly(masks[i])
        # weave in the measurements
        idx = 0
        measurements = workspace.measurements
        assert isinstance(measurements, cpmeas.Measurements)
        for i in range(self.operand_count):
            if not wants_image[i]:
                value = measurements.get_current_image_measurement(
                    self.images[i].measurement.value)
                pixel_data.insert(i, value)
                masks.insert(i, True)
        #
        # Multiply images by their factors
        #
        for i in range(len(image_factors)):
            pixel_data[i] = pixel_data[i] * image_factors[i]

        output_pixel_data = pixel_data[0]
        output_mask = masks[0]

        opval = self.operation.value
        if opval in (O_ADD, O_SUBTRACT, O_DIFFERENCE, O_MULTIPLY, O_DIVIDE, O_AVERAGE, O_MAXIMUM):
            # Binary operations
            if opval in (O_ADD, O_AVERAGE):
                op = np.add
            elif opval == O_SUBTRACT:
                op = np.subtract
            elif opval == O_DIFFERENCE:
                op = lambda x, y: np.abs(np.subtract(x, y))
            elif opval == O_MULTIPLY:
                op = np.multiply
            elif opval == O_MAXIMUM:
                op = np.maximum
            else:
                op = np.divide
            for pd, mask in zip(pixel_data[1:], masks[1:]):
                output_pixel_data = op(output_pixel_data, pd)
                if self.ignore_mask == True:
                    continue
                else:
                    if output_mask is None:
                        output_mask = mask
                    elif mask is not None:
                        output_mask = (output_mask & mask)
            if opval == O_AVERAGE:
                output_pixel_data /= sum(image_factors)
        elif opval == O_INVERT:
            output_pixel_data = 1 - output_pixel_data 
        elif opval == O_LOG_TRANSFORM:
            output_pixel_data = np.log2(output_pixel_data)
        elif opval == O_NONE:
            pass
        else:
            raise NotImplementedException("The operation %s has not been implemented"%opval)

        # Check to see if there was a measurement & image w/o mask. If so
        # set mask to none
        if np.isscalar(output_mask):
            output_mask = None
        #
        # Post-processing: exponent, multiply, add
        #
        if self.exponent.value != 1:
            output_pixel_data **= self.exponent.value
        if self.after_factor.value != 1:
            output_pixel_data *= self.after_factor.value
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
        # add the output image to the workspace
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

        #
        # Display results
        #
        if workspace.frame is not None:
            display_pixel_data = [image.pixel_data for image in images] + [output_pixel_data]
            display_names = image_names + [self.output_image_name.value]
            columns = (len(display_pixel_data) + 1 ) / 2
            figure = workspace.create_or_find_figure(title="ImageMath, image cycle #%d"%(
                workspace.measurements.image_set_number),subplots=(columns, 2))
            for i in range(len(display_pixel_data)):
                if display_pixel_data[i].ndim == 3:
                    figure.subplot_imshow(i%columns, int(i / columns),
                                          display_pixel_data[i],
                                          title=display_names[i],
                                          sharex = figure.subplot(0,0),
                                          sharey = figure.subplot(0,0))
                else:
                    figure.subplot_imshow_bw(i%columns, int(i / columns),
                                             display_pixel_data[i],
                                             title=display_names[i],
                                             sharex = figure.subplot(0,0),
                                             sharey = figure.subplot(0,0))

    def validate_module(self, pipeline):
        '''Guarantee that at least one operand is an image'''
        for i in range(self.operand_count):
            op = self.images[i]
            if op.image_or_measurement == IM_IMAGE:
                return
        raise cps.ValidationError("At least one of the operands must be an image",
                                  op.image_or_measurement)
    
    def upgrade_settings(self, setting_values, variable_revision_number, 
                         module_name, from_matlab):
        if (from_matlab and module_name == 'Subtract' and 
            variable_revision_number == 3):
            subtract_image_name, basic_image_name, resulting_image_name,\
            multiply_factor_1, multiply_factor_2, truncate = setting_values
            setting_values = [ O_SUBTRACT,
                               1, # exponent
                               1, # post-multiply factor
                               0, # addend
                               truncate, # truncate low
                               cps.NO, # truncate high
                               resulting_image_name,
                               basic_image_name,
                               multiply_factor_2,
                               subtract_image_name,
                               multiply_factor_1]
            module_name = 'ImageMath'
            from_matlab = False
            variable_revision_number = 1
        if (from_matlab and module_name == 'Combine' and
            variable_revision_number == 3):
            names_and_weights = [ 
                (name, weight)
                for name, weight in zip(setting_values[:3],
                                        setting_values[4:])
                if name.lower() != cps.DO_NOT_USE.lower()]
            
            multiplier = 1.0/sum([float(weight) 
                                  for name, weight in names_and_weights])
            output_image = setting_values[3]
            setting_values = [O_ADD,  # Operation
                              "1",    # Exponent
                              str(multiplier),    # Post-operation multiplier
                              "0",    # Post-operation offset
                              cps.NO, # Truncate low
                              cps.NO, # Truncate high
                              output_image]
            for name, weight in names_and_weights:
                setting_values += [name, weight]
            module_name = 'ImageMath'
            variable_revision_number = 1
            from_matlab = False
        if (from_matlab and module_name == 'InvertIntensity' and
            variable_revision_number == 1):
            image_name, output_image = setting_values
            setting_values = [image_name, cps.DO_NOT_USE, cps.DO_NOT_USE,
                              'Invert',
                              1,1,1,1,1,cps.NO,cps.NO, output_image]
            module_name = 'ImageMath'
            variable_revision_number = 2
        if (from_matlab and module_name == 'Multiply' and 
            variable_revision_number == 1):
            image1, image2, output_image = setting_values
            setting_values = [image1, image2, cps.DO_NOT_USE,
                              'Multiply', 1,1,1,1,1,cps.NO, cps.NO,
                              output_image]
            module_name = 'ImageMath'
            variable_revision_number = 2
            
        if (from_matlab and variable_revision_number == 1 and
            module_name == 'ImageMath'):
            image_names = [setting_values[1]]
            input_factors = [float(setting_values[4])]
            operation = setting_values[3]
            factors = []
            # The user could type in a constant for the second image name
            try:
                factors += [float(setting_values[2]) *
                            float(setting_values[5])]    
            except:
                if setting_values[2] != cps.DO_NOT_USE:
                    image_names += [setting_values[2]]  
                    input_factors += [float(setting_values[5])]
            exponent = 1.0
            multiplier = 1.0
            addend = 0
            wants_truncate_low = setting_values[6]
            wants_truncate_high = setting_values[7]
            output_image_name = setting_values[0]
            old_operation = operation
            if operation == O_DIVIDE and len(factors):
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
            if operation == O_COMPLEMENT:
                image_names = image_names[:1]
                input_factors = input_factors[:1]
            elif old_operation in (O_ADD, O_SUBTRACT, O_MULTIPLY, O_DIVIDE):
                if len(image_names) < 2:
                    setting_values[0] = O_NONE
                image_names = image_names[:2]
                input_factors = input_factors[:2]
            for image_name, input_factor in zip(image_names, input_factors):
                setting_values += [image_name, input_factor]
            from_matlab = False
            variable_revision_number = 1
            
        if (from_matlab and variable_revision_number == 2 and
            module_name == 'ImageMath'):
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
                # Fix for variable_revision_number 2: subtract reversed operands
                if old_operation == O_SUBTRACT:
                    image_names.reverse()
                    input_factors.reverse()
            for image_name, input_factor in zip(image_names, input_factors):
                setting_values += [image_name, input_factor]
            from_matlab = False
            variable_revision_number = 1
        if (not from_matlab) and variable_revision_number == 1:
            # added image_or_measurement and measurement
            new_setting_values = setting_values[:FIXED_SETTING_COUNT_1]
            for i in range(FIXED_SETTING_COUNT_1, len(setting_values),
                           IMAGE_SETTING_COUNT_1):
                new_setting_values += [IM_IMAGE, setting_values[i],
                                       setting_values[i+1], ""]
            setting_values = new_setting_values
            variable_revision_number = 2
        if (not from_matlab) and variable_revision_number == 2:
            # added the ability to ignore the mask
            new_setting_values = setting_values
            new_setting_values.insert(6, 'No')
            setting_values = new_setting_values
            variable_revision_number = 3
        return setting_values, variable_revision_number, from_matlab
