'''<b>Image Math<b/> performs simple mathematical operations on image intensities.
<hr>
<ul>
ImageMath can perform addition, subtraction, multiplication, division, or averaging
of two or more images, as well as inversion, log transform, or scaling by 
a constant for individual images.

<li><i>Average</i> calculates the mean intensity of the images loaded in the module.  
This is equivalent to the "add" option divided by the number of images loaded 
by this module.  If you would like to average many images (all of the images in 
an entire pipeline, i.e. across cycles), please use the CorrectIllumination_Calculate module 
and choose the 'All' (vs. 'Each') option.</li>

<li><i>Invert</i> subtracts the image intensities from 1. This makes the darkest
color the brightest and vice-versa.</li>

<li><i>Multiply factors</i> The final image may have a substantially different range of pixel
intensities than the originals, so each image can be multiplied by a 
factor prior to the operation. This factor can be any real number.</li>

<li><i>Do you want values in the image to be set to zero/one?</i>
Values outside the range 0 to 1 might not be handled well by other
modules. Here, you have the option of setting negative values to 0, and values greater than 1 to a maximum value of 1.
See the Rescale Intensity module for more scaling options.</li>

See also SubtractBackground, RescaleIntensity, CorrectIllumination_Calculate.
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
from contrib.english import ordinal

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
# Combine is now obsolete - done by Add now, but we need the string for upgrade_settings
O_COMBINE = "Combine"


# The number of settings per image
IMAGE_SETTING_COUNT = 2

# The number of settings other than for images
FIXED_SETTING_COUNT = 7


class ImageMath(cpm.CPModule):
    
    category = "Image Processing"
    variable_revision_number = 1
    module_name = "ImageMath"

    def create_settings(self):
        # the list of per image settings (name & scaling factor)
        self.images = []
        # create the first two images (the default number)
        self.add_image()
        self.add_image()

        # other settings
        self.operation = cps.Choice("Operation", 
                                    [O_ADD, O_SUBTRACT, O_MULTIPLY, O_DIVIDE, O_INVERT, O_LOG_TRANSFORM, O_AVERAGE, O_NONE], doc=
                                    """What operation would you like performed?""")
        self.divider_top = cps.Divider(line=False)
        self.exponent = cps.Float("Raise to exponent", 1, doc="""Enter an exponent to raise the result to *after* the chosen operation""")
        self.after_factor = cps.Float("Multiply by", 1, doc="""Enter a factor to multiply the result by *after* the chosen operation""")
        self.addend = cps.Float("Add to result", 0, doc ="""Enter a number to add to the result *after* the chosen operation""")
        self.truncate_low = cps.Binary("Set values<0 to 0?", True, doc="""Do you want negative values in the image to be set to zero?""")
        self.truncate_high = cps.Binary("Set values>1 to 1?", True, doc ="""Do you want values greater than one to be set to one?""")
        self.output_image_name = cps.ImageNameProvider("Name the output image", "ImageAfterMath", doc="""What do you want to call the resulting image?""")
        self.add_button = cps.DoSomething("Add another image","Add image", self.add_image, True)
        self.divider_bottom = cps.Divider(line=False)
    
    def add_image(self, removable=False):
        # The text for these settings will be replaced in renumber_settings()
        group = cps.SettingsGroup()
        group.append("image_name", cps.ImageNameSubscriber("", ""))
        group.append("factor", cps.Float("", 1))
        if removable:
            group.append("remover", cps.RemoveSettingButton("Remove the image above", "Remove", self.images, group))
        group.append("divider", cps.Divider())
        self.images.append(group)
        
    def renumber_settings(self):
        for idx, image in enumerate(self.images):
            image.image_name.text = "Choose %s image:"%(ordinal(idx + 1))
            image.factor.text = "Enter a factor to multiply the %s image by (before other operations):"%ordinal(idx + 1)

    def settings(self):
        result = [self.operation, self.exponent, self.after_factor, self.addend,
                  self.truncate_low, self.truncate_high, 
                  self.output_image_name]
        for image in self.images:
            result += [image.image_name, image.factor]
        return result

    def visible_settings(self):
        result = [self.operation, self.divider_top]
        self.renumber_settings()
        single_image = self.operation.value in (O_INVERT, O_LOG_TRANSFORM, O_NONE)
        for index, image in enumerate(self.images):
            if (index >= 1) and single_image:
            # these operations use the first image only
                break 
            result += image.unpack_group()

        if single_image:
            result[-1] = self.divider_bottom # this looks better when there's just one image
        else:
            result += [self.add_button, self.divider_bottom]

        result += [self.output_image_name, self.exponent, self.after_factor, 
                   self.addend, self.truncate_low, self.truncate_high]
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
        image_names = [image.image_name.value for image in self.images]
        image_factors = [image.factor.value for image in self.images]
        if self.operation.value in (O_INVERT, O_LOG_TRANSFORM, O_NONE):
            # these only operate on the first image
            image_names = image_names[:1]
            image_factors = image_factors[:1]

        images = [workspace.image_set.get_image(x) for x in image_names]
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

        #
        # Multiply images by their factors
        #
        for i in range(len(image_factors)):
            pixel_data[i] = pixel_data[i] * image_factors[i]

        output_pixel_data = pixel_data[0]
        output_mask = masks[0]

        opval = self.operation.value
        if opval in (O_ADD, O_SUBTRACT, O_MULTIPLY, O_DIVIDE, O_AVERAGE):
            # Binary operations
            if opval in (O_ADD, O_AVERAGE):
                op = np.add
            elif opval == O_SUBTRACT:
                op = np.subtract
            elif opval == O_MULTIPLY:
                op = np.multiply
            else:
                op = np.divide
            for pd, mask in zip(pixel_data[1:], masks[1:]):
                output_pixel_data = op(output_pixel_data, pd)
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
            display_pixel_data = pixel_data + [output_pixel_data]
            display_names = image_names + [self.output_image_name.value]
            columns = (len(display_pixel_data) + 1 ) / 2
            figure = workspace.create_or_find_figure(subplots=(columns, 2))
            for i in range(len(display_pixel_data)):
                if display_pixel_data[i].ndim == 3:
                    figure.subplot_imshow_color(i%columns, int(i / columns),
                                                display_pixel_data[i],
                                                title=display_names[i])
                else:
                    figure.subplot_imshow_bw(i%columns, int(i / columns),
                                             display_pixel_data[i],
                                             title=display_names[i])

    def upgrade_settings(self, setting_values, variable_revision_number, 
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
