'''
Performs simple mathematical operations on image intensities.

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

See also SubtractBackground, RescaleIntensity.'''

__version__="$Revision: 7763 $"

import numpy as np
from contrib.english import ordinal
import uuid

import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
from cellprofiler.settings import *

####################################################
# Module information
####################################################
module_name = "ImageMath"
category = "Image Processing"
variable_revision_number = 1

####################################################
# Constants for settings
####################################################
O_ADD = "Add"
O_SUBTRACT = "Subtract"
O_MULTIPLY = "Multiply"
O_DIVIDE = "Divide"
O_INVERT = "Invert"
O_LOG_TRANSFORM = "Log transform (base 2)"
O_AVERAGE = "Average"
O_NONE = "None"
# Combine is now Add, but we need this for backwards_compatibilize
O_COMBINE = "Combine"

# The number of settings per image.
IMAGE_SETTING_COUNT = 2
# How many settings are there other than per image.
FIXED_SETTING_COUNT = 7

# Global variable for the per-image settings
image_settings = []

# These functions are needed to create the initial settings below
def subscriber_question(place):
    return "Choose %s image:"%ordinal(place)

def factor_question(place):
    return "Enter a factor to multiply the %s image by (before other operations):"%ordinal(place)

def add_image(removable=True):
    settings = {}
    num = len(image_settings)
    settings['name'] = ImageNameSubscriber(subscriber_question(num + 1), "None")
    settings['factor'] = Float(factor_question(num + 1), 1)
    if removable:
        settings['remove'] = DoSomething("Remove the above image", "Remove", remove_image, settings['name'])
    image_settings.append(settings)

def remove_image(key_to_remove):
    global image_settings
    image_settings = [s for s in image_settings if s['name'] != key_to_remove]
    # adjust questions after removal
    for i, settings in enumerate(image_settings):
        settings['name'].text = subscriber_question(i + 1)
        settings['factor'].text = factor_question(i + 1)


####################################################
# Initial Settings
####################################################
operation = Choice("What operation would you like performed?",
                   [O_ADD, O_SUBTRACT, O_MULTIPLY, O_DIVIDE, O_INVERT, 
                    O_LOG_TRANSFORM, O_AVERAGE, O_NONE])
exponent = Float("Enter an exponent to raise the the result to *after* the chosen operation:", 1)
factor = Float("Enter a factor to multiply the result by *after* the chosen operation:", 1)
addend = Float("Enter a number to add to the result *after* the chosen operation:", 0)
truncate_low = Binary("Do you want negative values in the image to be set to zero?", True)
truncate_high = Binary("Do you want values greater than one to be set to one?", True)
output_image_name = ImageNameProvider("What do you want to call the resulting image?", "ImageAfterMath")
add_button = DoSomething("Add another image", "Add image", add_image)

# Add settings for two images.  Because at least two images are
# required, these fields should not be removable.
add_image(removable=False)
add_image(removable=False)

####################################################
# Module functions
####################################################
def settings():
    result = [operation, exponent, factor, addend, truncate_low, 
              truncate_high, output_image_name]
    assert len(result) == FIXED_SETTING_COUNT
    assert IMAGE_SETTING_COUNT == 2
    for settings in image_settings:
        # Only add the first two settings, not the remove button.
        result.extend([settings['name'], settings['factor']])
    return result

def visible_settings():
    result = [image_settings[0]['name'], image_settings[0]['factor'], operation]
    if operation.value not in (O_INVERT, O_LOG_TRANSFORM, O_NONE):
        # Only these operations have multiple images
        for settings in image_settings[1:]:
            result = result + [settings['name'], settings['factor'], settings['remove']]
    result += [output_image_name, exponent, factor, addend, truncate_low, truncate_high]
    return result

def prepare_to_set_values(setting_values):
    global image_settings
    value_count = len(setting_values)
    assert (value_count - FIXED_SETTING_COUNT) % IMAGE_SETTING_COUNT == 0
    image_count = (value_count - FIXED_SETTING_COUNT) / IMAGE_SETTING_COUNT
    # always keep the first two images
    del image_settings[2:]
    while len(image_settings) < image_count:
        add_image()

def run(workspace):
    if operation.value in (O_INVERT, O_LOG_TRANSFORM, O_NONE):
        # Operations on a single image
        cur_image_settings = image_settings[:1]
    else:
        cur_image_settings = image_settings
        
    image_names = [settings['name'].value for settings in cur_image_settings]
    image_factors = [settings['factor'].value for settings in cur_image_settings]
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
    
    opval = operation.value
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
    if exponent.value != 1:
        output_pixel_data **= exponent.value
    if factor.value != 1:
        output_pixel_data *= factor.value
    if addend.value != 0:
        output_pixel_data += addend.value

    #
    # truncate values
    #
    if truncate_low.value:
        output_pixel_data[output_pixel_data < 0] = 0
    if truncate_high.value:
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
    workspace.image_set.add(output_image_name.value, output_image)

    #
    # Display results
    #
    if workspace.frame is not None:
        display_pixel_data = pixel_data + [output_pixel_data]
        display_names = image_names + [output_image_name]
        columns = (len(display_pixel_data) + 1 ) / 2
        figure = workspace.create_or_find_figure(subplots=(columns, 2))
        for i in range(len(display_pixel_data)):
            if display_pixel_data[i].ndim == 3:
                figure.subplot_imshow_color(i%columns, int(i / columns),
                                            display_pixel_data[i],
                                            title=display_name[i])
            else:
                figure.subplot_imshow_bw(i%columns, int(i / columns),
                                         display_pixel_data[i],
                                         title=display_name[i])

def backwards_compatibilize(setting_values, variable_revision_number, 
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
                if setting_values[i] != DO_NOT_USE:
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

