'''
Inverts Fluorescent-looking images into brightfield-looking images.

This module turns a single or multi-channel immunofluorescent-stained
image into an image that resembles a brightfield image stained with
similarly- colored stains, which generally prints better.
    
You have the option of combining up to three grayscale images
(representing the red, green and blue channels of a color image) or of
operating on a single color image. The module can produce either three
grayscale images or one color image on output.
'''

__version__ = "$Revision: 8009 $"

verbose_name = 'Invert for printing'
category = 'Image Processing'

import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps

CC_GRAYSCALE = "Grayscale"
CC_COLOR = "Color"

# Input settings
input_color_choice = Choice("Do you want to combine grayscale images or load a single color image?",
                            [CC_COLOR, CC_GRAYSCALE])
wants_red_input = Binary("Do you want to load an image for the red channel?",
                             True)
red_input_image = ImageNameSubscriber("What did you call the red image?",
                                          "None")
wants_green_input = Binary("Do you want to load an image for the green channel?",
                               True)
green_input_image = ImageNameSubscriber("What did you call the green image?", 
                                        "None")
wants_blue_input = Binary("Do you want to load an image for the blue channel?", 
                          True)
blue_input_image = ImageNameSubscriber("What did you call the blue image?", 
                                       "None")
color_input_image = cps.ImageNameSubscriber("What did you call the color image?", 
                                            "None")

# Output settings
output_color_choice = Choice("Do you want to produce several grayscale images or one color image?",
                             [CC_COLOR, CC_GRAYSCALE])
wants_red_output = Binary("Do you want to produce an image for the red channel?",
                          True)
red_output_image = ImageNameProvider("What do you want to call the red image?",
                                     "InvertedRed")
wants_green_output = Binary("Do you want to produce an image for the green channel?",
                                True)
green_output_image = ImageNameProvider("What do you want to call the green image?",
                                           "InvertedGreen")
wants_blue_output = Binary("Do you want to produce an image for the blue channel?", 
                           True)
blue_output_image = ImageNameProvider("What do you want to call the blue image?", 
                                          "InvertedBlue")
color_output_image = ImageNameProvider("What do you want to call the inverted color image?",
                                           "InvertedColor")

    
def convert_settings_from_old_versions(setting_values, 
                                       variable_revision_number, module_name, 
                                       from_matlab):
    if from_matlab and variable_revision_number == 1:
        setting_values = dict(
            input_color_choice=CC_GRAYSCALE,
            CC_GRAYSCALE,                # input_color_choice
            setting_values[0] != 'None', # wants_red_input
            setting_values[0],           # red_input_image
            setting_values[1] != 'None',
            setting_values[1],
            setting_values[2] != 'None',
            setting_values[2],
            'None',                      # color
            CC_GRAYSCALE,                # output_color_choice
            setting_values[3] != 'None',
            setting_values[3],
            setting_values[4] != 'None',
            setting_values[4],
            setting_values[5] != 'None',
            setting_values[5],
            'InvertedColor']
        from_matlab = False
        variable_revision_number = 1

    return setting_values, variable_revision_number, from_matlab
        
def visible_settings:
    '''Return a list of the names of settings to display in the user interface.'''
    result = [input_color_choice]
    if input_color_choice.value.is(CC_GRAYSCALE):
        result.append(wants_red_input)
        if wants_red_input.is_yes:
            results.append(red_input_image)
        result.append(wants_green_input)
        if wants_green_input.is_yes:
            results.append(green_input_image)
        result.append(wants_blue_input)
        if wants_blue_input.is_yes:
            results.append(blue_input_image)
    else:
        result.append(color_input_image)
    result.append(output_color_choice)
    if output_color_choice.is(CC_GRAYSCALE):
        result.append(wants_red_output)
        if wants_red_output.is_yes:
            results.append(red_output_image)
        result.append(wants_green_output)
        if wants_green_output.is_yes.is_yes:
            results.append(green_output_image)
        result.append(wants_blue_output)
        if wants_blue_output.is_yes:
            results.append(blue_output_image)
    else:
        result.append(color_output_image)
    return result

def validate_module(pipeline):
    '''Make sure the user has at least one of the grayscale boxes checked'''
    if (input_color_choice.is(CC_GRAYSCALE) and
        wants_red_input.value.is_no and
        wants_green_input.value.is_no and
        wants_blue_input.value.is_no):
        raise cps.ValidationError("You must supply at least one grayscale input",
                                  wants_red_input)
        
def run(workspace):
    image_set = workspace.image_set
    assert isinstance(image_set, cpi.ImageSet)

    def input_images_grayscale():
        red_image = green_image = blue_image = None
        if wants_red_input.is_yes:
            red_image = image_set.get_image(red_input_image.value,
                                            must_be_grayscale=True).pixel_data
            shape = red_image.shape
        if wants_green_input.is_yes:
            green_image = image_set.get_image(green_input_image.value,
                                              must_be_grayscale=True).pixel_data
            shape = green_image.shape
        if wants_blue_input.is_yes:
            blue_image = image_set.get_image(blue_input_image.value,
                                             must_be_grayscale=True).pixel_data
            shape = blue_image.shape
        return (red_image or np.zeros(shape),
                green_image or np.zeros(shape),
                blue_image or np.zeros(shape))

    def input_images_color():
        color_image = image_set.get_image(color_input_image.value,
                                          must_be_color=True).pixel_data
        return color_image[:,:,0], color_image[:,:,1], color_image[:,:,2]

    if input_color_choice.is(CC_GRAYSCALE):
            red, green, blue = input_images_grayscale()
    elif input_color_choice.is(CC_COLOR):
            red, green, blue, input_images_color()
    else:
        raise ValueError("Unimplemented color choice: %s" %
                         input_color_choice.value)

    inverted_red = (1 - green) * (1 - blue)
    inverted_green = (1 - red) * (1 - blue)
    inverted_blue = (1 - red) * (1 - green)
    inverted_color = np.dstack((inverted_red, inverted_green, inverted_blue))

    if output_color_choice.is(CC_GRAYSCALE):
        if wants_red_output.is_yes:
            image_set.add(red_output_image.value, cpi.Image(inverted_red))
        if wants_green_output.is_yes:
            image_set.add(green_output_image.value, cpi.Image(inverted_green))
        if wants_blue_output.is_yes:
            image_set.add(blue_output_image.value, cpi.Image(inverted_blue))
    elif output_color_choice.is(CC_COLOR):
        image_set.add(color_output_image.value, cpi.Image(inverted_color))
    else:
        raise ValueError("Unimplemented color choice: %s" %
                         output_color_choice.value)

def display(workspace):
    figure = workspace.create_or_find_figure(subplots=(2, 1))
    color = np.dstack((red, green, blue))
    figure.subplot_imshow_color(0, 0, color, "Original image")
    figure.subplot_imshow_color(1, 0, inverted_color, "Color-inverted image")
