'''<b>Example2</b> - an image processing module
<hr>
This is the boilerplate for an image processing module. You can implement
your own filter in the "run" method.
'''
import numpy as np

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.cpimage as cpi

class Example2(cpm.CPModule):
    variable_revision_number = 1
    module_name = "Example2"
    category = "Image Processing"
    
    def create_settings(self):
        #
        # The ImageNameSubscriber knows about the images that were provided
        # by all of the previous modules and will display those to the user
        # in a drop-down choice box.
        #
        self.input_image_name = cps.ImageNameSubscriber("Input image")
        #
        # The ImageNameProvider tells CellProfiler that this module will
        # provide an image.
        #
        self.output_image_name = cps.ImageNameProvider("Output image", 
                                                       "Sharpened")
        #
        # If you have a image processing filter, there's a good chance that
        # there are some parameters such as the sigma of a Gaussian or
        # some other sort of scale. You can add those to create_settings
        # on the lines below.
        #
    def settings(self):
        #
        # Add your settings to the list below.
        #
        return [self.input_image_name, self.output_image_name]
    
    def run(self, workspace):
        image_set = workspace.image_set
        #
        # Get your image from the image set using the ImageNameProvider
        #
        image = workspace.image_set.get_image(self.input_image_name.value)
        #
        # Get the pixel data from the image. I've chosen to make a copy
        # of the pixel data for safety's sake. It's easy to inadvertantly
        # change the input image's data and that will go against the
        # expectations of your users.
        #
        pixel_data = image.pixel_data.copy()
        #
        # This is where your transformation code goes. Right here, I'm
        # creating a gradient filter in the vertical direction. I take the
        # difference between the previous and next rows in the image.
        #
        # One row will be missing here by necessity - the last row has no next
        # and the first has no previous. We'll leave that row blank.
        #
        pixel_data[:-1, :] = np.abs(pixel_data[:-1, :] - pixel_data[1:, :])
        #
        # Make a cpi.Image using the transformed pixel data
        # put your image back in the image set.
        #
        output_image = cpi.Image(pixel_data)
        image_set.add(self.output_image_name.value, output_image)
        #
        # Store the input and output images in the workspace so that
        # they can be displayed later
        #
        if workspace.show_frame:
            workspace.display_data.input_image = image.pixel_data
            workspace.display_data.output_image = pixel_data

    #
    # The display interface is changing / has changed.
    # This is a recipe to make yours work with both
    #
    def display(self, workspace, figure=None):
        if figure is None:
            figure = workspace.create_or_find_figure(subplots=(2, 1))
        else:
            figure.set_subplots((2, 1))
        figure.subplot_imshow_grayscale(
            0, 0, workspace.display_data.input_image,
            title = self.input_image_name.value)
        figure.subplot_imshow_grayscale(
            1, 0, workspace.display_data.output_image,
            title = self.output_image_name.value)        
        
    def is_interactive(self):
        return False
    