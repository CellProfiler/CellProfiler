'''<b>Example2</b> - an image processing module
<hr>
This is the boilerplate for an image processing module.
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
        # Put your ImageNameProvider and ImageNameSubscriber here
        #
        # use self.input_image_name as the ImageNameSubscriber
        # use self.output_image_name as the ImageNameProvider
        #
        # Those are the names that are expected by the unit tests.
        #
        pass
    
    def settings(self):
        #
        # Add your ImageNameProvider and ImageNameSubscriber to the
        # settings that are returned.
        #
        return []
    
    def run(self, workspace):
        image_set = workspace.image_set
        #
        # Get your image from the image set using the ImageNameProvider
        # Get the pixel data from the image
        # Do something creative if you'd like
        # Make a cpi.Image using the transformed pixel data
        # put your image back in the image set.
        #
        # If you need help figuring out which methods to use,
        # you can always use the Python help:
        #
        # help(cpi.Image)
        # help(cpi.ImageSet)
        #
        
