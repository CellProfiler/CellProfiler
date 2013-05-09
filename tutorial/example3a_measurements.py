'''<b>Example3</b> boilerplate for measurement exercises
<hr>
'''

import numpy as np

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.settings as cps

'''This is the measurement category (you shouldn't change to Example3a)'''
C_EXAMPLE3 = "Example3"

'''This is the name of the feature'''
FTR_VARIANCE = "Variance"

class Example3a(cpm.CPModule):
    variable_revision_number = 1
    module_name = "Example3a"
    category = "Measurement"
    
    def create_settings(self):
        self.input_image_name = cps.ImageNameSubscriber("Input image")
        
    def settings(self):
        return [self.input_image_name]
    
    def run(self, workspace):
        image_set = workspace.image_set
        m = workspace.measurements
        image = image_set.get_image(self.input_image_name.value,
                                    must_be_grayscale=True)
        pixel_data = image.pixel_data
        #
        # Calculate the variance here and store it
        # in the measurements using add_measurement.
        #
        # Use cpmeas.IMAGE as the object name
        #
        # The code below makes use of an image's mask if it exists. The
        # "Crop" and "MaskImage" modules add masks to images to delineate
        # the image's region of interest. If the image has a mask,
        # you should honor that by only processing the pixels of the image
        # whose value in the mask is "True"
        #
        ##if image.has_mask:
            #
            # Use mask indexing (http://docs.scipy.org/doc/numpy/user/basics.indexing.html#boolean-or-mask-index-arrays)
            # to only take the variance of the masked region. The variance
            # will be "not a number" or np.NaN if all or all but one pixel
            # is masked out. That's OK.
            # 
        ##    variance = np.var(pixel_data[image.mask])
        ##else:
        ##    variance = np.var(pixel_data)
        #
        # Add the measurement. self.get_feature_name() returns a feature
        # name to use, e.g. "Example3_Variance_DNA" if the input image name
        # is "DNA".
        #
        # An alternative way of doing this (only in the latest versions of CP):
        #
        # m[cpmeas.IMAGE, self.get_feature_name()] = variance.
        #
        ##m.add_measurement(cpmeas.IMAGE, self.get_feature_name(), variance)
        #
        # Save the variance in the workspace display data
        #
        if workspace.show_frame:
            workspace.display_data.variance = variance
            
    def is_interactive(self):
        return True
    
    def display(self, workspace, frame=None):
        if frame is not None:
            #
            # New style: tell the figure frame that we want to break the frame
            # into a 1 x 1 grid of axes.
            #
            # Use the *new* version of subplot table to make a table that's
            # much prettier than the old one.
            #
            frame.set_subplots((1,1))
        else:
            #
            # The old version
            #
            frame = workspace.create_or_find_figure(subplots=(1,1))
        frame.subplot_table(
            0, 0, [[ "Value = %f" % workspace.display_data.variance]])
        
    def get_feature_name(self):
        '''Return the name to be used to store the feature
        
        Returns CATEGORY_FEATURE_IMAGENAME
        where CATEGORY is Example3
              FEATURE is Variance
              IMAGENAME is the name of the input image.
        '''
        return "_".join([C_EXAMPLE3, FTR_VARIANCE, self.input_image_name.value])
        
    def get_measurement_columns(self, pipeline):
        #
        # Return a list of one tuple - that tuple should have
        # cpmeas.IMAGE as it's first element, the feature name as the
        # second and the datatype which is cpmeas.COLTYPE_FLOAT as
        # it's third.
        #
        ##return [(cpmeas.IMAGE, self.get_feature_name(), cpmeas.COLTYPE_FLOAT)]
        
        