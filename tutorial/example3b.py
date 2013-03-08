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

class Example3b(cpm.CPModule):
    variable_revision_number = 1
    module_name = "Example3b"
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
        if image.has_mask:
            variance = np.var(pixel_data[image.mask])
        else:
            variance = np.var(pixel_data)
        m.add_measurement(cpmeas.IMAGE, self.get_feature_name(), variance)
        
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
        return [(cpmeas.IMAGE, self.get_feature_name(), cpmeas.COLTYPE_FLOAT)]
    
    def get_categories(self, pipeline, object_name):
        result = []
        if object_name == cpmeas.IMAGE:
            result.append(C_EXAMPLE3)
        return result
    
    def get_measurements(self, pipeline, object_name, category):
        result = []
        if object_name == cpmeas.IMAGE and category == C_EXAMPLE3:
            result.append(FTR_VARIANCE)
        return result
    
    def get_measurement_images(self, pipeline, object_name, category, measurement):
        result = []
        if (object_name == cpmeas.IMAGE 
            and category == C_EXAMPLE3 
            and measurement == FTR_VARIANCE):
            result.append(self.input_image_name.value)
        return result

        
        
        