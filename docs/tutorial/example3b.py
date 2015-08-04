'''<b>Example3b</b> Being nice to CellProfiler when adding a measurement
<hr>
This module demonstrates how to fully inform CellProfiler about the
measurements produced by your module. At a minimum, you must implement
"get_measurement_columns()" to have your measurements handled by
ExportToSpreadsheet and ExportToDatabase. Modules such as "DisplayDataOnImage"
and "CalculateMath" need more context when they display the Measurement setting.
There are four methods that supply that context:
<br><ul><li><i>get_categories()</i> gives the categories of measurements
supplied for a given object (or the "Image" "object"). An example is "Intensity"
for "Intensity_IntegratedIntensity_DNA".</li>
<li><i>get_measurements()</i> gives the feature name part of the measurement.
An example is "IntegratedIntensity" for "Intensity_IntegratedIntensity_DNA".</li>
<li><i>get_measurement_images()</i> and <i>get_measurement_objects()</i>
give the image name or the object name part of the measurement. An example is
"DNA" for "Intensity_IntegratedIntensigy_DNA" where the measurement was made
on the "Nucleus" object and used the "DNA" image. Some object measurements
can reference a secondary object, such as parent / child measurements and
those are documented using get_measurement_objects(). These two methods
should only be implemented for measurements that do make reference to
a particular image or secondary objecct.</li>
<li><i>get_measurement_scales()</i> gives the scale part of a measurement.
We've broadened the concept of "Scale" to include any major parameter condition
where a user might be expected to sample among a variety of parameter 
configurations. An example of a measurement with a scale is the texture measurement,
"Texture_Variance_DNA_3_90" which has a scale of "3_90", meaning "pixels offset
by 3 in the vertical direction".
</li></ul>
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
    #
    # get_categories should return a list of the category parts of measurements
    # made by the module. You should only return measurements whose name
    # matches the object name given. In this case, we're making image
    # measurements, so "object_name" must match "Image"
    #
    ##def get_categories(self, pipeline, object_name):
    ##    result = []
    ##    if object_name == cpmeas.IMAGE:
    ##        result.append(C_EXAMPLE3)
    ##    return result
    #
    # get_measurements should return a list of the features that will be
    # produced for a given object name and category.
    #
    ##def get_measurements(self, pipeline, object_name, category):
    ##    result = []
    ##    if object_name == cpmeas.IMAGE and category == C_EXAMPLE3:
    ##        result.append(FTR_VARIANCE)
    ##    return result
    #
    # get_measurement_images should return a list of the images that will
    # be produced, given a particular object name, category and feature name.
    #
    ##def get_measurement_images(self, pipeline, object_name, category, measurement):
    ##    result = []
    ##    if (object_name == cpmeas.IMAGE 
    ##        and category == C_EXAMPLE3 
    ##        and measurement == FTR_VARIANCE):
    ##        result.append(self.input_image_name.value)
    ##    return result

        
        
        