'''measureimageintensity.py - Measure imagewide intensity

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version__="$Revision$"

import numpy as np
import uuid

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.measurements as cpmeas

'''Number of settings saved/loaded per image measured'''
SETTINGS_PER_IMAGE = 3

'''Measurement category for this module'''
INTENSITY = 'Intensity'

'''Name for the measurement that's the sum of all measured pixels'''
TOTAL_INTENSITY = 'TotalIntensity'

'''Name for the measurement that's the mean of all measured pixels'''
MEAN_INTENSITY = 'MeanIntensity'

'''Name for the measurement that's the number of measured pixels'''
TOTAL_AREA = 'TotalArea'

class MeasureImageIntensity(cpm.CPModule):
    '''Measures the total image intensity by summing pixel intensity
****************************************************************

This module will sum all pixel values to measure the total image
intensity. The user can measure all pixels in the image or can restrict
the measurement to pixels within objects. If the image has a mask, only
unmasked pixels will be measured.

Features measured:
TotalIntensity    - Sum of all pixel values
MeanIntensity     - Sum of all pixel values divided by number of pixels
                    measured
TotalArea         - number of pixels measured

Settings:
What did you call the images whose intensity you want to measure?
Choose an image name from the drop-down to calculate intensity for that
image. Use the "Add image" button below to add additional images which will be
measured. You can add the same image multiple times if you want to measure
the intensity within several different objects.

Do you want to measure intensity only from areas of the image that contain
objects you've identified?
Check this option to restrict the pixels being measured to those within
the boundaries of some object.
What is the name of the objects to use?
The intensity will be restricted to within the objects you name here.

    '''

    category = "Measurement"
    variable_revision_number = 1
    
    def create_settings(self):
        '''Create the settings & name the module'''
        self.module_name = 'MeasureImageIntensity'
        self.images = []
        self.add_image_measurement()
        self.add_button = cps.DoSomething("Add another image","Add image",
                                          self.add_image_measurement)
    
    def add_image_measurement(self):
        class ImageMeasurement(object):
            '''Represents the information needed to take an image measurement
            
            '''
            def __init__(self, uber_self):
                self.__key = uuid.uuid4() 
                self.__image_name = cps.ImageNameSubscriber("What did you call the images whose intensity you want to measure?",
                                                            "None")
                self.__wants_objects = cps.Binary("Do you want to measure intensity only from areas of the image that contain objects you've identified?",
                                                  False)
                self.__object_name = cps.ObjectNameSubscriber("What is the name of the objects to use?","None")
                self.remove_button = cps.DoSomething("Remove this image from the list of images to be measured", 
                                                     "Remove image",
                                                     uber_self.remove_image_measurement,
                                                     self.__key)
            
            def name(self, feature):
                '''Return the feature name produced by this measurement'''  
                if self.wants_objects.value:
                    return '_'.join([INTENSITY, feature,
                                     self.image_name.value,
                                     self.object_name.value])
                else:
                    return '_'.join([INTENSITY, feature, self.image_name.value])
                
            @property
            def key(self):
                '''The key that uniquely identifies this image measurement
                
                Use this to find the image within the list of images
                when removing.
                '''
                return self.__key
            
            @property
            def image_name(self):
                '''The setting that holds the name of the image to measure'''
                return self.__image_name
            
            @property
            def wants_objects(self):
                '''The setting that chooses to restrict measurement to objects'''
                return self.__wants_objects
            
            @property
            def object_name(self):
                '''The setting that holds the name of the restricting object'''
                return self.__object_name
            
            def settings(self):
                '''The settings that should be saved or loaded from pipeline'''
                return [ self.image_name, self.wants_objects, self.object_name ]
            
            def visible_settings(self):
                '''The settings seen by the user'''
                return ([ self.image_name, self.wants_objects] +
                        ([self.object_name] if self.wants_objects.value
                         else []) + [self.remove_button])
            
        self.images.append(ImageMeasurement(self))
    
    def remove_image_measurement(self, key):
        '''Remove an image measurement from the image list'''
        idx = [x.key for x in self.images].index(key)
        del self.images[idx]

    def backwards_compatibilize(self, setting_values, 
                                variable_revision_number, 
                                module_name, from_matlab):
        '''Account for prior versions when loading
        
        We handle Matlab revision # 2 here. We don't support thresholding
        because it was generally unused. The first setting is the image name.
        '''
        if from_matlab and variable_revision_number == 2:
            setting_values = [setting_values[0], # image name
                              cps.NO,            # wants objects
                              "None" ]           # object name
            variable_revision_number = 1
            from_matlab = False
        return setting_values, variable_revision_number, from_matlab


    def prepare_to_set_values(self, setting_values):
        assert len(setting_values) % SETTINGS_PER_IMAGE == 0
        image_count = len(setting_values) / SETTINGS_PER_IMAGE
        while image_count > len(self.images):
            self.add_image_measurement()
        while image_count < len(self.images):
            self.remove_image_measurement(self.images[-1].key)

    def settings(self):
        '''The settings as saved and loaded from the pipeline'''
        return reduce(lambda x,y: x+y,
                      [x.settings() for x in self.images])

    def visible_settings(self):
        '''The settings as seen by the user'''
        return (reduce(lambda x,y: x+y,
                       [x.visible_settings() for x in self.images]) +
                [self.add_button])

    def get_non_redundant_image_measurements(self):
        '''Return a non-redundant sequence of image measurement objects'''
        dict = {}
        for im in self.images:
            key = ((im.image_name, im.object_name) if im.wants_objects.value
                   else (im.image_name,))
            dict[key] = im
        return dict.values()
        
    def run(self, workspace):
        '''Perform the measurements on the imageset'''
        #
        # Then measure each
        #
        statistics = [["Image","Masking object","Feature","Value"]]
        for im in self.get_non_redundant_image_measurements():
            statistics += self.measure(im, workspace)
        if not workspace.frame is None:
            figure = workspace.create_or_find_figure(subplots=(1,1))
            figure.subplot_table(0, 0, statistics, ratio=(.25,.25,.25,.25))
    
    def measure(self, im, workspace):
        '''Perform measurements according to the image measurement in im
        
        im - image measurement info (see ImageMeasurement class above)
        workspace - has all the details for current image set
        '''
        image = workspace.image_set.get_image(im.image_name.value,
                                              must_be_grayscale=True)
        pixels = image.pixel_data
        if im.wants_objects.value:
            objects = workspace.get_objects(im.object_name.value)
            if image.has_mask:
                pixels = pixels[np.logical_and(objects.segmented != 0,
                                               image.mask)]
            else:
                pixels = pixels[objects.segmented != 0]
        elif image.has_mask:
            pixels = pixels[image.mask]

        pixel_count = np.product(pixels.shape)
        pixel_sum = np.sum(pixels)
        pixel_mean = pixel_sum/float(pixel_count)
        m = workspace.measurements
        m.add_image_measurement(im.name(TOTAL_INTENSITY), pixel_sum)
        m.add_image_measurement(im.name(MEAN_INTENSITY), pixel_mean)
        m.add_image_measurement(im.name(TOTAL_AREA), pixel_count)
        return [[im.image_name.value, 
                 im.object_name.value if im.wants_objects.value else "",
                 feature_name, str(value)]
                for feature_name, value in (('Total intensity', pixel_sum),
                                            ('Mean intensity', pixel_mean),
                                            ('Total area', pixel_count))]
    
    def get_measurement_columns(self):
        '''Return column definitions for measurements made by this module'''
        columns = []
        for im in self.get_non_redundant_image_measurements():
            for feature, coltype in ((TOTAL_INTENSITY, cpmeas.COLTYPE_FLOAT),
                                     (MEAN_INTENSITY, cpmeas.COLTYPE_FLOAT),
                                     (TOTAL_AREA, cpmeas.COLTYPE_INTEGER)):
                columns.append((cpmeas.IMAGE, im.name(feature), coltype))
        return columns
                        
    def get_categories(self, pipeline, object_name):
        if object_name == cpmeas.IMAGE:
            return [INTENSITY]
        else:
            return []

    def get_measurements(self, pipeline, object_name, category):
        if (object_name == cpmeas.IMAGE and
            category == INTENSITY):
            return [TOTAL_INTENSITY, MEAN_INTENSITY, TOTAL_AREA]
        return []

    def get_measurement_objects(self, pipeline, object_name, 
                                category, measurement):
        if (object_name == cpmeas.IMAGE and
            category == INTENSITY and
            measurement in [TOTAL_INTENSITY, MEAN_INTENSITY, TOTAL_AREA]):
            return [ im.object_name.value for im in self.images
                    if im.wants_objects.value]
        return []

    def get_measurement_images(self, pipeline, object_name, 
                               category, measurement):
        if (object_name == cpmeas.IMAGE and
            category == INTENSITY and
            measurement in [TOTAL_INTENSITY, MEAN_INTENSITY, TOTAL_AREA]):
            return [im.image_name.value for im in self.images]
        return []
