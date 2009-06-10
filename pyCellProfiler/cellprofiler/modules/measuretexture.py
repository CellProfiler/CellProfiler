"""
Measure texture features for an object.

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

__version__="$Revision: 1 $"

import numpy as np
import scipy.ndimage as scind
import uuid

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
from cellprofiler.cpmath.cpmorphology import fixup_scipy_ndimage_result
from cellprofiler.cpmath.haralick import Haralick

"""The category of the per-object measurements made by this module"""
TEXTURE = 'Texture'

"""The "name" slot in the object group dictionary entry"""
OG_NAME = 'name'
"""The "remove"slot in the object group dictionary entry"""
OG_REMOVE = 'remove'

F_HARALICK = """AngularSecondMoment Contrast Correlation Variance 
InverseDifferenceMoment SumAverage SumVariance SumEntropy Entropy
DifferenceVariance DifferenceEntropy InfoMeas1 InfoMeas2""".split()

class MeasureTexture(cpm.CPModule):
    """TODO: Docstring"""

    variable_revision_number = 1
    category = 'Measurement'

    def create_settings(self):
        """Create the settings for the module at startup.
        
        The module allows for an unlimited number of measured objects, each
        of which has an entry in self.object_groups.
        """ 
        self.module_name = "MeasureTexture"
        self.image_groups = []
        self.object_groups = []
        self.scale_groups = []
        self.image_count = cps.HiddenCount(self.image_groups)
        self.object_count = cps.HiddenCount(self.object_groups)
        self.scale_count = cps.HiddenCount(self.scale_groups)
        self.add_image_cb()
        self.add_images = cps.DoSomething("Add another image", "Add",
                                          self.add_image_cb)
        self.add_object_cb()
        self.add_objects = cps.DoSomething("Add another object", "Add",
                                           self.add_object_cb)
        self.add_scale_cb()
        self.add_scales = cps.DoSomething("Add another scale", "Add",
                                          self.add_scale_cb)

    def settings(self):
        """The settings as they appear in the save file."""
        result = [self.image_count, self.object_count, self.scale_count]
        for groups in (self.image_groups, self.object_groups, 
                       self.scale_groups):
            for group in groups:
                result += group.settings()
        return result

    def backwards_compatibilize(self,setting_values,variable_revision_number,
                                module_name,from_matlab):
        """Adjust the setting_values for older save file versions
        
        setting_values - a list of strings representing the settings for
                         this module.
        variable_revision_number - the variable revision number of the module
                                   that saved the settings
        module_name - the name of the module that saved the settings
        from_matlab - true if it was a Matlab module that saved the settings
        
        returns the modified settings, revision number and "from_matlab" flag
        """
        if from_matlab and variable_revision_number == 2:
            #
            # The first 3 settings are:
            # image count (1 for legacy)
            # object count (calculated)
            # scale_count (calculated)
            #
            object_names = [name for name in setting_values[1:7]
                            if name.upper() != cps.DO_NOT_USE.upper()] 
            scales = setting_values[7].split(',')
            setting_values = [ "1", str(len(object_names)), str(len(scales)),
                               setting_values[0]] + object_names + scales
            variable_revision_number = 1
            from_matlab = False
        return setting_values, variable_revision_number, from_matlab

    def prepare_to_set_values(self,setting_values):
        """Adjust the number of object groups based on the number of
        setting_values"""
        for count, sequence, fn in\
            ((int(setting_values[0]), self.image_groups, self.add_image_cb),
             (int(setting_values[1]), self.object_groups, self.add_object_cb),
             (int(setting_values[2]), self.scale_groups, self.add_scale_cb)):
            while len(sequence) > count:
                del sequence[count]
            while len(sequence) < count:
                fn()
        
    def visible_settings(self):
        """The settings as they appear in the module viewer"""
        result = []
        for groups, add_button in ((self.image_groups, self.add_images),
                                   (self.object_groups, self.add_objects),
                                   (self.scale_groups, self.add_scales)):
            for group in groups:
                result += group.visible_settings()
            result += [add_button]
        return result

    def add_image_cb(self):
        """Add a slot for another image"""
        class ImageSettings(object):
            def __init__(self, sequence):
                self.key = uuid.uuid4()
                def remove(sequence=sequence, key=self.key):
                    index = [x.key for x in sequence].index(key)
                    del sequence[index]
                
                self.image_name = cps.ImageNameSubscriber("What did you call the images you want to measure?","None")
                self.remove_button = cps.DoSomething("Remove the above image",
                                                     "Remove",remove)
            def settings(self):
                return [self.image_name]
            def visible_settings(self):
                return [self.image_name, self.remove_button]
        self.image_groups.append(ImageSettings(self.image_groups))
        
    def add_object_cb(self):
        """Add a slot for another object"""
        class ObjectSettings(object):
            def __init__(self, sequence):
                self.key = uuid.uuid4()
                def remove(sequence=sequence, key=self.key):
                    index = [x.key for x in sequence].index(key)
                    del sequence[index]
                
                self.object_name = cps.ObjectNameSubscriber("What did you call the objects you want to measure?","None")
                self.remove_button = cps.DoSomething("Remove the above objects",
                                                     "Remove",remove)
            def settings(self):
                return [self.object_name]
            def visible_settings(self):
                return [self.object_name, self.remove_button]
            
        self.object_groups.append(ObjectSettings(self.object_groups))

    def add_scale_cb(self):
        '''Add another scale to be measured'''
        class ScaleSettings(object):
            def __init__(self, sequence):
                self.key = uuid.uuid4()
                def remove(sequence=sequence, key=self.key):
                    index = [x.key for x in sequence].index(key)
                    del sequence[index]
                
                self.scale = cps.Integer("What is the scale of the texture?",
                                         len(sequence)+3)
                self.remove_button = cps.DoSomething("Remove the above scale",
                                                     "Remove",remove)
            def settings(self):
                return [self.scale]
            def visible_settings(self):
                return [self.scale, self.remove_button]
            
        self.scale_groups.append(ScaleSettings(self.scale_groups))

    def get_categories(self,pipeline, object_name):
        """Get the measurement categories supplied for the given object name.
        
        pipeline - pipeline being run
        object_name - name of labels in question (or 'Images')
        returns a list of category names
        """
        if any([object_name == og.object_name for og in self.object_groups]):
            return [TEXTURE]
        else:
            return []

    def get_measurements(self, pipeline, object_name, category):
        '''Get the measurements made on the given object in the given category
        
        pipeline - pipeline being run
        object_name - name of objects being measured
        category - measurement category
        '''
        if len(self.get_categories(pipeline, object_name)) > 0:
            return F_HARALICK
        return []

    def get_measurement_images(self, pipeline, object_name, category, measurement):
        '''Get the list of images measured
        
        pipeline - pipeline being run
        object_name - name of objects being measured
        category - measurement category
        measurement - measurement made on images
        '''
        measurements = self.get_measurements(pipeline, object_name, category)
        if measurement in measurements:
            return [x.image_name.value for x in self.image_groups]
        return []

    def get_measurement_scales(self, pipeline, object_name, category, 
                               measurement, image_name):
        '''Get the list of scales at which the measurement was taken

        pipeline - pipeline being run
        object_name - name of objects being measured
        category - measurement category
        measurement - name of measurement made
        image_name - name of image that was measured
        '''
        if len(self.get_measurement_images(pipeline, object_name, category,
                                           measurement)) > 0:
            return [x.scale.value for x in self.scale_groups]
        return []

    def run(self, workspace):
        """Run, computing the area measurements for the objects"""
        
        for image_group in self.image_groups:
            for object_group in self.object_groups:
                for scale_group in self.scale_groups:
                    self.run_one(image_group.image_name.value,
                                 object_group.object_name.value,
                                 scale_group.scale.value, workspace)
    
    def run_one(self, image_name, object_name, scale, workspace):
        """Run, computing the area measurements for a single map of objects"""
        image = workspace.image_set.get_image(image_name,
                                              must_be_grayscale=True)
        objects = workspace.get_objects(object_name)
        labels = image.crop_image_similarly(objects.segmented)
        for name, value in zip(F_HARALICK, Haralick(image.pixel_data,
                                                    labels,
                                                    scale).all()):
            self.record_measurement(workspace, image_name, object_name, scale,
                                    name, value)

    def record_measurement(self, workspace,  
                           image_name, object_name, scale,
                           feature_name, result):
        """Record the result of a measurement in the workspace's
        measurements"""
        data = fixup_scipy_ndimage_result(result)
        data[~np.isfinite(data)] = 0
        workspace.add_measurement(object_name, 
                                  "%s_%s_%s_%d"%
                                  (TEXTURE, feature_name,image_name, scale), 
                                  data)
