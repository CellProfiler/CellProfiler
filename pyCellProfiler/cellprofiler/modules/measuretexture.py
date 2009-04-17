"""
Measure texture features for an object.
Not tested -- at all!

"""

__version__="$Revision: 1 $"

import numpy as np
import scipy.ndimage as scind

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
        self.object_groups = []
        self.add_object_cb()
        self.add_objects = cps.DoSomething("Add another object", "Add",
                                           self.add_object_cb)
        self.scale = cps.Integer("Scale of the texture", 3, minval=1)

    def settings(self):
        """The settings as they appear in the save file."""
        return [og[OG_NAME] for og in self.object_groups] + [self.scale]

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
            # Remove the "Do not use" objects from the list
            setting_values = np.array(setting_values)
            setting_values = list(setting_values[setting_values !=
                                                 cps.DO_NOT_USE])
            variable_revision_number = 1
            from_matlab = False
        return setting_values, variable_revision_number, from_matlab

    def prepare_to_set_values(self,setting_values):
        """Adjust the number of object groups based on the number of
        setting_values"""
        object_group_count = len(setting_values)-1
        while len(self.object_groups) > object_group_count:
            self.remove_object_cb(object_group_count)
        
        while len(self.object_groups) < object_group_count:
            self.add_object_cb()
        
    def visible_settings(self):
        """The settings as they appear in the module viewer"""
        result = []
        for og in self.object_groups:
            result.extend([og[OG_NAME],og[OG_REMOVE]])
        result.extend([self.add_objects, self.scale])
        return result

    def add_object_cb(self):
        """Add a slot for another object"""
        index = len(self.object_groups)
        self.object_groups.append({OG_NAME:cps.ObjectNameSubscriber("What did you call the objects you want to measure?","None"),
                                   OG_REMOVE:cps.DoSomething("Remove the above objects","Remove",self.remove_object_cb,index)})
        
    def remove_object_cb(self, index):
        """Remove the indexed object from the to-do list"""
        del self.object_groups[index]

    def get_categories(self,pipeline, object_name):
        """Get the measurement categories supplied for the given object name.
        
        pipeline - pipeline being run
        object_name - name of labels in question (or 'Images')
        returns a list of category names
        """
        if any([object_name == og[OG_NAME] for og in self.object_groups]):
            return [TEXTURE]
        else:
            return []

    def run(self, workspace):
        """Run, computing the area measurements for the objects"""
        
        for object_group in self.object_groups:
            self.run_on_objects(object_group[OG_NAME].value, workspace)
    
    def run_on_objects(self, object_name, workspace):
        """Run, computing the area measurements for a single map of objects"""
        objects = workspace.get_objects(object_name)
        image = objects.get_parent_image()
        print "Computing haralick..."
        for name, value in zip(F_HARALICK, Haralick(image.pixel_data,
                                                    objects.segmented,
                                                    self.scale.value).all()):
            self.record_measurement(workspace, object_name, name, value)
        print "done"

    def record_measurement(self, workspace,  
                           object_name, feature_name, result):
        """Record the result of a measurement in the workspace's
        measurements"""
        data = fixup_scipy_ndimage_result(result)
        workspace.add_measurement(object_name, 
                                  "%s_%s"%(TEXTURE, feature_name), 
                                  data)
