'''<b>RelabelObjects</b>
Relabels objects so that objects within a specified distance of each
other, or objects with a straight line connecting
their centroids that has a relatively uniform intensity, 
get the same label and thereby become the same object.
Optionally, if an object consists of two or more unconnected components, this
module can relabel them so that the components become separate objects.
<hr>
Right now, this module converts the input labels matrix into an output
which all have the same label - other stuff to be implemented.
'''
#CellProfiler is distributed under the GNU General Public License.
#See the accompanying file LICENSE for details.
#
#Developed by the Broad Institute
#Copyright 2003-2009
#
#Please see the AUTHORS file for credits.
#
#Website: http://www.cellprofiler.org


__version__="$Revision$"

import numpy as np

import cellprofiler.cpmodule as cpm
import cellprofiler.objects as cpo
import cellprofiler.settings as cps
from cellprofiler.modules.identify import get_object_measurement_columns
from cellprofiler.modules.identify import add_object_count_measurements
from cellprofiler.modules.identify import add_object_location_measurements
class RelabelObjects(cpm.CPModule):
    module_name = "RelabelObjects"
    category = "Object Processing"
    variable_revision_number = 1
    
    def create_settings(self):
        self.objects_name = cps.ObjectNameSubscriber("Enter original objects name:",
                                                    "None")
        self.output_objects_name = cps.ObjectNameProvider("Enter new objects name:",
                                                          "None")
        
    def settings(self):
        return [self.objects_name, self.output_objects_name]
    
    def run(self, workspace):
        objects = workspace.object_set.get_objects(self.objects_name.value)
        output_objects = cpo.Objects()
        output_objects.segmented = (objects.segmented > 0).astype(np.uint8)
        output_objects.parent_image = objects.parent_image
        workspace.object_set.add_objects(output_objects, self.output_objects_name.value)
        add_object_count_measurements(workspace.measurements,
                                      self.output_objects_name.value,
                                      np.max(output_objects.segmented))
        add_object_location_measurements(workspace.measurements,
                                         self.output_objects_name.value,
                                         output_objects.segmented)
                                                 
    def get_measurement_columns(self, pipeline):
        return get_object_measurement_columns(self.output_objects_name.value)
    
    def get_categories(self,pipeline, object_name):
        """Return the categories of measurements that this module produces
        
        object_name - return measurements made on this object (or 'Image' for image measurements)
        """
        if object_name == 'Image':
            return ['Count']
        elif object_name == self.output_objects_name.value:
            return ['Location']
        return []
      
    def get_measurements(self, pipeline, object_name, category):
        """Return the measurements that this module produces
        
        object_name - return measurements made on this object (or 'Image' for image measurements)
        category - return measurements made in this category
        """
        if object_name == 'Image' and category == 'Count':
            return [ self.output_objects_name.value ]
        elif object_name == self.output_objects_name.value and category == 'Location':
            return ['Center_X','Center_Y']
        return []

        