'''<b>Example4</b> Object processing
<hr>
'''

import cellprofiler.cpmodule as cpm
import cellprofiler.objects as cpo
import cellprofiler.settings as cps

#
# cellprofiler.cpmath.cpmorphology has many useful image processing algorithms.
# 
# skeletonize_labels performs the skeletonization (medial axis transform) for
# each labeled object in a labels matrix. It can skeletonize thousands of
# objects in an image almost as rapidly as skeletonizing a single object
# of the same complexity.
#
from cellprofiler.cpmath.cpmorphology import skeletonize_labels

class Example4(cpm.CPModule):
    module_name = "Example4"
    variable_revision_number = 1
    category = "Object Processing"
    
    def create_settings(self):
        #
        # The ObjectNameSubscriber is aware of all objects published by
        # modules upstream of this one. You use it to let the user choose
        # the objects produced by a prior module.
        #
        self.input_objects_name = cps.ObjectNameSubscriber(
            "Input objects", "Nuclei")
        #
        # The ObjectNamePublisher lets downstream modules know that this
        # module will produce objects with the name entered by the user.
        #
        self.output_objects_name = cps.ObjectNameProvider(
            "Output objects", "Skeletons")
        
    def settings(self):
        return [self.input_objects_name, self.output_objects_name]
    
    def run(self, workspace):
        #
        # The object_set keeps track of the objects produced during a cycle
        #
        # Crucial methods:
        #
        # object_set.get_objects(name) returns an instance of cpo.Objects
        #
        # object_set.add(objects, name) adds objects with the given name to
        #           the object set
        #
        #
        # Create objects in three steps:
        #     make a labels matrix
        #     create an instance of cpo.Objects()
        #     set cpo.Objects.segmented = labels matrix
        #
        # You can be "nicer" by giving more information, but this is not
        # absolutely necessary. See subsequent exercises for how to be nice.
        #     
        object_set = workspace.object_set
        
        