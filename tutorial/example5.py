'''<b>Example 5</b> Object measurements
<hr>
'''

import numpy as np
import scipy.ndimage

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps

C_EXAMPLE5 = "Example5"
FTR_MEAN_DISTANCE = "MeanDistance"
M_MEAN_DISTANCE = "_".join((C_EXAMPLE5, FTR_MEAN_DISTANCE))

class Example5(cpm.CPModule):
    variable_revision_number = 1
    module_name = "Example5"
    category = "Measurement"
    
    def create_settings(self):
        self.objects_name = cps.ObjectNameSubscriber("Objects name", "Nuclei")
        
    def settings(self):
        return [self.objects_name]
    
    def run(self, workspace):
        #
        # Get some things we need from the workspace
        #
        measurements = workspace.measurements
        object_set = workspace.object_set
        #
        # Get the objects
        #
        objects_name = self.objects_name.value
        objects = object_set.get_objects(objects_name)
        labels = objects.segmented
        #
        # The indices are the integer values representing each of the objects
        # in the labels matrix. scipy.ndimage functions often take an optional
        # argument that tells them which objects should be analyzed.
        # For instance, scipy.ndimage.mean takes an input image, a labels matrix
        # and the indices. If you don't supply the indices, it will just take
        # the mean of all labeled pixels, returning a single number.
        #
        indices = objects.indices
        #
        # Find the labeled pixels using labels != 0
        #
        ####
        #
        # use scipy.ndimage.distance_transform_edt to find the distance of
        # every foreground pixel from the object edge
        #
        ####
        #
        # call scipy.ndimage.mean(distance, labels, indices) to find the
        # mean distance in each object from its edge
        #
        ####
        #
        # record the measurement using measurements.add_measurement
        # with an object name of "objects_name" and a measurement name
        # of M_MEAN_DISTANCE
        #
        