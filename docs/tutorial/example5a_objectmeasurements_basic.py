'''<b>Example 5a</b> Object measurements
<hr>
This version does not support overlapping objects.
'''

import numpy as np
import scipy.ndimage

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.settings as cps

C_EXAMPLE5 = "Example5"
FTR_MEAN_DISTANCE = "MeanDistance"
M_MEAN_DISTANCE = "_".join((C_EXAMPLE5, FTR_MEAN_DISTANCE))

'''Run Example5a using an algorithm that only supports non-overlapping, non-touching'''
SUPPORT_BASIC = "Basic"
'''Run Example5a using an algorithm that supports overlapping'''
SUPPORT_OVERLAPPING = "Overlapping"
'''Run Example5a using an algorithm that supports overlapping and touching'''
SUPPORT_TOUCHING = "Overlapping and touching"

class Example5aSimple(cpm.CPModule):
    variable_revision_number = 1
    module_name = "Example5aSimple"
    category = "Measurement"
    
    def create_settings(self):
        self.objects_name = cps.ObjectNameSubscriber("Objects name", "Nuclei")
        self.method = cps.Choice("Algorithm method",
                                 [SUPPORT_BASIC, SUPPORT_OVERLAPPING, 
                                  SUPPORT_TOUCHING], SUPPORT_TOUCHING)
        
    def settings(self):
        return [self.objects_name, self.method]
    
    def visible_settings(self):
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
        #
        # The module will fail and the user will see an error dialog
        # if cpo.Objects.segmented is referenced and there are overlapping
        # objects.
        # On the other hand, the code is very simple.
        #
        ##labels = objects.segmented
        #
        # The indices are the integer values representing each of the objects
        # in the labels matrix. scipy.ndimage functions often take an optional
        # argument that tells them which objects should be analyzed.
        # For instance, scipy.ndimage.mean takes an input image, a labels matrix
        # and the indices. If you don't supply the indices, it will just take
        # the mean of all labeled pixels, returning a single number.
        #
        ##indices = objects.indices
        #
        # Find the labeled pixels using labels != 0
        #
        ##foreground = labels != 0
        #
        # use scipy.ndimage.distance_transform_edt to find the distance of
        # every foreground pixel from the object edge
        #
        ##distance = scipy.ndimage.distance_transform_edt(foreground)
        #
        # call scipy.ndimage.mean(distance, labels, indices) to find the
        # mean distance in each object from its edge
        #
        ##values = scipy.ndimage.mean(distance, labels, indices)
        #
        # record the measurement using measurements.add_measurement
        # with an object name of "objects_name" and a measurement name
        # of M_MEAN_DISTANCE
        #
        ##measurements.add_measurement(objects_name,
        ##                             M_MEAN_DISTANCE,
        ##                             values)
        if workspace.show_frame:
            workspace.display_data.dt_image = distance
            workspace.display_data.labels = labels.copy()
            workspace.display_data.values = values
            i, j = np.mgrid[0:labels.shape[0], 0:labels.shape[1]]
            workspace.display_data.center_x = \
                scipy.ndimage.mean(j.astype(float), labels, indices)
            workspace.display_data.center_y = \
                scipy.ndimage.mean(i.astype(float), labels, indices)
            
    def display(self, workspace, frame):
        frame.set_subplots((1, 1))
        ax = frame.subplot_imshow_grayscale(
            0, 0, workspace.display_data.dt_image, "Distance from edge")
        for x, y, v in zip(workspace.display_data.center_x,
                           workspace.display_data.center_y,
                           workspace.display_data.values):
            ax.text(x, y, "%0.1f" % v, color="red", ha="center", va="center")
            
    def get_measurement_columns(self, pipeline):
        return [(self.objects_name.value, M_MEAN_DISTANCE, cpmeas.COLTYPE_FLOAT)]

    def get_categories(self, pipeline, object_name):
        if object_name == self.objects_name:
            return [C_EXAMPLE5]
        return []
    
    def get_measurements(self, pipeline, object_name, category):
        if object_name == self.objects_name and category == C_EXAMPLE5:
            return [FTR_MEAN_DISTANCE]
        return []