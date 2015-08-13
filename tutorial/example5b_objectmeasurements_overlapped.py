'''<b>Example 5</b> Object measurements
<hr>
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

class Example5b(cpm.CPModule):
    variable_revision_number = 1
    module_name = "Example5bOverlapped"
    category = "Measurement"
    
    def create_settings(self):
        self.objects_name = cps.ObjectNameSubscriber("Objects name", "Nuclei")
        self.method = cps.Choice("Algorithm method",
                                 [SUPPORT_BASIC, SUPPORT_OVERLAPPING, 
                                  SUPPORT_TOUCHING], SUPPORT_TOUCHING)
        
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
        #
        # It's just a little expensive finding out which labels are
        # touching others. The trick here is to use a function from
        # cpmorphology called "color_labels". This is akin to the
        # four color theorem - you want to color objects so that no
        # two adjacent ones have the same color.
        #
        # After we've done that, we process each of the colors in turn,
        # knowing that each object is colored only once and none of its
        # neighbors have the same color.
        #
        # This is a good demo of why Python and Numpy are good choices
        # for image processing. We're handling some pretty abstract
        # concepts in just a few lines of code and the result, I hope,
        # is clear and readable.
        #
        from centrosome.cpmorphology import color_labels
        #
        # First, reserve a vector that we'll use within the loop to store
        # the measurement values. I cheat a little here in order to use
        # object indexing. 0 is the background and values[0] will never
        # be written. values[object_number] will be the measurement for
        # that object number. Consequently, we need an extra slot in the
        # vector because we are indexing by 1.
        #
        ##values = np.zeros(objects.count+1)
        #
        # Each outer loop iteration gets a labels matrix and the object numbers
        # represented within the labels matrix. If there are no overlaps,
        # there will only be one iteration.
        # 
        ##for labels, indices in objects.get_labels():
            #
            # Now we use color_labels to magically produce an array in which
            # all the background pixels have the value, "0", and the pixels
            # of touching objects have different numbers
            #
            ##clabels = color_labels(labels)
            #
            # np.unique returns the unique #s in an array.
            #
            ##colors = np.unique(clabels)
            ##for color in colors:
                # 0 = background, so ignore it.
                ##if color == 0:
                ##    continue
                #
                # Ok, here's a trick. clabels == color gets converted
                # to either 1 (is the current color) or 0 (is not) and
                # we can use that to mask only the labels for the current
                # color by multiplying (0 * anything = 0)
                #
                ##foreground = clabels == color
                ##mini_labels = labels * foreground
                ##distance = scipy.ndimage.distance_transform_edt(foreground)
                #
                # And here's another trick - scipy.ndimage.mean returns
                # NaN for any index that doesn't appear because the
                # mean isn't computable. How lucky!
                #
                ##v1 = scipy.ndimage.mean(distance, mini_labels, indices)
                ##good_v1 = ~ np.isnan(v1)
                ##values[indices[good_v1]] = v1[good_v1]
        measurements.add_measurement(objects_name,
                                     M_MEAN_DISTANCE,
                                     values[1:])
        if workspace.show_frame:
            workspace.display_data.ijv = objects.ijv.copy()
            workspace.display_data.values = values[1:]
            
    def display(self, workspace, frame):
        frame.set_subplots((1,1))
        ax = frame.subplot_imshow_ijv(0, 0, workspace.display_data.ijv,
                                      title=self.objects_name.value)
        indices = np.unique(workspace.display_data.ijv[:, 2])
        x = scipy.ndimage.mean(workspace.display_data.ijv[:, 1],
                               workspace.display_data.ijv[:, 2],
                               indices)  
        y = scipy.ndimage.minimum(workspace.display_data.ijv[:, 0],
                                  workspace.display_data.ijv[:, 2],
                                  indices)
    
        for xi, yi, v in zip(x, y, workspace.display_data.values):
            ax.text(xi, yi, "%0.1f" % v, color="red", ha="center", va="bottom")
            
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