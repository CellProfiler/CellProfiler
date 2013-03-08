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

class Example5a(cpm.CPModule):
    variable_revision_number = 1
    module_name = "Example5a"
    category = "Measurement"
    
    def create_settings(self):
        self.objects_name = cps.ObjectNameSubscriber("Objects name", "Nuclei")
        self.method = cps.Choice("Algorithm method",
                                 [SUPPORT_BASIC, SUPPORT_OVERLAPPING, 
                                  SUPPORT_TOUCHING], SUPPORT_TOUCHING)
        
    def settings(self):
        return [self.objects_name, self.method]
    
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
        # First, I do it the (1) way to show how that code should look.
        # Later, I do it the (3) way and that will work even if objects.has_ijv
        # is False. 
        if self.method == SUPPORT_BASIC:
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
            foreground = labels != 0
            #
            # use scipy.ndimage.distance_transform_edt to find the distance of
            # every foreground pixel from the object edge
            #
            distance = scipy.ndimage.distance_transform_edt(foreground)
            #
            # call scipy.ndimage.mean(distance, labels, indices) to find the
            # mean distance in each object from its edge
            #
            values = scipy.ndimage.mean(distance, labels, indices)
            #
            # record the measurement using measurements.add_measurement
            # with an object name of "objects_name" and a measurement name
            # of M_MEAN_DISTANCE
            #
            measurements.add_measurement(objects_name,
                                         M_MEAN_DISTANCE,
                                         values)
        elif self.method == SUPPORT_OVERLAPPING:
            #
            # I'll use objects.get_labels to get labels matrices. This involves
            # a little extra work coallating the values, but not so bad.
            #
            # First of all, labels indices start at 1, but arrays start at
            # zero, so for "values", I'm going to cheat and waste values[0].
            # Later, I'll only use values[1:]
            #
            values = np.zeros(objects.count+1)
            #
            # Now for the loop
            #
            for labels, indices in objects.get_labels():
                foreground = labels != 0
                distance = scipy.ndimage.distance_transform_edt(foreground)
                v1 = scipy.ndimage.mean(distance, labels, indices)
                #
                # We copy the values above into the appropriate slots
                #
                values[indices] = v1
            measurements.add_measurement(objects_name,
                                         M_MEAN_DISTANCE,
                                         values[1:])
        else:
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
            from cellprofiler.cpmath.cpmorphology import color_labels
            
            values = np.zeros(objects.count+1)
            for labels, indices in objects.get_labels():
                clabels = color_labels(labels)
                #
                # np.unique returns the unique #s in an array.
                #
                colors = np.unique(clabels)
                for color in colors:
                    # 0 = background, so ignore it.
                    if color == 0:
                        continue
                    #
                    # Ok, here's a trick. clabels == color gets converted
                    # to either 1 (is the current color) or 0 (is not) and
                    # we can use that to mask only the labels for the current
                    # color by multiplying (0 * anything = 0)
                    #
                    foreground = clabels == color
                    mini_labels = labels * foreground
                    distance = scipy.ndimage.distance_transform_edt(foreground)
                    #
                    # And here's another trick - scipy.ndimage.mean returns
                    # NaN for any index that doesn't appear because the
                    # mean isn't computable. How lucky!
                    #
                    v1 = scipy.ndimage.mean(distance, mini_labels, indices)
                    good_v1 = ~ np.isnan(v1)
                    values[indices[good_v1]] = v1[good_v1]
            measurements.add_measurement(objects_name,
                                         M_MEAN_DISTANCE,
                                         values[1:])
            
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