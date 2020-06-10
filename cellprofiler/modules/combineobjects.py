# coding=utf-8

"""
CombineObjects
=============

**CombineObjects** allows you to merge two object sets into a single object set.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO          NO
============ ============ ===============

"""

import cellprofiler_core.module
import cellprofiler_core.object
import cellprofiler_core.setting
import numpy
import skimage.segmentation
import skimage.morphology
from scipy.ndimage import distance_transform_edt

class CombineObjects(cellprofiler_core.module.image_segmentation.ObjectProcessing):
    category = "Object Processing"

    module_name = "CombineObjects"

    variable_revision_number = 1

    def create_settings(self):
        self.objects_x = cellprofiler_core.setting.ObjectNameSubscriber(
            "Select initial object set",
            "None",
            doc="""\
        Select the object sets which you want to merge.""",
        )

        self.objects_y = cellprofiler_core.setting.ObjectNameSubscriber(
            "Select object set to combine",
            "None",
            doc="""\
        Select the object sets which you want to merge.""",
        )

        self.merge_method = cellprofiler_core.setting.Choice(
            "Select how to handle overlapping objects",
            choices=["Merge", "Preserve", "Discard", "Segment"],
            doc="""\
        When combining sets of objects, it is possible that both sets had an object in the
        same location. Use this setting to choose how to handle objects which overlap with
        eachother.
        
        - Selecting "Merge" will make overlapping objects combine into a single object.
        This can work well if you expect the same object to appear in both sets.
        Note that this method is not suitable for combining dense object sets with multiple
        overlapping neighbours.
        
        - Selecting "Preserve" will protect the initial object set. Any overlapping regions
        from the second set will be cut out in favour of the object from the initial set.
        
        - Selecting "Discard" will only add objects which do not have any overlap with objects
        in the initial object set.
        
        - Selecting "Segment" will combine both object sets and attempt to re-draw lines to
        separate objects which overlapped. Note: This becomes less reliable when more than
        two objects were overlapping. 
         """
        )

        self.output_object = cellprofiler_core.setting.ObjectNameProvider(
            "Name the combined object set",
            "CombinedObjects",
            doc="""\
Enter the name for the combined object set. These objects will be available for use by
subsequent modules.""",
        )

    def settings(self):
        return [self.objects_x, self.objects_y, self.merge_method, self.output_object]

    def visible_settings(self):
        return [self.objects_x, self.objects_y, self.merge_method, self.output_object]

    def run(self, workspace):
        for object_name in (self.objects_x.value, self.objects_y.value):
            if object_name not in workspace.object_set.object_names:
                raise ValueError(
                    "The %s objects are missing from the pipeline." % object_name
                )
        objects_x = workspace.object_set.get_objects(self.objects_x.value)

        objects_y = workspace.object_set.get_objects(self.objects_y.value)

        assert objects_x.shape == objects_y.shape,\
            "Objects sets must have the same dimensions"

        overlay_matrix = numpy.zeros_like(objects_x.segmented)

        overlay_matrix[objects_x.segmented > 0] += 1
        overlay_matrix[objects_y.segmented > 0] += 1

        # Ensure array Y's labels don't conflict with array X.
        labels_x = objects_x.segmented.copy()
        labels_y = objects_y.segmented.copy()

        output = self.combine_arrays(labels_x, labels_y)
        output_labels = skimage.morphology.label(output)
        output_objects = cellprofiler_core.object.Objects()
        output_objects.segmented = output_labels

        workspace.object_set.add_objects(output_objects, self.output_object.value)

        if self.show_window:
            workspace.display_data.input_object_x_name = self.objects_x.value
            workspace.display_data.input_object_x = objects_x.segmented
            workspace.display_data.input_object_y_name = self.objects_y.value
            workspace.display_data.input_object_y = objects_y.segmented
            workspace.display_data.output_object_name = self.output_object.value
            workspace.display_data.output_object = output_objects.segmented

    def display(self, workspace, figure):
        figure.set_subplots((2, 2))

        # Display image 3 times w/ input object a, input object b, and merged output object:
        ax = figure.subplot_imshow_labels(0, 0, workspace.display_data.input_object_x,
                                     workspace.display_data.input_object_x_name
                                          )
        figure.subplot_imshow_labels(1, 0, workspace.display_data.input_object_y,
                                     workspace.display_data.input_object_y_name,
                                     sharexy=ax)
        figure.subplot_imshow_labels(0, 1, workspace.display_data.output_object,
                                     workspace.display_data.output_object_name,
                                     sharexy=ax)

    def combine_arrays(self, labels_x, labels_y):
        output = numpy.zeros_like(labels_x)
        method = self.merge_method.value

        if method == "Preserve":
            return numpy.where(labels_x > 0, labels_x, labels_y)

        indices_x = numpy.unique(labels_x)
        indices_x = indices_x[indices_x > 0]
        indices_y = numpy.unique(labels_y)
        indices_y = indices_y[indices_y > 0]

        # Resolve non-conflicting and totally overlapped labels first
        undisputed = numpy.logical_xor(labels_x > 0, labels_y > 0)
        disputed = numpy.logical_and(labels_x > 0, labels_y > 0)
        for label in indices_x:
            mapped = labels_x == label
            if numpy.all(undisputed[mapped]):
                # Only appeared in one object set
                output[mapped] = label
                labels_x[mapped] = 0
                indices_x = indices_x[indices_x != label]
            elif numpy.all(disputed[mapped]):
                # Completely covered by objects in other set
                labels_x[mapped] = 0
                indices_x = indices_x[indices_x != label]
        # Recalcualate overlapping areas
        disputed = numpy.logical_and(labels_x > 0, labels_y > 0)
        for label in indices_y:
            mapped = labels_y == label
            if numpy.all(undisputed[mapped]):
                output[mapped] = label
                labels_y[mapped] = 0
            elif numpy.all(disputed[mapped]):
                labels_x[mapped] = 0
                indices_x = indices_x[indices_x != label]

        # Resolve conflicting labels
        if method == "Merge":
            for x_label in indices_x:
                mapped = labels_x == x_label
                target_labels = numpy.unique(labels_y[mapped])
                target_labels = target_labels[target_labels != 0]
                output[mapped] = x_label
                for y_label in target_labels:
                    mask = labels_y == y_label
                    output[mask] = x_label
                    labels_y[mask] = 0
                    labels_x[mask] = 0

        elif method == "Discard":
            print(True)
            output2 = numpy.where(labels_x > 0, labels_x, labels_y)
            print(True)

        elif self.merge_method.value == "Segment":
            to_segment = numpy.logical_or(labels_x > 0, labels_y > 0)
            undisputed = numpy.logical_xor(labels_x > 0, labels_y > 0)
            seeds = numpy.add(labels_x, labels_y)
            seeds[~undisputed] = 0

            distances, (i, j) = distance_transform_edt(
                ~undisputed, return_indices=True
            )

            output[to_segment] = seeds[i[to_segment], j[to_segment]]


        return output
