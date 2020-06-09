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
            "Select first object set to merge",
            "None",
            doc="""\
        Select the object sets which you want to merge.""",
        )

        self.objects_y = cellprofiler_core.setting.ObjectNameSubscriber(
            "Select second object set to merge",
            "None",
            doc="""\
        Select the object sets which you want to merge.""",
        )

        self.merge_method = cellprofiler_core.setting.Choice(
            "Choose how to handle overlapping objects",
            choices=["Merge", "Separate"],
            doc="""\
        Use this setting to choose how to handle objects which overlap with eachother.
         Selecting "Merge" will combine touching objects into a single object. Selecting
         "Separate" will attempt to divide the objects.
         """
        )

        self.output_object = cellprofiler_core.setting.ObjectNameProvider(
            "Output object set name",
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

        labels_x = objects_x.segmented.copy()
        indices_x = objects_x.indices
        labels_y = objects_y.segmented.copy()
        labels_y[labels_y > 0] += labels_x.max()
        indices_y = [ind + labels_x.max() for ind in objects_y.indices]

        output = numpy.zeros_like(labels_x)

        # Resolve non-conflicting labels first
        undisputed = numpy.logical_xor(labels_x > 0, labels_y > 0)
        for label in indices_x:
            mapped = labels_x == label
            if numpy.all(undisputed[mapped]):
                output[mapped] = label
                labels_x[mapped] = 0
                indices_x = indices_x[indices_x != label]
        for label in indices_y:
            mapped = labels_y == label
            if numpy.all(undisputed[mapped]):
                output[mapped] = label
                labels_y[mapped] = 0

        # Resolve conflicting labels
        if self.merge_method.value == "Merge":
            for label in indices_x:
                mapped = labels_x == label
                target_labels = numpy.unique(labels_y[mapped])
                target_labels = target_labels[target_labels != 0]
                for indice in target_labels:
                    mask = labels_y == indice
                    output[mask] = label
                    labels_y[mask] = 0
                    labels_x[mask] = 0
        elif self.merge_method.value == "Separate":
            for label in indices_x:
                only_labels_x = labels_x.copy()
                only_labels_x[only_labels_x != label] = 0
                target_labels = numpy.unique(labels_y[only_labels_x > 0])
                target_labels = target_labels[target_labels != 0]
                only_labels_y = numpy.zeros_like(only_labels_x)
                for indice in target_labels:
                    only_labels_y[labels_y == indice] = indice
                mask = numpy.logical_or(only_labels_x > 0, only_labels_y > 0)
                undisputed = numpy.logical_xor(only_labels_x > 0, only_labels_y > 0)
                seeds = numpy.add(only_labels_x, only_labels_y)
                seeds[~undisputed] = 0
                distance = distance_transform_edt(mask)
                watershed = skimage.morphology.watershed(distance, seeds, mask=mask)
                output = numpy.where(watershed > 0, watershed, output)
                # Do a seeded watershed with non conflicting areas as seeds
                # flaw = multiple overlap

        output_labels, _, _ = skimage.segmentation.relabel_sequential(output)
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

    #
    # display lets you use matplotlib to display your results.
    #
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
