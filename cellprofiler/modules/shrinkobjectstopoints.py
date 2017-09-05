# coding=utf-8

"""
ShrinkObjectsToPoints
======================

**ShrinkObjectsToPoints** will transform a set of objects into a label image with single points
representing each object. The location of each point corresponds to the centroid of the input objects.
"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import numpy
import skimage.measure


class ShrinkObjectsToPoints(cellprofiler.module.ObjectProcessing):

    module_name = "ShrinkObjectsToPoints"

    category = "Advanced"

    variable_revision_number = 1

    def create_settings(self):

        super(ShrinkObjectsToPoints, self).create_settings()

        self.x_name.text = "Input objects"

        self.x_name.doc = "Select the objects that you want to shrink to points."


    def settings(self):

        __settings__ = super(ShrinkObjectsToPoints, self).settings()

        return __settings__


    def visible_settings(self):

        __settings__ = super(ShrinkObjectsToPoints, self).visible_settings()

        return __settings__


    def display(self, workspace, figure):

        if not self.show_window:

            return

        input_objects_segmented = workspace.display_data.input_objects_segmented

        output_objects_segmented = workspace.display_data.output_objects_segmented

        dimensions = workspace.display_data.dimensions

        figure.set_subplots((2, 1), dimensions=dimensions)

        figure.subplot_imshow_labels(
            0,
            0,
            input_objects_segmented,
            title=self.x_name.value,
            dimensions=dimensions

        )

        figure.subplot_imshow_labels(
            1,
            0,
            output_objects_segmented,
            title=self.y_name.value,
            dimensions=dimensions
        )


    def run(self, workspace):

        input_objects = workspace.object_set.get_objects(self.x_name.value)

        output_objects = cellprofiler.object.Objects()

        output_objects.segmented = self.shrink_ray(input_objects.segmented)

        if input_objects.has_small_removed_segmented:

            output_objects.small_removed_segmented = self.shrink_ray(input_objects.small_removed_segmented)

        if input_objects.has_unedited_segmented:

            output_objects.unedited_segmented = self.shrink_ray(input_objects.unedited_segmented)

        output_objects.parent_image = input_objects.parent_image

        workspace.object_set.add_objects(output_objects, self.y_name.value)

        cellprofiler.modules.identify.add_object_count_measurements(
            workspace.measurements,
            self.y_name.value,
            numpy.max(output_objects.segmented)
        )

        if self.show_window:

            workspace.display_data.input_objects_segmented = input_objects.segmented

            workspace.display_data.output_objects_segmented = output_objects.segmented

            workspace.display_data.dimensions = input_objects.dimensions


    def shrink_ray(self, label_image):

        input_props = skimage.measure.regionprops(label_image, intensity_image=None, cache=True)

        input_centroids = [numpy.int_(obj["centroid"]) for obj in input_props]

        output_segmented = numpy.zeros_like(label_image)

        for ind, arr in enumerate(input_centroids):

            output_segmented[tuple(arr)] = ind + 1

        return output_segmented