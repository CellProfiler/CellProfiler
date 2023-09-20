"""
ShrinkToObjectCenters
======================

**ShrinkToObjectCenters** will transform a set of objects into a label image with single points
representing each object. The location of each point corresponds to the centroid of the input objects.

Note that if the object is not sufficiently *round*, the resulting single pixel will reside outside the
original object. For example, a 'U' shaped object, perhaps a *C. Elegans*, could potentially lead to this
special case. This could be a concern if these points are later used as seeds or markers for a **Watershed**
operation further in the pipeline.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

"""

import cellprofiler_core.object
import numpy
import skimage.measure
from cellprofiler_core.module.image_segmentation import ObjectProcessing


class ShrinkToObjectCenters(ObjectProcessing):
    module_name = "ShrinkToObjectCenters"

    category = "Advanced"

    variable_revision_number = 1

    def run(self, workspace):
        input_objects = workspace.object_set.get_objects(self.x_name.value)

        output_objects = cellprofiler_core.object.Objects()

        output_objects.segmented = self.find_centroids(input_objects.segmented)

        if input_objects.has_small_removed_segmented:
            output_objects.small_removed_segmented = self.find_centroids(
                input_objects.small_removed_segmented
            )

        if input_objects.has_unedited_segmented:
            output_objects.unedited_segmented = self.find_centroids(
                input_objects.unedited_segmented
            )

        output_objects.parent_image = input_objects.parent_image

        workspace.object_set.add_objects(output_objects, self.y_name.value)

        self.add_measurements(workspace)

        if self.show_window:
            workspace.display_data.x_data = input_objects.segmented

            workspace.display_data.y_data = output_objects.segmented

            workspace.display_data.dimensions = input_objects.dimensions

    @staticmethod
    def find_centroids(label_image):
        input_props = skimage.measure.regionprops(
            label_image, intensity_image=None, cache=True
        )

        input_centroids = [numpy.int_(obj["centroid"]) for obj in input_props]

        output_segmented = numpy.zeros_like(label_image)

        for ind, arr in enumerate(input_centroids):
            output_segmented[tuple(arr)] = ind + 1

        return output_segmented
