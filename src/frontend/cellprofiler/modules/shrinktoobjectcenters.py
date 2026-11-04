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
from cellprofiler_core.module.image_segmentation import ObjectProcessing
from cellprofiler_library.modules._shrinktoobjectcenters import shrink_to_object_centers

class ShrinkToObjectCenters(ObjectProcessing):
    module_name = "ShrinkToObjectCenters"

    category = "Advanced"

    variable_revision_number = 1

    def run(self, workspace):
        #
        # Get all inputs
        #
        input_objects = workspace.object_set.get_objects(self.x_name.value)
        input_segemnted = input_objects.segmented
        output_segmented = None

        input_small_removed_segmented = None
        output_small_removed_segmented = None
        if input_objects.has_small_removed_segmented:
            input_small_removed_segmented = input_objects.small_removed_segmented
        
        input_unedited_segmented = None
        output_unedited_segmented = None
        if input_objects.has_unedited_segmented:
            input_unedited_segmented = input_objects.unedited_segmented

        #
        # Perform shrinking
        #
        output_segmented = shrink_to_object_centers(input_segemnted)

        if input_small_removed_segmented is not None:
            output_small_removed_segmented = shrink_to_object_centers(input_small_removed_segmented)

        if input_unedited_segmented is not None:
            output_unedited_segmented = shrink_to_object_centers(input_unedited_segmented)
        
        #
        # Create output objects
        #
        output_objects = cellprofiler_core.object.Objects()

        output_objects.segmented = output_segmented
        output_objects.small_removed_segmented = output_small_removed_segmented
        output_objects.unedited_segmented = output_unedited_segmented

        output_objects.parent_image = input_objects.parent_image

        workspace.object_set.add_objects(output_objects, self.y_name.value)

        self.add_measurements(workspace)

        if self.show_window:
            workspace.display_data.x_data = input_objects.segmented
            workspace.display_data.y_data = output_objects.segmented
            workspace.display_data.dimensions = input_objects.dimensions

