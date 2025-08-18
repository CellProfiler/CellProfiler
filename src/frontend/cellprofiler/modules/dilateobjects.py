"""
DilateObjects
=============

**DilateObjects** expands objects based on the structuring element provided.
This function is similar to the "Expand" function of **ExpandOrShrinkObjects**,
with two major distinctions-

1. **DilateObjects** supports 3D objects, unlike **ExpandOrShrinkObjects**. 
2. In **ExpandOrShrinkObjects**, two objects closer than the expansion distance
   will expand until they meet and then stop there. In this module, the object with
   the larger object number (the object that is lower in the image) will be expanded
   on top of the object with the smaller object number.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

"""

from cellprofiler_core.module.image_segmentation import ObjectProcessing
from cellprofiler_core.setting import StructuringElement

import cellprofiler.utilities.morphology
from cellprofiler.modules._help import HELP_FOR_STREL


class DilateObjects(ObjectProcessing):
    category = "Advanced"

    module_name = "DilateObjects"

    variable_revision_number = 1

    def create_settings(self):
        super(DilateObjects, self).create_settings()

        self.structuring_element = StructuringElement(
            allow_planewise=True, doc=HELP_FOR_STREL
        )

    def settings(self):
        __settings__ = super(DilateObjects, self).settings()

        return __settings__ + [self.structuring_element]

    def visible_settings(self):
        __settings__ = super(DilateObjects, self).visible_settings()

        return __settings__ + [self.structuring_element]

    def run(self, workspace):
        from cellprofiler_library.modules._dilateobjects import dilate_objects
        
        x_name = self.x_name.value
        y_name = self.y_name.value
        objects = workspace.object_set
        x = objects.get_objects(x_name)
        x_data = x.segmented

        y_data = dilate_objects(
            labels=x_data,
            structuring_element=self.structuring_element.value
        )

        from cellprofiler_core.object import Objects
        y = Objects()
        y.segmented = y_data
        y.parent_image = x.parent_image
        objects.add_objects(y, y_name)
        self.add_measurements(workspace)

        if self.show_window:
            workspace.display_data.x_data = x_data
            workspace.display_data.y_data = y_data
            workspace.display_data.dimensions = x.dimensions
