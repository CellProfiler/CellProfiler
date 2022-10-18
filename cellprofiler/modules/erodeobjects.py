# coding=utf-8

"""
ErodeObjects
=============

**ErodeObjects** shrinks objects based on the structuring element provided.
This function is similar to the "Shrink" function of **ExpandOrShrinkObjects**,
with two major distinctions-

1. **ErodeObjects** supports 3D objects, unlike **ExpandOrShrinkObjects**.
2. In **ExpandOrShrinkObjects**, a small object will only ever be shrunk down to a
   single pixel. In this module, an object smaller than the structuring element will
   be removed entirely unless 'Prevent object removal' is enabled.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

"""

import numpy
import scipy.ndimage
import skimage.measure
import skimage.morphology
from cellprofiler_core.module.image_segmentation import ObjectProcessing
from cellprofiler_core.object import Objects
from cellprofiler_core.setting import StructuringElement, Binary

import cellprofiler.utilities.morphology
from cellprofiler.modules._help import HELP_FOR_STREL


class ErodeObjects(ObjectProcessing):
    category = "Advanced"

    module_name = "ErodeObjects"

    variable_revision_number = 1

    def create_settings(self):
        super(ErodeObjects, self).create_settings()

        self.structuring_element = StructuringElement(
            allow_planewise=True, doc=HELP_FOR_STREL
        )

        self.preserve_midpoints = Binary(
            "Prevent object removal",
            True,
            doc="""
If set to "Yes", the central pixels for each object will not be eroded. This ensures that 
objects are not lost. The preserved pixels are those furtherst from the object's edge, so 
in some objects this may be a cluster of pixels with equal distance to the edge.
If set to "No", erosion can completely remove smaller objects.""",
        )

        self.relabel_objects = Binary(
            "Relabel resulting objects",
            False,
            doc="""
Large erosion filters can sometimes remove a small object or cause an irregularly shaped object 
to be split into two. This can cause problems in some other modules. Selecting "Yes" will assign 
new label numbers to resulting objects. This will ensure that there are no 'missing' labels 
(if object '3' is gone, object '4' will be reassigned to that number). However, this also means 
that parts of objects which were split and are no longer touching will be given new, individual 
label numbers.""",
        )

    def settings(self):
        __settings__ = super(ErodeObjects, self).settings()

        return __settings__ + [
            self.structuring_element,
            self.preserve_midpoints,
            self.relabel_objects,
        ]

    def visible_settings(self):
        __settings__ = super(ErodeObjects, self).settings()

        return __settings__ + [
            self.structuring_element,
            self.preserve_midpoints,
            self.relabel_objects,
        ]

    def run(self, workspace):
        x_name = self.x_name.value
        y_name = self.y_name.value
        objects = workspace.object_set
        x = objects.get_objects(x_name)
        dimensions = x.dimensions
        x_data = x.segmented

        contours = cellprofiler.utilities.morphology.morphological_gradient(x_data, self.structuring_element.value)
        y_data = x_data * (contours == 0)

        if self.preserve_midpoints.value:
            missing_labels = numpy.setxor1d(x_data, y_data)
            if self.structuring_element.value_text == "Disk,1":
                y_data += x_data * numpy.isin(x_data, missing_labels)
            else:
                for label in missing_labels:
                    binary = x_data == label
                    midpoint = scipy.ndimage.morphology.distance_transform_edt(binary)
                    y_data[midpoint == numpy.max(midpoint)] = label

        if self.relabel_objects.value:
            y_data = skimage.morphology.label(y_data)

        y = Objects()
        y.segmented = y_data
        y.parent_image = x.parent_image
        objects.add_objects(y, y_name)
        self.add_measurements(workspace)

        if self.show_window:
            workspace.display_data.x_data = x_data
            workspace.display_data.y_data = y_data
            workspace.display_data.dimensions = dimensions
