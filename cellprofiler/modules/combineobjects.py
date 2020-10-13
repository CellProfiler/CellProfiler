"""
CombineObjects
==============

**CombineObjects** allows you to combine two object sets into a single object set.

This moduled is geared towards situations where a set of objects was identified
using multiple instances of an Identify module, typically to account for large
variability in size or intensity. Using this module will combine object sets to
create a new set of objects which can be used in other modules.

CellProfiler can only handle a single object in each location of an image, so
it is important to carefully choose how to handle objects which would be
overlapping.

When performing operations, this module treats the first selected object set, termed
"initial objects" as the starting point for a joined set. CellProfiler will try to add
objects from the second selected set to the initial set.

Object label numbers are re-assigned after merging the object sets. This can mean that
if your settings result in one object being cut into two by another object, the divided
segments will be reassigned as seperate objects.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           NO
============ ============ ===============

"""

import numpy
import scipy.ndimage
import skimage.morphology
import skimage.segmentation
from cellprofiler_core.module import Identify
from cellprofiler_core.object import Objects
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.subscriber import LabelSubscriber
from cellprofiler_core.setting.text import LabelName
from cellprofiler_core.utilities.core.module.identify import add_object_count_measurements
from cellprofiler_core.utilities.core.module.identify import get_object_measurement_columns
from cellprofiler_core.utilities.core.module.identify import add_object_location_measurements


class CombineObjects(Identify):
    category = "Object Processing"

    module_name = "CombineObjects"

    variable_revision_number = 1

    def create_settings(self):
        self.objects_x = LabelSubscriber(
            "Select initial object set",
            "None",
            doc="""Select an object set which you want to add objects to.""",
        )

        self.objects_y = LabelSubscriber(
            "Select object set to combine",
            "None",
            doc="""Select an object set which you want to add to the initial set.""",
        )

        self.merge_method = Choice(
            "Select how to handle overlapping objects",
            choices=["Merge", "Preserve", "Discard", "Segment"],
            doc="""\
When combining sets of objects, it is possible that both sets had an object in the
same location. Use this setting to choose how to handle objects which overlap with
eachother.
        
- Selecting "Merge" will make overlapping objects combine into a single object, taking
  on the label of the object from the initial set. When an added object would overlap
  with multiple objects from the initial set, each pixel of the added object will be
  assigned to the closest object from the initial set. This is primarily useful when
  the same objects appear in both sets.
        
- Selecting "Preserve" will protect the initial object set. Any overlapping regions
  from the second set will be ignored in favour of the object from the initial set.
        
- Selecting "Discard" will only add objects which do not have any overlap with objects
  in the initial object set.
        
- Selecting "Segment" will combine both object sets and attempt to re-draw segmentation to
  separate objects which overlapped. Note: This is less reliable when more than
  two objects were overlapping. If two object sets genuinely occupy the same space
  it may be better to consider them seperately.
         """,
        )

        self.output_object = LabelName(
            "Name the combined object set",
            "CombinedObjects",
            doc="""\
Enter the name for the combined object set. These objects will be available for use in
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

        assert (
            objects_x.shape == objects_y.shape
        ), "Objects sets must have the same dimensions"

        labels_x = objects_x.segmented.copy()
        labels_y = objects_y.segmented.copy()

        output = self.combine_arrays(labels_x, labels_y)
        output_labels = skimage.morphology.label(output)
        output_objects = Objects()
        output_objects.segmented = output_labels

        workspace.object_set.add_objects(output_objects, self.output_object.value)

        m = workspace.measurements
        object_count = numpy.max(output_labels)
        add_object_count_measurements(m, self.output_object.value, object_count)
        add_object_location_measurements(m, self.output_object.value, output_labels)

        if self.show_window:
            workspace.display_data.input_object_x_name = self.objects_x.value
            workspace.display_data.input_object_x = objects_x.segmented
            workspace.display_data.input_object_y_name = self.objects_y.value
            workspace.display_data.input_object_y = objects_y.segmented
            workspace.display_data.output_object_name = self.output_object.value
            workspace.display_data.output_object = output_objects.segmented

    def display(self, workspace, figure):
        figure.set_subplots((2, 2))
        cmap = figure.return_cmap()

        ax = figure.subplot_imshow_labels(
            0,
            0,
            workspace.display_data.input_object_x,
            workspace.display_data.input_object_x_name,
            colormap=cmap,
        )
        figure.subplot_imshow_labels(
            1,
            0,
            workspace.display_data.input_object_y,
            workspace.display_data.input_object_y_name,
            sharexy=ax,
            colormap=cmap,
        )
        figure.subplot_imshow_labels(
            0,
            1,
            workspace.display_data.output_object,
            workspace.display_data.output_object_name,
            sharexy=ax,
            colormap=cmap,
        )

    def combine_arrays(self, labels_x, labels_y):
        output = numpy.zeros_like(labels_x)
        method = self.merge_method.value

        # Ensure labels in each set are unique
        labels_y[labels_y > 0] += labels_x.max()

        if method == "Preserve":
            return numpy.where(labels_x > 0, labels_x, labels_y)

        indices_x = numpy.unique(labels_x)
        indices_x = indices_x[indices_x > 0]
        indices_y = numpy.unique(labels_y)
        indices_y = indices_y[indices_y > 0]

        # Resolve non-conflicting labels first
        undisputed = numpy.logical_xor(labels_x > 0, labels_y > 0)
        for label in indices_x:
            mapped = labels_x == label
            if numpy.all(undisputed[mapped]):
                output[mapped] = label
                labels_x[mapped] = 0
        for label in indices_y:
            mapped = labels_y == label
            if numpy.all(undisputed[mapped]):
                output[mapped] = label
                labels_y[mapped] = 0

        # Resolve conflicting labels
        if method == "Discard":
            return numpy.where(labels_x > 0, labels_x, output)

        elif method == "Segment":
            to_segment = numpy.logical_or(labels_x > 0, labels_y > 0)
            disputed = numpy.logical_and(labels_x > 0, labels_y > 0)
            seeds = numpy.add(labels_x, labels_y)
            seeds[disputed] = 0
            distances, (i, j) = scipy.ndimage.distance_transform_edt(
                seeds == 0, return_indices=True
            )
            output[to_segment] = seeds[i[to_segment], j[to_segment]]

        elif method == "Merge":
            to_segment = numpy.logical_or(labels_x > 0, labels_y > 0)
            distances, (i, j) = scipy.ndimage.distance_transform_edt(
                labels_x == 0, return_indices=True
            )
            output[to_segment] = labels_x[i[to_segment], j[to_segment]]

        return output

    def get_categories(self, pipeline, object_name):
        return self.get_object_categories(pipeline, object_name, {self.output_object.value: []})

    def get_measurements(self, pipeline, object_name, category):
        return self.get_object_measurements(
            pipeline, object_name, category, {self.output_object.value: []}
        )

    def get_measurement_columns(self, pipeline):
        return get_object_measurement_columns(self.output_object.value)
