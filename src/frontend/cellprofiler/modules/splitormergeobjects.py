import numpy
from cellprofiler_core.constants.measurement import (
    C_PARENT,
    FF_CHILDREN_COUNT,
    FF_PARENT,
    COLTYPE_INTEGER,
)
from cellprofiler_core.module import Module
from cellprofiler_core.object import Objects
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting import ValidationError
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.subscriber import LabelSubscriber, ImageSubscriber
from cellprofiler_core.setting.text import Integer, Float, LabelName
from cellprofiler_core.utilities.core.module.identify import (
    add_object_count_measurements,
    add_object_location_measurements,
    get_object_measurement_columns,
)

from cellprofiler.modules import _help
from cellprofiler_library.opts.splitormergeobjects import RelabelOption, MergeOption, MergingMethod, ObjectIntensityMethod
from cellprofiler_library.modules._splitormergeobjects import split_or_merge_objects
from cellprofiler_library.measurement_model import LibraryMeasurements
from cellprofiler_library.functions.segmentation import copy_labels
__doc__ = """\
SplitOrMergeObjects
===================

**SplitOrMergeObjects** separates or combines a set of objects that
were identified earlier in a pipeline.

Objects and their measurements are associated with each other based on
their object numbers (also known as *labels*). Typically, each object is
assigned a single unique number, such that the exported measurements are
ordered by this numbering. This module allows the reassignment of object
numbers by either merging separate objects to share the same label, or
splitting portions of separate objects that previously had the same
label.

There are many options in this module. For example, objects that share a
label, but are not touching can be relabeled into separate objects.
Objects that share a boundary can be combined into a single object.
Children of the same parent can be given the same label.

Note that this module does not *physically* connect/bridge/merge objects
that are separated by background pixels,
it simply assigns the same object number to the portions of the object.
The new, "merged" object may therefore consist of two or more unconnected
components. If you want to add pixels around objects, see
**ExpandOrShrink** or **Morph**.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           YES
============ ============ ===============

See also
^^^^^^^^

See also **RelateObjects**.

{HELP_ON_SAVING_OBJECTS}

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Parent object measurements:**

-  *Children Count:* The number of relabeled objects created from each
   parent object.

**Reassigned object measurements:**

-  *Parent:* The label number of the parent object.
-  *Location\_X, Location\_Y:* The pixel (X,Y) coordinates of the center
   of mass of the reassigned objects.

Technical notes
^^^^^^^^^^^^^^^

Reassignment means that the numerical value of every pixel within an
object (in the label matrix version of the image) gets changed, as
specified by the module settings. In order to ensure that objects are
labeled consecutively without gaps in the numbering (which other modules
may depend on), **SplitOrMergeObjects** will typically result in most
of the objects having their numbers reordered. This reassignment
information is stored as a per-object measurement with both the original
input and reassigned output objects, in case you need to track the
reassignment.
""".format(
    **{"HELP_ON_SAVING_OBJECTS": _help.HELP_ON_SAVING_OBJECTS}
)

class SplitOrMergeObjects(Module):
    module_name = "SplitOrMergeObjects"
    category = "Object Processing"
    variable_revision_number = 6

    def create_settings(self):
        self.objects_name = LabelSubscriber(
            "Select the input objects",
            "None",
            doc="""\
Select the objects you would like to split or merge (that is,
whose object numbers you want to reassign). You can
use any objects that were created in previous modules, such as
**IdentifyPrimaryObjects** or **IdentifySecondaryObjects**.""",
        )

        self.output_objects_name = LabelName(
            "Name the new objects",
            "RelabeledNuclei",
            doc="""\
Enter a name for the objects that have been split or merged (that is,
whose numbers have been reassigned).
You can use this name in subsequent modules that take objects as inputs.""",
        )

        self.relabel_option = Choice(
            "Operation",
            [RelabelOption.MERGE.value, RelabelOption.SPLIT.value],
            doc="""\
You can choose one of the following options:

-  *{OPTION_MERGE}:* Assign adjacent or nearby objects the same label
   based on certain criteria. It can be useful, for example, to merge
   together touching objects that were incorrectly split into two pieces
   by an **Identify** module.
-  *{OPTION_SPLIT}:* Assign a unique number to separate objects that
   currently share the same label. This can occur if you applied certain
   operations in the **Morph** module to objects.""".format(
        **{"OPTION_MERGE": RelabelOption.MERGE.value, "OPTION_SPLIT": RelabelOption.SPLIT.value}
   ),
        )

        self.merge_option = Choice(
            "Merging method",
            [MergeOption.UNIFY_DISTANCE.value, MergeOption.UNIFY_PARENT.value],
            doc="""\
*(Used only with the "{OPTION_MERGE}" option)*

You can merge objects in one of two ways:

-  *{UNIFY_DISTANCE}:* All objects within a certain pixel radius from
   each other will be merged.
-  *{UNIFY_PARENT}:* All objects which share the same parent
   relationship to another object will be merged. This is not to be
   confused with using the **RelateObjects** module, in which the
   related objects remain as individual objects. See **RelateObjects**
   for more details.""".format(
       **{
           "OPTION_MERGE": RelabelOption.MERGE.value,
           "UNIFY_DISTANCE": MergeOption.UNIFY_DISTANCE.value,
           "UNIFY_PARENT": MergeOption.UNIFY_PARENT.value
       }
   ),
        )

        self.merging_method = Choice(
            "Output object type",
            [MergingMethod.DISCONNECTED.value, MergingMethod.CONVEX_HULL.value],
            doc="""\
*(Used only with the "{UNIFY_PARENT}" merging method)*

**SplitOrMergeObjects** can either merge the child objects and keep them
disconnected or it can find the smallest convex polygon (the convex
hull) that encloses all of a parent’s child objects. The convex hull
will be truncated to include only those pixels in the parent - in that
case it may not truly be convex. Choose *{UM_DISCONNECTED}* to leave
the children as disconnected pieces. Choose *{UM_CONVEX_HULL}* to
create an output object that is the convex hull around them all.""".format(
    **{"UNIFY_PARENT": MergeOption.UNIFY_PARENT.value, "UM_DISCONNECTED": MergingMethod.DISCONNECTED.value, "UM_CONVEX_HULL": MergingMethod.CONVEX_HULL.value}
),
        )

        self.parent_object = Choice(
            "Select the parent object",
            ["None"],
            choices_fn=self.get_parent_choices,
            doc="""\
Select the parent object that will be used to merge the child objects.
Please note the following:

-  You must have established a parent-child relationship between the
   objects using a prior **RelateObjects** module.
-  Primary objects and their associated secondary objects are already in
   a one-to-one parent-child relationship, so it makes no sense to merge
   them here.""",
        )

        self.distance_threshold = Integer(
            "Maximum distance within which to merge objects",
            0,
            minval=0,
            doc="""\
*(Used only with the "{OPTION_MERGE}" option and the "{UNIFY_DISTANCE}"
method)*

Objects that are less than or equal to the distance you enter here, in
pixels, will be merged. If you choose zero (the default), only objects
that are touching will be merged. Note that *{OPTION_MERGE}* will
not actually connect or bridge the two objects by adding any new pixels;
it simply assigns the same object number to the portions of the object.
The new, merged object may therefore consist of two or more unconnected
components. If you want to add pixels around objects, see
**ExpandOrShrink** or **Morph**.""".format(
    **{"OPTION_MERGE": RelabelOption.MERGE.value, "UNIFY_DISTANCE": MergeOption.UNIFY_DISTANCE.value}
),
        )

        self.wants_image = Binary(
            "Merge using a grayscale image?",
            False,
            doc="""\
*(Used only with the "{OPTION_MERGE}" option)*

Select *Yes* to use the objects’ intensity features to determine
whether two objects should be merged. If you choose to use a grayscale
image, *{OPTION_MERGE}* will merge two objects only if they are
within the distance you have specified *and* certain criteria about the
objects within the grayscale image are met.""".format(
    **{"OPTION_MERGE": RelabelOption.MERGE.value}
),
        )

        self.image_name = ImageSubscriber(
            "Select the grayscale image to guide merging",
            "None",
            doc="""\
*(Used only if a grayscale image is to be used as a guide for
merging)*

Select the name of an image loaded or created by a previous module.""",
        )

        self.minimum_intensity_fraction = Float(
            "Minimum intensity fraction",
            0.9,
            minval=0,
            maxval=1,
            doc="""\
*(Used only if a grayscale image is to be used as a guide for
merging)*

Select the minimum acceptable intensity fraction. This will be used as
described for the method you choose in the next setting.""",
        )

        self.where_algorithm = Choice(
            "Method to find object intensity",
            [ObjectIntensityMethod.CLOSEST_POINT.value, ObjectIntensityMethod.CENTROIDS.value],
            doc="""\
*(Used only if a grayscale image is to be used as a guide for
merging)*

You can use one of two methods to determine whether two objects should
merged, assuming they meet the distance criteria (as specified
above):

-  *{CA_CENTROIDS}:* When the module considers merging two objects,
   this method identifies the centroid of each object, records the
   intensity value of the dimmer of the two centroids, multiplies this
   value by the *minimum intensity fraction* to generate a threshold,
   and draws a line between the centroids. The method will merge the two
   objects only if the intensity of every point along the line is above
   the threshold. For instance, if the intensity of one centroid is 0.75
   and the other is 0.50 and the *minimum intensity fraction* has been
   chosen to be 0.9, all points along the line would need to have an
   intensity of min(0.75, 0.50) \* 0.9 = 0.50 \* 0.9 = 0.45.
   This method works well for round cells whose maximum intensity is in
   the center of the cell: a single cell that was incorrectly segmented
   into two objects will typically not have a dim line between the
   centroids of the two halves and will be correctly merged.
-  *{CA_CLOSEST_POINT}:* This method is useful for unifying
   irregularly shaped cells that are connected. It starts by assigning
   background pixels in the vicinity of the objects to the nearest
   object. Objects are then merged if each object has background pixels
   that are:

   -  Within a distance threshold from each object;
   -  Above the minimum intensity fraction of the nearest object pixel;
   -  Adjacent to background pixels assigned to a neighboring object.

   An example of a feature that satisfies the above constraints is a
   line of pixels that connects two neighboring objects and is roughly
   the same intensity as the boundary pixels of both (such as an axon
   connecting two neurons' soma).""".format(
       **{"CA_CENTROIDS": ObjectIntensityMethod.CENTROIDS.value, "CA_CLOSEST_POINT": ObjectIntensityMethod.CLOSEST_POINT.value}
   ),
        )

    def get_parent_choices(self, pipeline):
        columns = pipeline.get_measurement_columns()
        choices = ["None"]
        for column in columns:
            object_name, feature, coltype = column[:3]
            if object_name == self.objects_name.value and feature.startswith(C_PARENT):
                choices.append(feature[(len(C_PARENT) + 1) :])
        return choices

    def validate_module(self, pipeline):
        if (
            self.relabel_option == RelabelOption.MERGE.value
            and self.merge_option == MergeOption.UNIFY_PARENT.value
            and self.parent_object.value == "None"
        ):
            raise ValidationError(
                "%s is not a valid object name" % "None", self.parent_object
            )

    def settings(self):
        return [
            self.objects_name,
            self.output_objects_name,
            self.relabel_option,
            self.distance_threshold,
            self.wants_image,
            self.image_name,
            self.minimum_intensity_fraction,
            self.where_algorithm,
            self.merge_option,
            self.parent_object,
            self.merging_method,
        ]

    def visible_settings(self):
        result = [self.objects_name, self.output_objects_name, self.relabel_option]
        if self.relabel_option == RelabelOption.MERGE.value:
            result += [self.merge_option]
            if self.merge_option == MergeOption.UNIFY_DISTANCE.value:
                result += [self.distance_threshold, self.wants_image]
                if self.wants_image:
                    result += [
                        self.image_name,
                        self.minimum_intensity_fraction,
                        self.where_algorithm,
                    ]
            elif self.merge_option == MergeOption.UNIFY_PARENT.value:
                result += [self.merging_method, self.parent_object]
        return result

    def run(self, workspace):
        #
        # Construct arguments for split_or_merge_objects
        #
        objects_name = self.objects_name.value
        objects = workspace.object_set.get_objects(objects_name)
        assert isinstance(objects, Objects)
        labels = objects.segmented

        parent_measurements = LibraryMeasurements()
        if self.merge_option == MergeOption.UNIFY_PARENT.value:
            feature_name = "_".join((C_PARENT, self.parent_object.value))
            parent_measurements.add_measurement(objects_name, feature_name, workspace.measurements[objects_name, feature_name])
        image = None
        if self.relabel_option.value != RelabelOption.SPLIT.value and self.merge_option.value == MergeOption.UNIFY_DISTANCE.value and self.wants_image.value:
            image = self.get_image(workspace)
        if self.show_window:
            # Save the image for display
            workspace.display_data.image = image

        #
        # Run split_or_merge_objects
        #
        output_labels = split_or_merge_objects(
            labels=labels,
            relabel_option=self.relabel_option.value,
            merge_option=self.merge_option.value,
            distance_threshold=self.distance_threshold.value,
            objects_name=objects_name,
            parent_name=self.parent_object.value,
            relaitonship_measurement=parent_measurements,
            merge_using_image=self.wants_image.value,
            merging_method=self.merging_method.value,
            image=image,
            where_algorithm=self.where_algorithm.value,
            minimum_intensity_fraction=self.minimum_intensity_fraction.value
        )

        #
        # Add the outputs of the split_or_merge_objects to the workspace
        #
        output_objects = Objects()
        output_objects.segmented = output_labels
        if objects.has_small_removed_segmented:
            output_objects.small_removed_segmented = copy_labels(
                objects.small_removed_segmented, output_labels
            )
        if objects.has_unedited_segmented:
            output_objects.unedited_segmented = copy_labels(
                objects.unedited_segmented, output_labels
            )
        output_objects.parent_image = objects.parent_image
        workspace.object_set.add_objects(output_objects, self.output_objects_name.value)

        # TODO: #5117 move these to library from <here>
        measurements = workspace.measurements
        add_object_count_measurements(
            measurements,
            self.output_objects_name.value,
            numpy.max(output_objects.segmented),
        )
        add_object_location_measurements(
            measurements, self.output_objects_name.value, output_objects.segmented
        )

        #
        # Relate the output objects to the input ones and record
        # the relationship.
        #
        children_per_parent, parents_of_children = objects.relate_children(
            output_objects
        )
        measurements.add_measurement(
            self.objects_name.value,
            FF_CHILDREN_COUNT % self.output_objects_name.value,
            children_per_parent,
        )
        measurements.add_measurement(
            self.output_objects_name.value,
            FF_PARENT % self.objects_name.value,
            parents_of_children,
        )
        # TODO: #5117 move these to library to </here>

        if self.show_window:
            workspace.display_data.orig_labels = objects.segmented
            workspace.display_data.output_labels = output_objects.segmented
            if self.merge_option == MergeOption.UNIFY_PARENT.value:
                workspace.display_data.parent_labels = workspace.object_set.get_objects(
                    self.parent_object.value
                ).segmented

    def display(self, workspace, figure):
        """Display the results of relabeling

        workspace - workspace containing saved display data
        """

        figure.set_subplots((2, 1))
        ax = figure.subplot_imshow_labels(
            0, 0, workspace.display_data.orig_labels, title=self.objects_name.value
        )

        if self.relabel_option == RelabelOption.MERGE.value and (
            (self.merge_option == MergeOption.UNIFY_DISTANCE.value and self.wants_image)
            or (self.merge_option == MergeOption.UNIFY_PARENT.value)
        ):
            if self.merge_option == MergeOption.UNIFY_DISTANCE.value and self.wants_image:
                image = workspace.display_data.image
                cplabels = [
                    dict(
                        name=self.output_objects_name.value,
                        labels=[workspace.display_data.output_labels],
                    ),
                    dict(
                        name=self.objects_name.value,
                        labels=[workspace.display_data.orig_labels],
                    ),
                ]

            elif self.merge_option == MergeOption.UNIFY_PARENT.value:
                image = numpy.zeros(workspace.display_data.output_labels.shape)
                cplabels = [
                    dict(
                        name=self.output_objects_name.value,
                        labels=[workspace.display_data.output_labels],
                    ),
                    dict(
                        name=self.parent_object.value,
                        labels=[workspace.display_data.parent_labels],
                    ),
                    dict(
                        name=self.objects_name.value,
                        labels=[workspace.display_data.orig_labels],
                        mode="none",
                    ),
                ]
            if image.ndim == 2:
                figure.subplot_imshow_grayscale(
                    1,
                    0,
                    image,
                    title=self.output_objects_name.value,
                    cplabels=cplabels,
                    sharexy=ax,
                )
            else:
                figure.subplot_imshow_color(
                    1,
                    0,
                    image,
                    title=self.output_objects_name.value,
                    cplabels=cplabels,
                    sharexy=ax,
                )
        else:
            figure.subplot_imshow_labels(
                1,
                0,
                workspace.display_data.output_labels,
                title=self.output_objects_name.value,
                sharexy=ax,
            )

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 1:
            # Added outline options
            setting_values += ["No", "RelabeledNucleiOutlines"]
            variable_revision_number = 2

        if variable_revision_number == 1:
            # Added per-parent unification
            setting_values += [MergeOption.UNIFY_DISTANCE.value, "None"]
            variable_revision_number = 3

        if variable_revision_number == 3:
            setting_values = setting_values + [MergingMethod.DISCONNECTED.value]
            variable_revision_number = 4

        if variable_revision_number == 4:
            setting_values = setting_values[:8] + setting_values[10:]
            variable_revision_number = 5

        if variable_revision_number == 5:
            # Unify --> Merge
            if setting_values[2] == "Unify":
                setting_values[2] = "Merge"

            variable_revision_number = 6

        return setting_values, variable_revision_number

    def get_image(self, workspace):
        """Get the image for image-directed merging"""
        objects = workspace.object_set.get_objects(self.objects_name.value)
        image = workspace.image_set.get_image(
            self.image_name.value, must_be_grayscale=True
        )
        image = objects.crop_image_similarly(image.pixel_data)
        return image

    def get_measurement_columns(self, pipeline):
        columns = get_object_measurement_columns(self.output_objects_name.value)
        columns += [
            (
                self.output_objects_name.value,
                FF_PARENT % self.objects_name.value,
                COLTYPE_INTEGER,
            ),
            (
                self.objects_name.value,
                FF_CHILDREN_COUNT % self.output_objects_name.value,
                COLTYPE_INTEGER,
            ),
        ]
        return columns

    def get_categories(self, pipeline, object_name):
        """Return the categories of measurements that this module produces

        object_name - return measurements made on this object (or 'Image' for image measurements)
        """
        if object_name == "Image":
            return ["Count"]
        elif object_name == self.output_objects_name.value:
            return ["Location", "Parent", "Number"]
        elif object_name == self.objects_name.value:
            return ["Children"]
        return []

    def get_measurements(self, pipeline, object_name, category):
        """Return the measurements that this module produces

        object_name - return measurements made on this object (or 'Image' for image measurements)
        category - return measurements made in this category
        """
        if object_name == "Image" and category == "Count":
            return [self.output_objects_name.value]
        elif object_name == self.output_objects_name.value and category == "Location":
            return ["Center_X", "Center_Y"]
        elif object_name == self.output_objects_name.value and category == "Parent":
            return [self.objects_name.value]
        elif object_name == self.output_objects_name.value and category == "Number":
            return ["Object_Number"]
        elif object_name == self.objects_name.value and category == "Children":
            return ["%s_Count" % self.output_objects_name.value]
        return []

