import re

import cellprofiler_core.object
import numpy
import scipy.ndimage
import skimage.segmentation
from cellprofiler_core.constants.measurement import (
    C_PARENT,
    C_CHILDREN,
    FF_PARENT,
    FF_CHILDREN_COUNT,
    R_PARENT,
    R_CHILD,
    MCA_AVAILABLE_EACH_CYCLE,
    C_COUNT,
    C_LOCATION,
    C_NUMBER,
    FTR_CENTER_X,
    FTR_CENTER_Y,
    FTR_CENTER_Z,
    FTR_OBJECT_NUMBER,
    M_NUMBER_OBJECT_NUMBER,
    COLTYPE_FLOAT,
)
from cellprofiler_core.module.image_segmentation import ObjectProcessing
from cellprofiler_core.setting import Binary, SettingsGroup, ValidationError
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.do_something import DoSomething, RemoveSettingButton
from cellprofiler_core.setting.subscriber import LabelSubscriber
from cellprofiler_core.setting.text import LabelName

from cellprofiler.modules import _help

__doc__ = """\
RelateObjects
=============

**RelateObjects** assigns relationships; all objects (e.g., speckles)
within a parent object (e.g., nucleus) become its children.

This module allows you to associate *child* objects with *parent*
objects. This is useful for counting the number of children associated
with each parent, and for calculating mean measurement values for all
children that are associated with each parent.

An object will be considered a child even if the edge is the only partly
touching a parent object. If a child object is touching multiple parent
objects, the object will be assigned to the parent with maximal overlap.
For an alternate approach to assigning parent/child relationships,
consider using the **MaskObjects** module.

If you want to include child objects that lie outside but still near
parent objects, you might want to expand the parent objects using
**ExpandOrShrink** or **IdentifySecondaryObjects**.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============

See also
^^^^^^^^

See also: **SplitOrMergeObjects**, **MaskObjects**.

{HELP_ON_SAVING_OBJECTS}

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Parent object measurements:**

-  *Count:* The number of child sub-objects for each parent object.
-  *Mean measurements:* The mean of the child object measurements,
   calculated for each parent object.

**Child object measurements:**

-  *Parent:* The label number of the parent object, as assigned by an
   **Identify** or **Watershed** module.
-  *Distances:* The distance of each child object to its respective parent.
""".format(
    **{"HELP_ON_SAVING_OBJECTS": _help.HELP_ON_SAVING_OBJECTS}
)

D_NONE = "None"
D_CENTROID = "Centroid"
D_MINIMUM = "Minimum"
D_BOTH = "Both"

D_ALL = [D_NONE, D_CENTROID, D_MINIMUM, D_BOTH]

C_MEAN = "Mean"

FF_MEAN = "%s_%%s_%%s" % C_MEAN

"""Distance category"""
C_DISTANCE = "Distance"

"""Centroid distance feature"""
FEAT_CENTROID = "Centroid"

"""Minimum distance feature"""
FEAT_MINIMUM = "Minimum"

"""Centroid distance measurement (FF_DISTANCE % parent)"""
FF_CENTROID = "%s_%s_%%s" % (C_DISTANCE, FEAT_CENTROID)

"""Minimum distance measurement (FF_MINIMUM % parent)"""
FF_MINIMUM = "%s_%s_%%s" % (C_DISTANCE, FEAT_MINIMUM)

FIXED_SETTING_COUNT = 7
VARIABLE_SETTING_COUNT = 1


class RelateObjects(ObjectProcessing):
    module_name = "RelateObjects"

    variable_revision_number = 5

    def create_settings(self):
        super(RelateObjects, self).create_settings()

        self.x_name.text = "Parent objects"

        self.x_name.doc = """\
Parent objects are defined as those objects which encompass the child object.
For example, when relating speckles to the nuclei that contain them,
the nuclei are the parents.
        """

        self.y_name = LabelSubscriber(
            "Child objects",
            doc="""\
Child objects are defined as those objects contained within the parent object. For example, when relating
speckles to the nuclei that contains them, the speckles are the children.
            """,
        )

        self.find_parent_child_distances = Choice(
            "Calculate child-parent distances?",
            D_ALL,
            doc="""\
Choose the method to calculate distances of each child to its parent.
For example, these measurements can tell you whether nuclear speckles
are located more closely to the center of the nucleus or to the nuclear
periphery.

-  *{D_NONE}:* Do not calculate any distances. This saves computation time.
-  *{D_MINIMUM}:* The distance from the centroid of the child object to
   the closest perimeter point on the parent object.
-  *{D_CENTROID}:* The distance from the centroid of the child object
   to the centroid of the parent.
-  *{D_BOTH}:* Calculate both the *{D_MINIMUM}* and *{D_CENTROID}*
   distances.""".format(
                **{
                    "D_NONE": D_NONE,
                    "D_MINIMUM": D_MINIMUM,
                    "D_CENTROID": D_CENTROID,
                    "D_BOTH": D_BOTH,
                }
            ),
        )

        self.wants_step_parent_distances = Binary(
            "Calculate distances to other parents?",
            False,
            doc="""\
*(Used only if calculating distances)*

Select "*{YES}*" to calculate the distances of the child objects to some
other objects. These objects must be either parents or children of your
parent object in order for this module to determine the distances. For
instance, you might find “Nuclei” using **IdentifyPrimaryObjects**, find
“Cells” using **IdentifySecondaryObjects** and find “Cytoplasm” using
**IdentifyTertiaryObjects**. You can use **Relate** to relate speckles
to cells and then measure distances to nuclei and cytoplasm. You could
not use **RelateObjects** to relate speckles to cytoplasm and then
measure distances to nuclei, because nuclei are neither a direct parent
nor child of cytoplasm.""".format(
                **{"YES": "Yes"}
            ),
        )

        self.step_parent_names = []

        self.add_step_parent(can_delete=False)

        self.add_step_parent_button = DoSomething(
            "", "Add another parent", self.add_step_parent
        )

        self.wants_per_parent_means = Binary(
            "Calculate per-parent means for all child measurements?",
            False,
            doc="""\
Select "*{YES}*" to calculate the per-parent mean values of every upstream
measurement made with the children objects and store them as a
measurement for the parent; the nomenclature of this new measurement is
“Mean_<child>_<category>_<feature>”. This module
must be placed *after* all **Measure** modules that make measurements
of the children objects.""".format(
                **{"YES": "Yes"}
            ),
        )

        self.wants_child_objects_saved = Binary(
            "Do you want to save the children with parents as a new object set?",
            False,
            doc="""\
Select "*{YES}*" to save the children objects that do have parents as new
object set. Objects with no parents will be discarded""".format(
                **{"YES": "Yes"}
            ),
        )

        self.output_child_objects_name = LabelName(
            "Name the output object",
            "RelateObjects",
            doc="""\
Enter the name you want to call the object produced by this module. """,
        )

    def add_step_parent(self, can_delete=True):
        group = SettingsGroup()

        group.append(
            "step_parent_name",
            Choice(
                "Parent name",
                ["None"],
                choices_fn=self.get_step_parents,
                doc="""\
*(Used only if calculating distances to another parent)*

Choose the name of the other parent. The **RelateObjects** module will
measure the distance from this parent to the child objects in the same
manner as it does to the primary parents. You can only choose the
parents or children of the parent object.""",
            ),
        )

        if can_delete:
            group.append(
                "remove",
                RemoveSettingButton(
                    "", "Remove this object", self.step_parent_names, group
                ),
            )

        self.step_parent_names.append(group)

    def get_step_parents(self, pipeline):
        """Return the possible step-parents associated with the parent"""
        step_parents = set()

        parent_name = self.x_name.value

        for module in pipeline.modules():
            if module.module_num == self.module_num:
                return list(step_parents)

            # Objects that are the parent of the parents
            grandparents = module.get_measurements(pipeline, parent_name, C_PARENT)

            step_parents.update(grandparents)

            # Objects that are the children of the parents
            siblings = module.get_measurements(pipeline, parent_name, C_CHILDREN)

            for sibling in siblings:
                match = re.match("^([^_]+)_Count", sibling)

                if match is not None:
                    sibling_name = match.groups()[0]

                    if parent_name in module.get_measurements(
                        pipeline, sibling_name, C_PARENT
                    ):
                        step_parents.add(sibling_name)

        return list(step_parents)

    @property
    def has_step_parents(self):
        """True if there are possible step-parents for the parent object"""
        return (
            len(self.step_parent_names) > 0
            and len(self.step_parent_names[0].step_parent_name.choices) > 0
        )

    def settings(self):
        settings = super(RelateObjects, self).settings()

        settings += [
            self.find_parent_child_distances,
            self.wants_per_parent_means,
            self.wants_step_parent_distances,
            self.wants_child_objects_saved,
            self.output_child_objects_name,
        ]

        settings += [group.step_parent_name for group in self.step_parent_names]

        return settings

    def visible_settings(self):
        visible_settings = super(RelateObjects, self).visible_settings()

        visible_settings += [
            self.wants_per_parent_means,
            self.find_parent_child_distances,
            self.wants_child_objects_saved,
        ]

        if self.wants_child_objects_saved:
            visible_settings += [self.output_child_objects_name]

        if self.find_parent_child_distances != D_NONE and self.has_step_parents:
            visible_settings += [self.wants_step_parent_distances]

            if self.wants_step_parent_distances:
                for group in self.step_parent_names:
                    visible_settings += group.visible_settings()

                visible_settings += [self.add_step_parent_button]

        return visible_settings

    def run(self, workspace):
        parents = workspace.object_set.get_objects(self.x_name.value)

        children = workspace.object_set.get_objects(self.y_name.value)

        child_count, parents_of = parents.relate_children(children)

        m = workspace.measurements

        m.add_measurement(
            self.y_name.value, FF_PARENT % self.x_name.value, parents_of,
        )

        m.add_measurement(
            self.x_name.value, FF_CHILDREN_COUNT % self.y_name.value, child_count,
        )

        good_parents = parents_of[parents_of != 0]

        image_numbers = numpy.ones(len(good_parents), int) * m.image_set_number

        good_children = numpy.argwhere(parents_of != 0).flatten() + 1

        if numpy.any(good_parents):
            m.add_relate_measurement(
                self.module_num,
                R_PARENT,
                self.x_name.value,
                self.y_name.value,
                image_numbers,
                good_parents,
                image_numbers,
                good_children,
            )

            m.add_relate_measurement(
                self.module_num,
                R_CHILD,
                self.y_name.value,
                self.x_name.value,
                image_numbers,
                good_children,
                image_numbers,
                good_parents,
            )

        parent_names = self.get_parent_names()

        for parent_name in parent_names:
            if self.find_parent_child_distances in (D_BOTH, D_CENTROID):
                self.calculate_centroid_distances(workspace, parent_name)

            if self.find_parent_child_distances in (D_BOTH, D_MINIMUM):
                self.calculate_minimum_distances(workspace, parent_name)

        if self.wants_per_parent_means.value:
            parent_indexes = numpy.arange(numpy.max(parents.segmented)) + 1

            for feature_name in m.get_feature_names(self.y_name.value):
                if not self.should_aggregate_feature(feature_name):
                    continue

                data = m.get_current_measurement(self.y_name.value, feature_name)

                if data is not None and len(data) > 0:
                    if len(parents_of) > 0:
                        means = scipy.ndimage.mean(
                            data.astype(float), parents_of, parent_indexes
                        )
                    else:
                        means = numpy.zeros((0,))
                else:
                    # No child measurements - all NaN
                    means = numpy.ones(len(parents_of)) * numpy.nan

                mean_feature_name = FF_MEAN % (self.y_name.value, feature_name)

                m.add_measurement(self.x_name.value, mean_feature_name, means)

        if self.wants_child_objects_saved.value:
            # most of this is lifted wholesale from FilterObjects
            parent_labels = parents.segmented

            child_labels = children.segmented

            children_with_parents = numpy.where(parent_labels > 0, child_labels, 0)

            indexes = numpy.unique(children_with_parents)[1:]

            # Create an array that maps label indexes to their new values
            # All labels to be deleted have a value in this array of zero
            #
            new_object_count = len(indexes)
            max_label = numpy.max(child_labels)
            label_indexes = numpy.zeros((max_label + 1,), int)
            label_indexes[indexes] = numpy.arange(1, new_object_count + 1)

            target_labels = children.segmented.copy()
            #
            # Reindex the labels of the old source image
            #
            target_labels[target_labels > max_label] = 0
            target_labels = label_indexes[target_labels]
            #
            # Make a new set of objects - retain the old set's unedited
            # segmentation for the new and generally try to copy stuff
            # from the old to the new.
            #
            target_objects = cellprofiler_core.object.Objects()
            target_objects.segmented = target_labels
            target_objects.unedited_segmented = children.unedited_segmented
            #
            # Remove the filtered objects from the small_removed_segmented
            # if present. "small_removed_segmented" should really be
            # "filtered_removed_segmented".
            #
            small_removed = children.small_removed_segmented.copy()
            small_removed[(target_labels == 0) & (children.segmented != 0)] = 0
            target_objects.small_removed_segmented = small_removed
            if children.has_parent_image:
                target_objects.parent_image = children.parent_image
            workspace.object_set.add_objects(
                target_objects, self.output_child_objects_name.value
            )
            self.add_measurements(
                workspace, self.y_name.value, self.output_child_objects_name.value
            )

        if self.show_window:
            workspace.display_data.parent_labels = parents.segmented

            workspace.display_data.parent_count = parents.count

            workspace.display_data.child_labels = children.segmented

            workspace.display_data.parents_of = parents_of

            workspace.display_data.dimensions = parents.dimensions

    def display(self, workspace, figure):
        if not self.show_window:
            return

        dimensions = workspace.display_data.dimensions

        figure.set_subplots((2, 2), dimensions=dimensions)

        child_labels = workspace.display_data.child_labels

        parents_of = workspace.display_data.parents_of

        parent_labels = workspace.display_data.parent_labels

        #
        # discover the mapping so that we can apply it to the children
        #
        mapping = numpy.arange(workspace.display_data.parent_count + 1)

        mapping[parent_labels] = parent_labels

        parent_labeled_children = numpy.zeros(child_labels.shape, int)

        mask = child_labels > 0

        parent_labeled_children[mask] = mapping[parents_of[child_labels[mask] - 1]]

        max_label = max(
            parent_labels.max(), child_labels.max(), parent_labeled_children.max()
        )

        seed = numpy.random.randint(256)

        cmap = figure.return_cmap(max_label)

        figure.subplot_imshow_labels(
            0,
            0,
            parent_labels,
            title=self.x_name.value,
            max_label=max_label,
            seed=seed,
            colormap=cmap,
        )

        figure.subplot_imshow_labels(
            1,
            0,
            child_labels,
            title=self.y_name.value,
            sharexy=figure.subplot(0, 0),
            max_label=max_label,
            seed=seed,
            colormap=cmap,
        )

        figure.subplot_imshow_labels(
            0,
            1,
            parent_labeled_children,
            title="{} labeled by {}".format(self.y_name.value, self.x_name.value),
            sharexy=figure.subplot(0, 0),
            max_label=max_label,
            seed=seed,
            colormap=cmap,
        )

    def get_parent_names(self):
        parent_names = [self.x_name.value]

        if self.wants_step_parent_distances.value:
            parent_names += [
                group.step_parent_name.value for group in self.step_parent_names
            ]

        return parent_names

    def calculate_centroid_distances(self, workspace, parent_name):
        """Calculate the centroid-centroid distance between parent & child"""
        meas = workspace.measurements

        sub_object_name = self.y_name.value

        parents = workspace.object_set.get_objects(parent_name)

        children = workspace.object_set.get_objects(sub_object_name)

        parents_of = self.get_parents_of(workspace, parent_name)

        pcenters = parents.center_of_mass()

        ccenters = children.center_of_mass()

        if pcenters.shape[0] == 0 or ccenters.shape[0] == 0:
            dist = numpy.array([numpy.NaN] * len(parents_of))
        else:
            #
            # Make indexing of parents_of be same as pcenters
            #
            parents_of = parents_of - 1

            mask = (parents_of != -1) | (parents_of > pcenters.shape[0])

            dist = numpy.array([numpy.NaN] * ccenters.shape[0])

            dist[mask] = numpy.sqrt(
                numpy.sum((ccenters[mask, :] - pcenters[parents_of[mask], :]) ** 2, 1)
            )

        meas.add_measurement(sub_object_name, FF_CENTROID % parent_name, dist)

    def calculate_minimum_distances(self, workspace, parent_name):
        """Calculate the distance from child center to parent perimeter"""
        meas = workspace.measurements

        sub_object_name = self.y_name.value

        parents = workspace.object_set.get_objects(parent_name)

        children = workspace.object_set.get_objects(sub_object_name)

        parents_of = self.get_parents_of(workspace, parent_name)

        if len(parents_of) == 0:
            dist = numpy.zeros((0,))
        elif numpy.all(parents_of == 0):
            dist = numpy.array([numpy.NaN] * len(parents_of))
        else:
            mask = parents_of > 0

            ccenters = children.center_of_mass()

            ccenters = ccenters[mask, :]

            parents_of_masked = parents_of[mask] - 1

            pperim = (
                skimage.segmentation.find_boundaries(parents.segmented, mode="inner")
                * parents.segmented
            )

            # Get a list of all points on the perimeter
            perim_loc = numpy.argwhere(pperim != 0)

            # Get the label # for each point
            # multidimensional indexing with non-tuple values not allowed as of numpy 1.23
            perim_loc_t = tuple(map(tuple, perim_loc.transpose()))
            perim_idx = pperim[perim_loc_t]

            # Sort the points by label #
            reverse_column_order = list(range(children.dimensions))[::-1]

            coordinates = perim_loc[:, reverse_column_order].transpose().tolist()

            coordinates.append(perim_idx)

            idx = numpy.lexsort(coordinates)

            perim_loc = perim_loc[idx, :]

            perim_idx = perim_idx[idx]

            # Get counts and indexes to each run of perimeter points
            counts = scipy.ndimage.sum(
                numpy.ones(len(perim_idx)),
                perim_idx,
                numpy.arange(1, perim_idx[-1] + 1),
            ).astype(numpy.int32)

            indexes = numpy.cumsum(counts) - counts

            # For the children, get the index and count of the parent
            ccounts = counts[parents_of_masked]

            cindexes = indexes[parents_of_masked]

            # Now make an array that has an element for each of that child's perimeter points
            clabel = numpy.zeros(numpy.sum(ccounts), int)

            # cfirst is the eventual first index of each child in the clabel array
            cfirst = numpy.cumsum(ccounts) - ccounts

            clabel[cfirst[1:]] += 1

            clabel = numpy.cumsum(clabel)

            # Make an index that runs from 0 to ccounts for each child label.
            cp_index = numpy.arange(len(clabel)) - cfirst[clabel]

            # then add cindexes to get an index to the perimeter point
            cp_index += cindexes[clabel]

            # Now, calculate the distance from the centroid of each label to each perimeter point in the parent.
            dist = numpy.sqrt(
                numpy.sum((perim_loc[cp_index, :] - ccenters[clabel, :]) ** 2, 1)
            )

            # Finally, find the minimum distance per child
            min_dist = scipy.ndimage.minimum(dist, clabel, numpy.arange(len(ccounts)))

            # Account for unparented children
            dist = numpy.array([numpy.NaN] * len(mask))

            dist[mask] = min_dist

        meas.add_measurement(sub_object_name, FF_MINIMUM % parent_name, dist)

    def get_parents_of(self, workspace, parent_name):
        """Return the parents_of measurement or equivalent
        parent_name - name of parent objects

        Return a vector of parent indexes to the given parent name using
        the Parent measurement. Look for a direct parent / child link first
        and then look for relationships between self.parent_name and the
        named parent.
        """
        meas = workspace.measurements

        parent_feature = FF_PARENT % parent_name

        primary_parent = self.x_name.value

        sub_object_name = self.y_name.value

        primary_parent_feature = FF_PARENT % primary_parent

        if parent_feature in meas.get_feature_names(sub_object_name):
            parents_of = meas.get_current_measurement(sub_object_name, parent_feature)
        elif parent_feature in meas.get_feature_names(primary_parent):
            #
            # parent_name is the grandparent of the sub-object via
            # the primary parent.
            #
            primary_parents_of = meas.get_current_measurement(
                sub_object_name, primary_parent_feature
            )

            grandparents_of = meas.get_current_measurement(
                primary_parent, parent_feature
            )

            mask = primary_parents_of != 0

            parents_of = numpy.zeros(primary_parents_of.shape[0], grandparents_of.dtype)

            if primary_parents_of.shape[0] > 0:
                parents_of[mask] = grandparents_of[primary_parents_of[mask] - 1]
        elif primary_parent_feature in meas.get_feature_names(parent_name):
            primary_parents_of = meas.get_current_measurement(
                sub_object_name, primary_parent_feature
            )

            primary_parents_of_parent = meas.get_current_measurement(
                parent_name, primary_parent_feature
            )

            if len(primary_parents_of_parent) == 0:
                return primary_parents_of_parent

            #
            # There may not be a 1-1 relationship, but we attempt to
            # construct one
            #
            reverse_lookup_len = max(
                numpy.max(primary_parents_of) + 1, len(primary_parents_of_parent)
            )

            reverse_lookup = numpy.zeros(reverse_lookup_len, int)

            if primary_parents_of_parent.shape[0] > 0:
                reverse_lookup[primary_parents_of_parent] = numpy.arange(
                    1, len(primary_parents_of_parent) + 1
                )

            if primary_parents_of.shape[0] > 0:
                parents_of = reverse_lookup[primary_parents_of]
        else:
            raise ValueError(
                "Don't know how to relate {} to {}".format(primary_parent, parent_name)
            )

        return parents_of

    ignore_features = set(M_NUMBER_OBJECT_NUMBER)

    def should_aggregate_feature(self, feature_name):
        """Return True if aggregate measurements should be made on a feature

        feature_name - name of a measurement, such as Location_Center_X
        """
        if feature_name.startswith(C_MEAN):
            return False

        if feature_name.startswith(C_PARENT):
            return False

        if feature_name in self.ignore_features:
            return False

        return True

    def validate_module(self, pipeline):
        """Validate the module's settings

        Relate will complain if the children and parents are related
        by a prior module or if a step-parent is named twice"""
        for module in pipeline.modules():
            if module == self:
                break

            parent_features = module.get_measurements(
                pipeline, self.y_name.value, "Parent"
            )

            if self.x_name.value in parent_features:
                raise ValidationError(
                    "{} and {} were related by the {} module".format(
                        self.y_name.value, self.x_name.value, module.module_name
                    ),
                    self.x_name,
                )

        if self.has_step_parents and self.wants_step_parent_distances:
            step_parents = set()
            for group in self.step_parent_names:
                if group.step_parent_name.value in step_parents:
                    raise ValidationError(
                        "{} has already been chosen".format(
                            group.step_parent_name.value
                        ),
                        group.step_parent_name,
                    )

                step_parents.add(group.step_parent_name.value)

    def get_child_columns(self, pipeline):
        child_columns = list(
            filter(
                lambda column: column[0] == self.y_name.value
                and self.should_aggregate_feature(column[1]),
                pipeline.get_measurement_columns(self),
            )
        )

        child_columns += self.get_child_measurement_columns(pipeline)

        return child_columns

    def get_child_measurement_columns(self, pipeline):
        columns = []
        if self.find_parent_child_distances in (D_BOTH, D_CENTROID):
            for parent_name in self.get_parent_names():
                columns += [(self.y_name.value, FF_CENTROID % parent_name, "integer",)]

        if self.find_parent_child_distances in (D_BOTH, D_MINIMUM):
            for parent_name in self.get_parent_names():
                columns += [(self.y_name.value, FF_MINIMUM % parent_name, "integer",)]

        return columns

    def get_saved_child_measurement_columns(self, pipeline):
        """Return measurements for saved child objects"""
        columns_to_return = []
        if self.wants_child_objects_saved:
            columns = super(RelateObjects, self).get_measurement_columns(
                pipeline,
                additional_objects=[
                    (self.y_name.value, self.output_child_objects_name.value)
                ],
            )
            columns_to_return = []
            for column in columns:
                if (
                    column[0] == self.output_child_objects_name.value
                    or self.output_child_objects_name.value in column[1]
                ):
                    columns_to_return.append(column)
        return columns_to_return

    def get_measurement_columns(self, pipeline):
        """Return the column definitions for this module's measurements"""

        columns = [
            (self.y_name.value, FF_PARENT % self.x_name.value, "integer",),
            (self.x_name.value, FF_CHILDREN_COUNT % self.y_name.value, "integer",),
        ]

        if self.wants_child_objects_saved:
            columns += self.get_saved_child_measurement_columns(pipeline)

        if self.wants_per_parent_means.value:
            child_columns = self.get_child_columns(pipeline)

            columns += [
                (
                    self.x_name.value,
                    FF_MEAN % (self.y_name.value, column[1]),
                    COLTYPE_FLOAT,
                )
                for column in child_columns
            ]

        columns += self.get_child_measurement_columns(pipeline)

        return columns

    def get_object_relationships(self, pipeline):
        """Return the object relationships produced by this module"""
        parent_name = self.x_name.value

        sub_object_name = self.y_name.value

        return [
            (R_PARENT, parent_name, sub_object_name, MCA_AVAILABLE_EACH_CYCLE,),
            (R_CHILD, sub_object_name, parent_name, MCA_AVAILABLE_EACH_CYCLE,),
        ]

    def get_categories(self, pipeline, object_name):
        result = []
        if object_name == self.x_name.value:
            if self.wants_per_parent_means:
                result += ["Mean_{}".format(self.y_name.value), "Children"]
            else:
                result += ["Children"]
        elif object_name == self.y_name.value:
            result = ["Parent"]

            if self.find_parent_child_distances != D_NONE:
                result += [C_DISTANCE]
        elif object_name == "Image":
            result += [C_COUNT]
        elif object_name == self.output_child_objects_name.value:
            result += [
                C_LOCATION,
                C_NUMBER,
            ]
        return result

    def get_measurements(self, pipeline, object_name, category):
        if object_name == self.x_name.value:
            if category == "Mean_{}".format(self.y_name.value):
                measurements = []

                child_columns = self.get_child_columns(pipeline)

                measurements += [column[1] for column in child_columns]

                return measurements
            elif category == "Children":
                return ["%s_Count" % self.y_name.value]
        elif object_name == self.y_name.value and category == "Parent":
            return [self.x_name.value]
        elif object_name == self.y_name.value and category == C_DISTANCE:
            result = []

            if self.find_parent_child_distances in (D_BOTH, D_CENTROID):
                result += [
                    "{}_{}".format(FEAT_CENTROID, parent_name)
                    for parent_name in self.get_parent_names()
                ]

            if self.find_parent_child_distances in (D_BOTH, D_MINIMUM):
                result += [
                    "{}_{}".format(FEAT_MINIMUM, parent_name)
                    for parent_name in self.get_parent_names()
                ]

            return result
        elif object_name == self.output_child_objects_name.value:
            if category == C_LOCATION:
                return [
                    FTR_CENTER_X,
                    FTR_CENTER_Y,
                    FTR_CENTER_Z,
                ]

            if category == C_NUMBER:
                return [FTR_OBJECT_NUMBER]

        elif (
            object_name == "Image"
            and self.wants_child_objects_saved.value
            and category == C_COUNT
        ):
            return [self.output_child_objects_name.value]

        return []

    def prepare_settings(self, setting_values):
        setting_count = len(setting_values)

        step_parent_count = (
            setting_count - FIXED_SETTING_COUNT
        ) // VARIABLE_SETTING_COUNT

        assert len(self.step_parent_names) > 0

        self.step_parent_names = self.step_parent_names[:1]

        for i in range(1, step_parent_count):
            self.add_step_parent()

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 1:
            #
            # Added other distance parents
            #
            if setting_values[2] == "Do not use":
                find_parent_distances = D_NONE
            else:
                find_parent_distances = setting_values[2]

            if setting_values[3].upper() == "Do not use".upper():
                wants_step_parent_distances = "No"
            else:
                wants_step_parent_distances = "Yes"

            setting_values = setting_values[:2] + [
                find_parent_distances,
                setting_values[4],
                wants_step_parent_distances,
                setting_values[3],
            ]

            variable_revision_number = 2

        if variable_revision_number == 2:
            setting_values = [setting_values[1], setting_values[0]] + setting_values[2:]

            variable_revision_number = 3

        if variable_revision_number == 3:
            setting_values = setting_values[:5] + ["No"] + setting_values[5:]

            variable_revision_number = 5

        if variable_revision_number == 4:
            setting_values = (
                setting_values[0:2]
                + setting_values[3:6]
                + ["Yes"]
                + [setting_values[2]]
                + setting_values[6:]
            )

            variable_revision_number = 5

        return setting_values, variable_revision_number


Relate = RelateObjects
