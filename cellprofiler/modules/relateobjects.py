# coding=utf-8

import re
import numpy
import scipy.ndimage
import skimage.segmentation
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.setting
import cellprofiler.object
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
""".format(**{
    "HELP_ON_SAVING_OBJECTS": _help.HELP_ON_SAVING_OBJECTS
})

D_NONE = "None"
D_CENTROID = "Centroid"
D_MINIMUM = "Minimum"
D_BOTH = "Both"

D_ALL = [D_NONE, D_CENTROID, D_MINIMUM, D_BOTH]

C_MEAN = "Mean"

FF_MEAN = '%s_%%s_%%s' % C_MEAN

'''Distance category'''
C_DISTANCE = 'Distance'

'''Centroid distance feature'''
FEAT_CENTROID = 'Centroid'

'''Minimum distance feature'''
FEAT_MINIMUM = 'Minimum'

'''Centroid distance measurement (FF_DISTANCE % parent)'''
FF_CENTROID = '%s_%s_%%s' % (C_DISTANCE, FEAT_CENTROID)

'''Minimum distance measurement (FF_MINIMUM % parent)'''
FF_MINIMUM = '%s_%s_%%s' % (C_DISTANCE, FEAT_MINIMUM)

FIXED_SETTING_COUNT = 5
VARIABLE_SETTING_COUNT = 1

class RelateObjects(cellprofiler.module.ObjectProcessing):
    module_name = "RelateObjects"

    variable_revision_number = 4

    def create_settings(self):
        super(RelateObjects, self).create_settings()

        self.x_name.text = "Parent objects"

        self.x_name.doc = """\
Parent objects are defined as those objects which encompass the child object.
For example, when relating speckles to the nuclei that contain them,
the nuclei are the parents.
        """

        self.x_child_name = cellprofiler.setting.ObjectNameSubscriber(
            "Child objects",
            doc="""\
Child objects are defined as those objects contained within the parent object. For example, when relating
speckles to the nuclei that contains them, the speckles are the children.
            """
        )

        self.find_parent_child_distances = cellprofiler.setting.Choice(
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
   distances.""".format(**{
                "D_NONE": D_NONE,
                "D_MINIMUM": D_MINIMUM,
                "D_CENTROID": D_CENTROID,
                "D_BOTH": D_BOTH
            })
        )

        self.wants_step_parent_distances = cellprofiler.setting.Binary(
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
nor child of cytoplasm.""".format(**{
                "YES": cellprofiler.setting.YES
            })
        )

        self.step_parent_names = []

        self.add_step_parent(can_delete=False)

        self.add_step_parent_button = cellprofiler.setting.DoSomething(
            "",
            "Add another parent",
            self.add_step_parent
        )

        self.wants_per_parent_means = cellprofiler.setting.Binary(
            "Calculate per-parent means for all child measurements?",
            False,
            doc="""\
Select "*{YES}*" to calculate the per-parent mean values of every upstream
measurement made with the children objects and store them as a
measurement for the parent; the nomenclature of this new measurement is
“Mean_<child>_<category>_<feature>”. This module
must be placed *after* all **Measure** modules that make measurements
of the children objects.""".format(**{
                "YES": cellprofiler.setting.YES
            })
        )

    def add_step_parent(self, can_delete=True):
        group = cellprofiler.setting.SettingsGroup()

        group.append(
            "step_parent_name",
            cellprofiler.setting.Choice(
                "Parent name",
                [cellprofiler.setting.NONE],
                choices_fn=self.get_step_parents,
                doc="""\
*(Used only if calculating distances to another parent)*

Choose the name of the other parent. The **RelateObjects** module will
measure the distance from this parent to the child objects in the same
manner as it does to the primary parents. You can only choose the
parents or children of the parent object."""
            )
        )

        if can_delete:
            group.append(
                "remove",
                cellprofiler.setting.RemoveSettingButton(
                    "",
                    "Remove this object",
                    self.step_parent_names, group
                )
            )

        self.step_parent_names.append(group)

    def get_step_parents(self, pipeline):
        '''Return the possible step-parents associated with the parent'''
        step_parents = set()

        parent_name = self.x_name.value

        for module in pipeline.modules():
            if module.module_num == self.module_num:
                return list(step_parents)

            # Objects that are the parent of the parents
            grandparents = module.get_measurements(
                pipeline,
                parent_name,
                cellprofiler.measurement.C_PARENT
            )

            step_parents.update(grandparents)

            # Objects that are the children of the parents
            siblings = module.get_measurements(
                pipeline,
                parent_name,
                cellprofiler.measurement.C_CHILDREN
            )

            for sibling in siblings:
                match = re.match("^([^_]+)_Count", sibling)

                if match is not None:
                    sibling_name = match.groups()[0]

                    if parent_name in module.get_measurements(
                            pipeline,
                            sibling_name,
                            cellprofiler.measurement.C_PARENT
                    ):
                        step_parents.add(sibling_name)

        return list(step_parents)

    @property
    def has_step_parents(self):
        '''True if there are possible step-parents for the parent object'''
        return (len(self.step_parent_names) > 0 and len(self.step_parent_names[0].step_parent_name.choices) > 0)

    def settings(self):
        __settings__ = super(RelateObjects, self).settings()

        # Because we're subscribing to multiple objects and we still want to have a provider,
        # we insert the child subscriber before the output provider
        __settings__.insert(1, self.x_child_name)

        __settings__ += [
            self.find_parent_child_distances,
            self.wants_per_parent_means,
            self.wants_step_parent_distances
        ]

        __settings__ += [group.step_parent_name for group in self.step_parent_names]

        return __settings__

    def visible_settings(self):

        __settings__ = super(RelateObjects, self).visible_settings()

        # See settings for insert rationale
        __settings__.insert(1, self.x_child_name)

        __settings__ += [
            self.wants_per_parent_means,
            self.find_parent_child_distances
        ]

        if (self.find_parent_child_distances != D_NONE and self.has_step_parents):
            __settings__ += [self.wants_step_parent_distances]

            if self.wants_step_parent_distances:
                for group in self.step_parent_names:
                    __settings__ += group.visible_settings()

                __settings__ += [self.add_step_parent_button]

        return __settings__

    def run(self, workspace):
        objects = workspace.object_set

        parents = objects.get_objects(self.x_name.value)

        children = objects.get_objects(self.x_child_name.value)

        child_count, parents_of = parents.relate_children(children)

        m = workspace.measurements

        m.add_measurement(
            self.x_child_name.value,
            cellprofiler.measurement.FF_PARENT % self.x_name.value,
            parents_of
        )

        m.add_measurement(
            self.x_name.value,
            cellprofiler.measurement.FF_CHILDREN_COUNT % self.x_child_name.value,
            child_count
        )

        good_parents = parents_of[parents_of != 0]

        image_numbers = numpy.ones(len(good_parents), int) * m.image_set_number

        good_children = numpy.argwhere(parents_of != 0).flatten() + 1

        if numpy.any(good_parents):
            m.add_relate_measurement(
                self.module_num,
                cellprofiler.measurement.R_PARENT,
                self.x_name.value,
                self.x_child_name.value,
                image_numbers,
                good_parents,
                image_numbers,
                good_children
            )

            m.add_relate_measurement(
                self.module_num,
                cellprofiler.measurement.R_CHILD,
                self.x_child_name.value,
                self.x_name.value,
                image_numbers,
                good_children,
                image_numbers,
                good_parents
            )

        parent_names = self.get_parent_names()

        for parent_name in parent_names:
            if self.find_parent_child_distances in (D_BOTH, D_CENTROID):
                self.calculate_centroid_distances(workspace, parent_name)

            if self.find_parent_child_distances in (D_BOTH, D_MINIMUM):
                self.calculate_minimum_distances(workspace, parent_name)

        if self.wants_per_parent_means.value:
            parent_indexes = numpy.arange(numpy.max(parents.segmented)) + 1

            for feature_name in m.get_feature_names(self.x_child_name.value):
                if not self.should_aggregate_feature(feature_name):
                    continue

                data = m.get_current_measurement(self.x_child_name.value, feature_name)

                if data is not None and len(data) > 0:
                    if len(parents_of) > 0:
                        means = scipy.ndimage.mean(data.astype(float), parents_of, parent_indexes)
                    else:
                        means = numpy.zeros((0,))
                else:
                    # No child measurements - all NaN
                    means = numpy.ones(len(parents_of)) * numpy.nan

                mean_feature_name = FF_MEAN % (self.x_child_name.value, feature_name)

                m.add_measurement(self.x_name.value, mean_feature_name, means)

        # Discover the mapping so that we can apply it to the children
        parent_labels = parents.segmented

        child_labels = children.segmented

        mapping = numpy.arange(parents.count + 1)

        mapping[parent_labels] = parent_labels

        parent_labeled_children = numpy.zeros(child_labels.shape, int)

        mask = child_labels > 0

        parent_labeled_children[mask] = mapping[parents_of[child_labels[mask] - 1]]

        y_name = self.y_name.value

        y = cellprofiler.object.Objects()

        y.segmented = parent_labeled_children

        y.parent_image = children.parent_image

        objects.add_objects(y, y_name)

        if self.show_window:
            workspace.display_data.parent_labels = parents.segmented

            workspace.display_data.parent_count = parents.count

            workspace.display_data.child_labels = children.segmented

            workspace.display_data.related_children = parent_labeled_children

            workspace.display_data.dimensions = parents.dimensions

    def display(self, workspace, figure):
        if not self.show_window:
            return

        dimensions = workspace.display_data.dimensions

        figure.set_subplots((2, 2), dimensions=dimensions)

        child_labels = workspace.display_data.child_labels

        related_children = workspace.display_data.related_children

        parent_labels = workspace.display_data.parent_labels

        max_label = max(
            parent_labels.max(),
            child_labels.max(),
            related_children.max()
        )

        seed = numpy.random.randint(256)

        figure.subplot_imshow(
            0,
            0,
            parent_labels,
            title=self.x_name.value,
            # max_label=max_label,
            # seed=seed
        )

        figure.subplot_imshow(
            1,
            0,
            child_labels,
            title=self.x_child_name.value,
            sharexy=figure.subplot(0, 0),
            # max_label=max_label,
            # seed=seed
        )

        figure.subplot_imshow(
            0,
            1,
            related_children,
            title="{} labeled by {}".format(self.x_child_name.value, self.x_name.value),
            sharexy=figure.subplot(0, 0),
            # max_label=max_label,
            # seed=seed
        )

    def get_parent_names(self):
        parent_names = [self.x_name.value]

        if self.wants_step_parent_distances.value:
            parent_names += [group.step_parent_name.value for group in self.step_parent_names]

        return parent_names

    def calculate_centroid_distances(self, workspace, parent_name):
        '''Calculate the centroid-centroid distance between parent & child'''
        meas = workspace.measurements

        sub_object_name = self.x_child_name.value

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

            dist[mask] = numpy.sqrt(numpy.sum((ccenters[mask, :] - pcenters[parents_of[mask], :]) ** 2, 1))

        meas.add_measurement(sub_object_name, FF_CENTROID % parent_name, dist)

    def calculate_minimum_distances(self, workspace, parent_name):
        '''Calculate the distance from child center to parent perimeter'''
        meas = workspace.measurements

        sub_object_name = self.x_child_name.value

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

            pperim = skimage.segmentation.find_boundaries(parents.segmented, mode="inner") * parents.segmented

            # Get a list of all points on the perimeter
            perim_loc = numpy.argwhere(pperim != 0)

            # Get the label # for each point
            perim_idx = pperim[perim_loc.transpose().tolist()]

            # Sort the points by label #
            reverse_column_order = range(children.dimensions)[::-1]

            coordinates = perim_loc[:, reverse_column_order].transpose().tolist()

            coordinates.append(perim_idx)

            idx = numpy.lexsort(coordinates)

            perim_loc = perim_loc[idx, :]

            perim_idx = perim_idx[idx]

            # Get counts and indexes to each run of perimeter points
            counts = scipy.ndimage.sum(
                numpy.ones(len(perim_idx)), perim_idx, numpy.arange(1, perim_idx[-1] + 1)
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
            dist = numpy.sqrt(numpy.sum((perim_loc[cp_index, :] - ccenters[clabel, :]) ** 2, 1))

            # Finally, find the minimum distance per child
            min_dist = scipy.ndimage.minimum(dist, clabel, numpy.arange(len(ccounts)))

            # Account for parentless children
            dist = numpy.array([numpy.NaN] * len(mask))

            dist[mask] = min_dist

        meas.add_measurement(sub_object_name, FF_MINIMUM % parent_name, dist)

    def get_parents_of(self, workspace, parent_name):
        '''Return the parents_of measurement or equivalent

        parent_name - name of parent objects

        Return a vector of parent indexes to the given parent name using
        the Parent measurement. Look for a direct parent / child link first
        and then look for relationships between self.parent_name and the
        named parent.
        '''
        meas = workspace.measurements

        parent_feature = cellprofiler.measurement.FF_PARENT % parent_name

        primary_parent = self.x_name.value

        sub_object_name = self.x_child_name.value

        primary_parent_feature = cellprofiler.measurement.FF_PARENT % primary_parent

        if parent_feature in meas.get_feature_names(sub_object_name):
            parents_of = meas.get_current_measurement(sub_object_name, parent_feature)
        elif parent_feature in meas.get_feature_names(primary_parent):
            #
            # parent_name is the grandparent of the sub-object via
            # the primary parent.
            #
            primary_parents_of = meas.get_current_measurement(sub_object_name, primary_parent_feature)

            grandparents_of = meas.get_current_measurement(primary_parent, parent_feature)

            mask = primary_parents_of != 0

            parents_of = numpy.zeros(primary_parents_of.shape[0], grandparents_of.dtype)

            if primary_parents_of.shape[0] > 0:
                parents_of[mask] = grandparents_of[primary_parents_of[mask] - 1]
        elif primary_parent_feature in meas.get_feature_names(parent_name):
            primary_parents_of = meas.get_current_measurement(sub_object_name, primary_parent_feature)

            primary_parents_of_parent = meas.get_current_measurement(parent_name, primary_parent_feature)

            if len(primary_parents_of_parent) == 0:
                return primary_parents_of_parent

            #
            # There may not be a 1-1 relationship, but we attempt to
            # construct one
            #
            reverse_lookup_len = max(numpy.max(primary_parents_of) + 1, len(primary_parents_of_parent))

            reverse_lookup = numpy.zeros(reverse_lookup_len, int)

            if primary_parents_of_parent.shape[0] > 0:
                reverse_lookup[primary_parents_of_parent] = numpy.arange(1, len(primary_parents_of_parent) + 1)

            if primary_parents_of.shape[0] > 0:
                parents_of = reverse_lookup[primary_parents_of]
        else:
            raise ValueError("Don't know how to relate {} to {}".format(primary_parent, parent_name))

        return parents_of

    ignore_features = set(cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER)

    def should_aggregate_feature(self, feature_name):
        '''Return True if aggregate measurements should be made on a feature

        feature_name - name of a measurement, such as Location_Center_X
        '''
        if feature_name.startswith(C_MEAN):
            return False

        if feature_name.startswith(cellprofiler.measurement.C_PARENT):
            return False

        if feature_name in self.ignore_features:
            return False

        return True

    def validate_module(self, pipeline):
        '''Validate the module's settings

        Relate will complain if the children and parents are related
        by a prior module or if a step-parent is named twice'''
        for module in pipeline.modules():
            if module == self:
                break

            parent_features = module.get_measurements(pipeline, self.x_child_name.value, "Parent")

            if self.x_name.value in parent_features:
                raise cellprofiler.setting.ValidationError(
                    "{} and {} were related by the {} module".format(
                        self.x_child_name.value,
                        self.x_name.value,
                        module.module_name
                    ),
                    self.x_name
                )

        if self.has_step_parents and self.wants_step_parent_distances:
            step_parents = set()
            for group in self.step_parent_names:
                if group.step_parent_name.value in step_parents:
                    raise cellprofiler.setting.ValidationError(
                        u"{} has already been chosen".format(
                            group.step_parent_name.value
                        ),
                        group.step_parent_name
                    )

                step_parents.add(group.step_parent_name.value)

    def get_child_columns(self, pipeline):
        child_columns = list(filter(
            lambda column: column[0] == self.x_child_name.value and self.should_aggregate_feature(column[1]),
            pipeline.get_measurement_columns(self)

        ))

        child_columns += self.get_child_measurement_columns(pipeline)

        return child_columns

    def get_child_measurement_columns(self, pipeline):
        columns = []
        if self.find_parent_child_distances in (D_BOTH, D_CENTROID):
            for parent_name in self.get_parent_names():
                columns += [
                    (self.x_child_name.value, FF_CENTROID % parent_name, cellprofiler.measurement.COLTYPE_INTEGER)
                ]

        if self.find_parent_child_distances in (D_BOTH, D_MINIMUM):
            for parent_name in self.get_parent_names():
                columns += [
                    (self.x_child_name.value, FF_MINIMUM % parent_name, cellprofiler.measurement.COLTYPE_INTEGER)
                ]

        return columns

    def get_measurement_columns(self, pipeline):
        '''Return the column definitions for this module's measurements'''
        columns = [
            (
                self.x_child_name.value,
                cellprofiler.measurement.FF_PARENT % self.x_name.value,
                cellprofiler.measurement.COLTYPE_INTEGER
            ),
            (
                self.x_name.value,
                cellprofiler.measurement.FF_CHILDREN_COUNT % self.x_child_name.value,
                cellprofiler.measurement.COLTYPE_INTEGER
            )
        ]

        if self.wants_per_parent_means.value:
            child_columns = self.get_child_columns(pipeline)

            columns += [
                (
                    self.x_name.value,
                    FF_MEAN % (self.x_child_name.value, column[1]),
                    cellprofiler.measurement.COLTYPE_FLOAT
                 ) for column in child_columns
                ]

        columns += self.get_child_measurement_columns(pipeline)

        return columns

    def get_object_relationships(self, pipeline):
        '''Return the object relationships produced by this module'''
        parent_name = self.x_name.value

        sub_object_name = self.x_child_name.value

        return [
            (
                cellprofiler.measurement.R_PARENT,
                parent_name,
                sub_object_name,
                cellprofiler.measurement.MCA_AVAILABLE_EACH_CYCLE
            ),
            (
                cellprofiler.measurement.R_CHILD,
                sub_object_name,
                parent_name,
                cellprofiler.measurement.MCA_AVAILABLE_EACH_CYCLE
            )
        ]

    def get_categories(self, pipeline, object_name):
        if object_name == self.x_name.value:
            if self.wants_per_parent_means:
                return [
                    "Mean_{}".format(self.x_child_name),
                    "Children"
                ]
            else:
                return ["Children"]
        elif object_name == self.x_child_name.value:
            result = ["Parent"]

            if self.find_parent_child_distances != D_NONE:
                result += [C_DISTANCE]

            return result

        return []

    def get_measurements(self, pipeline, object_name, category):
        if object_name == self.x_name.value:
            if category == u"Mean_{}".format(self.x_child_name.value):
                measurements = []

                child_columns = self.get_child_columns(pipeline)

                measurements += [column[1] for column in child_columns]

                return measurements
            elif category == "Children":
                return [u"{}_Count".format(self.x_child_name.value)]
        elif object_name == self.x_child_name.value and category == "Parent":
            return [self.x_name.value]
        elif (object_name == self.x_child_name.value and category == C_DISTANCE):
            result = []

            if self.find_parent_child_distances in (D_BOTH, D_CENTROID):
                result += ["{}_{}".format(FEAT_CENTROID, parent_name) for parent_name in self.get_parent_names()]

            if self.find_parent_child_distances in (D_BOTH, D_MINIMUM):
                result += ["{}_{}".format(FEAT_MINIMUM, parent_name) for parent_name in self.get_parent_names()]

            return result

        return []

    def prepare_settings(self, setting_values):
        setting_count = len(setting_values)

        step_parent_count = ((setting_count - FIXED_SETTING_COUNT) / VARIABLE_SETTING_COUNT)

        assert len(self.step_parent_names) > 0

        self.step_parent_names = self.step_parent_names[:1]

        for i in range(1, step_parent_count):
            self.add_step_parent()

    def upgrade_settings(self, setting_values, variable_revision_number, module_name, from_matlab):
        if variable_revision_number == 1:
            #
            # Added other distance parents
            #
            if setting_values[2] == cellprofiler.setting.DO_NOT_USE:
                find_parent_distances = D_NONE
            else:
                find_parent_distances = setting_values[2]

            if setting_values[3].upper() == cellprofiler.setting.DO_NOT_USE.upper():
                wants_step_parent_distances = cellprofiler.setting.NO
            else:
                wants_step_parent_distances = cellprofiler.setting.YES

            setting_values = (setting_values[:2] + [
                find_parent_distances,
                setting_values[4],
                wants_step_parent_distances,
                setting_values[3]
            ])

            variable_revision_number = 2

        if variable_revision_number == 2:
            setting_values = [setting_values[1], setting_values[0]] + setting_values[2:]

            variable_revision_number = 3

        if variable_revision_number == 3:
            # Added an output provider after the parent/child selection
            setting_values.insert(2, module_name)

            variable_revision_number = 4

        return setting_values, variable_revision_number, from_matlab


Relate = RelateObjects
