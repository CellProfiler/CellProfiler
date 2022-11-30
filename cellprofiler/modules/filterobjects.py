from cellprofiler_core.constants.measurement import (
    M_LOCATION_CENTER_X,
    M_LOCATION_CENTER_Y, C_CHILDREN, C_PARENT, C_LOCATION, C_NUMBER, FTR_OBJECT_NUMBER, C_COUNT, FTR_CENTER_X,
    FTR_CENTER_Y, FTR_CENTER_Z,
)
from cellprofiler_core.module.image_segmentation import ObjectProcessing
from cellprofiler_core.preferences import (
    DEFAULT_INPUT_FOLDER_NAME,
    ABSOLUTE_FOLDER_NAME,
    DEFAULT_INPUT_SUBFOLDER_NAME,
    DEFAULT_OUTPUT_SUBFOLDER_NAME,
)
from cellprofiler_core.setting import (
    Divider,
    HiddenCount,
    SettingsGroup,
    Measurement,
    Binary,
    ValidationError,
)
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.do_something import DoSomething, RemoveSettingButton
from cellprofiler_core.setting.subscriber import LabelSubscriber
from cellprofiler_core.setting.text import Directory, Filename, Float, LabelName

from cellprofiler.modules import _help

__doc__ = """\
FilterObjects
=============

**FilterObjects** eliminates objects based on their measurements (e.g.,
area, shape, texture, intensity).

This module removes selected objects based on measurements produced by
another module (e.g., **MeasureObjectSizeShape**,
**MeasureObjectIntensity**, **MeasureTexture**, etc). All objects that
do not satisfy the specified parameters will be discarded.

This module also may remove objects touching the image border or edges
of a mask. This is useful if you would like to unify images via
**SplitOrMergeObjects** before deciding to discard these objects.

Please note that the objects that pass the filtering step comprise a new
object set, and hence do not inherit the measurements associated with
the original objects. Any measurements on the new object set will need
to be made post-filtering by the desired measurement modules.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============

See also
^^^^^^^^

See also any of the **MeasureObject** modules, **MeasureTexture**,
**MeasureColocalization**, and **CalculateMath**.

{HELP_ON_SAVING_OBJECTS}

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Image measurements:**

-  *Count:* The number of objects remaining after filtering.

**Object measurements:**

-  *Parent:* The identity of the input object associated with each
   filtered (remaining) object.
-  *Location\_X, Location\_Y, Location\_Z:* The pixel (X,Y,Z)
   coordinates of the center of mass of the filtered (remaining) objects.
""".format(
    **{"HELP_ON_SAVING_OBJECTS": _help.HELP_ON_SAVING_OBJECTS}
)

import logging
import os

import numpy
import scipy
import scipy.ndimage
import scipy.sparse

import cellprofiler.gui.help
import cellprofiler_core.object
from cellprofiler.utilities.rules import Rules

LOGGER = logging.getLogger(__name__)


"""Minimal filter - pick a single object per image by minimum measured value"""
FI_MINIMAL = "Minimal"

"""Maximal filter - pick a single object per image by maximum measured value"""
FI_MAXIMAL = "Maximal"

"""Pick one object per containing object by minimum measured value"""
FI_MINIMAL_PER_OBJECT = "Minimal per object"

"""Pick one object per containing object by maximum measured value"""
FI_MAXIMAL_PER_OBJECT = "Maximal per object"

"""Keep all objects whose values fall between set limits"""
FI_LIMITS = "Limits"

FI_ALL = [
    FI_MINIMAL,
    FI_MAXIMAL,
    FI_MINIMAL_PER_OBJECT,
    FI_MAXIMAL_PER_OBJECT,
    FI_LIMITS,
]

"""The number of settings for this module in the pipeline if no additional objects"""
FIXED_SETTING_COUNT_V6 = 12

"""The location of the setting count"""
ADDITIONAL_OBJECT_SETTING_INDEX = 9

"""The location of the measurements count setting"""
MEASUREMENT_COUNT_SETTING_INDEX = 8

MODE_RULES = "Rules"
MODE_CLASSIFIERS = "Classifiers"
MODE_MEASUREMENTS = "Measurements"
MODE_BORDER = "Image or mask border"

DIR_CUSTOM = "Custom folder"

PO_BOTH = "Both parents"
PO_PARENT_WITH_MOST_OVERLAP = "Parent with most overlap"
PO_ALL = [PO_BOTH, PO_PARENT_WITH_MOST_OVERLAP]


class FilterObjects(ObjectProcessing):
    module_name = "FilterObjects"

    variable_revision_number = 10

    def __init__(self):
        self.rules = Rules()

        super(FilterObjects, self).__init__()

    def create_settings(self):
        super(FilterObjects, self).create_settings()

        self.x_name.text = """Select the objects to filter"""

        self.x_name.doc = """\
Select the set of objects that you want to filter. This setting also
controls which measurement choices appear for filtering: you can only
filter based on measurements made on the object you select. Be sure
the **FilterObjects** module is downstream of the necessary **Measure**
modules. If you
intend to use a measurement calculated by the **CalculateMath** module
to to filter objects, select the first operand’s object here, because
**CalculateMath** measurements are stored with the first operand’s
object."""

        self.y_name.text = """Name the output objects"""

        self.y_name.doc = "Enter a name for the collection of objects that are retained after applying the filter(s)."

        self.spacer_1 = Divider(line=False)

        self.mode = Choice(
            "Select the filtering mode",
            [MODE_MEASUREMENTS, MODE_RULES, MODE_BORDER, MODE_CLASSIFIERS],
            doc="""\
You can choose from the following options:

-  *{MODE_MEASUREMENTS}*: Specify a per-object measurement made by an
   upstream module in the pipeline.
-  *{MODE_BORDER}*: Remove objects touching the border of the image
   and/or the edges of an image mask.
-  *{MODE_RULES}*: Use a file containing rules generated by
   CellProfiler Analyst. You will need to ensure that the measurements
   specified by the rules file are produced by upstream modules in the
   pipeline. This setting is not compatible with data processed as 3D.
-  *{MODE_CLASSIFIERS}*: Use a file containing a trained classifier from
   CellProfiler Analyst. You will need to ensure that the measurements
   specified by the file are produced by upstream modules in the
   pipeline. This setting is not compatible with data processed as 3D.""".format(
                **{
                    "MODE_MEASUREMENTS": MODE_MEASUREMENTS,
                    "MODE_RULES": MODE_RULES,
                    "MODE_BORDER": MODE_BORDER,
                    "MODE_CLASSIFIERS": MODE_CLASSIFIERS,
                }
            ),
        )

        self.spacer_2 = Divider(line=False)

        self.measurements = []

        self.measurement_count = HiddenCount(self.measurements, "Measurement count")

        self.add_measurement(False)

        self.add_measurement_button = DoSomething(
            "", "Add another measurement", self.add_measurement
        )

        self.filter_choice = Choice(
            "Select the filtering method",
            FI_ALL,
            FI_LIMITS,
            doc="""\
*(Used only if filtering using measurements)*

There are five different ways to filter objects:

-  *{FI_LIMITS}:* Keep an object if its measurement value falls within
   a range you specify.
-  *{FI_MAXIMAL}:* Keep the object with the maximum value for the
   measurement of interest. If multiple objects share a maximal value,
   retain one object selected arbitrarily per image.
-  *{FI_MINIMAL}:* Keep the object with the minimum value for the
   measurement of interest. If multiple objects share a minimal value,
   retain one object selected arbitrarily per image.
-  *{FI_MAXIMAL_PER_OBJECT}:* This option requires you to choose a
   parent object. The parent object might contain several child objects
   of choice (for instance, mitotic spindles within a cell or FISH probe
   spots within a nucleus). Only the child object whose measurements
   equal the maximum child-measurement value among that set of child
   objects will be kept (for example, the longest spindle in each cell).
   You do not have to explicitly relate objects before using this
   module.
-  *{FI_MINIMAL_PER_OBJECT}:* Same as *Maximal per object*, except
   filtering is based on the minimum value.""".format(
                **{
                    "FI_LIMITS": FI_LIMITS,
                    "FI_MAXIMAL": FI_MAXIMAL,
                    "FI_MINIMAL": FI_MINIMAL,
                    "FI_MAXIMAL_PER_OBJECT": FI_MAXIMAL_PER_OBJECT,
                    "FI_MINIMAL_PER_OBJECT": FI_MINIMAL_PER_OBJECT,
                }
            ),
        )

        self.per_object_assignment = Choice(
            "Assign overlapping child to",
            PO_ALL,
            doc="""\
*(Used only if filtering per object)*

A child object can overlap two parent objects and can have the
maximal/minimal measurement of all child objects in both parents. This
option controls how an overlapping maximal/minimal child affects
filtering of other children of its parents and to which parent the
maximal child is assigned. The choices are:

-  *{PO_BOTH}*: The child will be assigned to both parents and all
   other children of both parents will be filtered. Only the maximal
   child per parent will be left, but if **RelateObjects** is used to
   relate the maximal child to its parent, one or the other of the
   overlapping parents will not have a child even though the excluded
   parent may have other child objects. The maximal child can still be
   assigned to both parents using a database join via the relationships
   table if you are using **ExportToDatabase** and separate object
   tables.
-  *{PO_PARENT_WITH_MOST_OVERLAP}*: The child will be assigned to
   the parent with the most overlap and a child with a less
   maximal/minimal measurement, if available, will be assigned to other
   parents. Use this option to ensure that parents with an alternate
   non-overlapping child object are assigned some child object by a
   subsequent **RelateObjects** module.""".format(
                **{
                    "PO_BOTH": PO_BOTH,
                    "PO_PARENT_WITH_MOST_OVERLAP": PO_PARENT_WITH_MOST_OVERLAP,
                }
            ),
        )

        self.enclosing_object_name = LabelSubscriber(
            "Select the objects that contain the filtered objects",
            "None",
            doc="""\
*(Used only if a per-object filtering method is selected)*

This setting selects the container (i.e., parent) objects for the
*{FI_MAXIMAL_PER_OBJECT}* and *{FI_MINIMAL_PER_OBJECT}* filtering
choices.""".format(
                **{
                    "FI_MAXIMAL_PER_OBJECT": FI_MAXIMAL_PER_OBJECT,
                    "FI_MINIMAL_PER_OBJECT": FI_MINIMAL_PER_OBJECT,
                }
            ),
        )

        self.rules_directory = Directory(
            "Select the location of the rules or classifier file",
            doc="""\
*(Used only when filtering using {MODE_RULES} or {MODE_CLASSIFIERS})*

Select the location of the rules or classifier file that will be used for
filtering.

{IO_FOLDER_CHOICE_HELP_TEXT}
""".format(
                **{
                    "MODE_CLASSIFIERS": MODE_CLASSIFIERS,
                    "MODE_RULES": MODE_RULES,
                    "IO_FOLDER_CHOICE_HELP_TEXT": _help.IO_FOLDER_CHOICE_HELP_TEXT,
                }
            ),
        )

        self.rules_class = Choice(
            "Class number",
            choices=["1", "2"],
            choices_fn=self.get_class_choices,
            doc="""\
*(Used only when filtering using {MODE_RULES} or {MODE_CLASSIFIERS})*

Select which of the classes to keep when filtering. The CellProfiler
Analyst classifier user interface lists the names of the classes in
left-to-right order. **FilterObjects** uses the first class from
CellProfiler Analyst if you choose “1”, etc.

Please note the following:

-  The object is retained if the object falls into the selected class.
-  You can make multiple class selections. If you do so, the module will
   retain the object if the object falls into any of the selected
   classes.""".format(
                **{"MODE_CLASSIFIERS": MODE_CLASSIFIERS, "MODE_RULES": MODE_RULES}
            ),
        )

        def get_directory_fn():
            """Get the directory for the rules file name"""
            return self.rules_directory.get_absolute_path()

        def set_directory_fn(path):
            dir_choice, custom_path = self.rules_directory.get_parts_from_path(path)

            self.rules_directory.join_parts(dir_choice, custom_path)

        self.rules_file_name = Filename(
            "Rules or classifier file name",
            "rules.txt",
            get_directory_fn=get_directory_fn,
            set_directory_fn=set_directory_fn,
            doc="""\
*(Used only when filtering using {MODE_RULES} or {MODE_CLASSIFIERS})*

The name of the rules or classifier file.

A rules file is a plain text file containing the complete set of rules.

Each line of the rules file should be a rule naming a measurement to be made
on the object you selected, for instance:

    IF (Nuclei_AreaShape_Area < 351.3, [0.79, -0.79], [-0.94, 0.94])

The above rule will score +0.79 for the positive category and -0.94
for the negative category for nuclei whose area is less than 351.3
pixels and will score the opposite for nuclei whose area is larger.
The filter adds positive and negative and keeps only objects whose
positive score is higher than the negative score.

A classifier file is a trained classifier exported from CellProfiler Analyst.
You will need to ensure that the measurements specified by the file are
produced by upstream modules in the pipeline. This setting is not compatible
with data processed as 3D.
""".format(
                **{"MODE_CLASSIFIERS": MODE_CLASSIFIERS, "MODE_RULES": MODE_RULES}
            ),
        )


        self.keep_removed_objects = Binary(
            "Keep removed objects as a separate set?",
            False,
            doc="""
Select *Yes* to create an object set from objects that did not pass your filter.
            
This may be useful if you want to make use of the negative (filtered out) population as well."""
        )

        self.removed_objects_name = LabelName(
            "Name the objects removed by the filter",
            "RemovedObjects",
            doc="Enter the name you want to call the objects removed by the filter.",

        )

        self.additional_objects = []

        self.additional_object_count = HiddenCount(
            self.additional_objects, "Additional object count"
        )

        self.spacer_3 = Divider(line=True)

        self.spacer_4 = Divider(line=False)

        self.additional_object_button = DoSomething(
            "Relabel additional objects to match the filtered object?",
            "Add an additional object",
            self.add_additional_object,
            doc="""\
Click this button to add an object to receive the same post-filtering labels as
the filtered object. This is useful in making sure that labeling is maintained
between related objects (e.g., primary and secondary objects) after filtering.""",
        )

        self.rules.create_settings()

        self.allow_fuzzy = self.rules.settings()[0]

    def get_class_choices(self, pipeline):
        if self.mode == MODE_CLASSIFIERS:
            return self.get_bin_labels()
        elif self.mode == MODE_RULES:
            rules = self.get_rules()
            nclasses = len(rules.rules[0].weights[0])
            return [str(i) for i in range(1, nclasses + 1)]

    def get_rules_class_choices(self, pipeline):
        try:
            rules = self.get_rules()
            nclasses = len(rules.rules[0].weights[0])
            return [str(i) for i in range(1, nclasses + 1)]
        except:
            return [str(i) for i in range(1, 3)]

    def add_measurement(self, can_delete=True):
        """Add another measurement to the filter list"""
        group = SettingsGroup()

        group.append(
            "measurement",
            Measurement(
                "Select the measurement to filter by",
                self.x_name.get_value,
                "AreaShape_Area",
                doc="""\
*(Used only if filtering using {MODE_MEASUREMENTS})*

See the **Measurements** modules help pages for more information on the
features measured.""".format(
                    **{"MODE_MEASUREMENTS": MODE_MEASUREMENTS}
                ),
            ),
        )

        group.append(
            "wants_minimum",
            Binary(
                "Filter using a minimum measurement value?",
                True,
                doc="""\
*(Used only if {FI_LIMITS} is selected for filtering method)*

Select "*{YES}*" to filter the objects based on a minimum acceptable
object measurement value. Objects which are greater than or equal to
this value will be retained.""".format(
                    **{"FI_LIMITS": FI_LIMITS, "YES": "Yes"}
                ),
            ),
        )

        group.append("min_limit", Float("Minimum value", 0))

        group.append(
            "wants_maximum",
            Binary(
                "Filter using a maximum measurement value?",
                True,
                doc="""\
*(Used only if {FI_LIMITS} is selected for filtering method)*

Select "*{YES}*" to filter the objects based on a maximum acceptable
object measurement value. Objects which are less than or equal to this
value will be retained.""".format(
                    **{"FI_LIMITS": FI_LIMITS, "YES": "Yes"}
                ),
            ),
        )

        group.append("max_limit", Float("Maximum value", 1))

        group.append("divider", Divider())

        self.measurements.append(group)

        if can_delete:
            group.append(
                "remover",
                RemoveSettingButton(
                    "", "Remove this measurement", self.measurements, group
                ),
            )

    def add_additional_object(self):
        group = SettingsGroup()

        group.append(
            "object_name",
            LabelSubscriber("Select additional object to relabel", "None"),
        )

        group.append(
            "target_name", LabelName("Name the relabeled objects", "FilteredGreen"),
        )

        group.append(
            "remover",
            RemoveSettingButton(
                "", "Remove this additional object", self.additional_objects, group
            ),
        )

        group.append("divider", Divider(line=False))

        self.additional_objects.append(group)

    def prepare_settings(self, setting_values):
        """Make sure the # of slots for additional objects matches
           the anticipated number of additional objects"""
        additional_object_count = int(setting_values[ADDITIONAL_OBJECT_SETTING_INDEX])
        while len(self.additional_objects) > additional_object_count:
            self.remove_additional_object(self.additional_objects[-1].key)
        while len(self.additional_objects) < additional_object_count:
            self.add_additional_object()

        measurement_count = int(setting_values[MEASUREMENT_COUNT_SETTING_INDEX])
        while len(self.measurements) > measurement_count:
            del self.measurements[-1]
        while len(self.measurements) < measurement_count:
            self.add_measurement()

    def settings(self):
        settings = super(FilterObjects, self).settings()

        settings += [
            self.mode,
            self.filter_choice,
            self.enclosing_object_name,
            self.rules_directory,
            self.rules_file_name,
            self.rules_class,
            self.measurement_count,
            self.additional_object_count,
            self.per_object_assignment,
            self.keep_removed_objects,
            self.removed_objects_name,
        ]

        for x in self.measurements:
            settings += x.pipeline_settings()

        for x in self.additional_objects:
            settings += [x.object_name, x.target_name]

        settings += [self.allow_fuzzy]

        return settings

    def help_settings(self):
        return [
            self.x_name,
            self.y_name,
            self.mode,
            self.filter_choice,
            self.per_object_assignment,
            self.rules_directory,
            self.rules_file_name,
            self.rules_class,
            self.keep_removed_objects,
            self.removed_objects_name,
            self.enclosing_object_name,
            self.additional_object_button,
            self.allow_fuzzy,
        ]

    def visible_settings(self):
        visible_settings = super(FilterObjects, self).visible_settings()

        visible_settings += [self.spacer_2, self.mode]

        if self.mode == MODE_RULES or self.mode == MODE_CLASSIFIERS:
            visible_settings += [
                self.allow_fuzzy,
                self.rules_file_name,
                self.rules_directory,
                self.rules_class,
            ]
            self.rules_class.text = (
                "Class number" if self.mode == MODE_RULES else "Class name"
            )
            try:
                self.rules_class.test_valid(None)
            except:
                pass

        elif self.mode == MODE_MEASUREMENTS:
            visible_settings += [self.spacer_1, self.filter_choice]
            if self.filter_choice in (FI_MINIMAL, FI_MAXIMAL):
                visible_settings += [
                    self.measurements[0].measurement,
                    self.measurements[0].divider,
                ]
            elif self.filter_choice in (FI_MINIMAL_PER_OBJECT, FI_MAXIMAL_PER_OBJECT):
                visible_settings += [
                    self.per_object_assignment,
                    self.measurements[0].measurement,
                    self.enclosing_object_name,
                    self.measurements[0].divider,
                ]
            elif self.filter_choice == FI_LIMITS:
                for i, group in enumerate(self.measurements):
                    visible_settings += [group.measurement, group.wants_minimum]
                    if group.wants_minimum:
                        visible_settings.append(group.min_limit)
                    visible_settings.append(group.wants_maximum)
                    if group.wants_maximum.value:
                        visible_settings.append(group.max_limit)
                    if i > 0:
                        visible_settings += [group.remover]
                    visible_settings += [group.divider]
                visible_settings += [self.add_measurement_button]
        visible_settings += [self.spacer_3, self.keep_removed_objects]
        if self.keep_removed_objects.value:
            visible_settings += [self.removed_objects_name]
        visible_settings += [self.spacer_4]
        for x in self.additional_objects:
            visible_settings += x.visible_settings()
        visible_settings += [self.additional_object_button]
        return visible_settings

    def validate_module(self, pipeline):
        """Make sure that the user has selected some limits when filtering"""
        if self.mode == MODE_MEASUREMENTS and self.filter_choice == FI_LIMITS:
            for group in self.measurements:
                if not (group.wants_minimum.value or group.wants_maximum.value):
                    raise ValidationError(
                        "Please enter a minimum and/or maximum limit for your measurement",
                        group.wants_minimum,
                    )
        if self.mode == MODE_RULES:
            try:
                rules = self.get_rules()
            except Exception as instance:
                LOGGER.warning(
                    "Failed to load rules: %s", str(instance), exc_info=True
                )
                raise ValidationError(str(instance), self.rules_file_name)
            for r in rules.rules:
                if self.rules.Rule.return_fuzzy_measurement_name(
                    pipeline.get_measurement_columns(self),
                    r.object_name,
                    r.feature,
                    True,
                    self.allow_fuzzy
                    ) == '':
                    raise ValidationError(
                        (
                            "The rules file, %s, uses the measurement, %s "
                            "for object %s, but that measurement is not available "
                            "at this stage of the pipeline. Consider editing the "
                            "rules to match the available measurements or adding "
                            "measurement modules to produce the measurement."
                        )
                        % (self.rules_file_name, r.feature, r.object_name),
                        self.rules_file_name,
                    )
        elif self.mode == MODE_CLASSIFIERS:
            try:
                self.get_classifier()
                self.get_bin_labels()
                self.get_classifier_features()
            except IOError:
                raise ValidationError(
                    "Failed to load classifier file %s" % self.rules_file_name.value,
                    self.rules_file_name,
                )
            except:
                raise ValidationError(
                    "Unable to load %s as a classifier file"
                    % self.rules_file_name.value,
                    self.rules_file_name,
                )
            features = self.get_classifier_features()

            for feature in features:
                fuzzy_feature = self.rules.Rule.return_fuzzy_measurement_name(
                    pipeline.get_measurement_columns(),
                    feature[:feature.index('_')],
                    feature[feature.index('_'):],
                    True,
                    self.allow_fuzzy
                    )
                if fuzzy_feature == '':
                    raise ValidationError(
                        f"""The classifier {self.rules_file_name}, requires the measurement "{feature}", but that 
measurement is not available at this stage of the pipeline. Consider adding modules to produce the measurement.""",
                        self.rules_file_name
                    )              

    def run(self, workspace):
        """Filter objects for this image set, display results"""
        src_objects = workspace.get_objects(self.x_name.value)
        if self.mode == MODE_RULES:
            indexes = self.keep_by_rules(workspace, src_objects)
        elif self.mode == MODE_MEASUREMENTS:
            if self.filter_choice in (FI_MINIMAL, FI_MAXIMAL):
                indexes = self.keep_one(workspace, src_objects)
            if self.filter_choice in (FI_MINIMAL_PER_OBJECT, FI_MAXIMAL_PER_OBJECT):
                indexes = self.keep_per_object(workspace, src_objects)
            if self.filter_choice == FI_LIMITS:
                indexes = self.keep_within_limits(workspace, src_objects)
        elif self.mode == MODE_BORDER:
            indexes = self.discard_border_objects(src_objects)
        elif self.mode == MODE_CLASSIFIERS:
            indexes = self.keep_by_class(workspace, src_objects)
        else:
            raise ValueError("Unknown filter choice: %s" % self.mode.value)

        #
        # Create an array that maps label indexes to their new values
        # All labels to be deleted have a value in this array of zero
        #
        new_object_count = len(indexes)
        max_label = numpy.max(src_objects.segmented)
        label_indexes = numpy.zeros((max_label + 1,), int)
        label_indexes[indexes] = numpy.arange(1, new_object_count + 1)
        #
        # Loop over both the primary and additional objects
        #
        object_list = [(self.x_name.value, self.y_name.value)] + [
            (x.object_name.value, x.target_name.value) for x in self.additional_objects
        ]
        m = workspace.measurements
        first_set = True
        for src_name, target_name in object_list:
            src_objects = workspace.get_objects(src_name)
            target_labels = src_objects.segmented.copy()
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
            target_objects.unedited_segmented = src_objects.unedited_segmented
            #
            # Remove the filtered objects from the small_removed_segmented
            # if present. "small_removed_segmented" should really be
            # "filtered_removed_segmented".
            #
            small_removed = src_objects.small_removed_segmented.copy()
            small_removed[(target_labels == 0) & (src_objects.segmented != 0)] = 0
            target_objects.small_removed_segmented = small_removed
            if src_objects.has_parent_image:
                target_objects.parent_image = src_objects.parent_image
            workspace.object_set.add_objects(target_objects, target_name)

            self.add_measurements(workspace, src_name, target_name)
            if self.show_window and first_set:
                workspace.display_data.src_objects_segmented = src_objects.segmented
                workspace.display_data.target_objects_segmented = target_objects.segmented
                workspace.display_data.dimensions = src_objects.dimensions
                first_set = False

        if self.keep_removed_objects.value:
            # Isolate objects removed by the filter
            removed_indexes = [x for x in range(1, max_label+1) if x not in indexes]
            removed_object_count = len(removed_indexes)
            removed_label_indexes = numpy.zeros((max_label + 1,), int)
            removed_label_indexes[removed_indexes] = numpy.arange(1, removed_object_count + 1)

            src_objects = workspace.get_objects(self.x_name.value)
            removed_labels = src_objects.segmented.copy()
            #
            # Reindex the labels of the old source image
            #
            removed_labels[removed_labels > max_label] = 0
            removed_labels = removed_label_indexes[removed_labels]
            #
            # Make a new set of objects - retain the old set's unedited
            # segmentation for the new and generally try to copy stuff
            # from the old to the new.
            #
            removed_objects = cellprofiler_core.object.Objects()
            removed_objects.segmented = removed_labels
            removed_objects.unedited_segmented = src_objects.unedited_segmented
            #
            # Remove the filtered objects from the small_removed_segmented
            # if present. "small_removed_segmented" should really be
            # "filtered_removed_segmented".
            #
            small_removed = src_objects.small_removed_segmented.copy()
            small_removed[(removed_labels == 0) & (src_objects.segmented != 0)] = 0
            removed_objects.small_removed_segmented = small_removed
            if src_objects.has_parent_image:
                removed_objects.parent_image = src_objects.parent_image
            workspace.object_set.add_objects(removed_objects, self.removed_objects_name.value)

            self.add_measurements(workspace, self.x_name.value, self.removed_objects_name.value)
            if self.show_window:
                workspace.display_data.removed_objects_segmented = removed_objects.segmented

    def display(self, workspace, figure):
        """Display what was filtered"""
        src_name = self.x_name.value
        src_objects_segmented = workspace.display_data.src_objects_segmented
        target_objects_segmented = workspace.display_data.target_objects_segmented
        dimensions = workspace.display_data.dimensions

        target_name = self.y_name.value

        figure.set_subplots((2, 2), dimensions=dimensions)

        figure.subplot_imshow_labels(
            0, 0, src_objects_segmented, title="Original: %s" % src_name
        )

        figure.subplot_imshow_labels(
            1,
            0,
            target_objects_segmented,
            title="Filtered: %s" % target_name,
            sharexy=figure.subplot(0, 0),
        )

        pre = numpy.max(src_objects_segmented)
        post = numpy.max(target_objects_segmented)

        statistics = [[pre], [post], [pre - post]]

        figure.subplot_table(
            0,
            1,
            statistics,
            row_labels=(
                "Number of objects pre-filtering",
                "Number of objects post-filtering",
                "Number of objects removed",
            ),
        )

        if self.keep_removed_objects:
            removed_objects_segmented = workspace.display_data.removed_objects_segmented
            figure.subplot_imshow_labels(
                1,
                1,
                removed_objects_segmented,
                title="Removed: %s" % self.removed_objects_name,
                sharexy=figure.subplot(0, 0),
            )


    def keep_one(self, workspace, src_objects):
        """Return an array containing the single object to keep

        workspace - workspace passed into Run
        src_objects - the Objects instance to be filtered
        """
        measurement = self.measurements[0].measurement.value
        src_name = self.x_name.value
        values = workspace.measurements.get_current_measurement(src_name, measurement)
        if len(values) == 0:
            return numpy.array([], int)
        best_idx = (
            numpy.argmax(values)
            if self.filter_choice == FI_MAXIMAL
            else numpy.argmin(values)
        ) + 1
        return numpy.array([best_idx], int)

    def keep_per_object(self, workspace, src_objects):
        """Return an array containing the best object per enclosing object

        workspace - workspace passed into Run
        src_objects - the Objects instance to be filtered
        """
        measurement = self.measurements[0].measurement.value
        src_name = self.x_name.value
        enclosing_name = self.enclosing_object_name.value
        src_objects = workspace.get_objects(src_name)
        enclosing_objects = workspace.get_objects(enclosing_name)
        enclosing_labels = enclosing_objects.segmented
        enclosing_max = enclosing_objects.count
        if enclosing_max == 0:
            return numpy.array([], int)
        enclosing_range = numpy.arange(1, enclosing_max + 1)
        #
        # Make a vector of the value of the measurement per label index.
        # We can then label each pixel in the image with the measurement
        # value for the object at that pixel.
        # For unlabeled pixels, put the minimum value if looking for the
        # maximum value and vice-versa
        #
        values = workspace.measurements.get_current_measurement(src_name, measurement)
        wants_max = self.filter_choice == FI_MAXIMAL_PER_OBJECT
        src_labels = src_objects.segmented
        src_count = src_objects.count
        if self.per_object_assignment == PO_PARENT_WITH_MOST_OVERLAP:
            #
            # Find the number of overlapping pixels in enclosing
            # and source objects
            #
            mask = enclosing_labels * src_labels != 0
            enclosing_labels = enclosing_labels[mask]
            src_labels = src_labels[mask]
            order = numpy.lexsort((enclosing_labels, src_labels))
            src_labels = src_labels[order]
            enclosing_labels = enclosing_labels[order]
            firsts = numpy.hstack(
                (
                    [0],
                    numpy.where(
                        (src_labels[:-1] != src_labels[1:])
                        | (enclosing_labels[:-1] != enclosing_labels[1:])
                    )[0]
                    + 1,
                    [len(src_labels)],
                )
            )
            areas = firsts[1:] - firsts[:-1]
            enclosing_labels = enclosing_labels[firsts[:-1]]
            src_labels = src_labels[firsts[:-1]]
            #
            # Re-sort by source label value and area descending
            #
            if wants_max:
                svalues = -values
            else:
                svalues = values
            order = numpy.lexsort((-areas, svalues[src_labels - 1]))
            src_labels, enclosing_labels, areas = [
                x[order] for x in (src_labels, enclosing_labels, areas)
            ]
            firsts = numpy.hstack(
                (
                    [0],
                    numpy.where(src_labels[:-1] != src_labels[1:])[0] + 1,
                    src_labels.shape[:1],
                )
            )
            counts = firsts[1:] - firsts[:-1]
            #
            # Process them in order. The maximal or minimal child
            # will be assigned to the most overlapping parent and that
            # parent will be excluded.
            #
            best_src_label = numpy.zeros(enclosing_max + 1, int)
            for idx, count in zip(firsts[:-1], counts):
                for i in range(count):
                    enclosing_object_number = enclosing_labels[idx + i]
                    if best_src_label[enclosing_object_number] == 0:
                        best_src_label[enclosing_object_number] = src_labels[idx]
                        break
            #
            # Remove best source labels = 0 and sort to get the list
            #
            best_src_label = best_src_label[best_src_label != 0]
            best_src_label.sort()
            return best_src_label
        else:
            tricky_values = numpy.zeros((len(values) + 1,))
            tricky_values[1:] = values
            if wants_max:
                tricky_values[0] = -numpy.Inf
            else:
                tricky_values[0] = numpy.Inf
            src_values = tricky_values[src_labels]
            #
            # Now find the location of the best for each of the enclosing objects
            #
            fn = (
                scipy.ndimage.maximum_position
                if wants_max
                else scipy.ndimage.minimum_position
            )
            best_pos = fn(src_values, enclosing_labels, enclosing_range)
            best_pos = numpy.array(
                (best_pos,) if isinstance(best_pos, tuple) else best_pos
            )
            best_pos = best_pos.astype(numpy.uint32)
            #
            # Get the label of the pixel at each location
            #
            # Multidimensional indexing with non-tuple values is not allowed as of numpy 1.23
            best_pos = tuple(map(tuple, best_pos.transpose()))
            indexes = src_labels[best_pos]
            indexes = set(indexes)
            indexes = list(indexes)
            indexes.sort()
            return indexes[1:] if len(indexes) > 0 and indexes[0] == 0 else indexes

    def keep_within_limits(self, workspace, src_objects):
        """Return an array containing the indices of objects to keep

        workspace - workspace passed into Run
        src_objects - the Objects instance to be filtered
        """
        src_name = self.x_name.value
        hits = None
        m = workspace.measurements
        for group in self.measurements:
            measurement = group.measurement.value
            values = m.get_current_measurement(src_name, measurement)
            if hits is None:
                hits = numpy.ones(len(values), bool)
            elif len(hits) < len(values):
                temp = numpy.ones(len(values), bool)
                temp[~hits] = False
                hits = temp
            low_limit = group.min_limit.value
            high_limit = group.max_limit.value
            if group.wants_minimum.value:
                hits[values < low_limit] = False
            if group.wants_maximum.value:
                hits[values > high_limit] = False
        indexes = numpy.argwhere(hits)[:, 0]
        indexes = indexes + 1
        return indexes

    def discard_border_objects(self, src_objects):
        """Return an array containing the indices of objects to keep

        workspace - workspace passed into Run
        src_objects - the Objects instance to be filtered
        """
        labels = src_objects.segmented

        if src_objects.has_parent_image and src_objects.parent_image.has_mask:

            mask = src_objects.parent_image.mask

            interior_pixels = scipy.ndimage.binary_erosion(mask)

        else:

            interior_pixels = scipy.ndimage.binary_erosion(numpy.ones_like(labels))

        border_pixels = numpy.logical_not(interior_pixels)

        border_labels = set(labels[border_pixels])

        if (
            border_labels == {0}
            and src_objects.has_parent_image
            and src_objects.parent_image.has_mask
        ):
            # The assumption here is that, if nothing touches the border,
            # the mask is a large, elliptical mask that tells you where the
            # well is. That's the way the old Matlab code works and it's duplicated here
            #
            # The operation below gets the mask pixels that are on the border of the mask
            # The erosion turns all pixels touching an edge to zero. The not of this
            # is the border + formerly masked-out pixels.

            mask = src_objects.parent_image.mask

            interior_pixels = scipy.ndimage.binary_erosion(mask)

            border_pixels = numpy.logical_not(interior_pixels)

            border_labels = set(labels[border_pixels])

        return list(set(labels.ravel()).difference(border_labels))

    def get_rules(self):
        """Read the rules from a file"""
        rules_file = self.rules_file_name.value
        rules_directory = self.rules_directory.get_absolute_path()
        path = os.path.join(rules_directory, rules_file)
        if not os.path.isfile(path):
            raise ValidationError("No such rules file: %s" % path, self.rules_file_name)
        else:
            rules = Rules(allow_fuzzy=self.allow_fuzzy)
            rules.parse(path)
            return rules

    def load_classifier(self):
        """Load the classifier pickle if not cached

        returns classifier, bin_labels, name and features
        """
        d = self.get_dictionary()
        file_ = self.rules_file_name.value
        directory_ = self.rules_directory.get_absolute_path()
        path_ = os.path.join(directory_, file_)
        if path_ not in d:
            if not os.path.isfile(path_):
                raise ValidationError(
                    "No such classifier file: %s" % path_, self.rules_file_name
                )
            else:
                if not file_.endswith('.txt'):
                    # Probably a model file
                    import joblib
                    d[path_] = joblib.load(path_)
                    if len(d[path_]) < 3:
                        raise IOError("The selected model file doesn't look like a CellProfiler Analyst classifier."
                                      "See the help dialog for more info on model formats.")
                    if d[path_][2] == "FastGentleBoosting":
                        # FGB model files are not sklearn-based, we'll load it as rules instead.
                        rules = Rules(allow_fuzzy=self.allow_fuzzy)
                        rules.load(d[path_][0])
                        d[path_] = (rules,
                                    d[path_][1],
                                    "Rules",
                                    [f"{rule.object_name}_{rule.feature}" for rule in rules.rules])
                else:
                    # Probably a rules list
                    rules = Rules(allow_fuzzy=self.allow_fuzzy)
                    rules.parse(path_)
                    # Construct a classifier-like object
                    d[path_] = (rules,
                                rules.get_classes(),
                                "Rules",
                                [f"{rule.object_name}_{rule.feature}" for rule in rules.rules])
        return d[path_]

    def get_classifier(self):
        return self.load_classifier()[0]

    def get_bin_labels(self):
        return self.load_classifier()[1]

    def get_classifier_type(self):
        return self.load_classifier()[2]

    def get_classifier_features(self):
        return self.load_classifier()[3]

    def keep_by_rules(self, workspace, src_objects, rules=None):
        """Keep objects according to rules

        workspace - workspace holding the measurements for the rules
        src_objects - filter these objects (uses measurement indexes instead)
        rules - supply pre-generated rules loaded from a classifier model file

        Open the rules file indicated by the settings and score the
        objects by the rules. Return the indexes of the objects that pass.
        """
        if not rules:
            rules = self.get_rules()
            rules_class = int(self.rules_class.value) - 1
        else:
            rules_class = self.get_bin_labels().index(self.rules_class.value)
        scores = rules.score(workspace.measurements)
        if len(scores) > 0:
            is_not_nan = numpy.any(~numpy.isnan(scores), 1)
            best_class = numpy.argmax(scores[is_not_nan], 1).flatten()
            hits = numpy.zeros(scores.shape[0], bool)
            hits[is_not_nan] = best_class == rules_class
            indexes = numpy.argwhere(hits).flatten() + 1
        else:
            indexes = numpy.array([], int)
        return indexes

    def keep_by_class(self, workspace, src_objects):
        """ Keep objects according to their predicted class
        :param workspace: workspace holding the measurements for the rules
        :param src_objects: filter these objects (uses measurement indexes instead)
        :return: indexes (base 1) of the objects that pass
        """
        classifier = self.get_classifier()
        if self.get_classifier_type() == "Rules":
            return self.keep_by_rules(workspace, src_objects, rules=classifier)
        target_idx = self.get_bin_labels().index(self.rules_class.value)
        target_class = classifier.classes_[target_idx]
        features = self.split_feature_names(self.get_classifier_features(), workspace.object_set.get_object_names())
        feature_vector = numpy.column_stack(
            [
                workspace.measurements[
                    object_name, 
                    self.rules.Rule.return_fuzzy_measurement_name(
                        workspace.measurements.get_measurement_columns(),
                        object_name,
                        feature_name,
                        False,
                        self.allow_fuzzy
                        )
                        ]
                for object_name, feature_name in features
            ]
        )
        if hasattr(classifier, 'scaler') and classifier.scaler is not None:
            feature_vector = classifier.scaler.transform(feature_vector)
        numpy.nan_to_num(feature_vector, copy=False)
        predicted_classes = classifier.predict(feature_vector)
        hits = predicted_classes == target_class
        indexes = numpy.argwhere(hits) + 1
        return indexes.flatten()

    def get_measurement_columns(self, pipeline):
        return super(FilterObjects, self).get_measurement_columns(
            pipeline,
            additional_objects=[
                (x.object_name.value, x.target_name.value)
                for x in self.additional_objects
            ] + [(self.x_name.value,self.removed_objects_name.value)] if self.keep_removed_objects.value else [],
        )

    def get_categories(self, pipeline, object_name):
        categories = super(FilterObjects, self).get_categories(pipeline, object_name)
        if self.keep_removed_objects.value and object_name == self.removed_objects_name.value:
            categories += [C_PARENT, C_LOCATION, C_NUMBER]
        return categories

    def get_measurements(self, pipeline, object_name, category):
        if object_name == self.x_name.value and category == C_CHILDREN:
            measures = ["%s_Count" % self.y_name.value]
            if self.keep_removed_objects.value and object_name == self.removed_objects_name.value:
                measures += ["%s_Count" % self.removed_objects_name.value]
            return measures

        if object_name == self.y_name.value or (
                self.keep_removed_objects.value and object_name == self.removed_objects_name.value):
            if category == C_NUMBER:
                return [FTR_OBJECT_NUMBER]
            if category == C_PARENT:
                return [self.x_name.value]
            if category == C_LOCATION:
                return [FTR_CENTER_X, FTR_CENTER_Y, FTR_CENTER_Z,]

        if object_name == "Image" and category == C_COUNT:
            measures = [self.y_name.value]
            if self.keep_removed_objects.value:
                measures.append(self.removed_objects_name.value)
            return measures
        return []

    def prepare_to_create_batch(self, workspace, fn_alter_path):
        """Prepare to create a batch file

        This function is called when CellProfiler is about to create a
        file for batch processing. It will pickle the image set list's
        "legacy_fields" dictionary. This callback lets a module prepare for
        saving.

        pipeline - the pipeline to be saved
        image_set_list - the image set list to be saved
        fn_alter_path - this is a function that takes a pathname on the local
                        host and returns a pathname on the remote host. It
                        handles issues such as replacing backslashes and
                        mapping mountpoints. It should be called for every
                        pathname stored in the settings or legacy fields.
        """
        self.rules_directory.alter_for_create_batch_files(fn_alter_path)
        return True

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 1:
            #
            # Added CPA rules
            #
            setting_values = (
                setting_values[:11]
                + [MODE_MEASUREMENTS, DEFAULT_INPUT_FOLDER_NAME, ".",]
                + setting_values[11:]
            )
            variable_revision_number = 2
        if variable_revision_number == 2:
            #
            # Forgot file name (???!!!)
            #
            setting_values = setting_values[:14] + ["rules.txt"] + setting_values[14:]
            variable_revision_number = 3
        if variable_revision_number == 3:
            #
            # Allowed multiple measurements
            # Structure changed substantially.
            #
            (
                target_name,
                object_name,
                measurement,
                filter_choice,
                enclosing_objects,
                wants_minimum,
                minimum_value,
                wants_maximum,
                maximum_value,
                wants_outlines,
                outlines_name,
                rules_or_measurements,
                rules_directory_choice,
                rules_path_name,
                rules_file_name,
            ) = setting_values[:15]
            additional_object_settings = setting_values[15:]
            additional_object_count = len(additional_object_settings) // 4

            setting_values = [
                target_name,
                object_name,
                rules_or_measurements,
                filter_choice,
                enclosing_objects,
                wants_outlines,
                outlines_name,
                rules_directory_choice,
                rules_path_name,
                rules_file_name,
                "1",
                str(additional_object_count),
                measurement,
                wants_minimum,
                minimum_value,
                wants_maximum,
                maximum_value,
            ] + additional_object_settings
            variable_revision_number = 4
        if variable_revision_number == 4:
            #
            # Used Directory to combine directory choice & custom path
            #
            rules_directory_choice = setting_values[7]
            rules_path_name = setting_values[8]
            if rules_directory_choice == DIR_CUSTOM:
                rules_directory_choice = ABSOLUTE_FOLDER_NAME
                if rules_path_name.startswith("."):
                    rules_directory_choice = DEFAULT_INPUT_SUBFOLDER_NAME
                elif rules_path_name.startswith("&"):
                    rules_directory_choice = DEFAULT_OUTPUT_SUBFOLDER_NAME
                    rules_path_name = "." + rules_path_name[1:]

            rules_directory = Directory.static_join_string(
                rules_directory_choice, rules_path_name
            )
            setting_values = setting_values[:7] + [rules_directory] + setting_values[9:]
            variable_revision_number = 5

        if variable_revision_number == 5:
            #
            # added rules class
            #
            setting_values = setting_values[:9] + ["1"] + setting_values[9:]
            variable_revision_number = 6

        if variable_revision_number == 6:
            #
            # Added per-object assignment
            #
            setting_values = (
                setting_values[:FIXED_SETTING_COUNT_V6]
                + [PO_BOTH]
                + setting_values[FIXED_SETTING_COUNT_V6:]
            )

            variable_revision_number = 7

        if variable_revision_number == 7:
            x_name = setting_values[1]

            y_name = setting_values[0]

            measurement_count = int(setting_values[10])

            additional_object_count = int(setting_values[11])

            n_measurement_settings = measurement_count * 5

            additional_object_settings = setting_values[13 + n_measurement_settings :]

            additional_object_names = additional_object_settings[::4]

            additional_target_names = additional_object_settings[1::4]

            new_additional_object_settings = sum(
                [
                    [object_name, target_name]
                    for object_name, target_name in zip(
                        additional_object_names, additional_target_names
                    )
                ],
                [],
            )

            setting_values = (
                [x_name, y_name]
                + setting_values[2:5]
                + setting_values[7 : 13 + n_measurement_settings]
                + new_additional_object_settings
            )

            variable_revision_number = 8

        if variable_revision_number == 8:
            # Add default values for "keep removed objects".
            setting_values.insert(11, "No")
            setting_values.insert(12, "RemovedObjects")
            variable_revision_number = 9

        slot_directory = 5

        setting_values[slot_directory] = Directory.upgrade_setting(
            setting_values[slot_directory]
        )

        if variable_revision_number == 9:
            setting_values.append(False)
            variable_revision_number = 10

        return setting_values, variable_revision_number

    def get_dictionary_for_worker(self):
        # Sklearn models can't be serialized, so workers will need to read them from disk.
        return {}

    def split_feature_names(self, features, available_objects):
        # Attempts to split measurement names into object and feature pairs. Tests against a list of available objects.
        features_list = []
        # We want to test the longest keys first, so that "Cells_Edited" is matched before "Cells".
        available_objects = tuple(sorted(available_objects, key=len, reverse=True))
        for feature_name in features:
            obj, feature_name = next(((s, feature_name.split(f"{s}_", 1)[-1]) for s in available_objects if
                                      feature_name.startswith(s)), feature_name.split("_", 1))
            features_list.append((obj, feature_name))
        return features_list
        
#
# backwards compatibility
#
FilterByObjectMeasurement = FilterObjects
