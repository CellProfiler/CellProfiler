"""
FlagImage
=========

**FlagImage** allows you to flag an image based on properties that you
specify, for example, quality control measurements.

This module allows you to assign a flag if an image meets certain
measurement criteria that you specify (for example, if the image fails a
quality control measurement). The value of the flag is 1 if the image
meets the selected criteria (for example, if it fails QC), and 0 if it
does not meet the criteria (if it passes QC).

The flag can be used in
post-processing to filter out images you do not want to analyze, e.g.,
in CellProfiler Analyst. In addition, you can use
**ExportToSpreadsheet** to generate a file that includes the flag as a
metadata measurement associated with the images. The **Metadata** module
can then use this flag to put images that pass QC into one group and
images that fail into another.

A flag can be based on one or more
measurements. If you create a flag based on more than one measurement,
you can choose between setting the flag if all measurements are outside
the bounds or if one of the measurements is outside of the bounds. This
module must be placed in the pipeline after the relevant measurement
modules upon which the flags are based.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============

"""

import logging
import os

import numpy
from cellprofiler_core.constants.measurement import IMAGE, COLTYPE_INTEGER
from cellprofiler_core.constants.module import IO_FOLDER_CHOICE_HELP_TEXT
from cellprofiler_core.constants.workspace import DISPOSITION_CONTINUE, DISPOSITION_SKIP
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.module import Module
from cellprofiler_core.preferences import DEFAULT_INPUT_FOLDER_NAME
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting import Divider
from cellprofiler_core.setting import HiddenCount
from cellprofiler_core.setting import Measurement
from cellprofiler_core.setting import SettingsGroup
from cellprofiler_core.setting import ValidationError
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.do_something import DoSomething, RemoveSettingButton
from cellprofiler_core.setting.multichoice import MultiChoice
from cellprofiler_core.setting.subscriber import LabelSubscriber
from cellprofiler_core.setting.text import Text, Filename, Directory, Float

from cellprofiler.utilities.rules import Rules

LOGGER = logging.getLogger(__name__)

C_ANY = "Flag if any fail"
C_ALL = "Flag if all fail"

S_IMAGE = "Whole-image measurement"
S_AVERAGE_OBJECT = "Average measurement for all objects in each image"
S_ALL_OBJECTS = "Measurements for all objects in each image"
S_RULES = "Rules"
S_CLASSIFIER = "Classifier"
S_ALL = [S_IMAGE, S_AVERAGE_OBJECT, S_ALL_OBJECTS, S_RULES, S_CLASSIFIER]

"""Number of settings in the module, aside from those in the flags"""
N_FIXED_SETTINGS = 1

"""Number of settings in each flag, aside from those in the measurements"""
N_FIXED_SETTINGS_PER_FLAG = 5

N_SETTINGS_PER_MEASUREMENT_V2 = 7
N_SETTINGS_PER_MEASUREMENT_V3 = 9
N_SETTINGS_PER_MEASUREMENT_V4 = 10
"""Number of settings per measurement"""
N_SETTINGS_PER_MEASUREMENT = 11


class FlagImage(Module):
    category = "Data Tools"
    variable_revision_number = 5
    module_name = "FlagImage"

    def __init__(self):
        self.rules = Rules()

        super(FlagImage, self).__init__()

    def create_settings(self):
        self.flags = []
        self.flag_count = HiddenCount(self.flags)
        self.add_flag_button = DoSomething("", "Add another flag", self.add_flag)
        self.spacer_1 = Divider()
        self.add_flag(can_delete=False)
        self.ignore_flag_on_last = Binary(
            "Ignore flag skips on last cycle?",
            False,
            doc="""\
When set to *{YES}*, this option allows you to bypass skipping on the last
cycle of an image group.  This behavior is usually not desired, but may be 
useful when using SaveImages 'Save on last cycle' option for an image made
by any other module than MakeProjection, CorrectIlluminationCalculate, and Tile.
""".format(
                **{"YES": "Yes"}
            ),
        )

    def add_flag(self, can_delete=True):
        group = SettingsGroup()
        group.append("divider1", Divider(line=False))
        group.append("measurement_settings", [])
        group.append("measurement_count", HiddenCount(group.measurement_settings))
        group.append(
            "category",
            Text(
                "Name the flag's category",
                "Metadata",
                doc="""\
Name a measurement category by which to categorize the flag. The
*Metadata* category is the default used in CellProfiler to store
information about images (referred to as *metadata*).

The flag is stored as a per-image measurement whose name is a
combination of the flag’s category and the flag name that you choose, separated by
underscores. For instance, if the measurement category is *Metadata* and
the flag name is *QCFlag*, then the default measurement name would be
*Metadata_QCFlag*.
""",
            ),
        )

        group.append(
            "feature_name",
            Text(
                "Name the flag",
                "QCFlag",
                doc="""\
The flag is stored as a per-image measurement whose name is a
combination of the flag’s category and the flag name that you choose, separated by
underscores. For instance, if the measurement category is *Metadata* and
the flag name is *QCFlag*, then the default measurement name would be
*Metadata_QCFlag*.
""",
            ),
        )

        group.append(
            "combination_choice",
            Choice(
                "How should measurements be linked?",
                [C_ANY, C_ALL],
                doc="""\
For combinations of measurements, you can set the criteria under which
an image set is flagged:

-  *%(C_ANY)s:* An image set will be flagged if any of its measurements
   fail. This can be useful for flagging images possessing multiple QC
   flaws; for example, you can flag all bright images and all out of
   focus images with one flag.
-  *%(C_ALL)s:* A flag will only be assigned if all measurements fail.
   This can be useful for flagging images that possess only a
   combination of QC flaws; for example, you can flag only images that
   are both bright and out of focus.
"""
                % globals(),
            ),
        )

        group.append(
            "wants_skip",
            Binary(
                "Skip image set if flagged?",
                False,
                doc="""\
Select *Yes* to skip the remainder of the pipeline for image sets
that are flagged. CellProfiler will not run subsequent modules in the
pipeline on the images for any image set that is flagged. Select *No*
for CellProfiler to continue to process the pipeline regardless of
flagging.

You may want to skip processing in order to filter out unwanted images.
For instance, you may want to exclude out of focus images when running
**CorrectIllumination_Calculate**. You can do this with a pipeline that
measures image quality and flags inappropriate images before it runs
**CorrectIllumination_Calculate**.
"""
                % globals(),
            ),
        )

        group.append(
            "add_measurement_button",
            DoSomething(
                "",
                "Add another measurement",
                self.add_measurement,
                group,
                doc="""Add another measurement as a criteria.""",
            ),
        )
        self.add_measurement(group, False if not can_delete else True)
        if can_delete:
            group.append(
                "remover",
                RemoveSettingButton("", "Remove this flag", self.flags, group),
            )
        group.append("divider2", Divider(line=True))
        self.flags.append(group)

    def add_measurement(self, flag_settings, can_delete=True):
        measurement_settings = flag_settings.measurement_settings

        group = SettingsGroup()
        group.append("divider1", Divider(line=False))
        group.append(
            "source_choice",
            Choice(
                "Flag is based on",
                S_ALL,
                doc="""\
-  *%(S_IMAGE)s:* A per-image measurement, such as intensity or
   granularity.
-  *%(S_AVERAGE_OBJECT)s:* The average of all object measurements in
   the image.
-  *%(S_ALL_OBJECTS)s:* All the object measurements in an image,
   without averaging. In other words, if *any* of the objects meet the
   criteria, the image will be flagged.
-  *%(S_RULES)s:* Use a text file of rules produced by CellProfiler
   Analyst. With this option, you will have to ensure that this pipeline
   produces every measurement in the rules file upstream of this module.
-  *%(S_CLASSIFIER)s:* Use a classifier built by CellProfiler Analyst.
"""
                % globals(),
            ),
        )

        group.append(
            "object_name",
            LabelSubscriber(
                "Select the object to be used for flagging",
                "None",
                doc="""\
*(Used only when flag is based on an object measurement)*

Select the objects whose measurements you want to use for flagging.
""",
            ),
        )

        def object_fn():
            if group.source_choice == S_IMAGE:
                return IMAGE
            return group.object_name.value

        group.append(
            "rules_directory",
            Directory(
                "Rules file location",
                doc="""\
*(Used only when flagging using "{rules}")*

Select the location of the rules file that will be used for flagging images.
{folder_choice}
""".format(
                    rules=S_RULES, folder_choice=IO_FOLDER_CHOICE_HELP_TEXT
                ),
            ),
        )

        def get_directory_fn():
            """Get the directory for the rules file name"""
            return group.rules_directory.get_absolute_path()

        def set_directory_fn(path):
            dir_choice, custom_path = group.rules_directory.get_parts_from_path(path)
            group.rules_directory.join_parts(dir_choice, custom_path)

        group.append(
            "rules_file_name",
            Filename(
                "Rules file name",
                "rules.txt",
                get_directory_fn=get_directory_fn,
                set_directory_fn=set_directory_fn,
                doc="""\
*(Used only when flagging using "%(S_RULES)s")*

The name of the rules file, most commonly from CellProfiler Analyst's
Classifier. This file should be a plain text file
containing the complete set of rules.

Each line of this file should be a rule naming a measurement to be made
on an image, for instance:

    IF (Image_ImageQuality_PowerLogLogSlope_DNA < -2.5, [0.79, -0.79], [-0.94, 0.94])

The above rule will score +0.79 for the positive category and -0.94
for the negative category for images whose power log slope is less
than -2.5 pixels and will score the opposite for images whose slope is
larger. The filter adds positive and negative and flags the images
whose positive score is higher than the negative score.
"""
                % globals(),
            ),
        )

        def get_rules_class_choices(group=group):
            """Get the available choices from the rules file"""
            try:
                if group.source_choice == S_CLASSIFIER:
                    return self.get_bin_labels(group)
                elif group.source_choice == S_RULES:
                    rules = self.get_rules(group)
                    nclasses = len(rules.rules[0].weights[0])
                    return [str(i) for i in range(1, nclasses + 1)]
                else:
                    return ["None"]
                rules = self.get_rules(group)
                nclasses = len(rules.rules[0].weights[0])
                return [str(i) for i in range(1, nclasses + 1)]
            except:
                return [str(i) for i in range(1, 3)]

        group.append(
            "rules_class",
            MultiChoice(
                "Class number",
                choices=["1", "2"],
                doc="""\
*(Used only when flagging using "%(S_RULES)s")*

Select which classes to flag when filtering. The CellProfiler Analyst
Classifier user interface lists the names of the classes in order. By
default, these are the positive (class 1) and negative (class 2)
classes. **FlagImage** uses the first class from CellProfiler Analyst
if you choose “1”, etc.

Please note the following:

-  The flag is set if the image falls into the selected class.
-  You can make multiple class selections. If you do so, the module will
   set the flag if the image falls into any of the selected classes.
"""
                % globals(),
            ),
        )

        group.rules_class.get_choices = get_rules_class_choices

        group.append(
            "measurement",
            Measurement(
                "Which measurement?",
                object_fn,
                doc="""Choose the measurement to be used as criteria.""",
            ),
        )

        group.append(
            "wants_minimum",
            Binary(
                "Flag images based on low values?",
                True,
                doc="""\
Select *Yes* to flag images with measurements below the specified
cutoff. If the measurement evaluates to Not-A-Number (NaN), then the
image is not flagged.
"""
                % globals(),
            ),
        )

        group.append(
            "minimum_value",
            Float("Minimum value", 0, doc="""Set a value as a lower limit."""),
        )

        group.append(
            "wants_maximum",
            Binary(
                "Flag images based on high values?",
                True,
                doc="""\
Select *Yes* to flag images with measurements above the specified
cutoff. If the measurement evaluates to Not-A-Number (NaN), then the
image is not flagged.
"""
                % globals(),
            ),
        )

        group.append(
            "maximum_value",
            Float("Maximum value", 1, doc="""Set a value as an upper limit."""),
        )

        if can_delete:
            group.append(
                "remover",
                RemoveSettingButton(
                    "", "Remove this measurement", measurement_settings, group
                ),
            )

        group.append("divider2", Divider(line=True))
        self.rules.create_settings()
        group.append("allow_fuzzy",self.rules.settings()[0])
        measurement_settings.append(group)

    def settings(self):
        result = [self.flag_count]
        for flag in self.flags:
            result += [
                flag.measurement_count,
                flag.category,
                flag.feature_name,
                flag.combination_choice,
                flag.wants_skip,
            ]
            for mg in flag.measurement_settings:
                result += [
                    mg.source_choice,
                    mg.object_name,
                    mg.measurement,
                    mg.wants_minimum,
                    mg.minimum_value,
                    mg.wants_maximum,
                    mg.maximum_value,
                    mg.rules_directory,
                    mg.rules_file_name,
                    mg.rules_class,
                    mg.allow_fuzzy,
                ]
        result += [self.ignore_flag_on_last,]
        return result

    def prepare_settings(self, setting_values):
        """Construct the correct number of flags"""
        flag_count = int(setting_values[0])
        del self.flags[:]
        self.add_flag(can_delete=False)
        while len(self.flags) < flag_count:
            self.add_flag()

        setting_values = setting_values[N_FIXED_SETTINGS:]
        for flag in self.flags:
            count = int(setting_values[0])
            # Adding a flag adds the first measurement automatically
            while len(flag.measurement_settings) < count:
                self.add_measurement(flag, can_delete=True)
            setting_values = setting_values[
                N_FIXED_SETTINGS_PER_FLAG + count * N_SETTINGS_PER_MEASUREMENT :
            ]

    def visible_settings(self):
        def measurement_visibles(m_g):
            if hasattr(m_g, "remover"):
                result = [Divider(line=True)]
            else:
                result = []
            result += [m_g.source_choice]

            if (
                m_g.source_choice == S_ALL_OBJECTS
                or m_g.source_choice == S_AVERAGE_OBJECT
            ):
                result += [m_g.object_name]
            if m_g.source_choice == S_RULES or m_g.source_choice == S_CLASSIFIER:
                result += [m_g.rules_directory, m_g.rules_file_name, m_g.rules_class]
                whatami = "Rules" if m_g.source_choice == S_RULES else "Classifier"
                for setting, s in (
                    (m_g.rules_directory, "%s file location"),
                    (m_g.rules_file_name, "%s file name"),
                ):
                    setting.text = s % whatami
                result += [m_g.allow_fuzzy]
            else:
                result += [m_g.measurement, m_g.wants_minimum]
                if m_g.wants_minimum.value:
                    result += [m_g.minimum_value]
                result += [m_g.wants_maximum]
                if m_g.wants_maximum.value:
                    result += [m_g.maximum_value]
            if hasattr(m_g, "remover"):
                result += [m_g.remover, Divider(line=True)]
            return result

        def flag_visibles(flag):
            if hasattr(flag, "remover"):
                result = [Divider(line=True), Divider(line=True)]
            else:
                result = []
            result += [flag.category, flag.feature_name, flag.wants_skip]
            if len(flag.measurement_settings) > 1:
                result += [flag.combination_choice]
            for measurement_settings in flag.measurement_settings:
                result += measurement_visibles(measurement_settings)
            result += [flag.add_measurement_button]
            if hasattr(flag, "remover"):
                result += [flag.remover, Divider(line=True), Divider(line=True)]
            return result

        result = []
        for flag in self.flags:
            result += flag_visibles(flag)

        result += [self.add_flag_button]
        result += [self.ignore_flag_on_last]
        return result

    def validate_module(self, pipeline):
        """If using rules, validate them"""
        for flag in self.flags:
            for measurement_setting in flag.measurement_settings:
                if measurement_setting.source_choice == S_RULES:
                    try:
                        rules = self.get_rules(measurement_setting)
                    except Exception as instance:
                        LOGGER.warning(
                            "Failed to load rules: %s", str(instance), exc_info=True
                        )
                        raise ValidationError(
                            str(instance), measurement_setting.rules_file_name
                        )
                    if not numpy.all([r.object_name == IMAGE for r in rules.rules]):
                        raise ValidationError(
                            "The rules listed in %s describe objects instead of images."
                            % measurement_setting.rules_file_name.value,
                            measurement_setting.rules_file_name,
                        )
                    for r in rules.rules:
                        if self.rules.Rule.return_fuzzy_measurement_name(
                            pipeline.get_measurement_columns(self),
                            "Image",
                            r.feature,
                            True,
                            measurement_setting.allow_fuzzy
                            ) == '':
                                raise ValidationError(
                                    "The rule described by %s has not been measured earlier in the pipeline."
                                    %r.feature,
                                    measurement_setting.rules_file_name,
                                )
                elif measurement_setting.source_choice == S_CLASSIFIER:
                    try:
                        self.get_classifier(measurement_setting)
                        self.get_classifier_features(measurement_setting)
                        self.get_bin_labels(measurement_setting)
                    except IOError:
                        raise ValidationError(
                            "Failed to load classifier file %s"
                            % measurement_setting.rules_file_name.value,
                            measurement_setting.rules_file_name,
                        )
                    except:
                        raise ValidationError(
                            "Unable to load %s as a classifier file"
                            % measurement_setting.rules_file_name.value,
                            measurement_setting.rules_file_name,
                        )

    def prepare_to_create_batch(self, workspace, fn_alter_path):
        for flag_settings in self.flags:
            for group in flag_settings.measurement_settings:
                group.rules_directory.alter_for_create_batch_files(fn_alter_path)

    def run(self, workspace):
        col_labels = ("Flag", "Source", "Measurement", "Value", "Pass/Fail")
        statistics = []
        for flag in self.flags:
            statistics += self.run_flag(workspace, flag)
        if self.show_window:
            workspace.display_data.statistics = statistics
            workspace.display_data.col_labels = col_labels

    def display(self, workspace, figure):
        figure.set_subplots((1, 1))
        figure.subplot_table(
            0,
            0,
            workspace.display_data.statistics,
            col_labels=workspace.display_data.col_labels,
        )

    def run_as_data_tool(self, workspace):
        m = workspace.measurements
        assert isinstance(m, Measurements)
        m.is_first_image = True
        image_set_count = m.image_set_count
        for i in range(image_set_count):
            self.run(workspace)
            img_stats = workspace.display_data.statistics
            if i == 0:
                header = ["Image set"]
                for flag_name, object_name, feature, value, pf in img_stats:
                    header.append(flag_name)
                header.append("Pass/Fail")
                statistics = [header]
            row = [str(i + 1)]
            ok = True
            for flag_name, object_name, feature, value, pf in img_stats:
                ok = ok and (pf == "Pass")
                row.append(str(value))
            row.append("Pass" if ok else "Fail")
            statistics.append(row)
            if i < image_set_count - 1:
                m.next_image_set()
        self.show_window = False
        if image_set_count > 0:
            import wx
            from wx.grid import Grid, PyGridTableBase, EVT_GRID_LABEL_LEFT_CLICK
            from cellprofiler.gui.utilities.icon import get_cp_icon

            frame = wx.Frame(workspace.frame, -1, "Flag image results")
            sizer = wx.BoxSizer(wx.VERTICAL)
            frame.SetSizer(sizer)
            grid = Grid(frame, -1)
            sizer.Add(grid, 1, wx.EXPAND)
            #
            # The flag table supplies the statistics to the grid
            # using the grid table interface
            #
            sort_order = numpy.arange(len(statistics) - 1)
            sort_col = [None]
            sort_ascending = [None]

            def on_label_clicked(event):
                col = event.GetCol()
                if sort_col[0] == col:
                    sort_ascending[0] = not sort_ascending[0]
                else:
                    sort_ascending[0] = True
                sort_col[0] = col
                data = [x[col] for x in statistics[1:]]
                try:
                    data = numpy.array(data, float)
                except ValueError:
                    data = numpy.array(data)
                if sort_ascending[0]:
                    sort_order[:] = numpy.lexsort((data,))
                else:
                    sort_order[::-1] = numpy.lexsort((data,))
                grid.ForceRefresh()

            grid.Bind(EVT_GRID_LABEL_LEFT_CLICK, on_label_clicked)

            class FlagTable(PyGridTableBase):
                def __init__(self):
                    PyGridTableBase.__init__(self)

                def GetColLabelValue(self, col):
                    if col == sort_col[0]:
                        if sort_ascending[0]:

                            return statistics[0][col] + " v"
                        else:
                            return statistics[0][col] + " ^"
                    return statistics[0][col]

                def GetNumberRows(self):
                    return len(statistics) - 1

                def GetNumberCols(self):
                    return len(statistics[0])

                def GetValue(self, row, col):
                    return statistics[sort_order[row] + 1][col]

            grid.SetTable(FlagTable())
            frame.Fit()
            max_size = int(wx.SystemSettings.GetMetric(wx.SYS_SCREEN_Y) * 3 / 4)
            if frame.Size[1] > max_size:
                frame.SetSize((frame.Size[0], max_size))
            frame.SetIcon(get_cp_icon())
            frame.Show()

    def measurement_name(self, flag):
        return "_".join((flag.category.value, flag.feature_name.value))

    def get_rules(self, measurement_group):
        """Read the rules from a file"""
        rules_file = measurement_group.rules_file_name.value
        rules_directory = measurement_group.rules_directory.get_absolute_path()
        path = os.path.join(rules_directory, rules_file)
        if not os.path.isfile(path):
            raise ValidationError("No such rules file: %s" % path, rules_file)
        else:
            rules = Rules(allow_fuzzy=measurement_group.allow_fuzzy)
            rules.parse(path)
            return rules

    def load_classifier(self, measurement_group):
        """Load the classifier pickle if not cached

        returns classifier, bin_labels, name and features
        """
        d = self.get_dictionary()
        file_ = measurement_group.rules_file_name.value
        directory_ = measurement_group.rules_directory.get_absolute_path()
        path_ = os.path.join(directory_, file_)
        if path_ not in d:
            if not os.path.isfile(path_):
                raise ValidationError(
                    "No such rules file: %s" % path_, self.rules_file_name
                )
            else:
                import joblib

                d[path_] = joblib.load(path_)
        return d[path_]

    def get_classifier(self, measurement_group):
        return self.load_classifier(measurement_group)[0]

    def get_bin_labels(self, measurement_group):
        return self.load_classifier(measurement_group)[1]

    def get_classifier_features(self, measurement_group):
        return self.load_classifier(measurement_group)[3]

    def run_flag(self, workspace, flag):
        ok, stats = self.eval_measurement(workspace, flag.measurement_settings[0])
        statistics = [tuple([self.measurement_name(flag)] + list(stats))]
        for measurement_setting in flag.measurement_settings[1:]:
            ok_1, stats = self.eval_measurement(workspace, measurement_setting)
            statistics += [tuple([self.measurement_name(flag)] + list(stats))]
            if flag.combination_choice == C_ALL:
                ok = ok or ok_1
            elif flag.combination_choice == C_ANY:
                ok = ok and ok_1
            else:
                raise NotImplementedError(
                    "Unimplemented combination choice: %s"
                    % flag.combination_choice.value
                )
        m = workspace.measurements
        assert isinstance(m, Measurements)
        m.add_image_measurement(self.measurement_name(flag), 0 if ok else 1)
        if (not ok) and flag.wants_skip:
            if self.ignore_flag_on_last and (m.group_length - m.group_index) == 0:
                workspace.disposition = DISPOSITION_CONTINUE
            else:
                workspace.disposition = DISPOSITION_SKIP
        return statistics

    def eval_measurement(self, workspace, ms):
        """Evaluate a measurement

        workspace - holds the measurements to be evaluated
        ms - the measurement settings indicating how to evaluate

        returns a tuple
           first tuple element is True = pass, False = Fail
           second tuple element has all of the statistics except for the
                        flag name
        """
        m = workspace.measurements
        assert isinstance(m, Measurements)
        fail = False
        if ms.source_choice == S_IMAGE:
            value = m.get_current_image_measurement(ms.measurement.value)
            min_value = max_value = value
            display_value = str(round(value, 3))
            source = IMAGE
        elif ms.source_choice == S_AVERAGE_OBJECT:
            data = m.get_current_measurement(ms.object_name.value, ms.measurement.value)
            if len(data) == 0:
                min_value = max_value = numpy.NaN
                fail = True
                display_value = "No objects"
            else:
                min_value = max_value = numpy.mean(data)
                display_value = str(round(min_value, 3))
            source = "Ave. %s" % ms.object_name.value
        elif ms.source_choice == S_ALL_OBJECTS:
            data = m.get_current_measurement(ms.object_name.value, ms.measurement.value)
            source = ms.object_name.value
            if len(data) == 0:
                min_value = max_value = numpy.NaN
                fail = True
                display_value = "No objects"
            else:
                min_value = numpy.min(data)
                max_value = numpy.max(data)
                if min_value == max_value:
                    display_value = str(min_value)
                else:
                    display_value = "%.3f - %.3f" % (min_value, max_value)
        elif ms.source_choice == S_RULES:
            rules = self.get_rules(ms)
            scores = rules.score(workspace.measurements)
            rules_classes = numpy.array(
                [int(x) - 1 for x in ms.rules_class.get_selections()]
            )
            #
            # There should only be one in the vector, but if not, take
            # a majority vote (e.g., are there more class 1 objects than
            # class 2?)
            #
            is_not_nan = numpy.any(~numpy.isnan(scores), 1)
            objclass = numpy.argmax(scores[is_not_nan, :], 1).flatten()
            hit_count = numpy.sum(
                objclass[:, numpy.newaxis] == rules_classes[numpy.newaxis, :]
            )
            fail = hit_count > scores.shape[0] - hit_count
            source = IMAGE
            if len(scores) > 1:
                display_value = "%d of %d" % (hit_count, scores.shape[0])
            else:
                display_value = "--"
        elif ms.source_choice == S_CLASSIFIER:
            classifier = self.get_classifier(ms)
            target_idxs = [
                self.get_bin_labels(ms).index(_)
                for _ in ms.rules_class.get_selections()
            ]
            features = []
            image_features = workspace.measurements.get_feature_names(IMAGE)
            measurement_columns = workspace.measurements.get_measurement_columns()
            for feature_name in self.get_classifier_features(ms):
                feature_name = self.rules.Rule.return_fuzzy_measurement_name(measurement_columns,IMAGE,feature_name,False,ms.allow_fuzzy)
                features.append(feature_name)

            feature_vector = numpy.array(
                [
                    0
                    if feature_name not in image_features
                    else workspace.measurements[IMAGE, feature_name]
                    for feature_name in features
                ]
            ).reshape(1, len(features))
            predicted_class = classifier.predict(feature_vector)[0]
            predicted_idx = numpy.where(classifier.classes_ == predicted_class)[0][0]
            fail = predicted_idx in target_idxs
            display_value = self.get_bin_labels(ms)[predicted_idx]
            source = IMAGE
        else:
            raise NotImplementedError(
                "Source choice of %s not implemented" % ms.source_choice
            )
        is_rc = ms.source_choice in (S_RULES, S_CLASSIFIER)
        is_meas = not is_rc
        fail = (
            is_meas
            and (
                fail
                or (ms.wants_minimum.value and min_value < ms.minimum_value.value)
                or (ms.wants_maximum.value and max_value > ms.maximum_value.value)
            )
        ) or (is_rc and fail)

        return (
            (not fail),
            (
                source,
                ms.measurement.value if is_meas else ms.source_choice.value,
                display_value,
                "Fail" if fail else "Pass",
            ),
        )

    def get_measurement_columns(self, pipeline):
        """Return column definitions for each flag measurment in the module"""
        return [
            (IMAGE, self.measurement_name(flag), COLTYPE_INTEGER) for flag in self.flags
        ]

    def get_categories(self, pipeline, object_name):
        if object_name == IMAGE:
            return [flag.category.value for flag in self.flags]
        return []

    def get_measurements(self, pipeline, object_name, category):
        if object_name != IMAGE:
            return []
        return [
            flag.feature_name.value
            for flag in self.flags
            if flag.category.value == category
        ]

    def volumetric(self):
        return True
    
    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 1:
            new_setting_values = [setting_values[0]]
            idx = 1
            for flag_idx in range(int(setting_values[0])):
                new_setting_values += setting_values[idx : idx + 4] + ["No"]
                meas_count = int(setting_values[idx])
                idx += 4
                for meas_idx in range(meas_count):
                    measurement_source = setting_values[idx]
                    if (
                        measurement_source.startswith("Measurement for all")
                        or measurement_source == "All objects"
                    ):
                        measurement_source = S_ALL_OBJECTS
                    elif measurement_source == "Average for objects":
                        measurement_source = S_AVERAGE_OBJECT
                    elif measurement_source == "Image":
                        measurement_source = S_IMAGE
                    new_setting_values += [measurement_source]
                    new_setting_values += setting_values[(idx + 1) : (idx + 7)]
                    idx += 7
            setting_values = new_setting_values
            variable_revision_number = 2

        if variable_revision_number == 2:
            # Added rules
            new_setting_values = [setting_values[0]]
            idx = 1
            for flag_idx in range(int(setting_values[0])):
                new_setting_values += setting_values[
                    idx : idx + N_FIXED_SETTINGS_PER_FLAG
                ]
                meas_count = int(setting_values[idx])
                idx += N_FIXED_SETTINGS_PER_FLAG
                for meas_idx in range(meas_count):
                    measurement_source = setting_values[idx]
                    new_setting_values += [measurement_source]
                    new_setting_values += setting_values[
                        (idx + 1) : (idx + N_SETTINGS_PER_MEASUREMENT_V2)
                    ] + [
                        Directory.static_join_string(DEFAULT_INPUT_FOLDER_NAME, "None"),
                        "rules.txt",
                    ]
                    idx += N_SETTINGS_PER_MEASUREMENT_V2
            setting_values = new_setting_values

            variable_revision_number = 3

        if variable_revision_number == 3:
            # Added rules_class
            new_setting_values = setting_values[:1]
            idx = 1
            for flag_idx in range(int(setting_values[0])):
                new_setting_values += setting_values[
                    idx : (idx + N_FIXED_SETTINGS_PER_FLAG)
                ]
                meas_count = int(setting_values[idx])
                idx += N_FIXED_SETTINGS_PER_FLAG
                for meas_idx in range(meas_count):
                    new_setting_values += setting_values[
                        idx : (idx + N_SETTINGS_PER_MEASUREMENT_V3)
                    ]
                    new_setting_values += ["1"]
                    idx += N_SETTINGS_PER_MEASUREMENT_V3
            setting_values = new_setting_values
            variable_revision_number = 4

        if variable_revision_number == 4:
            #Add ability to do fuzzy matching, skip flag on last prev added
            new_setting_values = setting_values[:1]
            idx = 1
            for flag_idx in range(int(setting_values[0])):
                new_setting_values += setting_values[
                    idx : (idx + N_FIXED_SETTINGS_PER_FLAG)
                ]
                meas_count = int(setting_values[idx])
                idx += N_FIXED_SETTINGS_PER_FLAG
                for meas_idx in range(meas_count):
                    new_setting_values += setting_values[
                        idx : (idx + N_SETTINGS_PER_MEASUREMENT_V4)
                    ]
                    new_setting_values += [False]
                    idx += N_SETTINGS_PER_MEASUREMENT_V4
            new_setting_values += setting_values[-1:]
            setting_values = new_setting_values
            variable_revision_number = 5

        return setting_values, variable_revision_number
