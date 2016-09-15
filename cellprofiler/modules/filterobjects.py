'''<b>Filter Objects</b> eliminates objects based on their measurements (e.g., area, shape,
texture, intensity).
<hr>
This module removes selected objects based on measurements produced by another module (e.g.,
<b>MeasureObjectSizeShape</b>, <b>MeasureObjectIntensity</b>, <b>MeasureTexture</b>, etc).
All objects that do not satisfy the specified parameters will be discarded.

<p>This module also may remove objects touching the image border or edges of a mask. This is useful if
you would like to unify images via <b>ReassignObjectNumbers</b> before deciding to discard these objects.</p>

<p>Please note that the objects that pass the filtering step comprise a new object set, and hence do
not inherit the measurements associated with the original objects. Any measurements on the new object
set will need to be made post-filtering by the desired measurement modules.</p>

<h4>Available measurements</h4>
<b>Image measurements:</b>
<ul>
<li><i>Count:</i> The number of objects remaining after filtering.</li>
</ul>
<b>Object measurements:</b>
<ul>
<li><i>Parent:</i> The identity of the input object associated with each filtered
object.</li>
<li><i>Location_X, Location_Y:</i> The pixel (X,Y) coordinates of the center of
mass of the remaining objects.</li>
</ul>

See also any of the <b>MeasureObject</b> modules, <b>MeasureTexture</b>,
<b>MeasureCorrelation</b>, and <b>CalculateRatios</b>.
'''

import logging

logger = logging.getLogger(__name__)
import numpy as np
import os
import scipy as sp
import scipy.ndimage as scind
from scipy.sparse import coo_matrix
import traceback

from cellprofiler.modules.identify import FF_PARENT, FF_CHILDREN_COUNT, \
     M_LOCATION_CENTER_X, M_LOCATION_CENTER_Y
import cellprofiler.image as cpi
import cellprofiler.module as cpm
import cellprofiler.object as cpo
import cellprofiler.setting as cps
from cellprofiler.setting import YES, NO
import cellprofiler.measurement as cpmeas
import cellprofiler.preferences as cpprefs
import cellprofiler.utilities.rules as cprules
from centrosome.outline import outline
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix
from cellprofiler.modules.identify import add_object_count_measurements
from cellprofiler.modules.identify import add_object_location_measurements
from cellprofiler.modules.identify import get_object_measurement_columns
from cellprofiler.preferences import IO_FOLDER_CHOICE_HELP_TEXT
from cellprofiler.gui.help import RETAINING_OUTLINES_HELP, NAMING_OUTLINES_HELP

'''Minimal filter - pick a single object per image by minimum measured value'''
FI_MINIMAL = "Minimal"

'''Maximal filter - pick a single object per image by maximum measured value'''
FI_MAXIMAL = "Maximal"

'''Pick one object per containing object by minimum measured value'''
FI_MINIMAL_PER_OBJECT = "Minimal per object"

'''Pick one object per containing object by maximum measured value'''
FI_MAXIMAL_PER_OBJECT = "Maximal per object"

'''Keep all objects whose values fall between set limits'''
FI_LIMITS = "Limits"

FI_ALL = [FI_MINIMAL, FI_MAXIMAL, FI_MINIMAL_PER_OBJECT,
          FI_MAXIMAL_PER_OBJECT, FI_LIMITS]

'''The number of settings for this module in the pipeline if no additional objects'''
FIXED_SETTING_COUNT = 13

FIXED_SETTING_COUNT_V6 = 12

'''The number of settings per additional object'''
ADDITIONAL_OBJECT_SETTING_COUNT = 4

'''The location of the setting count'''
ADDITIONAL_OBJECT_SETTING_INDEX = 11

'''The location of the measurements count setting'''
MEASUREMENT_COUNT_SETTING_INDEX = 10

MODE_RULES = "Rules"
MODE_CLASSIFIERS = "Classifiers"
MODE_MEASUREMENTS = "Measurements"
MODE_BORDER = "Image or mask border"

DIR_CUSTOM = "Custom folder"

PO_BOTH = "Both parents"
PO_PARENT_WITH_MOST_OVERLAP = "Parent with most overlap"
PO_ALL = [PO_BOTH, PO_PARENT_WITH_MOST_OVERLAP]


class FilterObjects(cpm.Module):
    module_name = 'FilterObjects'
    category = "Object Processing"
    variable_revision_number = 7

    def create_settings(self):
        '''Create the initial settings and name the module'''
        self.target_name = cps.ObjectNameProvider(
            'Name the output objects', 'FilteredBlue', doc="""
            Enter a name for the collection of objects that are retained after applying the filter(s).""")

        self.object_name = cps.ObjectNameSubscriber(
            'Select the object to filter', cps.NONE, doc="""
            Select the set of objects that you want to filter. This setting
            also controls which measurement choices appear for filtering:
            you can only filter based on measurements made on the object you select.
            If you intend to use a measurement
            calculated by the <b>CalculateMath</b> module to to filter objects, select
            the first operand's object here, because <b>CalculateMath</b> measurements
            are stored with the first operand's object.""")

        self.spacer_1 = cps.Divider(line=False)

        self.mode = cps.Choice(
            'Select the filtering mode',
            [MODE_MEASUREMENTS, MODE_RULES, MODE_BORDER, MODE_CLASSIFIERS],
            doc = """
            You can choose from the following options:
            <ul>
            <li><i>%(MODE_MEASUREMENTS)s</i>: Specify a per-object measurement made by an upstream
            module in the pipeline.</li>
            <li><i>%(MODE_RULES)s</i>: Use a file containing rules generated by CellProfiler Analyst.
            You will need to ensure that the measurements specified by the rules file are
            produced by upstream modules in the pipeline.</li>
            <li><i>%(MODE_BORDER)s</i>: Remove objects touching the border of the image and/or the
            edges of an image mask.</li>
            <li><i>%(MODE_CLASSIFIERS)s</i>: Use a file containing trained classifier from CellProfiler Analyst.
            You will need to ensure that the measurements specified by the file are
            produced by upstream modules in the pipeline.</li>
            </ul>""" % globals())
        self.spacer_2 = cps.Divider(line=False)

        self.measurements = []
        self.measurement_count = cps.HiddenCount(self.measurements,
                                                 "Measurement count")
        self.add_measurement(False)
        self.add_measurement_button = cps.DoSomething(
            "", "Add another measurement", self.add_measurement)
        self.filter_choice = cps.Choice(
            "Select the filtering method", FI_ALL, FI_LIMITS, doc="""
            <i>(Used only if filtering using measurements)</i><br>
            There are five different ways to filter objects:
            <ul>
            <li><i>%(FI_LIMITS)s:</i> Keep an object if its measurement value falls within a range you specify.</li>
            <li><i>%(FI_MAXIMAL)s:</i> Keep the object with the maximum value for the measurement
            of interest. If multiple objects share a maximal value, retain one object
            selected arbitrarily per image.</li>
            <li><i>%(FI_MINIMAL)s:</i> Keep the object with the minimum value for the measurement
            of interest. If multiple objects share a minimal value, retain one object
            selected arbitrarily per image.</li>
            <li><i>%(FI_MAXIMAL_PER_OBJECT)s:</i> This option requires you to choose a parent object.
            The parent object might contain several child objects of
            choice (for instance, mitotic spindles within a cell or FISH
            probe spots within a nucleus). Only the child object whose measurements equal the maximum child-measurement
            value among that set of child objects will be kept
            (for example, the longest spindle
            in each cell).  You do not have to explicitly relate objects before using this module.</li>
            <li><i>%(FI_MINIMAL_PER_OBJECT)s:</i> Same as <i>Maximal per object</i>, except filtering is based on the minimum value.</li>
            </ul>""" % globals())

        self.per_object_assignment = cps.Choice(
            "Assign overlapping child to", PO_ALL,
            doc="""
            <i>(Used only if filtering per object)</i>
            <br>A child object can overlap two parent objects and can have
            the maximal/minimal measurement of all child objects in both parents.
            This option controls how an overlapping maximal/minimal child
            affects filtering of other children of its parents and to which
            parent the maximal child is assigned. The choices are:
            <br><ul>
            <li><i>%(PO_BOTH)s</i>: The child will be assigned to both parents
            and all other children of both parents will be filtered. Only the
            maximal child per parent will be left, but if <b>RelateObjects</b>
            is used to relate the maximal child to its parent, one or the other
            of the overlapping parents will not have a child even though the
            excluded parent may have other child objects. The maximal child
            can still be assigned to both parents using a database join
            via the relationships table if you are using <b>ExportToDatabase</b>
            and separate object tables.</li>
            <li><i>%(PO_PARENT_WITH_MOST_OVERLAP)s</i>: The child will be
            assigned to the parent with the most overlap and a child with a
            less maximal/minimal measurement, if available, will be assigned
            to other parents. Use this option to ensure that parents with
            an alternate non-overlapping child object are assigned some child
            object by a subseequent <b>RelateObjects</b> module.</li></ul>
            """ % globals())

        self.enclosing_object_name = cps.ObjectNameSubscriber(
            'Select the objects that contain the filtered objects', cps.NONE, doc="""
            <i>(Used only if a per-object filtering method is selected)</i><br>
            This setting selects the container (i.e., parent) objects for the <i>%(FI_MAXIMAL_PER_OBJECT)s</i>
            and <i>%(FI_MINIMAL_PER_OBJECT)s</i> filtering choices.""" % globals())

        self.rules_directory = cps.DirectoryPath(
            "Rules file location", doc="""
            <i>(Used only when filtering using %(MODE_RULES)s)</i>
            <br>
            Select the location of the rules file that will be used for filtering.
            %(IO_FOLDER_CHOICE_HELP_TEXT)s""" % globals())

        self.rules_class = cps.Choice(
            "Class number",
            choices = ["1", "2"],
            choices_fn = self.get_class_choices, doc ="""
            <i>(Used only when filtering using %(MODE_RULES)s)</i><br>
            Select which of the classes to keep when filtering. The
            CellProfiler Analyst classifier user interface lists the names of
            the classes in left-to-right order. <b>FilterObjects</b> uses the
            first class from CellProfiler Analyst if you choose "1", etc.
            <p>Please note the following:
            <ul>
            <li>The object is retained if the object falls into the selected class.</li>
            <li>You can make multiple class selections. If you do so, the module
            will retain the object if the object falls into any of the selected classes.</li>
            </ul></p>""" % globals())

        def get_directory_fn():
            '''Get the directory for the rules file name'''
            return self.rules_directory.get_absolute_path()

        def set_directory_fn(path):
            dir_choice, custom_path = self.rules_directory.get_parts_from_path(path)
            self.rules_directory.join_parts(dir_choice, custom_path)

        self.rules_file_name = cps.FilenameText(
            "Rules file name", "rules.txt",
            get_directory_fn=get_directory_fn,
            set_directory_fn=set_directory_fn, doc="""
            <i>(Used only when filtering using %(MODE_RULES)s)</i><br>
            The name of the rules file. This file should be a plain text
            file containing the complete set of rules.
            <p>Each line of
            this file should be a rule naming a measurement to be made
            on the object you selected, for instance:
            <pre>IF (Nuclei_AreaShape_Area &lt; 351.3, [0.79, -0.79], [-0.94, 0.94])</pre>
            <br><br>
            The above rule will score +0.79 for the positive category and -0.94
            for the negative category for nuclei whose area is less than 351.3
            pixels and will score the opposite for nuclei whose area is larger.
            The filter adds positive and negative and keeps only objects whose
            positive score is higher than the negative score.</p>""" %
                                                                     globals())

        self.wants_outlines = cps.Binary(
            'Retain outlines of the identified objects?', False, doc="""
            %(RETAINING_OUTLINES_HELP)s""" % globals())

        self.outlines_name = cps.OutlineNameProvider(
            'Name the outline image', 'FilteredObjects', doc="""
            %(NAMING_OUTLINES_HELP)s""" % globals())

        self.additional_objects = []
        self.additional_object_count = cps.HiddenCount(self.additional_objects,
                                                       "Additional object count")
        self.spacer_3 = cps.Divider(line=False)

        self.additional_object_button = cps.DoSomething(
            'Relabel additional objects to match the filtered object?',
            'Add an additional object', self.add_additional_object, doc="""
            Click this button to add an object to receive the same post-filtering labels as
            the filtered object. This is useful in making sure that labeling is maintained
            between related objects (e.g., primary and secondary objects) after filtering.""")

    def get_class_choices(self, pipeline):
        if self.mode == MODE_CLASSIFIERS:
            return self.get_bin_labels()
        elif self.mode == MODE_RULES:
            rules = self.get_rules()
            nclasses = len(rules.rules[0].weights[0])
            return [str(i) for i in range(1, nclasses+1)]

    def get_rules_class_choices(self, pipeline):
        try:
            rules = self.get_rules()
            nclasses = len(rules.rules[0].weights[0])
            return [str(i) for i in range(1, nclasses + 1)]
        except:
            return [str(i) for i in range(1, 3)]

    def add_measurement(self, can_delete=True):
        '''Add another measurement to the filter list'''
        group = cps.SettingsGroup()
        group.append("measurement", cps.Measurement(
            'Select the measurement to filter by',
            self.object_name.get_value, "AreaShape_Area", doc="""
            <i>(Used only if filtering using %(MODE_MEASUREMENTS)s)</i><br>
            See the <b>Measurements</b> modules help pages
            for more information on the features measured.""" % globals()))

        group.append("wants_minimum", cps.Binary(
            'Filter using a minimum measurement value?', True, doc="""
            <i>(Used only if %(FI_LIMITS)s is selected for filtering method)</i><br>
            Select <i>%(YES)s</i> to filter the objects based on a minimum acceptable object
            measurement value. Objects which are greater than or equal to this value
            will be retained.""" % globals()))

        group.append("min_limit", cps.Float('Minimum value', 0))

        group.append("wants_maximum", cps.Binary(
            'Filter using a maximum measurement value?', True, doc="""
            <i>(Used only if %(FI_LIMITS)s is selected for filtering method)</i><br>
            Select <i>%(YES)s</i> to filter the objects based on a maximum acceptable object
            measurement value. Objects which are less than or equal to this value
            will be retained.""" % globals()))

        group.append("max_limit", cps.Float('Maximum value', 1))
        group.append("divider", cps.Divider())
        self.measurements.append(group)
        if can_delete:
            group.append("remover", cps.RemoveSettingButton(
                "", "Remove this measurement",
                self.measurements, group))

    def add_additional_object(self):
        group = cps.SettingsGroup()
        group.append("object_name",
                     cps.ObjectNameSubscriber('Select additional object to relabel',
                                              cps.NONE))
        group.append("target_name",
                     cps.ObjectNameProvider('Name the relabeled objects', 'FilteredGreen'))

        group.append("wants_outlines",
                     cps.Binary('Retain outlines of relabeled objects?', False))

        group.append("outlines_name",
                     cps.OutlineNameProvider('Name the outline image', 'OutlinesFilteredGreen'))

        group.append("remover",
                     cps.RemoveSettingButton("", "Remove this additional object", self.additional_objects, group))
        group.append("divider", cps.Divider(line=False))
        self.additional_objects.append(group)

    def prepare_settings(self, setting_values):
        '''Make sure the # of slots for additional objects matches
           the anticipated number of additional objects'''
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
        result = [self.target_name, self.object_name, self.mode,
                  self.filter_choice, self.enclosing_object_name,
                  self.wants_outlines, self.outlines_name,
                  self.rules_directory,
                  self.rules_file_name,
                  self.rules_class,
                  self.measurement_count, self.additional_object_count,
                  self.per_object_assignment]
        for x in self.measurements:
            result += x.pipeline_settings()
        for x in self.additional_objects:
            result += [x.object_name, x.target_name, x.wants_outlines, x.outlines_name]
        return result

    def help_settings(self):
        return [self.target_name, self.object_name, self.mode,
                self.filter_choice, self.per_object_assignment,
                self.rules_directory,
                self.rules_file_name, self.rules_class,
                self.enclosing_object_name,
                self.wants_outlines, self.outlines_name]

    def visible_settings(self):
        result =[self.target_name, self.object_name,
                 self.spacer_2, self.mode]

        if self.mode == MODE_RULES or self.mode == MODE_CLASSIFIERS:
            result += [self.rules_file_name, self.rules_directory,
                       self.rules_class]
            self.rules_class.text = "Class number" if self.mode == MODE_RULES \
                else "Class name"
            try:
                self.rules_class.test_valid(None)
            except:
                pass

        elif self.mode == MODE_MEASUREMENTS:
            result += [self.spacer_1, self.filter_choice]
            if self.filter_choice in (FI_MINIMAL, FI_MAXIMAL):
                result += [self.measurements[0].measurement,
                           self.measurements[0].divider]
            elif self.filter_choice in (FI_MINIMAL_PER_OBJECT,
                                        FI_MAXIMAL_PER_OBJECT):
                result += [self.per_object_assignment,
                           self.measurements[0].measurement,
                           self.enclosing_object_name,
                           self.measurements[0].divider]
            elif self.filter_choice == FI_LIMITS:
                for i, group in enumerate(self.measurements):
                    result += [group.measurement, group.wants_minimum]
                    if group.wants_minimum:
                        result.append(group.min_limit)
                    result.append(group.wants_maximum)
                    if group.wants_maximum.value:
                        result.append(group.max_limit)
                    if i > 0:
                        result += [group.remover]
                    result += [group.divider]
                result += [self.add_measurement_button]
        result.append(self.wants_outlines)
        if self.wants_outlines.value:
            result.append(self.outlines_name)
        result.append(self.spacer_3)
        for x in self.additional_objects:
            temp = x.visible_settings()
            if not x.wants_outlines.value:
                del temp[temp.index(x.wants_outlines) + 1]
            result += temp
        result += [self.additional_object_button]
        return result

    def validate_module(self, pipeline):
        '''Make sure that the user has selected some limits when filtering'''
        if (self.mode == MODE_MEASUREMENTS and
            self.filter_choice == FI_LIMITS):
            for group in self.measurements:
                if (group.wants_minimum.value == False and
                    group.wants_maximum.value == False):
                    raise cps.ValidationError(
                        'Please enter a minimum and/or maximum limit for your measurement',
                        group.wants_minimum)
        if self.mode == MODE_RULES:
            try:
                rules = self.get_rules()
            except Exception, instance:
                logger.warning("Failed to load rules: %s", str(instance), exc_info=True)
                raise cps.ValidationError(str(instance),
                                          self.rules_file_name)
            measurement_columns = pipeline.get_measurement_columns(self)
            for r in rules.rules:
                if not any([mc[0] == r.object_name and
                            mc[1] == r.feature for mc in measurement_columns]):
                    raise cps.ValidationError(
                        ("The rules file, %s, uses the measurement, %s "
                         "for object %s, but that measurement is not available "
                         "at this stage of the pipeline. Consider editing the "
                         "rules to match the available measurements or adding "
                         "measurement modules to produce the measurement.") %
                        (self.rules_file_name, r.feature, r.object_name),
                        self.rules_file_name)
        elif self.mode == MODE_CLASSIFIERS:
            try:
                self.get_classifier()
                self.get_bin_labels()
                self.get_classifier_features()
            except IOError:
                raise cps.ValidationError(
                    "Failed to load classifier file %s" %
                    self.rules_file_name.value, self.rules_file_name)
            except:
                raise cps.ValidationError(
                    "Unable to load %s as a classifier file" %
                    self.rules_file_name.value, self.rules_file_name)

    def run(self, workspace):
        '''Filter objects for this image set, display results'''
        src_objects = workspace.get_objects(self.object_name.value)
        if self.mode == MODE_RULES:
            indexes = self.keep_by_rules(workspace, src_objects)
        elif self.mode == MODE_MEASUREMENTS:
            if self.filter_choice in (FI_MINIMAL, FI_MAXIMAL):
                indexes = self.keep_one(workspace, src_objects)
            if self.filter_choice in (FI_MINIMAL_PER_OBJECT,
                                      FI_MAXIMAL_PER_OBJECT):
                indexes = self.keep_per_object(workspace, src_objects)
            if self.filter_choice == FI_LIMITS:
                indexes = self.keep_within_limits(workspace, src_objects)
        elif self.mode == MODE_BORDER:
            indexes = self.discard_border_objects(workspace, src_objects)
        elif self.mode == MODE_CLASSIFIERS:
            indexes = self.keep_by_class(workspace, src_objects)
        else:
            raise ValueError("Unknown filter choice: %s" %
                             self.mode.value)

        #
        # Create an array that maps label indexes to their new values
        # All labels to be deleted have a value in this array of zero
        #
        new_object_count = len(indexes)
        max_label = np.max(src_objects.segmented)
        label_indexes = np.zeros((max_label + 1,), int)
        label_indexes[indexes] = np.arange(1, new_object_count + 1)
        #
        # Loop over both the primary and additional objects
        #
        object_list = ([(self.object_name.value, self.target_name.value,
                         self.wants_outlines.value, self.outlines_name.value)] +
                       [(x.object_name.value, x.target_name.value,
                         x.wants_outlines.value, x.outlines_name.value)
                        for x in self.additional_objects])
        m = workspace.measurements
        for src_name, target_name, wants_outlines, outlines_name in object_list:
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
            target_objects = cpo.Objects()
            target_objects.segmented = target_labels
            target_objects.unedited_segmented = src_objects.unedited_segmented
            #
            # Remove the filtered objects from the small_removed_segmented
            # if present. "small_removed_segmented" should really be
            # "filtered_removed_segmented".
            #
            small_removed = src_objects.small_removed_segmented.copy()
            small_removed[(target_labels == 0) &
                          (src_objects.segmented != 0)] = 0
            target_objects.small_removed_segmented = small_removed
            if src_objects.has_parent_image:
                target_objects.parent_image = src_objects.parent_image
            workspace.object_set.add_objects(target_objects, target_name)
            #
            # Add measurements for the new objects
            add_object_count_measurements(m, target_name, new_object_count)
            add_object_location_measurements(m, target_name, target_labels)
            #
            # Relate the old numbering to the new numbering
            #
            m.add_measurement(target_name,
                              FF_PARENT % src_name,
                              np.array(indexes))
            #
            # Count the children (0 / 1)
            #
            child_count = (label_indexes[1:] > 0).astype(int)
            m.add_measurement(src_name,
                              FF_CHILDREN_COUNT % target_name,
                              child_count)
            #
            # Add an outline if asked to do so
            #
            if wants_outlines:
                outline_image = cpi.Image(outline(target_labels) > 0,
                                          parent_image=target_objects.parent_image)
                workspace.image_set.add(outlines_name, outline_image)

        if self.show_window:
            src_objects = workspace.get_objects(src_name)
            image_names = \
                [image for image in
                 [m.measurement.get_image_name(workspace.pipeline)
                  for m in self.measurements]
                 if image is not None
                 and image in workspace.image_set.names]
            if len(image_names) == 0:
                # Measurement isn't image-based
                if src_objects.has_parent_image:
                    image = src_objects.parent_image.pixel_data
                else:
                    image = None
            else:
                image = workspace.image_set.get_image(image_names[0]).pixel_data

            workspace.display_data.src_objects_segmented = \
                src_objects.segmented
            workspace.display_data.image_names = image_names
            workspace.display_data.image = image
            workspace.display_data.target_objects_segmented = target_objects.segmented

    def display(self, workspace, figure):
        '''Display what was filtered'''
        src_name = self.object_name.value
        src_objects_segmented = workspace.display_data.src_objects_segmented
        image = workspace.display_data.image
        image_names = workspace.display_data.image_names
        target_objects_segmented = workspace.display_data.target_objects_segmented

        target_name = self.target_name.value

        if image is None:
            # Oh so sad - no image, just display the old and new labels
            figure.set_subplots((1, 2))
            figure.subplot_imshow_labels(0, 0, src_objects_segmented,
                                         title="Original: %s" % src_name)
            figure.subplot_imshow_labels(0, 1, target_objects_segmented,
                                         title="Filtered: %s" %
                                         target_name,
                                         sharexy=figure.subplot(0, 0))
        else:
            figure.set_subplots((2, 1))
            orig_minus_filtered = src_objects_segmented.copy()
            orig_minus_filtered[target_objects_segmented > 0] = 0
            cplabels = [
                dict(name=target_name,
                     labels=[target_objects_segmented]),
                dict(name="%s removed" % src_name,
                     labels=[orig_minus_filtered])]
            title = "Original: %s, Filtered: %s" % (src_name, target_name)
            if image.ndim == 3:
                figure.subplot_imshow_color(
                    0, 0, image, title=title, cplabels=cplabels)
            else:
                figure.subplot_imshow_grayscale(
                    0, 0, image, title=title, cplabels=cplabels)

            statistics = [[np.max(src_objects_segmented)],
                          [np.max(target_objects_segmented)]]
            figure.subplot_table(
                1, 0, statistics,
                row_labels=("Number of objects pre-filtering",
                            "Number of objects post-filtering"))

    def keep_one(self, workspace, src_objects):
        '''Return an array containing the single object to keep

        workspace - workspace passed into Run
        src_objects - the Objects instance to be filtered
        '''
        measurement = self.measurements[0].measurement.value
        src_name = self.object_name.value
        values = workspace.measurements.get_current_measurement(src_name,
                                                                measurement)
        if len(values) == 0:
            return np.array([], int)
        best_idx = (np.argmax(values) if self.filter_choice == FI_MAXIMAL
                    else np.argmin(values)) + 1
        return np.array([best_idx], int)

    def keep_per_object(self, workspace, src_objects):
        '''Return an array containing the best object per enclosing object

        workspace - workspace passed into Run
        src_objects - the Objects instance to be filtered
        '''
        measurement = self.measurements[0].measurement.value
        src_name = self.object_name.value
        enclosing_name = self.enclosing_object_name.value
        src_objects = workspace.get_objects(src_name)
        enclosing_objects = workspace.get_objects(enclosing_name)
        enclosing_labels = enclosing_objects.segmented
        enclosing_max = enclosing_objects.count
        if enclosing_max == 0:
            return np.array([], int)
        enclosing_range = np.arange(1, enclosing_max + 1)
        #
        # Make a vector of the value of the measurement per label index.
        # We can then label each pixel in the image with the measurement
        # value for the object at that pixel.
        # For unlabeled pixels, put the minimum value if looking for the
        # maximum value and vice-versa
        #
        values = workspace.measurements.get_current_measurement(src_name,
                                                                measurement)
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
            order = np.lexsort((enclosing_labels, src_labels))
            src_labels = src_labels[order]
            enclosing_labels = enclosing_labels[order]
            firsts = np.hstack(
                ([0],
                 np.where((src_labels[:-1] != src_labels[1:]) |
                          (enclosing_labels[:-1] != enclosing_labels[1:]))[0] + 1,
                 [len(src_labels)]))
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
            order = np.lexsort((-areas, svalues[src_labels - 1]))
            src_labels, enclosing_labels, areas = [
                x[order] for x in src_labels, enclosing_labels, areas]
            firsts = np.hstack((
                [0], np.where(src_labels[:-1] != src_labels[1:])[0] + 1,
                src_labels.shape[:1]))
            counts = firsts[1:] - firsts[:-1]
            #
            # Process them in order. The maximal or minimal child
            # will be assigned to the most overlapping parent and that
            # parent will be excluded.
            #
            best_src_label = np.zeros(enclosing_max + 1, int)
            for idx, count in zip(firsts[:-1], counts):
                for i in range(count):
                    enclosing_object_number = enclosing_labels[idx + i]
                    if best_src_label[enclosing_object_number] == 0:
                        best_src_label[enclosing_object_number] = \
                            src_labels[idx]
                        break
            #
            # Remove best source labels = 0 and sort to get the list
            #
            best_src_label = best_src_label[best_src_label != 0]
            best_src_label.sort()
            return best_src_label
        else:
            tricky_values = np.zeros((len(values) + 1,))
            tricky_values[1:] = values
            if wants_max:
                tricky_values[0] = -np.Inf
            else:
                tricky_values[0] = np.Inf
            src_values = tricky_values[src_labels]
            #
            # Now find the location of the best for each of the enclosing objects
            #
            fn = scind.maximum_position if wants_max else scind.minimum_position
            best_pos = fn(src_values, enclosing_labels, enclosing_range)
            best_pos = np.array((best_pos,) if isinstance(best_pos, tuple)
                                else best_pos)
            best_pos = best_pos.astype(np.uint32)
            #
            # Get the label of the pixel at each location
            #
            indexes = src_labels[best_pos[:, 0], best_pos[:, 1]]
            indexes = set(indexes)
            indexes = list(indexes)
            indexes.sort()
            return indexes[1:] if len(indexes) > 0 and indexes[0] == 0 else indexes

    def keep_within_limits(self, workspace, src_objects):
        '''Return an array containing the indices of objects to keep

        workspace - workspace passed into Run
        src_objects - the Objects instance to be filtered
        '''
        src_name = self.object_name.value
        hits = None
        m = workspace.measurements
        for group in self.measurements:
            measurement = group.measurement.value
            values = m.get_current_measurement(src_name,
                                               measurement)
            if hits is None:
                hits = np.ones(len(values), bool)
            elif len(hits) < len(values):
                temp = np.ones(len(values), bool)
                temp[~ hits] = False
                hits = temp
            low_limit = group.min_limit.value
            high_limit = group.max_limit.value
            if group.wants_minimum.value:
                hits[values < low_limit] = False
            if group.wants_maximum.value:
                hits[values > high_limit] = False
        indexes = np.argwhere(hits)[:, 0]
        indexes = indexes + 1
        return indexes

    def discard_border_objects(self, workspace, src_objects):
        '''Return an array containing the indices of objects to keep

        workspace - workspace passed into Run
        src_objects - the Objects instance to be filtered
        '''
        labeled_image = src_objects.segmented

        border_labeled_image = labeled_image.copy()

        border_labels = list(border_labeled_image[0, :])
        border_labels.extend(border_labeled_image[:, 0])
        border_labels.extend(border_labeled_image[border_labeled_image.shape[0] - 1, :])
        border_labels.extend(border_labeled_image[:, border_labeled_image.shape[1] - 1])
        border_labels = np.array(border_labels)
        #
        # the following histogram has a value > 0 for any object
        # with a border pixel
        #
        histogram = sp.sparse.coo_matrix((np.ones(border_labels.shape),
                                          (border_labels,
                                           np.zeros(border_labels.shape))),
                                         shape=(np.max(border_labeled_image) + 1, 1)).todense()
        histogram = np.array(histogram).flatten()
        if any(histogram[1:] > 0):
            histogram_image = histogram[border_labeled_image]
            border_labeled_image[histogram_image > 0] = 0
        elif src_objects.has_parent_image:
            if src_objects.parent_image.has_mask:
                # The assumption here is that, if nothing touches the border,
                # the mask is a large, elliptical mask that tells you where the
                # well is. That's the way the old Matlab code works and it's duplicated here
                #
                # The operation below gets the mask pixels that are on the border of the mask
                # The erosion turns all pixels touching an edge to zero. The not of this
                # is the border + formerly masked-out pixels.
                image = src_objects.parent_image
                mask_border = np.logical_not(scind.binary_erosion(image.mask))
                mask_border = np.logical_and(mask_border, image.mask)
                border_labels = labeled_image[mask_border]
                border_labels = border_labels.flatten()
                histogram = sp.sparse.coo_matrix(
                    (np.ones(border_labels.shape),
                     (border_labels,
                      np.zeros(border_labels.shape))),
                    shape=(np.max(labeled_image) + 1, 1)).todense()
                histogram = np.array(histogram).flatten()
                if any(histogram[1:] > 0):
                    histogram_image = histogram[labeled_image]
                    border_labeled_image[histogram_image > 0] = 0

        return np.unique(border_labeled_image)[1:]

    def get_rules(self):
        '''Read the rules from a file'''
        rules_file = self.rules_file_name.value
        rules_directory = self.rules_directory.get_absolute_path()
        path = os.path.join(rules_directory, rules_file)
        if not os.path.isfile(path):
            raise cps.ValidationError("No such rules file: %s" % path,
                                      self.rules_file_name)
        else:
            rules = cprules.Rules()
            rules.parse(path)
            return rules

    def load_classifier(self):
        '''Load the classifier pickle if not cached

        returns classifier, bin_labels, name and features
        '''
        d = self.get_dictionary()
        file_ = self.rules_file_name.value
        directory_ = self.rules_directory.get_absolute_path()
        path_ = os.path.join(directory_, file_)
        if path_ not in d:
            if not os.path.isfile(path_):
                raise cps.ValidationError("No such rules file: %s" % path_,
                                          self.rules_file_name)
            else:
                from sklearn.externals import joblib
                d[path_] = joblib.load(path_)
        return d[path_]

    def get_classifier(self):
        return self.load_classifier()[0]

    def get_bin_labels(self):
        return self.load_classifier()[1]

    def get_classifier_features(self):
        return self.load_classifier()[3]

    def keep_by_rules(self, workspace, src_objects):
        '''Keep objects according to rules

        workspace - workspace holding the measurements for the rules
        src_objects - filter these objects (uses measurement indexes instead)

        Open the rules file indicated by the settings and score the
        objects by the rules. Return the indexes of the objects that pass.
        '''
        rules = self.get_rules()
        rules_class = int(self.rules_class.value) - 1
        scores = rules.score(workspace.measurements)
        if len(scores) > 0:
            is_not_nan = np.any(~ np.isnan(scores), 1)
            best_class = np.argmax(scores[is_not_nan], 1).flatten()
            hits = np.zeros(scores.shape[0], bool)
            hits[is_not_nan] = best_class == rules_class
            indexes = np.argwhere(hits).flatten() + 1
        else:
            indexes = np.array([], int)
        return indexes

    def keep_by_class(self, workspace, src_objects):
        ''' Keep objects according to their predicted class
        :param workspace: workspace holding the measurements for the rules
        :param src_objects: filter these objects (uses measurement indexes instead)
        :return: indexes (base 1) of the objects that pass
        '''
        classifier = self.get_classifier()
        target_idx = self.get_bin_labels().index(self.rules_class.value)
        target_class = classifier.classes_[target_idx]
        features = []
        for feature_name in self.get_classifier_features():
            feature_name = feature_name.split("_", 1)[1]
            if feature_name == "x_loc":
                feature_name = M_LOCATION_CENTER_X
            elif feature_name == "y_loc":
                feature_name = M_LOCATION_CENTER_Y
            features.append(feature_name)

        feature_vector = np.column_stack([
            workspace.measurements[self.object_name.value, feature_name]
            for feature_name in features])
        predicted_classes = classifier.predict(feature_vector)
        hits = predicted_classes == target_class
        indexes = np.argwhere(hits) + 1
        return indexes.flatten()

    def get_measurement_columns(self, pipeline):
        '''Return measurement column defs for the parent/child measurement'''
        object_list = ([(self.object_name.value, self.target_name.value)] +
                       [(x.object_name.value, x.target_name.value)
                        for x in self.additional_objects])
        columns = []
        for src_name, target_name in object_list:
            columns += [(target_name,
                         FF_PARENT % src_name,
                         cpmeas.COLTYPE_INTEGER),
                        (src_name,
                         FF_CHILDREN_COUNT % target_name,
                         cpmeas.COLTYPE_INTEGER)]
            columns += get_object_measurement_columns(target_name)
        return columns

    def get_categories(self, pipeline, object_name):
        """Return the categories of measurements that this module produces

        object_name - return measurements made on this object (or 'Image' for image measurements)
        """
        categories = []
        if object_name == cpmeas.IMAGE:
            categories += ["Count"]
        elif object_name == self.object_name:
            categories.append("Children")
        if object_name == self.target_name.value:
            categories += ("Parent", "Location", "Number")
        return categories

    def get_measurements(self, pipeline, object_name, category):
        """Return the measurements that this module produces

        object_name - return measurements made on this object (or 'Image' for image measurements)
        category - return measurements made in this category
        """
        result = []

        if object_name == cpmeas.IMAGE:
            if category == "Count":
                result += [self.target_name.value]
        if object_name == self.object_name and category == "Children":
            result += ["%s_Count" % self.target_name.value]
        if object_name == self.target_name:
            if category == "Location":
                result += ["Center_X", "Center_Y"]
            elif category == "Parent":
                result += [self.object_name.value]
            elif category == "Number":
                result += ["Object_Number"]
        return result

    def prepare_to_create_batch(self, workspace, fn_alter_path):
        '''Prepare to create a batch file

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
        '''
        self.rules_directory.alter_for_create_batch_files(fn_alter_path)
        return True

    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        '''Account for old save formats

        setting_values - the strings for the settings as saved in the pipeline
        variable_revision_number - the variable revision number at the time
                                   of saving
        module_name - this is either FilterByObjectMeasurement for pyCP
                      and Matlab's FilterByObjectMeasurement module or
                      it is KeepLargestObject for Matlab's module of that
                      name.
        from_matlab - true if file was saved by Matlab CP
        '''

        DIR_DEFAULT_INPUT = "Default input folder"
        DIR_DEFAULT_OUTPUT = "Default output folder"

        if (module_name == 'KeepLargestObject' and from_matlab
            and variable_revision_number == 1):
            #
            # This is a specialized case:
            # The filtering method is FI_MAXIMAL_PER_OBJECT to pick out
            # the largest. The measurement is AreaShape_Area.
            # The slots are as follows:
            # 0 - the source objects name
            # 1 - the enclosing objects name
            # 2 - the target objects name
            setting_values = [setting_values[1],
                              setting_values[2],
                              "AreaShape_Area",
                              FI_MAXIMAL_PER_OBJECT,
                              setting_values[0],
                              cps.YES, "0", cps.YES, "1",
                              cps.NO, cps.NONE]
            from_matlab = False
            variable_revision_number = 1
            module_name = self.module_name
        if (module_name == 'FilterByObjectMeasurement' and from_matlab and
            variable_revision_number == 5):
            #
            # Swapped first two measurements
            #
            setting_values = ([setting_values[1], setting_values[0]] +
                              setting_values[2:])
            variable_revision_number = 6

        if (module_name == 'FilterByObjectMeasurement' and from_matlab and
            variable_revision_number == 6):
            # The measurement may not be correct here - it will display
            # as an error, though
            measurement = '_'.join((setting_values[2],
                                    setting_values[3]))
            if setting_values[6] == 'No minimum':
                wants_minimum = cps.NO
                min_limit = "0"
            else:
                wants_minimum = cps.YES
                min_limit = setting_values[6]
            if setting_values[7] == 'No maximum':
                wants_maximum = cps.NO
                max_limit = "1"
            else:
                wants_maximum = cps.YES
                max_limit = setting_values[7]
            if setting_values[8] == cps.DO_NOT_USE:
                wants_outlines = cps.NO
                outlines_name = cps.NONE
            else:
                wants_outlines = cps.YES
                outlines_name = setting_values[8]

            setting_values = [setting_values[0], setting_values[1],
                              measurement, FI_LIMITS, cps.NONE,
                              wants_minimum, min_limit,
                              wants_maximum, max_limit,
                              wants_outlines, outlines_name]
            module_name = self.module_name
            from_matlab = False
            variable_revision_number = 1
        if (from_matlab and module_name == 'FilterByObjectMeasurement' and
            variable_revision_number == 7):
            #
            # Added rules file name and rules path name
            #
            target_name, object_name, category, feature, image, scale, \
                min_value1, max_value1, save_outlines, rules_file_name, \
                rules_path_name = setting_values

            parts = [category, feature]
            if len(image) > 0:
                parts.append(image)
            if len(scale) > 0:
                parts.append(scale)
            measurement = "_".join(parts)
            if rules_file_name == cps.DO_NOT_USE:
                rules_or_measurements = MODE_MEASUREMENTS
                rules_directory_choice = DIR_DEFAULT_INPUT
            else:
                rules_or_measurements = MODE_RULES
                if rules_path_name == '.':
                    rules_directory_choice = DIR_DEFAULT_OUTPUT
                elif rules_path_name == '&':
                    rules_directory_choice = DIR_DEFAULT_INPUT
                else:
                    rules_directory_choice = DIR_CUSTOM
            if min_value1 == 'No minimum':
                wants_minimum = cps.NO
                min_limit = "0"
            else:
                wants_minimum = cps.YES
                min_limit = min_value1
            if max_value1 == 'No maximum':
                wants_maximum = cps.NO
                max_limit = "1"
            else:
                wants_maximum = cps.YES
                max_limit = max_value1
            if save_outlines == cps.DO_NOT_USE:
                wants_outlines = cps.NO
                outlines_name = cps.NONE
            else:
                wants_outlines = cps.YES
                outlines_name = save_outlines
            setting_values = [target_name,
                              object_name,
                              measurement,
                              FI_LIMITS,
                              cps.NONE,  # enclosing object name
                              wants_minimum,
                              min_limit,
                              wants_maximum,
                              max_limit,
                              wants_outlines,
                              outlines_name,
                              rules_or_measurements,
                              rules_directory_choice,
                              rules_path_name,
                              rules_file_name]
            variable_revision_number = 3
            module_name = self.module_name
            from_matlab = False

        if (not from_matlab) and variable_revision_number == 1:
            #
            # Added CPA rules
            #
            setting_values = (setting_values[:11] +
                              [MODE_MEASUREMENTS, DIR_DEFAULT_INPUT, "."] +
                              setting_values[11:])
            variable_revision_number = 2
        if (not from_matlab) and variable_revision_number == 2:
            #
            # Forgot file name (???!!!)
            #
            setting_values = (setting_values[:14] + ["rules.txt"] +
                              setting_values[14:])
            variable_revision_number = 3
        if (not from_matlab) and variable_revision_number == 3:
            #
            # Allowed multiple measurements
            # Structure changed substantially.
            #
            target_name, object_name, measurement, filter_choice, \
                enclosing_objects, wants_minimum, minimum_value, \
                wants_maximum, maximum_value, wants_outlines, \
                outlines_name, rules_or_measurements, rules_directory_choice, \
                rules_path_name, rules_file_name = setting_values[:15]
            additional_object_settings = setting_values[15:]
            additional_object_count = len(additional_object_settings) / 4

            setting_values = [
                target_name, object_name, rules_or_measurements,
                filter_choice, enclosing_objects, wants_outlines,
                outlines_name, rules_directory_choice, rules_path_name,
                rules_file_name, "1", str(additional_object_count),
                measurement, wants_minimum, minimum_value,
                wants_maximum, maximum_value] + additional_object_settings
            variable_revision_number = 4
        if (not from_matlab) and variable_revision_number == 4:
            #
            # Used DirectoryPath to combine directory choice & custom path
            #
            rules_directory_choice = setting_values[7]
            rules_path_name = setting_values[8]
            if rules_directory_choice == DIR_CUSTOM:
                rules_directory_choice == cpprefs.ABSOLUTE_FOLDER_NAME
                if rules_path_name.startswith('.'):
                    rules_directory_choice = cps.DEFAULT_INPUT_SUBFOLDER_NAME
                elif rules_path_name.startswith('&'):
                    rules_directory_choice = cps.DEFAULT_OUTPUT_SUBFOLDER_NAME
                    rules_path_name = "." + rules_path_name[1:]

            rules_directory = cps.DirectoryPath.static_join_string(
                rules_directory_choice, rules_path_name)
            setting_values = (
                setting_values[:7] + [rules_directory] + setting_values[9:])
            variable_revision_number = 5

        if (not from_matlab) and variable_revision_number == 5:
            #
            # added rules class
            #
            setting_values = setting_values[:9] + ["1"] + setting_values[9:]
            variable_revision_number = 6

        if (not from_matlab) and variable_revision_number == 6:
            #
            # Added per-object assignment
            #
            setting_values = setting_values[:FIXED_SETTING_COUNT_V6] + \
                [PO_BOTH] + setting_values[FIXED_SETTING_COUNT_V6:]
            variable_revision_number = 7

        SLOT_DIRECTORY = 7
        setting_values[SLOT_DIRECTORY] = cps.DirectoryPath.upgrade_setting(
            setting_values[SLOT_DIRECTORY])

        return setting_values, variable_revision_number, from_matlab


#
# backwards compatability
#
FilterByObjectMeasurement = FilterObjects
