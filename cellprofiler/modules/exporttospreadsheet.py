from cellprofiler.gui.help import USING_METADATA_TAGS_REF

__doc__ = '''
<b>Export To Spreadsheet</b> exports measurements into one or more files that can be
opened in Excel or other spreadsheet programs.
<hr>
This module will convert the measurements to a comma-, tab-, or other
character-delimited text format and
save them to the hard drive in one or several files, as requested.
<h2>Using metadata tags for output</h2>
<b>ExportToSpreadsheet</b> can write out separate files for groups of images based
on their metadata tags. This is controlled by the directory and file names
that you enter. For instance, you might have applied two treatments
to each of your samples and labeled them with the metadata names "Treatment1"
and "Treatment2", and you might want to create separate files for each
combination of treatments, storing all measurements with a given "Treatment1"
in separate directories. You can do this by specifying metadata tags for the folder name and
file name:
<ul>
<li>Choose <i>Elsewhere...</i> or <i>Default Input/Output Folder sub-folder</i> for the output file location.</li>
<li>Insert the metadata tag of choice into the output path. %(USING_METADATA_TAGS_REF)s In this instance,
you would select the metadata tag "Treatment1"</li>
<li>Uncheck "Export all measurements?"</li>
<li>Uncheck <i>Use the object name for the file name?</i>.</li>
<li>Using the same approach as above, select the metadata tag "Treatment2", and complete
the filename by appending the text ".csv". </li>
</ul>
Here's an example table of the files that would be generated:<br><br><tt>
<table border="1"><tr><th>Treatment1</th><th>Treatment2</th><th>Path</th></tr>

<tr><td>1M_NaCl</td><td>20uM_DMSO</td><td>1M_NaCl/20uM_DMSO.csv</td></tr>
<tr><td>1M_NaCl</td><td>40uM_DMSO</td><td>1M_NaCl/40uM_DMSO.csv</td></tr>
<tr><td>2M_NaCl</td><td>20uM_DMSO</td><td>2M_NaCl/20uM_DMSO.csv</td></tr>
<tr><td>2M_NaCl</td><td>40uM_DMSO</td><td>2M_NaCl/40uM_DMSO.csv</td></tr>\
</table></tt>

<h4>Available measurements</h4>
For details on the nomenclature used by CellProfiler for the exported measurements,
see <i>Help > General Help > How Measurements Are Named</i>.

See also <b>ExportToDatabase</b>.
''' % globals()

import logging

logger = logging.getLogger(__name__)
import base64
import csv
import errno
import numpy as np
import os
import sys

import cellprofiler.module as cpm
import cellprofiler.measurement as cpmeas
import cellprofiler.pipeline as cpp
import cellprofiler.setting as cps
from cellprofiler.setting import YES, NO
from cellprofiler.measurement import IMAGE, EXPERIMENT
from cellprofiler.preferences import get_absolute_path, get_output_file_name
from cellprofiler.preferences import ABSPATH_OUTPUT, ABSPATH_IMAGE, get_headless
from cellprofiler.gui.help import USING_METADATA_TAGS_REF, USING_METADATA_HELP_REF, MEASUREMENT_NAMING_HELP
from cellprofiler.preferences import \
    standardize_default_folder_names, DEFAULT_INPUT_FOLDER_NAME, \
    DEFAULT_OUTPUT_FOLDER_NAME, ABSOLUTE_FOLDER_NAME, \
    DEFAULT_INPUT_SUBFOLDER_NAME, DEFAULT_OUTPUT_SUBFOLDER_NAME, \
    IO_FOLDER_CHOICE_HELP_TEXT, IO_WITH_METADATA_HELP_TEXT

MAX_EXCEL_COLUMNS = 256
MAX_EXCEL_ROWS = 65536

DELIMITER_TAB = "Tab"
DELIMITER_COMMA = 'Comma (",")'
DELIMITERS = (DELIMITER_COMMA, DELIMITER_TAB)

OBJECT_RELATIONSHIPS = "Object relationships"
RELATIONSHIPS = "Relationships"

SETTING_OG_OFFSET_V7 = 15
SETTING_OG_OFFSET_V8 = 16
SETTING_OG_OFFSET_V9 = 15
SETTING_OG_OFFSET_V10 = 17
SETTING_OG_OFFSET_V11 = 18
"""Offset of the first object group in the settings"""
SETTING_OG_OFFSET = 18

"""Offset of the object name setting within an object group"""
SETTING_OBJECT_NAME_IDX = 0

"""Offset of the previous file flag setting within an object group"""
SETTING_PREVIOUS_FILE_IDX = 1

"""Offset of the file name setting within an object group"""
SETTING_FILE_NAME_IDX = 2

SETTING_AUTOMATIC_FILE_NAME_IDX = 3

"""# of settings within an object group"""
SETTING_OBJECT_GROUP_CT = 4

"""The caption for the image set number"""
IMAGE_NUMBER = "ImageNumber"

"""The caption for the object # within an image set"""
OBJECT_NUMBER = "ObjectNumber"

"""The heading for the "Key" column in the experiment CSV"""
EH_KEY = "Key"

"""The heading for the "Value" column in the experiment CSV"""
EH_VALUE = "Value"

DIR_CUSTOM = "Custom folder"
DIR_CUSTOM_WITH_METADATA = "Custom folder with metadata"

"""Options for GenePattern GCT file export"""
GP_NAME_FILENAME = "Image filename"
GP_NAME_METADATA = "Metadata"
GP_NAME_OPTIONS = [GP_NAME_METADATA, GP_NAME_FILENAME]

NANS_AS_NULLS = "Null"
NANS_AS_NANS = "NaN"


class ExportToSpreadsheet(cpm.Module):
    module_name = 'ExportToSpreadsheet'
    category = ["File Processing", "Data Tools"]
    variable_revision_number = 11

    def create_settings(self):
        self.delimiter = cps.CustomChoice(
                "Select the column delimiter", DELIMITERS, doc="""
            Select the delimiter to use, i.e., the character that separates columns in a file. The
            two default choices are tab and comma, but you can type in any single character delimiter
            you would prefer. Be sure that the delimiter you choose is not a character that is present
            within your data (for example, in file names).""")

        self.directory = cps.DirectoryPath(
                "Output file location",
                dir_choices=[
                    ABSOLUTE_FOLDER_NAME,
                    DEFAULT_OUTPUT_FOLDER_NAME, DEFAULT_OUTPUT_SUBFOLDER_NAME,
                    DEFAULT_INPUT_FOLDER_NAME, DEFAULT_INPUT_SUBFOLDER_NAME], doc="""
            This setting lets you choose the folder for the output
            files. %(IO_FOLDER_CHOICE_HELP_TEXT)s

            <p>%(IO_WITH_METADATA_HELP_TEXT)s %(USING_METADATA_TAGS_REF)s<br>
            For instance, if you have a metadata tag named
            "Plate", you can create a per-plate folder by selecting one of the subfolder options
            and then specifying the subfolder name as "\g&lt;Plate&gt;". The module will
            substitute the metadata values for the current image set for any metadata tags in the
            folder name. %(USING_METADATA_HELP_REF)s.</p>""" % globals())
        self.directory.dir_choice = DEFAULT_OUTPUT_FOLDER_NAME

        self.wants_prefix = cps.Binary(
                "Add a prefix to file names?",
                True,
                doc="""This setting lets you choose whether or not to add
            a prefix to each of the .CSV filenames produced by
            <b>ExportToSpreadsheet</b>. A prefix may be useful if you use
            the same directory for the results of more than one pipeline; you
            can specify a different prefix in each pipeline.
            Select <i>%(YES)s</i> to add a prefix to each file name
            (e.g. "MyExpt_Images.csv"). Select <i>%(NO)s</i> to use filenames
            without prefixes (e.g. "Images.csv").
            """ % globals())

        self.prefix = cps.Text(
                "Filename prefix", "MyExpt_",
                doc="""(<i>Used only if "Add a prefix to file names?" is %(YES)s</i>)

            The text you enter here is prepended to the names of each file
            produced by <b>ExportToSpreadsheet</b>.
            """ % globals())

        self.wants_overwrite_without_warning = cps.Binary(
                "Overwrite existing files without warning?", False,
                doc="""This setting either prevents or allows overwriting of
            old .CSV files by <b>ExportToSpreadsheet</b> without confirmation.
            Select <i>%(YES)s</i> to overwrite without warning any .CSV file
            that already exists. Select <i>%(NO)s</i> to prompt before overwriting
            when running CellProfiler in the GUI and to fail when running
            headless.""" % globals())

        self.add_metadata = cps.Binary(
                "Add image metadata columns to your object data file?", False, doc=""""
            Image_Metadata_" columns are normally exported in the Image data file, but if you
            select <i>%(YES)s</i>, they will also be exported with the Object data file(s).""" % globals())

        self.excel_limits = cps.Binary(
                "Limit output to a size that is allowed in Excel?", False, doc="""
            If your output has more than 256 columns, select <i>%(YES)s</i> will open a window
            allowing you to select the columns you'd like to export. If your output exceeds
            65,000 rows, you can still open the CSV in Excel, but not all rows will be visible.""" % globals())

        self.nan_representation = cps.Choice(
                "Representation of Nan/Inf", [NANS_AS_NANS, NANS_AS_NULLS], doc="""
            This setting controls the output for numeric fields
            if the calculated value is infinite (<i>Inf</i>) or undefined (<i>NaN</i>).
            CellProfiler will produce Inf or NaN values under certain rare
            circumstances, for instance when calculating the mean intensity
            of an object within a masked region of an image.
            <ul>
            <li><i>%(NANS_AS_NULLS)s:</i> Output these values as empty fields.</li>
            <li><i>%(NANS_AS_NANS)s:</i> Output them as the strings "NaN", "Inf" or "-Inf".</li>
            </ul>
            """ % globals())

        self.pick_columns = cps.Binary(
                "Select the measurements to export", False, doc="""
            Select <i>%(YES)s</i> to provide a button that allows you to select which measurements you want to export.
            This is useful if you know exactly what measurements you want included in the final spreadheet(s). """ % globals())

        self.columns = cps.MeasurementMultiChoice(
                "", doc="""
            <i>(Used only when selecting the columns of measurements to export)</i><br>
            This setting controls the columns to be exported. Press
            the button and check the measurements or categories to export.""")

        self.wants_aggregate_means = cps.Binary(
                "Calculate the per-image mean values for object measurements?", False, doc="""
            Select <i>%(YES)s</i> for <b>ExportToSpreadsheet</b> to calculate population statistics over all the
            objects in each image and save that value as an aggregate
            measurement in the Image file.  For instance, if you are measuring
            the area of the Nuclei objects and you check the box for this option, <b>ExportToSpreadsheet</b> will
            create a column in the Image file called "Mean_Nuclei_AreaShape_Area".

            <p>You may not want to use <b>ExportToSpreadsheet</b> to calculate these
            measurements if your pipeline generates a large number of per-object
            measurements; doing so might exceed Excel's limits on the number of columns (256).</p>

            <p>Keep in mind that if you chose to select the specific measurements to export, the aggregate
            statistics will only be computed for the selected per-object measurements.</p>""" % globals())

        self.wants_aggregate_medians = cps.Binary("Calculate the per-image median values for object measurements?",
                                                  False)

        self.wants_aggregate_std = cps.Binary(
                "Calculate the per-image standard deviation values for object measurements?", False)

        self.wants_genepattern_file = cps.Binary(
                "Create a GenePattern GCT file?", False, doc="""
            Select <i>%(YES)s</i> to create a GCT file compatible with
            <a href="http://www.broadinstitute.org/cancer/software/genepattern/">GenePattern</a>.
            The GCT file format is a tab-delimited text file format that describes a gene
            expression dataset; the specifics of the format are described
            <a href="http://www.broadinstitute.org/cancer/software/genepattern/tutorial/gp_fileformats.html#gct">here</a>.
            By converting your measurements into a GCT file, you can make
            use of GenePattern's data visualization and clustering methods.

            <p>Each row in the GCT file represents (ordinarily) a gene and
            each column represents a sample (in this case, a per-image set
            of measurements). In addition to any other spreadsheets desired,
            enabling this option will produce a GCT file with the extension .gct,
            prepended with the text selection above. If per-image aggregate
            measurements are requested above, those measurements are included
            in the GCT file as well.</p>""" % globals())

        self.how_to_specify_gene_name = cps.Choice(
                "Select source of sample row name",
                GP_NAME_OPTIONS, GP_NAME_METADATA, doc="""
            <i>(Used only if a GenePattern file is requested)</i><br>
            The first column of the GCT file is the unique identifier for each
            sample, which is ordinarily the gene name. This information may be
            specified in one of two ways:
            <ul>
            <li><i>Metadata:</i> If you used the <b>Metadata</b> modules to
            add metadata to your images, you may specify a metadata tag
            that corresponds to the identifier for this column.
            %(USING_METADATA_HELP_REF)s.</li>
            <li><i>Image filename:</i> If the gene name is not available, the image
            filename can be used as a surrogate identifier.</li>
            </ul>""" % globals())

        self.gene_name_column = cps.Measurement(
                "Select the metadata to use as the identifier",
                lambda: cpmeas.IMAGE, doc="""
            <i>(Used only if a GenePattern file is requested and metadata is used
            to name each row)</i><br>
            Choose the measurement that corresponds to the identifier, such as
            metadata from the <b>Metadata</b> module.
            %(USING_METADATA_HELP_REF)s.""" % globals())

        self.use_which_image_for_gene_name = cps.ImageNameSubscriber(
                "Select the image to use as the identifier", cps.NONE, doc="""
            <i>(Used only if a GenePattern file is requested and image filename is used to name each row)</i><br>
            Select which image whose filename will be used to identify each sample row.""")

        self.wants_everything = cps.Binary(
                "Export all measurement types?", True, doc="""
            Select <i>%(YES)s</i> to export every category of measurement.
            <b>ExportToSpreadsheet</b> will create one data file for each object produced
            in the pipeline, as well as per-image, per-experiment and object relationships,
            if relevant. See <i>%(MEASUREMENT_NAMING_HELP)s</i> for more details on
            the various measurement types. The module will use the object name as the file name,
            optionally prepending the output file name if specified above.
            <p>Select <i>%(NO)s</i> if you want to do either (or both) of two things:
            <ul>
            <li>Specify which objects should be exported;</li>
            <li>Override the automatic nomenclature of the exported files.</li>
            </ul></p>""" % globals())

        self.object_groups = []
        self.add_object_group()
        self.add_button = cps.DoSomething("", "Add another data set",
                                          self.add_object_group)

    def add_object_group(self, can_remove=True):
        group = cps.SettingsGroup()
        group.append(
                "name", EEObjectNameSubscriber("Data to export", doc="""
            <i>(Used only when "Export all measurements?" is set to "%(NO)s")</i><br>
            Choose <i>Image</i>, <i>Experiment</i>, <i>Object relationships</i>
            or an object name from the list. <b>ExportToSpreadsheet</b> will write out a
            file of measurements for the given category. See <i>%(MEASUREMENT_NAMING_HELP)s</i>
            for more details on the various measurement types.""" % globals()))

        group.append(
                "previous_file", cps.Binary(
                        "Combine these object measurements with those of the previous object?", False, doc="""
            <i>(Used only when "Export all measurements?" is set to "%(NO)s")</i><br>
            Select <i>%(YES)s</i> to create a file composed
            of measurements made on this object and the one directly
            above it.
            <p>Select <i>%(NO)s</i> to create separate
            files for this and the previous object.</p>""" % globals()))

        group.append("wants_automatic_file_name", cps.Binary(
                "Use the object name for the file name?", True, doc="""
            <i>(Used only when "Export all measurements?" is set to "%(NO)s")</i><br>
            Select <i>%(YES)s</i> to use the object name as selected above to generate a file
            name for the spreadsheet. For example, if you selected <i>Image</i>,
            above and have not checked the <i>Prepend output file name</i> option,
            your output file will be named "Image.csv".
            <p>Select <i>%(NO)s</i> to name the file yourself.</p>""" % globals()))

        group.append("file_name", cps.Text(
                "File name", "DATA.csv",
                metadata=True, doc="""
            <i>(Used only when "Export all measurements?" is set to "%(NO)s")</i><br>
            Enter a file name for the named objects'
            measurements. <b>ExportToSpreadsheet</b> will
            prepend the name of the measurements file to this
            if you asked to do so above. If you have metadata
            associated with your images, this setting will also substitute
            metadata tags if desired. %(USING_METADATA_TAGS_REF)s%(USING_METADATA_HELP_REF)s.""" % globals()))

        group.append("remover", cps.RemoveSettingButton("", "Remove this data set", self.object_groups, group))
        group.append("divider", cps.Divider(line=False))

        self.object_groups.append(group)

    def prepare_settings(self, setting_values):
        """Add enough object groups to capture the settings"""
        setting_count = len(setting_values)
        assert ((setting_count - SETTING_OG_OFFSET) %
                SETTING_OBJECT_GROUP_CT == 0)
        group_count = int((setting_count - SETTING_OG_OFFSET) /
                          SETTING_OBJECT_GROUP_CT)
        del self.object_groups[group_count:]

        while len(self.object_groups) < group_count:
            self.add_object_group()

    def settings(self):
        """Return the settings in the order used when storing """
        result = [self.delimiter, self.add_metadata,
                  self.excel_limits, self.pick_columns,
                  self.wants_aggregate_means, self.wants_aggregate_medians,
                  self.wants_aggregate_std, self.directory,
                  self.wants_genepattern_file, self.how_to_specify_gene_name,
                  self.use_which_image_for_gene_name, self.gene_name_column,
                  self.wants_everything, self.columns, self.nan_representation,
                  self.wants_prefix, self.prefix,
                  self.wants_overwrite_without_warning]
        for group in self.object_groups:
            result += [group.name, group.previous_file, group.file_name,
                       group.wants_automatic_file_name]
        return result

    def visible_settings(self):
        """Return the settings as seen by the user"""
        result = [self.delimiter, self.directory, self.wants_prefix]
        if self.wants_prefix:
            result += [self.prefix]
        result += [
            self.wants_overwrite_without_warning, self.add_metadata,
            self.excel_limits, self.nan_representation, self.pick_columns]
        if self.pick_columns:
            result += [self.columns]
        result += [self.wants_aggregate_means, self.wants_aggregate_medians,
                   self.wants_aggregate_std, self.wants_genepattern_file]
        if self.wants_genepattern_file:
            result += [self.how_to_specify_gene_name]
            if self.how_to_specify_gene_name == GP_NAME_METADATA:
                result += [self.gene_name_column]
            elif self.how_to_specify_gene_name == GP_NAME_FILENAME:
                result += [self.use_which_image_for_gene_name]
        result += [self.wants_everything]
        if not self.wants_everything:
            previous_group = None
            for index, group in enumerate(self.object_groups):
                result += [group.name]
                append_file_name = True
                if is_object_group(group):
                    if ((not previous_group is None) and
                            is_object_group(previous_group)):
                        #
                        # Show the previous-group button if there was a previous
                        # group and it was an object group
                        #
                        result += [group.previous_file]
                        if group.previous_file.value:
                            append_file_name = False
                if append_file_name:
                    result += [group.wants_automatic_file_name]
                    if not group.wants_automatic_file_name:
                        result += [group.file_name]
                result += [group.remover, group.divider]
                previous_group = group
            result += [self.add_button]
        return result

    def validate_module(self, pipeline):
        '''Test the module settings to make sure they are internally consistent'''
        if (len(self.delimiter.value) != 1 and
                not self.delimiter.value in (DELIMITER_TAB, DELIMITER_COMMA)):
            raise cps.ValidationError("The CSV field delimiter must be a single character", self.delimiter)

        '''Make sure metadata tags exist'''
        for group in self.object_groups:
            if not group.wants_automatic_file_name:
                text_str = group.file_name.value
                undefined_tags = pipeline.get_undefined_metadata_tags(text_str)
                if len(undefined_tags) > 0:
                    raise cps.ValidationError(
                            "%s is not a defined metadata tag. Check the metadata specifications in your load modules" %
                            undefined_tags[0],
                            group.file_name)

    def validate_module_warnings(self, pipeline):
        '''Warn user re: Test mode '''
        if pipeline.test_mode:
            raise cps.ValidationError("ExportToSpreadsheet will not produce output in Test Mode",
                                      self.directory)

        '''Warn user that changing the extension may cause Excel to stuff everything into one column'''
        if not self.wants_everything.value:
            all_extensions = [os.path.splitext(group.file_name.value)[1] for group in self.object_groups]
            is_valid_extension = [not group.wants_automatic_file_name.value and (
                (extension == ".csv" and self.delimiter == DELIMITER_COMMA) or (
                    extension == ".txt" and self.delimiter == DELIMITER_TAB))
                                  for (extension, group) in zip(all_extensions, self.object_groups)]
            if not all(is_valid_extension):
                raise (cps.ValidationError(
                        "To avoid formatting problems in Excel, use the extension .csv for comma-delimited files and .txt for tab-delimited..",
                        self.object_groups[is_valid_extension.index(False)].file_name))

    @property
    def delimiter_char(self):
        if self.delimiter == DELIMITER_TAB:
            return "\t"
        elif self.delimiter == DELIMITER_COMMA:
            return ","
        else:
            return self.delimiter.value.encode("ascii")

    def prepare_run(self, workspace):
        '''Prepare an image set to be run

        workspace - workspace with image set populated (at this point)

        returns False if analysis can't be done
        '''
        return self.check_overwrite(workspace)

    def run(self, workspace):
        # all of the work is done in post_run()
        if self.show_window:
            image_set_number = workspace.measurements.image_set_number
            header = ["Objects", "Filename"]
            columns = []
            if self.wants_everything:
                for object_name in workspace.measurements.get_object_names():
                    path = self.make_objects_file_name(
                            object_name, workspace, image_set_number)
                    columns.append((object_name, path))
            else:
                first = True
                for i in range(len(self.object_groups)):
                    group = self.object_groups[i]
                    last_in_file = self.last_in_file(i)
                    if first:
                        filename = self.make_objects_file_name(
                                group.name.value, workspace, image_set_number, group)
                        first = False
                    columns.append((group.name.value, filename))
                    if last_in_file:
                        first = True
            workspace.display_data.header = header
            workspace.display_data.columns = columns

    def display(self, workspace, figure):
        figure.set_subplots((1, 1,))
        if workspace.display_data.columns is None:
            figure.subplot_table(
                    0, 0, [["Data written to spreadsheet"]])
        elif workspace.pipeline.test_mode:
            figure.subplot_table(
                    0, 0, [["Data not written to spreadsheets in test mode"]])
        else:
            figure.subplot_table(0, 0,
                                 workspace.display_data.columns,
                                 col_labels=workspace.display_data.header)

    def run_as_data_tool(self, workspace):
        '''Run the module as a data tool

        For ExportToSpreadsheet, we do the "post_run" method in order to write
        out the .csv files as if the experiment had just finished.
        '''
        #
        # Set the measurements to the end of the list to mimic the state
        # at the end of the run.
        #
        m = workspace.measurements
        m.image_set_number = m.image_set_count
        self.post_run(workspace)

    def post_run(self, workspace):
        '''Save measurements at end of run'''
        #
        # Don't export in test mode
        #
        if workspace.pipeline.test_mode:
            return
        #
        # Signal "display" that we are post_run
        #
        workspace.display_data.columns = None
        workspace.display_data.header = None
        #
        # Export all measurements if requested
        #
        if self.wants_everything:
            for object_name in workspace.measurements.get_object_names():
                self.run_objects([object_name], workspace)
            return

        object_names = []
        #
        # Loop, collecting names of objects that get included in the same file
        #
        for i in range(len(self.object_groups)):
            group = self.object_groups[i]
            last_in_file = self.last_in_file(i)
            if len(object_names) == 0:
                first_group = group
            object_names.append(group.name.value)
            if last_in_file:
                self.run_objects(object_names, workspace, first_group)
                object_names = []

    def last_in_file(self, i):
        '''Return true if the group is the last to be included in a csv file

        i - the index of the group being considered.

        Objects can be collected together in one file. Return true if
        this is the last object in a collection.
        '''

        group = self.object_groups[i]
        return ((i == len(self.object_groups) - 1) or
                (not is_object_group(group)) or
                (not is_object_group(self.object_groups[i + 1])) or
                (not self.object_groups[i + 1].previous_file.value))

    def should_stop_writing_measurements(self):
        '''All subsequent modules should not write measurements'''
        return True

    def get_metadata_groups(self, workspace, settings_group=None):
        '''Find the metadata groups that are relevant for creating the file name

        workspace - the workspace with the image set metadata elements and
                    grouping measurements populated.
        settings_group - if saving individual objects, this is the settings
                         group that controls naming the files.
        '''
        if settings_group is None or settings_group.wants_automatic_file_name:
            tags = []
        else:
            tags = cpmeas.find_metadata_tokens(settings_group.file_name.value)
        if self.directory.is_custom_choice:
            tags += cpmeas.find_metadata_tokens(self.directory.custom_path)
        metadata_groups = workspace.measurements.group_by_metadata(tags)
        return metadata_groups

    def run_objects(self, object_names, workspace, settings_group=None):
        """Create a file (or files if there's metadata) based on the object names

        object_names - a sequence of object names (or Image or Experiment)
                       which tell us which objects get piled into each file
        workspace - get the images from here.
        settings_group - if present, use the settings group for naming.

        """
        if len(object_names) == 1 and object_names[0] == EXPERIMENT:
            self.make_experiment_file(workspace, settings_group)
            return
        metadata_groups = self.get_metadata_groups(workspace, settings_group)
        for metadata_group in metadata_groups:
            if len(object_names) == 1 and object_names[0] == IMAGE:
                self.make_image_file(metadata_group.image_numbers,
                                     workspace, settings_group)
                if self.wants_genepattern_file.value:
                    self.make_gct_file(metadata_group.image_numbers,
                                       workspace, settings_group)
            elif len(object_names) == 1 and object_names[0] == OBJECT_RELATIONSHIPS:
                self.make_relationships_file(
                        metadata_group.image_numbers, workspace, settings_group)
            else:
                self.make_object_file(
                        object_names, metadata_group.image_numbers,
                        workspace, settings_group)

    def make_full_filename(self, file_name,
                           workspace=None, image_set_number=None):
        """Convert a file name into an absolute path

        We do a few things here:
        * apply metadata from an image set to the file name if an
          image set is specified
        * change the relative path into an absolute one using the "." and "&"
          convention
        * Create any directories along the path
        """
        if image_set_number is not None and workspace is not None:
            file_name = workspace.measurements.apply_metadata(file_name,
                                                              image_set_number)
        measurements = None if workspace is None else workspace.measurements
        path_name = self.directory.get_absolute_path(measurements,
                                                     image_set_number)
        if self.wants_prefix:
            file_name = self.prefix.value + file_name
        file_name = os.path.join(path_name, file_name)
        path, file = os.path.split(file_name)
        if not os.path.isdir(path):
            os.makedirs(path)
        return os.path.join(path, file)

    def extension(self):
        '''Return the appropriate extension for the CSV file name

        The appropriate extension is "csv" if comma is used as the
        delimiter, otherwise "txt"
        '''
        return "csv" if self.delimiter == DELIMITER_COMMA else "txt"

    def make_objects_file_name(
            self, object_name, workspace, image_set_number, settings_group=None):
        '''Concoct the .CSV filename for some object category

        :param object_name: name of the objects whose measurements are to be
                            saved (or IMAGES or EXPERIMENT)
        :param workspace: the current workspace
        :param image_set_number: the current image set number
        :param settings_group: the settings group used to name the file
        '''
        if self.wants_everything:
            filename = "%s.%s" % (object_name, self.extension())

            if object_name == EXPERIMENT:
                # No metadata substitution allowed for experiment file
                return self.make_full_filename(filename)
            return self.make_full_filename(
                    filename, workspace, image_set_number)
        if settings_group.wants_automatic_file_name:
            filename = "%s.%s" % (settings_group.name.value, self.extension())
        else:
            filename = settings_group.file_name.value
        filename = self.make_full_filename(
                filename, workspace, image_set_number)
        return filename

    def make_gct_file_name(self, workspace, image_set_number, settings_group=None):
        '''Concoct a name for the .gct file

        workspace - workspace containing metadata measurements
        image_number - the first image number in the group being written
        settings_group - the settings group asking for the file to be written
                        if not wants_everything
        '''
        file_name = self.make_objects_file_name(
                IMAGE, workspace, image_set_number, settings_group)
        if any([file_name.lower().endswith(x) for x in ".csv", "txt"]):
            file_name = file_name[:-3] + "gct"
        return file_name

    def check_overwrite(self, workspace):
        """Make sure it's ok to overwrite any existing files before starting run

        workspace - workspace with all image sets already populated

        returns True if ok to proceed, False if user cancels
        """
        if self.wants_overwrite_without_warning:
            return True

        files_to_check = []
        if self.wants_everything:
            object_names = set((IMAGE, EXPERIMENT, OBJECT_RELATIONSHIPS))
            object_providers = workspace.pipeline.get_provider_dictionary(
                    cps.OBJECT_GROUP, self)
            object_names.update(object_providers.keys())
            metadata_groups = self.get_metadata_groups(workspace)
            for object_name in object_names:
                for metadata_group in metadata_groups:
                    image_number = metadata_group.image_numbers[0]
                    if object_name == IMAGE and self.wants_genepattern_file:
                        files_to_check.append(self.make_gct_file_name(
                                workspace, image_number))
                    files_to_check.append(self.make_objects_file_name(
                            object_name, workspace, image_number))
        else:
            first_in_file = True
            for i, group in enumerate(self.object_groups):
                if first_in_file:
                    metadata_groups = self.get_metadata_groups(
                            workspace, group)
                    for metadata_group in metadata_groups:
                        image_number = metadata_group.image_numbers[0]
                        files_to_check.append(
                                self.make_objects_file_name(
                                        group.name.value, workspace, image_number,
                                        group))
                #
                # set first_in_file for next time around
                #
                first_in_file = self.last_in_file(i)

        files_to_overwrite = filter(os.path.isfile, files_to_check)
        if len(files_to_overwrite) > 0:
            if get_headless():
                logger.error(
                        "ExportToSpreadsheet is configured to refrain from overwriting files and the following file(s) already exist: %s" %
                        ", ".join(files_to_overwrite))
                return False
            msg = "Overwrite the following file(s)?\n" + \
                  "\n".join(files_to_overwrite)
            import wx
            result = wx.MessageBox(
                    msg, caption="ExportToSpreadsheet: Overwrite existing files",
                    style=wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION)
            if result != wx.YES:
                return False

        return True

    def make_experiment_file(self, workspace, settings_group=None):
        """Make a file containing the experiment measurements

        workspace - the workspace that has the measurements
        settings_group - the settings group used to choose the experiment
                         measurements for output or None if everything
                         is to be exported
        """
        m = workspace.measurements
        file_name = self.make_objects_file_name(
                EXPERIMENT, workspace, 1, settings_group)
        feature_names = [
            feature_name for feature_name in m.get_feature_names(EXPERIMENT)
            if feature_name != cpp.EXIT_STATUS]
        if len(feature_names) == 0:
            return
        fd = open(file_name, "wb")
        try:
            writer = csv.writer(fd, delimiter=self.delimiter_char)
            writer.writerow((EH_KEY, EH_VALUE))
            for feature_name in feature_names:
                v = m.get_all_measurements(EXPERIMENT, feature_name)
                if isinstance(v, np.ndarray) and \
                                v.dtype == np.uint8:
                    v = base64.b64encode(v.data)
                else:
                    unicode(v).encode('utf8')
                writer.writerow((feature_name, v))
        finally:
            fd.close()

    def make_image_file(self, image_set_numbers, workspace, settings_group=None):
        """Make a file containing image measurements

        image_set_numbers - the image sets whose data gets extracted
        workspace - workspace containing the measurements
        settings_group - the settings group used to choose the image
                         measurements for output or None if everything
                         is to be exported
        """
        m = workspace.measurements
        file_name = self.make_objects_file_name(
                IMAGE, workspace, image_set_numbers[0], settings_group)
        image_features = m.get_feature_names(IMAGE)
        image_features.insert(0, IMAGE_NUMBER)
        if not self.check_excel_limits(workspace, file_name,
                                       len(image_set_numbers),
                                       len(image_features)):
            return
        fd = open(file_name, "wb")
        try:
            writer = csv.writer(fd, delimiter=self.delimiter_char)
            for img_number in image_set_numbers:
                aggs = []
                if self.wants_aggregate_means:
                    aggs.append(cpmeas.AGG_MEAN)
                if self.wants_aggregate_medians:
                    aggs.append(cpmeas.AGG_MEDIAN)
                if self.wants_aggregate_std:
                    aggs.append(cpmeas.AGG_STD_DEV)
                agg_measurements = m.compute_aggregate_measurements(img_number,
                                                                    aggs)
                if img_number == image_set_numbers[0]:
                    ordered_agg_names = list(agg_measurements.keys())
                    ordered_agg_names.sort()
                    image_features += ordered_agg_names
                    image_features.sort()
                    image_features = self.filter_columns(image_features,
                                                         cpmeas.IMAGE)
                    if image_features is None:
                        return
                    writer.writerow(image_features)
                row = []
                for feature_name in image_features:

                    if feature_name == IMAGE_NUMBER:
                        row.append(str(img_number))
                    else:
                        if agg_measurements.has_key(feature_name):
                            value = agg_measurements[feature_name]
                        else:
                            value = m[IMAGE, feature_name, img_number]
                        if value is None:
                            row.append('')
                        elif isinstance(value, unicode):
                            row.append(value.encode('utf8'))
                        elif isinstance(value, basestring):
                            row.append(value)
                        elif isinstance(value, np.ndarray) and \
                                        value.dtype == np.uint8:
                            row.append(base64.b64encode(value.data))
                        elif np.isnan(value):
                            if self.nan_representation == NANS_AS_NULLS:
                                row.append('')
                            else:
                                row.append(str(np.NaN))
                        else:
                            row.append(str(value))
                writer.writerow(row)
        finally:
            fd.close()

    def make_gct_file(self, image_set_numbers, workspace, settings_group):
        """Make a GenePattern file containing image measurements
        Format specifications located at http://www.broadinstitute.org/cancer/software/genepattern/tutorial/gp_fileformats.html?gct

        file_name - create a file with this name
        image_set_numbers - the image sets whose data gets extracted
        workspace - workspace containing the measurements
        """
        from loaddata import is_path_name_feature, is_file_name_feature
        from loadimages import C_PATH_NAME, C_FILE_NAME, C_URL
        from loadimages import C_MD5_DIGEST, C_SCALING, C_HEIGHT, C_WIDTH

        file_name = self.make_gct_file_name(workspace, image_set_numbers[0],
                                            settings_group)

        def ignore_feature(feature_name):
            """Return true if we should ignore a feature"""
            if (is_file_name_feature(feature_name) or
                    is_path_name_feature(feature_name) or
                    feature_name.startswith('ImageNumber') or
                    feature_name.startswith("Group_Number") or
                    feature_name.startswith("Group_Index") or
                    feature_name.startswith('Description_') or
                    feature_name.startswith('ModuleError_') or
                    feature_name.startswith('TimeElapsed_') or
                    feature_name.startswith('ExecutionTime_') or
                    feature_name.startswith(C_URL) or
                    feature_name.startswith(C_MD5_DIGEST) or
                    feature_name.startswith(C_SCALING) or
                    feature_name.startswith(C_HEIGHT) or
                    feature_name.startswith(C_WIDTH)
                ):
                return True
            return False

        m = workspace.measurements
        image_features = m.get_feature_names(IMAGE)
        image_features.insert(0, IMAGE_NUMBER)
        if not self.check_excel_limits(workspace, file_name,
                                       len(image_set_numbers),
                                       len(image_features)):
            return

        fd = open(file_name, "wb")
        try:
            writer = csv.writer(fd, delimiter="\t")
            for img_number in image_set_numbers:
                aggs = []
                if self.wants_aggregate_means:
                    aggs.append(cpmeas.AGG_MEAN)
                if self.wants_aggregate_medians:
                    aggs.append(cpmeas.AGG_MEDIAN)
                if self.wants_aggregate_std:
                    aggs.append(cpmeas.AGG_STD_DEV)
                agg_measurements = m.compute_aggregate_measurements(img_number, aggs)

                if img_number == image_set_numbers[0]:
                    ordered_agg_names = list(agg_measurements.keys())
                    ordered_agg_names.sort()
                    image_features += ordered_agg_names
                    image_features.sort()
                    image_features = self.filter_columns(image_features,
                                                         cpmeas.IMAGE)
                    if image_features is None:
                        return

                    # Count # of actual measurements
                    num_measures = 0
                    for feature_name in image_features:
                        if not ignore_feature(feature_name) or agg_measurements.has_key(feature_name):
                            num_measures += 1

                    writer.writerow(['#1.2'])
                    writer.writerow([len(image_set_numbers), num_measures])

                    # Keep measurements only
                    measurement_feature_names = [x for x in image_features if not ignore_feature(x)]

                    # The first headers need to be 'NAME' and 'Description'
                    written_image_names = ['NAME', 'Description'] + measurement_feature_names
                    writer.writerow(written_image_names)

                    # Place the one of the paths and desired info column up front in image feature list
                    description_feature = [x for x in image_features if x.startswith(C_PATH_NAME + "_")]
                    if self.how_to_specify_gene_name == GP_NAME_METADATA:
                        name_feature = [self.gene_name_column.value]
                    elif self.how_to_specify_gene_name == GP_NAME_FILENAME:
                        name_feature = [x for x in image_features if
                                        x.startswith("_".join((C_FILE_NAME, self.use_which_image_for_gene_name.value)))]
                    image_features = [name_feature[0], description_feature[0]] + measurement_feature_names

                # Output all measurements
                row = [agg_measurements[feature_name]
                       if agg_measurements.has_key(feature_name)
                       else m.get_measurement(IMAGE, feature_name, img_number)
                       for feature_name in image_features]
                row = ['' if x is None
                       else x if np.isscalar(x)
                else x[0] for x in row]
                writer.writerow(row)
        finally:
            fd.close()

    def check_excel_limits(self, workspace, file_name, row_count, col_count):
        '''Return False if we shouldn't write because of Excel'''
        if self.excel_limits and self.show_window:
            message = None
            if col_count > MAX_EXCEL_COLUMNS:
                message = ("""The image file, "%s", will have %d columns, but Excel only supports %d.
Do you want to save it anyway?""" %
                           (file_name, col_count, MAX_EXCEL_COLUMNS))
            elif row_count > MAX_EXCEL_ROWS:
                message = ("""The image file, "%s", will have %d rows, but Excel only supports %d.
Do you want to save it anyway?""" %
                           (file_name, row_count, MAX_EXCEL_COLUMNS))
            if message is not None:
                # This is okay, as the only path to this function is via
                # post_run(), which is called in the main thread.
                import wx
                assert wx.Thread_IsMain(), "exporttospreadsheet.post_run() called in non-main thread."
                if wx.MessageBox(message, "Excel limits exceeded", wx.YES_NO) == wx.ID_NO:
                    return False
        return True

    def filter_columns(self, features, object_name):
        if self.pick_columns:
            columns = [
                self.columns.get_measurement_feature(x)
                for x in self.columns.selections
                if self.columns.get_measurement_object(x) == object_name]
            if object_name == cpmeas.IMAGE:
                if cpmeas.IMAGE_NUMBER not in columns:
                    columns.insert(0, cpmeas.IMAGE_NUMBER)
                for agg, wants_it in (
                        (cpmeas.AGG_MEAN, self.wants_aggregate_means),
                        (cpmeas.AGG_MEDIAN, self.wants_aggregate_medians),
                        (cpmeas.AGG_STD_DEV, self.wants_aggregate_std)):
                    if not wants_it:
                        continue
                    for column in self.columns.selections:
                        if self.columns.get_measurement_object(column) not in (
                                cpmeas.IMAGE, cpmeas.EXPERIMENT, cpmeas.NEIGHBORS):
                            columns += [cpmeas.get_agg_measurement_name(
                                    agg, self.columns.get_measurement_object(column),
                                    self.columns.get_measurement_feature(column))]

            columns = set(columns)
            features = [x for x in features if x in columns]
        return features

    def make_object_file(self, object_names, image_set_numbers, workspace,
                         settings_group=None):
        """Make a file containing object measurements

        object_names - sequence of names of the objects whose measurements
                       will be included
        image_set_numbers -  the image sets whose data gets extracted
        workspace - workspace containing the measurements
        settings_group - the settings group used to choose to make the file or
                         None if wants_everything
        """
        m = workspace.measurements
        file_name = self.make_objects_file_name(
                object_names[0], workspace, image_set_numbers[0], settings_group)
        features = [(IMAGE, IMAGE_NUMBER),
                    (object_names[0], OBJECT_NUMBER)]
        columns = map(
                (lambda c: c[:2]), workspace.pipeline.get_measurement_columns())
        if self.add_metadata.value:
            mdfeatures = [
                (IMAGE, name) for object_name, name in columns
                if name.startswith("Metadata_") and object_name == IMAGE]
            mdfeatures.sort()
            features += mdfeatures
        for object_name in object_names:
            ofeatures = [feature for col_object, feature in columns
                         if col_object == object_name]
            ofeatures = self.filter_columns(ofeatures, object_name)
            ofeatures = [(object_name, feature_name)
                         for feature_name in ofeatures]
            ofeatures.sort()
            features += ofeatures
        fd = open(file_name, "wb")
        if self.excel_limits:
            row_count = 1
            for img_number in image_set_numbers:
                object_count = \
                    np.max([m.get_measurement(IMAGE, "Count_%s" % name, img_number)
                            for name in object_names])
                row_count += int(object_count)
            if not self.check_excel_limits(workspace, file_name,
                                           row_count, len(features)):
                return

        try:
            writer = csv.writer(fd, delimiter=self.delimiter_char)
            #
            # We write the object names in the first row of headers if there are
            # multiple objects. Otherwise, we just write the feature names
            #
            for i in (0, 1) if len(object_names) > 1 else (1,):
                writer.writerow([x[i] for x in features])
            for img_number in image_set_numbers:
                object_count = \
                    np.max([m.get_measurement(IMAGE, "Count_%s" % name, img_number)
                            for name in object_names])
                object_count = int(object_count) if object_count else 0
                columns = [np.repeat(img_number, object_count)
                           if feature_name == IMAGE_NUMBER
                           else np.arange(1, object_count + 1)
                if feature_name == OBJECT_NUMBER
                else np.repeat(np.NAN, object_count)
                if not m.has_feature(object_name, feature_name)
                else np.repeat(m.get_measurement(IMAGE, feature_name,
                                                 img_number),
                               object_count)
                if object_name == IMAGE
                else m.get_measurement(object_name, feature_name,
                                       img_number)
                           for object_name, feature_name in features]
                for obj_index in range(object_count):
                    row = [column[obj_index]
                           if (column is not None and
                               obj_index < column.shape[0])
                           else np.NAN
                           for column in columns]
                    if self.nan_representation == NANS_AS_NULLS:
                        row = [
                            "" if (field is None) or
                                  (np.isreal(field) and not np.isfinite(field))
                            else field for field in row]
                    writer.writerow(row)
        finally:
            fd.close()

    def make_relationships_file(self, image_set_numbers, workspace,
                                settings_group=None):
        '''Create a CSV file documenting the relationships between objects'''

        file_name = self.make_objects_file_name(
                OBJECT_RELATIONSHIPS, workspace, image_set_numbers[0],
                settings_group)
        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        fd = open(file_name, "wb")
        module_map = {}
        for module in workspace.pipeline.modules():
            module_map[module.module_num] = module.module_name

        try:
            writer = csv.writer(fd, delimiter=self.delimiter_char)
            writer.writerow([
                "Module", "Module Number", "Relationship",
                "First Object Name", "First Image Number", "First Object Number",
                "Second Object Name", "Second Image Number", "Second Object Number"])
            for key in m.get_relationship_groups():
                r = m.get_relationships(
                        key.module_number, key.relationship,
                        key.object_name1, key.object_name2,
                        image_numbers=image_set_numbers)
                for image_number_1, image_number_2, \
                    object_number_1, object_number_2 in zip(
                        r[cpmeas.R_FIRST_IMAGE_NUMBER],
                        r[cpmeas.R_SECOND_IMAGE_NUMBER],
                        r[cpmeas.R_FIRST_OBJECT_NUMBER],
                        r[cpmeas.R_SECOND_OBJECT_NUMBER]):
                    module_name = module_map[key.module_number]
                    writer.writerow([
                        module_name, key.module_number, key.relationship,
                        key.object_name1, image_number_1, object_number_1,
                        key.object_name2, image_number_2, object_number_2])
        finally:
            fd.close()

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

        ExportToSpreadsheet has to convert the path to file names to
        something that can be used on the cluster.
        '''
        self.directory.alter_for_create_batch_files(fn_alter_path)
        return True

    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        """Adjust the setting values based on the version that saved them

        """

        if variable_revision_number == 1 and from_matlab:
            # Added create subdirectories questeion
            setting_values = list(setting_values)
            setting_values.append(cps.NO)
            variable_revision_number = 2
        if variable_revision_number == 2 and from_matlab:
            wants_subdirectories = (setting_values[8] == cps.YES)
            object_names = [x for x in setting_values[:-1]
                            if x != cps.DO_NOT_USE]
            setting_values = [DELIMITER_TAB, cps.YES, cps.NO, cps.NO,
                              cps.NO, cps.NO]
            for name in object_names:
                setting_values.extend([name, cps.NO, "%s.csv" % name])
            variable_revision_number = 1
            from_matlab = False
        if variable_revision_number == 3 and from_matlab:
            #
            # Variables 9 and 10 are the pathname and prefix and
            # are not yet replicated in pyCP
            #
            custom_directory = '.'
            if setting_values[8] == '.':
                directory_choice = DEFAULT_OUTPUT_FOLDER_NAME
            elif setting_values[8] == '&':
                directory_choice = DEFAULT_INPUT_FOLDER_NAME
            elif setting_values[8].find(r"\(?<"):
                directory_choice = DIR_CUSTOM_WITH_METADATA
                custom_directory = setting_values[8]
            else:
                directory_choice = DIR_CUSTOM
                custom_directory = setting_values[8]
            if setting_values[9] != cps.DO_NOT_USE:
                prefix = setting_values[9] + "_"
            else:
                prefix = ""
            object_names = [x for x in setting_values[:8]
                            if x != cps.DO_NOT_USE]
            setting_values = [DELIMITER_TAB, cps.YES, cps.NO, cps.NO,
                              cps.NO, cps.NO, cps.NO, cps.NO, cps.NO,
                              directory_choice, custom_directory]
            for name in object_names:
                setting_values.extend([name, cps.NO,
                                       "%s%s.csv" % (prefix, name)])
            variable_revision_number = 3
            from_matlab = False
        if variable_revision_number == 1 and not from_matlab:
            # Added aggregate questions
            setting_values = (setting_values[:6] + [cps.NO, cps.NO, cps.NO] +
                              setting_values[6:])
            variable_revision_number = 2
        if variable_revision_number == 2 and not from_matlab:
            # Added directory choice questions
            setting_values = (setting_values[:9] +
                              [DEFAULT_OUTPUT_FOLDER_NAME, "."] +
                              setting_values[9:])
            variable_revision_number = 3
        if variable_revision_number == 3 and not from_matlab:
            # Added "wants everything" setting
            #
            new_setting_values = setting_values[:11] + [cps.NO]
            for i in range(11, len(setting_values), 3):
                new_setting_values += setting_values[i:i + 3] + [cps.NO]

            setting_values = new_setting_values
            variable_revision_number = 4

        if variable_revision_number == 4 and not from_matlab:
            # Added column selector
            setting_values = setting_values[:12] + ['None|None'] + setting_values[12:]
            variable_revision_number = 5

        if variable_revision_number == 5 and not from_matlab:
            # Combined directory_choice and custom_directory
            # Removed add_indexes
            directory_choice = setting_values[9]
            custom_directory = setting_values[10]
            if directory_choice in (DIR_CUSTOM, DIR_CUSTOM_WITH_METADATA):
                if custom_directory.startswith('.'):
                    directory_choice = DEFAULT_OUTPUT_SUBFOLDER_NAME
                elif custom_directory.startswith('&'):
                    directory_choice = DEFAULT_INPUT_SUBFOLDER_NAME
                    custom_directory = '.' + custom_directory[1:]
                else:
                    directory_choice = ABSOLUTE_FOLDER_NAME
            directory = cps.DirectoryPath.static_join_string(
                    directory_choice, custom_directory)
            setting_values = (setting_values[:3] +
                              setting_values[4:9] +
                              [directory] +
                              setting_values[11:])
            variable_revision_number = 6

        if variable_revision_number == 6 and not from_matlab:
            ''' Add GenePattern export options
            self.wants_genepattern_file, self.how_to_specify_gene_name,
            self.use_which_image_for_gene_name,self.gene_name_column
            '''
            setting_values = (setting_values[:9] +
                              [cps.NO, GP_NAME_METADATA, cps.NONE, cps.NONE] +
                              setting_values[9:])
            variable_revision_number = 7

        if variable_revision_number == 7 and not from_matlab:
            # Add nan_representation
            setting_values = (
                setting_values[:SETTING_OG_OFFSET_V7] +
                [NANS_AS_NANS] + setting_values[SETTING_OG_OFFSET_V7:])
            variable_revision_number = 8

        if variable_revision_number == 8 and not from_matlab:
            # Removed output file prepend
            setting_values = setting_values[:1] + setting_values[2:]
            variable_revision_number = 9

        if variable_revision_number == 9 and not from_matlab:
            # Added prefix
            setting_values = setting_values[:SETTING_OG_OFFSET_V9] + \
                             [cps.NO, "MyExpt_"] + \
                             setting_values[SETTING_OG_OFFSET_V9:]
            variable_revision_number = 10

        if variable_revision_number == 10 and not from_matlab:
            # added overwrite choice - legacy value is "Yes"
            setting_values = setting_values[:SETTING_OG_OFFSET_V10] + \
                             [cps.YES] + \
                             setting_values[SETTING_OG_OFFSET_V10:]
            variable_revision_number = 11

        # Standardize input/output directory name references
        SLOT_DIRCHOICE = 7
        directory = setting_values[SLOT_DIRCHOICE]
        directory = cps.DirectoryPath.upgrade_setting(directory)
        setting_values = (setting_values[:SLOT_DIRCHOICE] +
                          [directory] +
                          setting_values[SLOT_DIRCHOICE + 1:])

        return setting_values, variable_revision_number, from_matlab

    def volumetric(self):
        return True


def is_object_group(group):
    """True if the group's object name is not one of the static names"""
    return not group.name.value in (IMAGE, EXPERIMENT, OBJECT_RELATIONSHIPS)


class EEObjectNameSubscriber(cps.ObjectNameSubscriber):
    """ExportToExcel needs to prepend "Image" and "Experiment" to the list of objects

    """

    def get_choices(self, pipeline):
        choices = [
            (s, '', 0, False) for s in [IMAGE, EXPERIMENT, OBJECT_RELATIONSHIPS]]
        choices += cps.ObjectNameSubscriber.get_choices(self, pipeline)
        return choices


ExportToExcel = ExportToSpreadsheet
