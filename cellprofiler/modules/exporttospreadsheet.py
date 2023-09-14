"""
ExportToSpreadsheet
===================

**ExportToSpreadsheet** exports measurements into one or more files
that can be opened in Excel or other spreadsheet programs.

This module will convert the measurements to a comma-, tab-, or other
character-delimited text format and save them to the hard drive in one
or several files, as requested.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============

Using metadata tags for output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**ExportToSpreadsheet** can write out separate files for groups of
images based on their metadata tags. This is controlled by the directory
and file names that you enter. For instance, you might have applied two
treatments to each of your samples and labeled them with the metadata
names “Treatment1” and “Treatment2”, and you might want to create
separate files for each combination of treatments, storing all
measurements with a given “Treatment1” in separate directories. You can
do this by specifying metadata tags for the folder name and file name:

-  Choose "*Elsewhere…*" or "*Default Input/Output Folder sub-folder*" for
   the output file location. Do note that regardless of your choice,
   the Experiment.csv is saved to the Default Input/Output Folder and
   *not* to individual subfolders. All other per-image and per-object
   .csv files are saved to the appropriate subfolders.
   See `Github issue #1110 <http://github.com/CellProfiler/CellProfiler/issues/1110>`__
   for details.

-  Insert the metadata tag of choice into the output path. You can
   insert a previously defined metadata tag by either using:

   -  The insert key
   -  A right mouse button click inside the control
   -  In Windows, the Context menu key, which is between the Windows key
      and Ctrl key

   The inserted metadata tag will appear in green. To change a
   previously inserted metadata tag, navigate the cursor to just before
   the tag and either:

   -  Use the up and down arrows to cycle through possible values.
   -  Right-click on the tag to display and select the available values.

   In this instance, you would select the metadata tag “Treatment1”
-  Uncheck "*Export all measurements?*"
-  Uncheck "*Use the object name for the file name?*"
-  Using the same approach as above, select the metadata tag
   “Treatment2”, and complete the filename by appending the text “.csv”.

| Here’s an example table of the files that would be generated:

+--------------+--------------+---------------------------+
| Treatment1   | Treatment2   | Path                      |
+==============+==============+===========================+
| 1M\_NaCl     | 20uM\_DMSO   | 1M\_NaCl/20uM\_DMSO.csv   |
+--------------+--------------+---------------------------+
| 1M\_NaCl     | 40uM\_DMSO   | 1M\_NaCl/40uM\_DMSO.csv   |
+--------------+--------------+---------------------------+
| 2M\_NaCl     | 20uM\_DMSO   | 2M\_NaCl/20uM\_DMSO.csv   |
+--------------+--------------+---------------------------+
| 2M\_NaCl     | 40uM\_DMSO   | 2M\_NaCl/40uM\_DMSO.csv   |
+--------------+--------------+---------------------------+

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For details on the nomenclature used by CellProfiler for the exported
measurements, see *Help > General Help > How Measurements Are Named*.
See also
^^^^^^^^

See also **ExportToDatabase**.
"""

import base64
import csv
import logging
import os

import numpy
from cellprofiler_core.constants.image import C_MD5_DIGEST, C_SCALING, C_HEIGHT, C_WIDTH
from cellprofiler_core.constants.measurement import (
    EXPERIMENT,
    IMAGE,
    AGG_MEAN,
    AGG_MEDIAN,
    AGG_STD_DEV,
    C_URL,
    C_PATH_NAME,
    C_FILE_NAME,
    NEIGHBORS,
    R_FIRST_IMAGE_NUMBER,
    R_SECOND_IMAGE_NUMBER,
    R_FIRST_OBJECT_NUMBER,
    R_SECOND_OBJECT_NUMBER,
)
from cellprofiler_core.constants.module import (
    IO_FOLDER_CHOICE_HELP_TEXT,
    IO_WITH_METADATA_HELP_TEXT,
    USING_METADATA_HELP_REF,
    USING_METADATA_TAGS_REF,
)
from cellprofiler_core.constants.pipeline import EXIT_STATUS
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.module import Module
from cellprofiler_core.preferences import ABSOLUTE_FOLDER_NAME
from cellprofiler_core.preferences import DEFAULT_INPUT_FOLDER_NAME
from cellprofiler_core.preferences import DEFAULT_INPUT_SUBFOLDER_NAME
from cellprofiler_core.preferences import DEFAULT_OUTPUT_FOLDER_NAME
from cellprofiler_core.preferences import DEFAULT_OUTPUT_SUBFOLDER_NAME
from cellprofiler_core.preferences import get_headless
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting import Divider
from cellprofiler_core.setting import Measurement
from cellprofiler_core.setting import SettingsGroup
from cellprofiler_core.setting import ValidationError
from cellprofiler_core.setting.choice import CustomChoice, Choice
from cellprofiler_core.setting.do_something import DoSomething, RemoveSettingButton
from cellprofiler_core.setting.multichoice import MeasurementMultiChoice
from cellprofiler_core.setting.subscriber import ImageSubscriber, LabelSubscriber
from cellprofiler_core.setting.text import Directory, Text
from cellprofiler_core.utilities.core.modules.load_data import (
    is_file_name_feature,
    is_path_name_feature,
)
from cellprofiler_core.utilities.measurement import (
    find_metadata_tokens,
    get_agg_measurement_name,
)

from cellprofiler.gui.help.content import MEASUREMENT_NAMING_HELP

LOGGER = logging.getLogger(__name__)

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


class ExportToSpreadsheet(Module):
    module_name = "ExportToSpreadsheet"
    category = ["File Processing", "Data Tools"]
    variable_revision_number = 13

    def create_settings(self):
        self.delimiter = CustomChoice(
            "Select the column delimiter",
            DELIMITERS,
            doc="""\
Select the delimiter to use, i.e., the character that separates columns in a file. The
two default choices are tab and comma, but you can type in any single character delimiter
you prefer. Be sure that the delimiter you choose is not a character that is present
within your data (for example, in file names).""",
        )

        self.directory = Directory(
            "Output file location",
            dir_choices=[
                ABSOLUTE_FOLDER_NAME,
                DEFAULT_OUTPUT_FOLDER_NAME,
                DEFAULT_OUTPUT_SUBFOLDER_NAME,
                DEFAULT_INPUT_FOLDER_NAME,
                DEFAULT_INPUT_SUBFOLDER_NAME,
            ],
            doc="""\
This setting lets you choose the folder for the output files. {folder_choice}

{metadata_help}
""".format(
                folder_choice=IO_FOLDER_CHOICE_HELP_TEXT,
                metadata_help=IO_WITH_METADATA_HELP_TEXT,
            ),
        )
        self.directory.dir_choice = DEFAULT_OUTPUT_FOLDER_NAME

        self.wants_prefix = Binary(
            "Add a prefix to file names?",
            True,
            doc="""\
This setting lets you choose whether or not to add a prefix to each of
the .CSV filenames produced by **ExportToSpreadsheet**. A prefix may be
useful if you use the same directory for the results of more than one
pipeline; you can specify a different prefix in each pipeline. Select
*"Yes"* to add a prefix to each file name (e.g., “MyExpt\_Images.csv”).
Select *"No"* to use filenames without prefixes (e.g., “Images.csv”).
            """
            % globals(),
        )

        self.prefix = Text(
            "Filename prefix",
            "MyExpt_",
            doc="""\
(*Used only if “Add a prefix to file names?” is "Yes"*)

The text you enter here is prepended to the names of each file produced by
**ExportToSpreadsheet**.
            """
            % globals(),
        )

        self.wants_overwrite_without_warning = Binary(
            "Overwrite existing files without warning?",
            False,
            doc="""\
This setting either prevents or allows overwriting of old .CSV files by
**ExportToSpreadsheet** without confirmation. Select *"Yes"* to
overwrite without warning any .CSV file that already exists. Select
*"No"* to prompt before overwriting when running CellProfiler in the
GUI and to fail when running headless."""
            % globals(),
        )

        self.add_metadata = Binary(
            "Add image metadata columns to your object data file?",
            False,
            doc="""\
“Image\_Metadata\_” columns are normally exported in the Image data
file, but if you select *"Yes"*, they will also be exported with the
Object data file(s)."""
            % globals(),
        )

        self.add_filepath = Binary(
            "Add image file and folder names to your object data file?",
            False,
            doc="""\
“Image\_PathName\_” and “Image\_FileName\_” columns are normally
exported in the Image data file, but if you select *"Yes"*, they will also
be exported with the Object data file(s)."""
            % globals(),
        )

        self.nan_representation = Choice(
            "Representation of Nan/Inf",
            [NANS_AS_NANS, NANS_AS_NULLS],
            doc="""\
This setting controls the output for numeric fields if the calculated
value is infinite (*Inf*) or undefined (*NaN*). CellProfiler will
produce Inf or NaN values under certain rare circumstances, for instance
when calculating the mean intensity of an object within a masked region
of an image.

-  *%(NANS_AS_NULLS)s:* Output these values as empty fields.
-  *%(NANS_AS_NANS)s:* Output them as the strings “NaN”, “Inf” or
   “-Inf”."""
            % globals(),
        )

        self.pick_columns = Binary(
            "Select the measurements to export",
            False,
            doc="""\
Select *{YES}* to provide a button that allows you to select which
measurements you want to export. This is useful if you know exactly what
measurements you want included in the final spreadsheet(s) and additional
measurements would be a nuisance.

Alternatively, this option can be helpful for viewing spreadsheets in
programs which limit the number of rows and columns.
""".format(
                **{"YES": "Yes"}
            ),
        )

        self.columns = MeasurementMultiChoice(
            "Press button to select measurements",
            doc="""\
*(Used only when selecting the columns of measurements to export)*

This setting controls the columns to be exported. Press the button and
check the measurements or categories to export.""",
        )

        self.wants_aggregate_means = Binary(
            "Calculate the per-image mean values for object measurements?",
            False,
            doc="""\
Select *"Yes"* for **ExportToSpreadsheet** to calculate population
statistics over all the objects in each image and save that value as an
aggregate measurement in the Image file. For instance, if you are
measuring the area of the Nuclei objects and you check the box for this
option, **ExportToSpreadsheet** will create a column in the Image file
called “Mean\_Nuclei\_AreaShape\_Area”. Note that this setting can
generate a very large number of columns of data.

However, if you chose to select the specific measurements to
export, the aggregate statistics will only be computed for the selected
per-object measurements."""
            % globals(),
        )

        self.wants_aggregate_medians = Binary(
            "Calculate the per-image median values for object measurements?",
            False,
            doc="""\
Select *"Yes"* for **ExportToSpreadsheet** to calculate population
statistics over all the objects in each image and save that value as an
aggregate measurement in the Image file. For instance, if you are
measuring the area of the Nuclei objects and you check the box for this
option, **ExportToSpreadsheet** will create a column in the Image file
called “Median\_Nuclei\_AreaShape\_Area”. Note that this setting can
generate a very large number of columns of data.

However, if you chose to select the specific measurements to
export, the aggregate statistics will only be computed for the selected
per-object measurements."""
            % globals(),
        )

        self.wants_aggregate_std = Binary(
            "Calculate the per-image standard deviation values for object measurements?",
            False,
            doc="""\
Select *"Yes"* for **ExportToSpreadsheet** to calculate population
statistics over all the objects in each image and save that value as an
aggregate measurement in the Image file. For instance, if you are
measuring the area of the Nuclei objects and you check the box for this
option, **ExportToSpreadsheet** will create a column in the Image file
called “StDev\_Nuclei\_AreaShape\_Area”. Note that this setting can
generate a very large number of columns of data.

However, if you chose to select the specific measurements to
export, the aggregate statistics will only be computed for the selected
per-object measurements."""
            % globals(),
        )

        self.wants_genepattern_file = Binary(
            "Create a GenePattern GCT file?",
            False,
            doc="""\
Select *"Yes"* to create a GCT file compatible with `GenePattern`_.
The GCT file format is a tab-delimited text file format designed for
gene expression datasets; the specifics of the format are described
`here`_. By converting your measurements into a GCT file, you can make
use of GenePattern’s data visualization and clustering methods.

Each row in the GCT file represents (ordinarily) a gene and each column
represents a sample (in this case, a per-image set of measurements). In
addition to any other spreadsheets desired, enabling this option will
produce a GCT file with the extension .gct, prepended with the text
selection above. If per-image aggregate measurements are requested
above, those measurements are included in the GCT file as well.

.. _GenePattern: http://www.broadinstitute.org/cancer/software/genepattern/
.. _here: http://software.broadinstitute.org/cancer/software/genepattern/file-formats-guide"""
            % globals(),
        )

        self.how_to_specify_gene_name = Choice(
            "Select source of sample row name",
            GP_NAME_OPTIONS,
            GP_NAME_METADATA,
            doc="""\
*(Used only if a GenePattern file is requested)*

The first column of the GCT file is the unique identifier for each
sample, which is ordinarily the gene name. This information may be
specified in one of two ways:

-  *Metadata:* If you used the **Metadata** modules to add metadata to
   your images, you may specify a metadata tag that corresponds to the
   identifier for this column.
-  *Image filename:* If the gene name is not available, the image
   filename can be used as a surrogate identifier.

{meta_help}
""".format(
                meta_help=USING_METADATA_HELP_REF
            ),
        )

        self.gene_name_column = Measurement(
            "Select the metadata to use as the identifier",
            lambda: IMAGE,
            doc="""\
*(Used only if a GenePattern file is requested and metadata is used to
name each row)*

Choose the measurement that corresponds to the identifier, such as
metadata from the **Metadata** module. {meta_help}""".format(
                meta_help=USING_METADATA_HELP_REF
            ),
        )

        self.use_which_image_for_gene_name = ImageSubscriber(
            "Select the image to use as the identifier",
            "None",
            doc="""\
*(Used only if a GenePattern file is requested and image filename is
used to name each row)*

Select which image whose filename will be used to identify each sample row.""",
        )

        self.wants_everything = Binary(
            "Export all measurement types?",
            True,
            doc="""\
Select *"Yes"* to export every category of measurement.
**ExportToSpreadsheet** will create one data file for each object
produced in the pipeline, as well as per-image, per-experiment and
object relationships, if relevant. See *{naming_help}*
for more details on the various measurement types. The module will use
the object name as the file name, optionally prepending the output file
name if specified above.

Select *"No"* if you want to do either (or both) of two things:

-  Specify which objects should be exported;
-  Override the automatic nomenclature of the exported files.""".format(
                naming_help=MEASUREMENT_NAMING_HELP
            ),
        )

        self.object_groups = []
        self.add_object_group()
        self.add_button = DoSomething("", "Add another data set", self.add_object_group)

    def add_object_group(self, can_remove=True):
        group = SettingsGroup()
        group.append(
            "name",
            EEObjectNameSubscriber(
                "Data to export",
                doc="""\
*(Used only when “Export all measurements?” is set to “No”)*

Choose *Image*, *Experiment*, *Object relationships* or an object name
from the list. **ExportToSpreadsheet** will write out a file of
measurements for the given category. See *{naming_help}*
for more details on the various measurement types.""".format(
                    naming_help=MEASUREMENT_NAMING_HELP
                ),
            ),
        )

        group.append(
            "previous_file",
            Binary(
                "Combine these object measurements with those of the previous object?",
                False,
                doc="""\
*(Used only when “Export all measurements?” is set to “No”)*

Select *"Yes"* to create a file composed of measurements made on
this object and the one directly above it. This can be convenient, for
example, if you measured Nuclei, Cells, and Cytoplasm objects, and you
want to look at the measurements for all of them in a single spreadsheet.

Select *"No"* to create separate files for this and the previous
object.""",
            ),
        )

        group.append(
            "wants_automatic_file_name",
            Binary(
                "Use the object name for the file name?",
                True,
                doc="""\
*(Used only when “Export all measurements?” is set to “No”)*

Select *"Yes"* to use the object name as selected above to generate
a file name for the spreadsheet. For example, if you selected *Image*
above and have not checked the "*Prepend output file name*" option, your
output file will be named “Image.csv”.
Select *"No"* to name the file yourself.""",
            ),
        )

        group.append(
            "file_name",
            Text(
                "File name",
                "DATA.csv",
                metadata=True,
                doc="""\
*(Used only when “Export all measurements?” is set to “No”)*

Enter a file name for the named objects’ measurements.
**ExportToSpreadsheet** will prepend the name of the measurements file
to this if you asked to do so above. If you have metadata associated
with your images, this setting will also substitute metadata tags if
desired.

{tags}

{help}
""".format(
                    tags=USING_METADATA_TAGS_REF, help=USING_METADATA_HELP_REF
                )
                % globals(),
            ),
        )

        group.append(
            "remover",
            RemoveSettingButton("", "Remove this data set", self.object_groups, group),
        )
        group.append("divider", Divider(line=False))

        self.object_groups.append(group)

    def prepare_settings(self, setting_values):
        """Add enough object groups to capture the settings"""
        setting_count = len(setting_values)
        assert (setting_count - SETTING_OG_OFFSET) % SETTING_OBJECT_GROUP_CT == 0
        group_count = int((setting_count - SETTING_OG_OFFSET) / SETTING_OBJECT_GROUP_CT)
        del self.object_groups[group_count:]

        while len(self.object_groups) < group_count:
            self.add_object_group()

    def settings(self):
        """Return the settings in the order used when storing """
        result = [
            self.delimiter,
            self.add_metadata,
            self.add_filepath,
            self.pick_columns,
            self.wants_aggregate_means,
            self.wants_aggregate_medians,
            self.wants_aggregate_std,
            self.directory,
            self.wants_genepattern_file,
            self.how_to_specify_gene_name,
            self.use_which_image_for_gene_name,
            self.gene_name_column,
            self.wants_everything,
            self.columns,
            self.nan_representation,
            self.wants_prefix,
            self.prefix,
            self.wants_overwrite_without_warning,
        ]
        for group in self.object_groups:
            result += [
                group.name,
                group.previous_file,
                group.file_name,
                group.wants_automatic_file_name,
            ]
        return result

    def visible_settings(self):
        """Return the settings as seen by the user"""
        result = [self.delimiter, self.directory, self.wants_prefix]
        if self.wants_prefix:
            result += [self.prefix]
        result += [
            self.wants_overwrite_without_warning,
            self.add_metadata,
            self.add_filepath,
            self.nan_representation,
            self.pick_columns,
        ]
        if self.pick_columns:
            result += [self.columns]
        result += [
            self.wants_aggregate_means,
            self.wants_aggregate_medians,
            self.wants_aggregate_std,
            self.wants_genepattern_file,
        ]
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
                    if (not previous_group is None) and is_object_group(previous_group):
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
        """Test the module settings to make sure they are internally consistent"""
        if len(self.delimiter.value) != 1 and not self.delimiter.value in (
            DELIMITER_TAB,
            DELIMITER_COMMA,
        ):
            raise ValidationError(
                "The CSV field delimiter must be a single character", self.delimiter
            )

        """Make sure metadata tags exist"""
        for group in self.object_groups:
            if not group.wants_automatic_file_name:
                text_str = group.file_name.value
                undefined_tags = pipeline.get_undefined_metadata_tags(text_str)
                if len(undefined_tags) > 0:
                    raise ValidationError(
                        "%s is not a defined metadata tag. Check the metadata specifications in your load modules"
                        % undefined_tags[0],
                        group.file_name,
                    )
      
        """Check if image features are exported if GCTs are being made"""
        if self.wants_genepattern_file:
            measurement_columns = pipeline.get_measurement_columns()
            image_features = self.filter_columns([x[1] for x in measurement_columns if x[0]==IMAGE],IMAGE)
            name_feature, _ = self.validate_image_features_exist(
                            image_features,
                            )

            if name_feature == []:
                raise ValidationError(
                    "At least one path measurement plus the feature selected in 'Select source of sample row name' must be enabled for GCT file creation. Use 'Press button to select measurements' to enable these measurements, or set 'Select measurements to export' to No.",
                    self.wants_genepattern_file
                )

    def validate_module_warnings(self, pipeline):
        """Warn user re: Test mode """
        if pipeline.test_mode:
            raise ValidationError(
                "ExportToSpreadsheet will not produce output in Test Mode",
                self.directory,
            )

        """Warn user that changing the extension may cause Excel to stuff everything into one column"""
        if not self.wants_everything.value:
            all_extensions = [
                os.path.splitext(group.file_name.value)[1]
                for group in self.object_groups
            ]
            is_valid_extension = [
                not group.wants_automatic_file_name.value
                and (
                    (extension == ".csv" and self.delimiter == DELIMITER_COMMA)
                    or (extension == ".txt" and self.delimiter == DELIMITER_TAB)
                )
                for (extension, group) in zip(all_extensions, self.object_groups)
            ]
            if not all(is_valid_extension):
                raise ValidationError(
                    "To avoid formatting problems in Excel, use the extension .csv for "
                    "comma-delimited files and .txt for tab-delimited..",
                    self.object_groups[is_valid_extension.index(False)].file_name,
                )

    @property
    def delimiter_char(self):
        if self.delimiter == DELIMITER_TAB:
            return "\t"
        elif self.delimiter == DELIMITER_COMMA:
            return ","
        else:
            return self.delimiter.value

    def prepare_run(self, workspace):
        """Prepare an image set to be run

        workspace - workspace with image set populated (at this point)

        returns False if analysis can't be done
        """
        maximum_image_sets = 500

        if workspace.measurements.has_groups():
            group_numbers = workspace.measurements["Image", "Group_Number", workspace.measurements.get_image_numbers()]
            max_image_set_len = max(numpy.bincount(group_numbers))
        elif workspace.measurements.has_measurements("Image", "Group_Length", 1):
            num_images = workspace.measurements.image_set_count
            max_image_set_len = max(workspace.measurements.get_measurement(
                "Image", "Group_Length", range(1, num_images + 1)))
        else:
            max_image_set_len = workspace.measurements.image_set_count
        if max_image_set_len > maximum_image_sets:
            if get_headless():
                LOGGER.warning("Given the large number of image sets, you may want to consider using "
                                "ExportToDatabase as opposed to ExportToSpreadsheet.")
            else:
                msg = (
                    f"You are using ExportToSpreadsheet to export {workspace.measurements.image_set_count} image sets. "
                    "Instead we suggest using ExportToDatabase because ExportToSpreadsheet"
                    " may fail on large image sets. Do you want to continue?"
                )
                import wx
                result = wx.MessageBox(
                    msg,
                    caption="ExportToSpreadsheet: Large number of image sets",
                    style=wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION,
                )
                if result == wx.NO:
                    return False
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
                        object_name, workspace, image_set_number
                    )
                    columns.append((object_name, path))
            else:
                first = True
                for i in range(len(self.object_groups)):
                    group = self.object_groups[i]
                    last_in_file = self.last_in_file(i)
                    if first:
                        filename = self.make_objects_file_name(
                            group.name.value, workspace, image_set_number, group
                        )
                        first = False
                    columns.append((group.name.value, filename))
                    if last_in_file:
                        first = True
            workspace.display_data.header = header
            workspace.display_data.columns = columns

    def display(self, workspace, figure):
        figure.set_subplots((1, 1))
        if workspace.display_data.columns is None:
            figure.subplot_table(0, 0, [["Data written to spreadsheet"]])
        elif workspace.pipeline.test_mode:
            figure.subplot_table(
                0, 0, [["Data not written to spreadsheets in test mode"]]
            )
        else:
            figure.subplot_table(
                0,
                0,
                workspace.display_data.columns,
                col_labels=workspace.display_data.header,
            )

    def run_as_data_tool(self, workspace):
        """Run the module as a data tool

        For ExportToSpreadsheet, we do the "post_run" method in order to write
        out the .csv files as if the experiment had just finished.
        """
        #
        # Set the measurements to the end of the list to mimic the state
        # at the end of the run.
        #
        m = workspace.measurements
        m.image_set_number = m.image_set_count
        self.post_run(workspace)

    def post_run(self, workspace):
        """Save measurements at end of run"""
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
        """Return true if the group is the last to be included in a csv file

        i - the index of the group being considered.

        Objects can be collected together in one file. Return true if
        this is the last object in a collection.
        """

        group = self.object_groups[i]
        return (
            (i == len(self.object_groups) - 1)
            or (not is_object_group(group))
            or (not is_object_group(self.object_groups[i + 1]))
            or (not self.object_groups[i + 1].previous_file.value)
        )

    def should_stop_writing_measurements(self):
        """All subsequent modules should not write measurements"""
        return True

    def get_metadata_groups(self, workspace, settings_group=None):
        """Find the metadata groups that are relevant for creating the file name

        workspace - the workspace with the image set metadata elements and
                    grouping measurements populated.
        settings_group - if saving individual objects, this is the settings
                         group that controls naming the files.
        """
        if settings_group is None or settings_group.wants_automatic_file_name:
            tags = []
        else:
            tags = find_metadata_tokens(settings_group.file_name.value)
        if self.directory.is_custom_choice:
            tags += find_metadata_tokens(self.directory.custom_path)
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
                self.make_image_file(
                    metadata_group.image_numbers, workspace, settings_group
                )
                if self.wants_genepattern_file.value:
                    self.make_gct_file(
                        metadata_group.image_numbers, workspace, settings_group
                    )
            elif len(object_names) == 1 and object_names[0] == OBJECT_RELATIONSHIPS:
                self.make_relationships_file(
                    metadata_group.image_numbers, workspace, settings_group
                )
            else:
                self.make_object_file(
                    object_names,
                    metadata_group.image_numbers,
                    workspace,
                    settings_group,
                )

    def make_full_filename(self, file_name, workspace=None, image_set_number=None):
        """Convert a file name into an absolute path

        We do a few things here:
        * apply metadata from an image set to the file name if an
          image set is specified
        * change the relative path into an absolute one using the "." and "&"
          convention
        * Create any directories along the path
        """
        if image_set_number is not None and workspace is not None:
            file_name = workspace.measurements.apply_metadata(
                file_name, image_set_number
            )
        measurements = None if workspace is None else workspace.measurements
        path_name = self.directory.get_absolute_path(measurements, image_set_number)
        if self.wants_prefix:
            file_name = self.prefix.value + file_name
        file_name = os.path.join(path_name, file_name)
        path, file = os.path.split(file_name)
        if not os.path.isdir(path):
            os.makedirs(path)
        return os.path.join(path, file)

    def extension(self):
        """Return the appropriate extension for the CSV file name

        The appropriate extension is "csv" if comma is used as the
        delimiter, otherwise "txt"
        """
        return "csv" if self.delimiter == DELIMITER_COMMA else "txt"

    def make_objects_file_name(
        self, object_name, workspace, image_set_number, settings_group=None
    ):
        """Concoct the .CSV filename for some object category

        :param object_name: name of the objects whose measurements are to be
                            saved (or IMAGES or EXPERIMENT)
        :param workspace: the current workspace
        :param image_set_number: the current image set number
        :param settings_group: the settings group used to name the file
        """
        if self.wants_everything:
            filename = "%s.%s" % (object_name, self.extension())

            if object_name == EXPERIMENT:
                # No metadata substitution allowed for experiment file
                return self.make_full_filename(filename)
            return self.make_full_filename(filename, workspace, image_set_number)
        if settings_group.wants_automatic_file_name:
            filename = "%s.%s" % (settings_group.name.value, self.extension())
        else:
            filename = settings_group.file_name.value
        filename = self.make_full_filename(filename, workspace, image_set_number)
        return filename

    def make_gct_file_name(self, workspace, image_set_number, settings_group=None):
        """Concoct a name for the .gct file

        workspace - workspace containing metadata measurements
        image_number - the first image number in the group being written
        settings_group - the settings group asking for the file to be written
                        if not wants_everything
        """
        file_name = self.make_objects_file_name(
            IMAGE, workspace, image_set_number, settings_group
        )
        if any([file_name.lower().endswith(x) for x in (".csv", "txt")]):
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
            object_names = {IMAGE, EXPERIMENT, OBJECT_RELATIONSHIPS}
            object_providers = workspace.pipeline.get_provider_dictionary(
                "objectgroup", self
            )
            object_names.update(list(object_providers.keys()))
            metadata_groups = self.get_metadata_groups(workspace)
            for object_name in object_names:
                for metadata_group in metadata_groups:
                    image_number = metadata_group.image_numbers[0]
                    if object_name == IMAGE and self.wants_genepattern_file:
                        files_to_check.append(
                            self.make_gct_file_name(workspace, image_number)
                        )
                    files_to_check.append(
                        self.make_objects_file_name(
                            object_name, workspace, image_number
                        )
                    )
        else:
            first_in_file = True
            for i, group in enumerate(self.object_groups):
                if first_in_file:
                    metadata_groups = self.get_metadata_groups(workspace, group)
                    for metadata_group in metadata_groups:
                        image_number = metadata_group.image_numbers[0]
                        files_to_check.append(
                            self.make_objects_file_name(
                                group.name.value, workspace, image_number, group
                            )
                        )
                #
                # set first_in_file for next time around
                #
                first_in_file = self.last_in_file(i)

        files_to_overwrite = list(filter(os.path.isfile, files_to_check))
        if len(files_to_overwrite) > 0:
            if get_headless():
                LOGGER.error(
                    "ExportToSpreadsheet is configured to refrain from overwriting files and the following file(s) already exist: %s"
                    % ", ".join(files_to_overwrite)
                )
                return False
            msg = "Overwrite the following file(s)?\n" + "\n".join(files_to_overwrite)
            import wx

            result = wx.MessageBox(
                msg,
                caption="ExportToSpreadsheet: Overwrite existing files",
                style=wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION,
            )
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
            EXPERIMENT, workspace, 1, settings_group
        )
        feature_names = [
            feature_name
            for feature_name in m.get_feature_names(EXPERIMENT)
            if feature_name != EXIT_STATUS
        ]
        if len(feature_names) == 0:
            return
        fd = open(file_name, "w", newline="")
        try:
            writer = csv.writer(fd, delimiter=self.delimiter_char)
            writer.writerow((EH_KEY, EH_VALUE))
            for feature_name in feature_names:
                v = m.get_all_measurements(EXPERIMENT, feature_name)
                if isinstance(v, numpy.ndarray) and v.dtype == numpy.uint8:
                    v = base64.b64encode(v.data)
                elif isinstance(v, bytes):
                    v = v.decode("unicode_escape", errors='ignore')
                else:
                    v = str(v)
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
            IMAGE, workspace, image_set_numbers[0], settings_group
        )
        image_features = m.get_feature_names(IMAGE)
        image_features.insert(0, IMAGE_NUMBER)

        fd = open(file_name, "w", newline="")
        try:
            writer = csv.writer(fd, delimiter=self.delimiter_char)
            for img_number in image_set_numbers:
                aggs = []
                if self.wants_aggregate_means:
                    aggs.append(AGG_MEAN)
                if self.wants_aggregate_medians:
                    aggs.append(AGG_MEDIAN)
                if self.wants_aggregate_std:
                    aggs.append(AGG_STD_DEV)
                agg_measurements = m.compute_aggregate_measurements(img_number, aggs)
                if img_number == image_set_numbers[0]:
                    ordered_agg_names = list(agg_measurements.keys())
                    ordered_agg_names.sort()
                    image_features += ordered_agg_names
                    image_features.sort()
                    image_features = self.filter_columns(image_features, IMAGE)
                    if image_features is None:
                        return
                    writer.writerow(image_features)
                row = []
                for feature_name in image_features:

                    if feature_name == IMAGE_NUMBER:
                        row.append(str(img_number))
                    else:
                        if feature_name in agg_measurements:
                            value = agg_measurements[feature_name]
                        else:
                            value = m[IMAGE, feature_name, img_number]
                        if value is None:
                            row.append("")
                        elif isinstance(value, str):
                            row.append(value)
                        elif isinstance(value, bytes):
                            row.append(value.decode())
                        elif (
                            isinstance(value, numpy.ndarray)
                            and value.dtype == numpy.uint8
                        ):
                            row.append(base64.b64encode(value.data))
                        elif numpy.isnan(value):
                            if self.nan_representation == NANS_AS_NULLS:
                                row.append("")
                            else:
                                row.append(str(numpy.NaN))
                        else:
                            row.append(str(value))
                writer.writerow(row)
        finally:
            fd.close()

    def validate_image_features_exist(self,image_features):
        # Place the one of the paths and desired info column up front in image feature list
        description_feature = [
            x for x in image_features if x.startswith(C_PATH_NAME + "_")
        ]
        if self.how_to_specify_gene_name == GP_NAME_METADATA:
            name_feature = [self.gene_name_column.value]
            if name_feature[0] not in image_features:
                name_feature = []
        elif self.how_to_specify_gene_name == GP_NAME_FILENAME:
            name_feature = [
                x
                for x in image_features
                if x.startswith(
                    "_".join(
                        (
                            C_FILE_NAME,
                            self.use_which_image_for_gene_name.value,
                        )
                    )
                )
            ]
        if len(name_feature) == 0 or len(description_feature) == 0:
            return [],[]
        else: 
            return name_feature, description_feature

    def make_gct_file(self, image_set_numbers, workspace, settings_group):
        """Make a GenePattern file containing image measurements
        Format specifications located at http://www.broadinstitute.org/cancer/software/genepattern/tutorial/gp_fileformats.html?gct

        file_name - create a file with this name
        image_set_numbers - the image sets whose data gets extracted
        workspace - workspace containing the measurements
        """

        file_name = self.make_gct_file_name(
            workspace, image_set_numbers[0], settings_group
        )

        def ignore_feature(feature_name):
            """Return true if we should ignore a feature"""
            if (
                is_file_name_feature(feature_name)
                or is_path_name_feature(feature_name)
                or feature_name.startswith("ImageNumber")
                or feature_name.startswith("Group_Number")
                or feature_name.startswith("Group_Index")
                or feature_name.startswith("Description_")
                or feature_name.startswith("ModuleError_")
                or feature_name.startswith("TimeElapsed_")
                or feature_name.startswith("ExecutionTime_")
                or feature_name.startswith(C_URL)
                or feature_name.startswith(C_MD5_DIGEST)
                or feature_name.startswith(C_SCALING)
                or feature_name.startswith(C_HEIGHT)
                or feature_name.startswith(C_WIDTH)
            ):
                return True
            return False

        m = workspace.measurements
        image_features = m.get_feature_names(IMAGE)
        image_features.insert(0, IMAGE_NUMBER)

        fd = open(file_name, "w", newline="")
        try:
            writer = csv.writer(fd, delimiter="\t")
            for img_number in image_set_numbers:
                aggs = []
                if self.wants_aggregate_means:
                    aggs.append(AGG_MEAN)
                if self.wants_aggregate_medians:
                    aggs.append(AGG_MEDIAN)
                if self.wants_aggregate_std:
                    aggs.append(AGG_STD_DEV)
                agg_measurements = m.compute_aggregate_measurements(img_number, aggs)

                if img_number == image_set_numbers[0]:
                    ordered_agg_names = list(agg_measurements.keys())
                    ordered_agg_names.sort()
                    image_features += ordered_agg_names
                    image_features.sort()
                    image_features = self.filter_columns(image_features, IMAGE)
                    if image_features is None:
                        return

                    # Count # of actual measurements
                    num_measures = 0
                    for feature_name in image_features:
                        if (
                            not ignore_feature(feature_name)
                            or feature_name in agg_measurements
                        ):
                            num_measures += 1

                    writer.writerow(["#1.2"])
                    writer.writerow([len(image_set_numbers), num_measures])

                    # Keep measurements only
                    measurement_feature_names = [
                        x for x in image_features if not ignore_feature(x)
                    ]

                    # The first headers need to be 'NAME' and 'Description'
                    written_image_names = [
                        "NAME",
                        "Description",
                    ] + measurement_feature_names
                    writer.writerow(written_image_names)

                    name_feature, description_feature = self.validate_image_features_exist(
                        image_features
                        )

                    if name_feature == []:
                        return

                    image_features = [
                        name_feature[0],
                        description_feature[0],
                    ] + measurement_feature_names

                # Output all measurements
                row = [
                    agg_measurements[feature_name]
                    if feature_name in agg_measurements
                    else m.get_measurement(IMAGE, feature_name, img_number)
                    for feature_name in image_features
                ]
                row = [
                    "" if x is None else x if numpy.isscalar(x) else x[0] for x in row
                ]
                writer.writerow(row)
        finally:
            fd.close()

    def filter_columns(self, features, object_name):
        if self.pick_columns:
            columns = [
                self.columns.get_measurement_feature(x)
                for x in self.columns.selections
                if self.columns.get_measurement_object(x) == object_name
            ]
            if object_name == IMAGE:
                if IMAGE_NUMBER not in columns:
                    columns.insert(0, IMAGE_NUMBER)
                for agg, wants_it in (
                    (AGG_MEAN, self.wants_aggregate_means),
                    (AGG_MEDIAN, self.wants_aggregate_medians),
                    (AGG_STD_DEV, self.wants_aggregate_std),
                ):
                    if not wants_it:
                        continue
                    for column in self.columns.selections:
                        if self.columns.get_measurement_object(column) not in (
                            IMAGE,
                            EXPERIMENT,
                            NEIGHBORS,
                        ):
                            columns += [
                                get_agg_measurement_name(
                                    agg,
                                    self.columns.get_measurement_object(column),
                                    self.columns.get_measurement_feature(column),
                                )
                            ]

            columns = set(columns)
            features = [x for x in features if x in columns]
        elif object_name == IMAGE:
            # Exclude any thumbnails if they've been created for ExportToDatabase
            features = [x for x in features if not x.startswith("Thumbnail_")]
        return features

    def make_object_file(
        self, object_names, image_set_numbers, workspace, settings_group=None
    ):
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
            object_names[0], workspace, image_set_numbers[0], settings_group
        )
        features = [(IMAGE, IMAGE_NUMBER), (object_names[0], OBJECT_NUMBER)]
        columns = list(
            map((lambda c: c[:2]), workspace.pipeline.get_measurement_columns())
        )
        if self.add_metadata.value:
            mdfeatures = [
                (IMAGE, name)
                for object_name, name in columns
                if name.startswith("Metadata_") and object_name == IMAGE
            ]
            mdfeatures.sort()
            features += mdfeatures
        if self.add_filepath.value:
            filefeatures = [
                (IMAGE, name)
                for object_name, name in columns
                if name.startswith(("PathName_", "FileName_")) and object_name == IMAGE
            ]
            filefeatures.sort()
            features += filefeatures
        for object_name in object_names:
            ofeatures = [
                feature for col_object, feature in columns if col_object == object_name
            ]
            ofeatures = self.filter_columns(ofeatures, object_name)
            ofeatures = [(object_name, feature_name) for feature_name in ofeatures]
            ofeatures.sort()
            features += ofeatures
        fd = open(file_name, "w", newline="")
        try:
            writer = csv.writer(fd, delimiter=self.delimiter_char)

            #
            # We write the object names in the first row of headers if there are
            # multiple objects. Otherwise, we just write the feature names
            #
            for i in (0, 1) if len(object_names) > 1 else (1,):
                writer.writerow([x[i] for x in features])

            for img_number in image_set_numbers:
                object_count = numpy.max(
                    [
                        # If no objects are found in the image, we can't find the max of None - 4653
                        m.get_measurement(IMAGE, "Count_%s" % name, img_number) or 0
                        for name in object_names
                    ]
                )
                object_count = int(object_count) if object_count and not numpy.isnan(object_count) else 0
                columns = [
                    numpy.repeat(img_number, object_count)
                    if feature_name == IMAGE_NUMBER
                    else numpy.arange(1, object_count + 1)
                    if feature_name == OBJECT_NUMBER
                    else numpy.repeat(numpy.NAN, object_count)
                    if not m.has_feature(object_name, feature_name)
                    else numpy.repeat(
                        m.get_measurement(IMAGE, feature_name, img_number), object_count
                    )
                    if object_name == IMAGE
                    else m.get_measurement(object_name, feature_name, img_number)
                    for object_name, feature_name in features
                ]
                for obj_index in range(object_count):
                    row = [
                        column[obj_index]
                        if (column is not None and obj_index < column.shape[0])
                        else numpy.NAN
                        for column in columns
                    ]
                    if self.nan_representation == NANS_AS_NULLS:
                        row = [
                            ""
                            if (field is None)
                            or (numpy.isreal(field) and not numpy.isfinite(field))
                            else field
                            for field in row
                        ]
                    writer.writerow(row)
        finally:
            fd.close()

    def make_relationships_file(
        self, image_set_numbers, workspace, settings_group=None
    ):
        """Create a CSV file documenting the relationships between objects"""

        file_name = self.make_objects_file_name(
            OBJECT_RELATIONSHIPS, workspace, image_set_numbers[0], settings_group
        )
        m = workspace.measurements
        assert isinstance(m, Measurements)
        fd = open(file_name, "w", newline="")
        module_map = {}
        for module in workspace.pipeline.modules():
            module_map[module.module_num] = module.module_name

        try:
            writer = csv.writer(fd, delimiter=self.delimiter_char)
            writer.writerow(
                [
                    "Module",
                    "Module Number",
                    "Relationship",
                    "First Object Name",
                    "First Image Number",
                    "First Object Number",
                    "Second Object Name",
                    "Second Image Number",
                    "Second Object Number",
                ]
            )
            for key in m.get_relationship_groups():
                r = m.get_relationships(
                    key.module_number,
                    key.relationship,
                    key.object_name1,
                    key.object_name2,
                    image_numbers=image_set_numbers,
                )
                for (
                    image_number_1,
                    image_number_2,
                    object_number_1,
                    object_number_2,
                ) in zip(
                    r[R_FIRST_IMAGE_NUMBER],
                    r[R_SECOND_IMAGE_NUMBER],
                    r[R_FIRST_OBJECT_NUMBER],
                    r[R_SECOND_OBJECT_NUMBER],
                ):
                    module_name = module_map[key.module_number]
                    writer.writerow(
                        [
                            module_name,
                            key.module_number,
                            key.relationship,
                            key.object_name1,
                            image_number_1,
                            object_number_1,
                            key.object_name2,
                            image_number_2,
                            object_number_2,
                        ]
                    )
        finally:
            fd.close()

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

        ExportToSpreadsheet has to convert the path to file names to
        something that can be used on the cluster.
        """
        self.directory.alter_for_create_batch_files(fn_alter_path)
        return True

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        """Adjust the setting values based on the version that saved them

        """

        if variable_revision_number == 1:
            # Added aggregate questions
            setting_values = (
                setting_values[:6] + ["No", "No", "No"] + setting_values[6:]
            )
            variable_revision_number = 2
        if variable_revision_number == 2:
            # Added directory choice questions
            setting_values = (
                setting_values[:9]
                + [DEFAULT_OUTPUT_FOLDER_NAME, "."]
                + setting_values[9:]
            )
            variable_revision_number = 3
        if variable_revision_number == 3:
            # Added "wants everything" setting
            #
            new_setting_values = setting_values[:11] + ["No"]
            for i in range(11, len(setting_values), 3):
                new_setting_values += setting_values[i : i + 3] + ["No"]

            setting_values = new_setting_values
            variable_revision_number = 4

        if variable_revision_number == 4:
            # Added column selector
            setting_values = setting_values[:12] + ["None|None"] + setting_values[12:]
            variable_revision_number = 5

        if variable_revision_number == 5:
            # Combined directory_choice and custom_directory
            # Removed add_indexes
            directory_choice = setting_values[9]
            custom_directory = setting_values[10]
            if directory_choice in (DIR_CUSTOM, DIR_CUSTOM_WITH_METADATA):
                if custom_directory.startswith("."):
                    directory_choice = DEFAULT_OUTPUT_SUBFOLDER_NAME
                elif custom_directory.startswith("&"):
                    directory_choice = DEFAULT_INPUT_SUBFOLDER_NAME
                    custom_directory = "." + custom_directory[1:]
                else:
                    directory_choice = ABSOLUTE_FOLDER_NAME
            directory = Directory.static_join_string(directory_choice, custom_directory)
            setting_values = (
                setting_values[:3]
                + setting_values[4:9]
                + [directory]
                + setting_values[11:]
            )
            variable_revision_number = 6

        if variable_revision_number == 6:
            """ Add GenePattern export options
            self.wants_genepattern_file, self.how_to_specify_gene_name,
            self.use_which_image_for_gene_name,self.gene_name_column
            """
            setting_values = (
                setting_values[:9]
                + ["No", GP_NAME_METADATA, "None", "None"]
                + setting_values[9:]
            )
            variable_revision_number = 7

        if variable_revision_number == 7:
            # Add nan_representation
            setting_values = (
                setting_values[:SETTING_OG_OFFSET_V7]
                + [NANS_AS_NANS]
                + setting_values[SETTING_OG_OFFSET_V7:]
            )
            variable_revision_number = 8

        if variable_revision_number == 8:
            # Removed output file prepend
            setting_values = setting_values[:1] + setting_values[2:]
            variable_revision_number = 9

        if variable_revision_number == 9:
            # Added prefix
            setting_values = (
                setting_values[:SETTING_OG_OFFSET_V9]
                + ["No", "MyExpt_"]
                + setting_values[SETTING_OG_OFFSET_V9:]
            )
            variable_revision_number = 10

        if variable_revision_number == 10:
            # added overwrite choice - legacy value is "Yes"
            setting_values = (
                setting_values[:SETTING_OG_OFFSET_V10]
                + ["Yes"]
                + setting_values[SETTING_OG_OFFSET_V10:]
            )
            variable_revision_number = 11

        if variable_revision_number == 11:
            setting_values = setting_values[:2] + setting_values[3:]

            variable_revision_number = 12
        if variable_revision_number == 12:
            # Add "add file path" setting.
            setting_values = setting_values[:2] + ["No"] + setting_values[2:]
            variable_revision_number = 13

        # Standardize input/output directory name references
        SLOT_DIRCHOICE = 7
        directory = setting_values[SLOT_DIRCHOICE]
        directory = Directory.upgrade_setting(directory)
        setting_values = (
            setting_values[:SLOT_DIRCHOICE]
            + [directory]
            + setting_values[SLOT_DIRCHOICE + 1 :]
        )

        return setting_values, variable_revision_number

    def volumetric(self):
        return True


def is_object_group(group):
    """True if the group's object name is not one of the static names"""
    return not group.name.value in (IMAGE, EXPERIMENT, OBJECT_RELATIONSHIPS)


class EEObjectNameSubscriber(LabelSubscriber):
    """ExportToExcel needs to prepend "Image" and "Experiment" to the list of objects

    """

    def get_choices(self, pipeline):
        choices = [(s, "", 0, False) for s in [IMAGE, EXPERIMENT, OBJECT_RELATIONSHIPS]]
        choices += LabelSubscriber.get_choices(self, pipeline)
        return choices


ExportToExcel = ExportToSpreadsheet
