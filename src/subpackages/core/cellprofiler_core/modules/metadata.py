import csv
import logging
import numbers
import os
import re
import string
import time
import urllib.request

from ..constants.image import MD_SIZE_C, MD_SIZE_T, MD_SIZE_Z, MD_SIZE_X, MD_SIZE_Y
from ..constants.measurement import COLTYPE_FLOAT, C_Z, C_T, C_C, FTR_WELL, ROW_KEYS, COL_KEYS
from ..constants.measurement import COLTYPE_INTEGER
from ..constants.measurement import COLTYPE_VARCHAR
from ..constants.measurement import COLTYPE_VARCHAR_FILE_NAME
from ..constants.measurement import C_FRAME
from ..constants.measurement import C_METADATA
from ..constants.measurement import C_SERIES
from ..constants.measurement import RESERVED_METADATA_KEYS
from ..constants.module import FILTER_RULES_BUTTONS_HELP
from ..constants.module import PROTIP_RECOMMEND_ICON
from ..constants.modules.metadata import COL_INDEX
from ..constants.modules.metadata import DEFAULT_METADATA_TAGS
from ..constants.modules.metadata import CSV_JOIN_NAME
from ..constants.modules.metadata import IPD_JOIN_NAME
from ..constants.modules.metadata import COL_PATH
from ..constants.modules.metadata import COL_URL
from ..constants.modules.metadata import COL_SERIES
from ..constants.modules.metadata import DTC_ALL
from ..constants.modules.metadata import DTC_CHOOSE
from ..constants.modules.metadata import DTC_TEXT
from ..constants.modules.metadata import F_ALL_IMAGES
from ..constants.modules.metadata import F_FILTERED_IMAGES
from ..constants.modules.metadata import IDX_EXTRACTION_METHOD_COUNT
from ..constants.modules.metadata import IDX_EXTRACTION_METHOD_COUNT_V1
from ..constants.modules.metadata import IDX_EXTRACTION_METHOD_COUNT_V2
from ..constants.modules.metadata import IDX_EXTRACTION_METHOD_COUNT_V3
from ..constants.modules.metadata import IDX_EXTRACTION_METHOD_V1
from ..constants.modules.metadata import IDX_EXTRACTION_METHOD_V2
from ..constants.modules.metadata import LEN_EXTRACTION_METHOD
from ..constants.modules.metadata import LEN_EXTRACTION_METHOD_V1
from ..constants.modules.metadata import XM_FILE_NAME
from ..constants.modules.metadata import XM_FOLDER_NAME
from ..constants.modules.metadata import X_ALL_EXTRACTION_METHODS
from ..constants.modules.metadata import X_AUTOMATIC_EXTRACTION
from ..constants.modules.metadata import X_IMPORTED_EXTRACTION
from ..constants.modules.metadata import X_MANUAL_EXTRACTION
from ..module import Module
from ..pipeline import Pipeline
from ..preferences import ABSOLUTE_FOLDER_NAME
from ..preferences import URL_FOLDER_NAME
from ..preferences import report_progress
from ..setting import Binary
from ..setting import DataTypes
from ..setting import Divider
from ..setting import FileCollectionDisplay
from ..setting import HiddenCount
from ..setting import Joiner
from ..setting import RegexpText
from ..setting import SettingsGroup
from ..setting import Table
from ..setting import ValidationError
from ..setting.choice import Choice
from ..setting.do_something import DoSomething
from ..setting.do_something import RemoveSettingButton
from ..setting.filter import (
    Filter,
    FilePredicate,
    DirectoryPredicate,
    ExtensionPredicate,
)
from cellprofiler_core.setting.text import Directory
from cellprofiler_core.setting.text import Filename
from cellprofiler_core.utilities.image import generate_presigned_url
from cellprofiler_core.utilities.image import image_resource
from cellprofiler_core.utilities.measurement import find_metadata_tokens


LOGGER = logging.getLogger(__name__)

__doc__ = """\
Metadata
========

The **Metadata** module allows you to extract and associate metadata
with your images. The metadata can be extracted from the image file
itself, from a part of the file name or location, and/or from a text
file you provide.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

What is “metadata”?
^^^^^^^^^^^^^^^^^^^

The term *metadata* refers to “data about data.” For many assays,
metadata is important in the context of tagging images with various
attributes, which can include (but is not limited to) items such as the
following:

-  The row and column of the microtiter plate that the image was
   acquired from.
-  The experimental treatment applied to the well that the image was
   acquired from.
-  The number of timepoints or channels contained in the image file.
-  The image type, i.e., RGB, indexed or separate channels.
-  The height and width of an image, in pixels.

It can be helpful to inform CellProfiler about certain metadata in order
to define a specific relationship between the images and the associated
metadata. For instance:

-  You want images with a common tag to be matched together so they are
   processed together during the pipeline run. E.g., the filenames for
   fluorescent DAPI and GFP images contain different tags indicating the
   wavelength but share ‘\_s1’ in the filename if they were acquired
   from site #1, ‘\_s2’ from site #2, and so on.
-  You want certain information attached to the output measurements and
   filenames for annotation or sample-tracking purposes. E.g., some
   images are to be identified as acquired from DMSO treated wells,
   whereas others were collected from wells treated with Compound 1, 2,…
   and so forth.

The underlying assumption in matching metadata values to image sets is
that there is an exact pairing (i.e., a one-to-one match) for a given
combination of metadata tags. A common example is that for a two-channel
microtiter plate assay, the values of the plate, well, and site tags
from one channel get matched uniquely to the plate, well, and site tag
values from the other channel.

What are the inputs?
^^^^^^^^^^^^^^^^^^^^

If you do not have metadata that is relevant to your analysis, you can
leave this module in the default "*No*" setting, and continue on to the
**NamesAndTypes** module If you do have relevant metadata, the
**Metadata** module receives the file list produced by the **Images**
module. It then associates information to each file in the file list,
which can be obtained from several sources:

-  From the image file name or location (e.g., as assigned by a
   microscope). In this case, you will provide the text search pattern
   to obtain this information.
-  In a text file created and filled out by you or a laboratory
   information management system. In this case, you will point the
   module to the location of this file.

You can extract metadata from all images loaded via the **Images**
module, or a subset of them by using rules to filter the list.

What do I get as output?
^^^^^^^^^^^^^^^^^^^^^^^^

The final product of the **Metadata** module is a list of files from the
**Images** module, accompanied by the associated metadata retrieved
from the source(s) provided and matched to the desired images.

As you are extracting metadata from your various sources, you can click
the “Update” button below the divider to display a table of results
using the current settings. Each row corresponds to an image file from
the **Images** module, and the columns display the metadata obtained for
each tag specified. You can press this button as many times as needed to
display the most current metadata obtained.

.. image:: {METADATA_DISPLAY_TABLE}
   :width: 100%

Some downstream use cases for metadata include the following:

-  If the metadata establishes how channels are related to one another,
   you can use them in the **NamesAndTypes** module to aid in creating
   an image set.
-  If the images need to be further sub-divided into groups of images
   that share a common metadata value, the **Groups** module can be used
   to specify which metadata is needed for this purpose.
-  You can also use the numerical values of pieces of metadata in later modules.
   Since the metadata is stored as an image measurement and can be
   assigned as an integer or floating-point number, any module which
   allows measurements as input can make use of it.
-  Several modules are also capable of using metadata for more specific
   purposes. Refer to the module setting help for additional information
   on how to use them in the context of the specific module.

If the metadata originates from an external source such as a CSV, there
are some caveats in the cases when metadata is either missing or
duplicated for the referenced images; see the **NamesAndTypes** module
for more details.

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  *Metadata:* The prefix of each metadata tag in the per-image table.
""".format(
    **{"METADATA_DISPLAY_TABLE": image_resource("Metadata_ExampleDisplayTable.png")}
)


class Metadata(Module):
    variable_revision_number = 7
    module_name = "Metadata"
    category = "File Processing"

    CSV_JOIN_NAME = "CSV Metadata"
    IPD_JOIN_NAME = "Image Metadata"

    def create_settings(self):
        self.pipeline = None
        self.filtered_file_list = None
        # Records whether upgrade process removed an automatic extraction group
        self.removed_automatic_extraction = False

        module_explanation = [
            "The %s module optionally allows you to extract information"
            % self.module_name,
            "describing your images (i.e, metadata) which will be stored along",
            "with your measurements. This information can be contained in the",
            "file name and/or location, or in an external file.",
        ]
        self.set_notes([" ".join(module_explanation)])

        self.wants_metadata = Binary(
            "Extract metadata?",
            False,
            doc="""\
Select "*{YES}*" if your file or path names contain
information (i.e., metadata) you would like to extract and store along
with your measurements. See the main module help for more details.
""".format(
                **{"YES": "Yes"}
            ),
        )

        self.extraction_methods = []

        self.add_extraction_method(False)

        self.extraction_method_count = HiddenCount(
            self.extraction_methods, "Extraction method count"
        )

        self.add_extraction_method_button = DoSomething(
            "", "Add another extraction method", self.add_extraction_method
        )

        self.dtc_divider = Divider()

        self.data_type_choice = Choice(
            "Metadata data type",
            DTC_ALL,
            tooltips={
                DTC_TEXT: "Save all metadata as text",
                DTC_CHOOSE: "Choose the data type (text or numeric) for each metadata category",
            },
            doc="""\
Metadata can be stored as either a text or numeric value:

-  *{DTC_TEXT}:* Save all metadata item as text.
-  *{DTC_CHOOSE}:* Choose the data type separately for each metadata
   entry. An example of when this approach would be necessary would be
   if a whole filename is captured as metadata but the file name is
   numeric, e.g., “0001101”. In this situation, if the file name needs
   to be used for an arithmetic calculation or index, the name would
   need to be converted to a number and you would select “Integer” as
   the data type. On the other hand, if it important that the leading
   zeroes be retained, setting it to an integer would remove them upon
   conversion to a number. In this case, storing the metadata values as
   “Text” would be more appropriate.
""".format(
                **{"DTC_CHOOSE": DTC_CHOOSE, "DTC_TEXT": DTC_TEXT}
            ),
        )

        self.data_types = DataTypes(
            "Metadata types",
            name_fn=self.get_metadata_keys,
            doc="""\
*(Used only when “{DTC_CHOOSE}” is selected for the metadata data type)*

This setting determines the data type of each metadata field when
stored as a measurement.

-  *Text:* Save the metadata as text.
-  *Integer:* Save the metadata as an integer.
-  *Float:* Save the metadata as a decimal number.
-  *None:* Do not save the metadata as a measurement.

If you are using the metadata to match images to create your image set,
the choice of metadata type here will determine the order of matching.
See **NamesAndTypes** for more details.
""".format(
                **{"DTC_CHOOSE": DTC_CHOOSE}
            ),
        )

        self.table = Table(
            "",
            use_sash=True,
            corner_button=dict(
                fn_clicked=self.update_table,
                label="Update",
                tooltip="Update the metadata table",
            ),
        )

    def add_extraction_method(self, can_remove=True):
        group = SettingsGroup()

        if can_remove:
            group.append("divider", Divider())

        group.append(
            "extraction_method",
            Choice(
                "Metadata extraction method",
                X_ALL_EXTRACTION_METHODS,
                X_MANUAL_EXTRACTION,
                doc="""\
Metadata can be stored in either or both of two ways:

-  *Internally:* This method is often through the file naming, directory
   structuring, or the file header information.
-  *Externally:* This is through an external index, such as spreadsheet
   or database of some kind.

The **Metadata** module can extract internal or external metadata from
the images in any of three ways:

-  *{X_MANUAL_EXTRACTION}*: This approach retrieves information based
   on the file nomenclature and/or location. A special syntax called
   “regular expressions” is used to match text patterns in the file name
   or path, and then assign this text as metadata for the images you
   specify. The tag for each metadata is assigned a name that is
   meaningful to you.

   |image0|  *When would you want to use this
   option?* If you want to take advantage of the fact that acquisition
   software often automatically assigns a regular nomenclature to the
   filenames or the containing folders. Alternately, the researcher
   acquiring the images may also have a specific nomenclature they
   adhere to for bookkeeping purposes.
-  *{X_IMPORTED_EXTRACTION}*: This option retrieves metadata from a
   comma-delimited file (known as a CSV file, for comma-separated
   values) of information; you will be prompted to specify the location
   of the CSV file. You can create such a file using a spreadsheet
   program such as Microsoft Excel.

   Please make sure your metadata file name does not contain parentheses
   or spaces.
   In general, avoid having spaces in your metadata file as it could cause 
   errors in later analysis steps. You can remove spaces using a spreadsheet
   program such as Microsoft Excel and using Find and Replace function.

   |image1|  *When would you want to
   use this option?* You have information curated in software that
   allows for export to a spreadsheet. This is commonly the case for
   laboratories that use data management systems that track samples and
   acquisition.

Specifics on the metadata extraction options are described below. Any or
all of these options may be used at time; press the “Add another
extraction method” button to add more.

.. |image0| image:: {PROTIP_RECOMMEND_ICON}
.. |image1| image:: {PROTIP_RECOMMEND_ICON}
""".format(
                    **{
                        "X_IMPORTED_EXTRACTION": X_IMPORTED_EXTRACTION,
                        "X_MANUAL_EXTRACTION": X_MANUAL_EXTRACTION,
                        "PROTIP_RECOMMEND_ICON": PROTIP_RECOMMEND_ICON,
                    }
                ),
            ),
        )

        group.append(
            "source",
            Choice(
                "Metadata source",
                [XM_FILE_NAME, XM_FOLDER_NAME],
                doc="You can extract the metadata from the image's file name or from its folder name.",
            ),
        )

        group.append(
            "file_regexp",
            RegexpText(
                "Regular expression to extract from file name",
                "^(?P<Plate>.*)_(?P<Well>[A-P][0-9]{2})_s(?P<Site>[0-9])_w(?P<ChannelNumber>[0-9])",
                get_example_fn=self.example_file_fn,
                doc="""\
*(Used only if you want to extract metadata from the file name)*

The regular expression to extract the metadata from the file name is
entered here. Please see the general module help for more information on
construction of a regular expression.

Clicking the magnifying glass icon to the right will bring up a tool for
checking the accuracy of your regular expression. The regular expression
syntax can be used to name different parts of your expression. The
syntax *(?P<fieldname>expr)* will extract whatever matches *expr* and
assign it to the measurement *fieldname* for the image.

For instance, a researcher uses plate names composed of a string of
letters and numbers, followed by an underscore, then the well,
followed by another underscore, followed by an “s” and a digit
representing the site taken within the well (e.g.,
*TE12345\_A05\_s1.tif*). The following regular expression will capture
the plate, well, and site in the fields “Plate”, “Well”, and “Site”:

+----------------------------------------------------------------+------------------------------------------------------------------+
| ^(?P<Plate>.\*)\_(?P<Well>[A-P][0-9]{1,2})\_s(?P<Site>[0-9])                                                                      |
+----------------------------------------------------------------+------------------------------------------------------------------+
| ^                                                              | Start only at beginning of the file name                         |
+----------------------------------------------------------------+------------------------------------------------------------------+
| (?P<Plate>                                                     | Name the captured field *Plate*                                  |
+----------------------------------------------------------------+------------------------------------------------------------------+
| .\*)                                                           | Capture as many characters as follow                             |
+----------------------------------------------------------------+------------------------------------------------------------------+
| \_                                                             | Discard the underbar separating plate from well                  |
+----------------------------------------------------------------+------------------------------------------------------------------+
| (?P<Well>                                                      | Name the captured field *Well*                                   |
+----------------------------------------------------------------+------------------------------------------------------------------+
| [A-P]                                                          | Capture exactly one letter between A and P                       |
+----------------------------------------------------------------+------------------------------------------------------------------+
| [0-9]{1,2} )                                                   | Capture one or two digits that follow                            |
+----------------------------------------------------------------+------------------------------------------------------------------+
| \_s                                                            | Discard the underbar followed by *s* separating well from site   |
+----------------------------------------------------------------+------------------------------------------------------------------+
| (?P<Site>                                                      | Name the captured field *Site*                                   |
+----------------------------------------------------------------+------------------------------------------------------------------+
| [0-9])                                                         | Capture one digit following                                      |
+----------------------------------------------------------------+------------------------------------------------------------------+

The regular expression can be typed in the upper text box, with a sample
file name given in the lower text box. Provided the syntax is correct,
the corresponding fields will be highlighted in the same color in the
two boxes. Press *Submit* to enter the typed regular expression.

Note that if you use the special fieldnames *<WellColumn>* and
*<WellRow>* together, LoadImages will automatically create a *<Well>*
metadata field by joining the two fieldname values together. For
example, if *<WellRow>* is “A” and *<WellColumn>* is “01”, a field
*<Well>* will be “A01”. This is useful if your well row and column names
are separated from each other in the filename, but you want to retain
the standard well nomenclature.
""",
            ),
        )

        group.append(
            "folder_regexp",
            RegexpText(
                "Regular expression to extract from folder name",
                "(?P<Date>[0-9]{4}_[0-9]{2}_[0-9]{2})$",
                get_example_fn=self.example_directory_fn,
                guess=RegexpText.GUESS_FOLDER,
                doc="""\
*(Used only if you want to extract metadata from the path)*

Enter the regular expression for extracting the metadata from the
path. Note that this field is available whether you have selected
*Text-Regular expressions* to load the files or not.

Clicking the magnifying glass icon to the right will bring up a tool
that will allow you to check the accuracy of your regular expression.
The regular expression syntax can be used to name different parts of
your expression. The syntax *(?P<fieldname>expr)* will extract whatever
matches *expr* and assign it to the image’s *fieldname* measurement.

For instance, a researcher uses folder names with the date and
subfolders containing the images with the run ID (e.g.,
*./2009\_10\_02/1234/*) The following regular expression will capture
the plate, well, and site in the fields *Date* and *Run*:

+-----------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| .\*[\\\\\\\\/](?P<Date>.\*)[\\\\\\\\/](?P<Run>.\*)$                                                                                                                                                                                          |
+-----------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| .\*[\\\\\\\\/]                                          | Skip characters at the beginning of the pathname until either a slash (/) or backslash (\\\\) is encountered (depending on the operating system)                                 |
+-----------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| (?P<Date>                                           | Name the captured field *Date*                                                                                                                                                 |
+-----------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| .\*)                                                | Capture as many characters that follow                                                                                                                                         |
+-----------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| [\\\\\\\\/]                                             | Discard the slash/backslash character                                                                                                                                          |
+-----------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| (?P<Run>                                            | Name the captured field *Run*                                                                                                                                                  |
+-----------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| .\* )                                               | Capture as many characters as follow                                                                                                                                           |
+-----------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| $                                                   | The *Run* field must be at the end of the path string, i.e., the last folder on the path. This also means that the Date field contains the parent folder of the Date folder.   |
+-----------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
""",
            ),
        )

        group.append(
            "filter_choice",
            Choice(
                "Extract metadata from",
                [F_ALL_IMAGES, F_FILTERED_IMAGES],
                doc="""\
Choose whether you want to extract metadata from all of the images
chosen by the **Images** module or a subset of the images.

This setting controls how different image types (e.g., an image of the
GFP stain and a brightfield image) have different metadata extracted.
There are two choices:

-  *{F_ALL_IMAGES}*: Extract metadata from all images specified in
   **Images**. This is the simplest choice and the appropriate one if
   you have only one kind of image (or only one image). CellProfiler
   will extract metadata from all images using the same method per
   iteration.
-  *{F_FILTERED_IMAGES}*: Extract metadata depending on specific
   file attributes. This is the appropriate choice if more than one
   image was taken of each imaging site. You can specify distinctive
   criteria for each image subset with matching metadata.
""".format(
                    **{
                        "F_ALL_IMAGES": F_ALL_IMAGES,
                        "F_FILTERED_IMAGES": F_FILTERED_IMAGES,
                    }
                ),
            ),
        )

        group.append(
            "filter",
            Filter(
                "Select the filtering criteria",
                [FilePredicate(), DirectoryPredicate(), ExtensionPredicate(), ],
                'and (file does contain "")',
                doc="""\
Select "*{YES}*" to display and use rules to select files for metadata
extraction.

{FILTER_RULES_BUTTONS_HELP}
""".format(
                    **{
                        "FILTER_RULES_BUTTONS_HELP": FILTER_RULES_BUTTONS_HELP,
                        "YES": "Yes",
                    }
                ),
            ),
        )

        group.append(
            "csv_location",
            Directory(
                "Metadata file location",
                support_urls=True,
                doc="""\
*(Used only if you want to extract metadata from a file)*

The file containing the metadata must be a comma-delimited file (CSV).
You can create or edit such a file using a spreadsheet program such as
Microsoft Excel.

The CSV file needs to conform to the following format:

-  Each column describes one type of metadata.
-  Each row describes the metadata for one image site.
-  The column headers are uniquely named. You can optionally prepend
   “Metadata\_” to the header name in order to insure that it is
   interpreted correctly.
-  The CSV must be plain text, i.e., without hidden file encoding
   information. If using Excel on a Mac to edit the file, choose to save
   the file as “Windows CSV” or “Windows Comma Separated”.

The file must be saved as plain text, i.e., without hidden file encoding
information. If using Excel on a Mac to edit the file, choose to save
the file as “Windows CSV” or “Windows Comma Separated”.
""",
            ),
        )

        group.append(
            "csv_filename",
            Filename(
                "Metadata file name",
                "None",
                browse_msg="Choose CSV file",
                exts=[("Data file (*.csv)", "*.csv")],
                doc="Provide the file name of the CSV file containing the metadata you want to load.",
                get_directory_fn=group.csv_location.get_absolute_path,
                set_directory_fn=lambda path: group.csv_location.join_parts(
                    *group.csv_location.get_parts_from_path(path)
                ),
            ),
        )

        group.append(
            "csv_joiner",
            Joiner(
                "Match file and image metadata",
                allow_none=False,
                doc="""\
*(Used only if you want to extract metadata from the file and/or folder name
AND you're extracting metadata from a file)*

Match columns in your .csv file to image metadata items. If you are
using a CSV in conjunction with the filename/path metadata matching, you
might want to capture the metadata in common with both sources. For
example, you might be extracting the well tag from the image filename
while your CSV contains treatment dosage information paired with each
well. Therefore, you would want to let CellProfiler know that the well
tag extracted from the image filename and the well tag noted in the CSV
are in fact the one and the same.

This setting controls how rows in your CSV file are matched to different
images. Set the drop-downs to pair the metadata tags of the images and
the CSV, such that each row contains the corresponding tags. This can be
done for as many metadata correspondences as you may have for each
source; press |image0| to add more rows.

.. |image0| image:: {MODULE_ADD_BUTTON}
""".format(
                    **{"MODULE_ADD_BUTTON": image_resource("module_add.png")}
                ),
            ),
        )

        group.append(
            "wants_case_insensitive",
            Binary(
                "Use case insensitive matching?",
                False,
                doc="""\
*(Used only if "Match file and image metadata" is set)*

This setting controls whether row matching takes the metadata case into
account when matching.

-  Select "*{NO}*" so that metadata entries that only differ by case (for
   instance, “A01” and “a01”) will not match.
-  Select "*{YES}*" to match metadata entries that only differ by case.

|image0| If you note that your CSV metadata is
not being applied, your choice on this setting may be the culprit.

.. |image0| image:: {PROTIP_RECOMMEND_ICON}
""".format(
                    **{
                        "NO": "No",
                        "YES": "Yes",
                        "PROTIP_RECOMMEND_ICON": PROTIP_RECOMMEND_ICON,
                    }
                ),
            ),
        )

        group.imported_metadata_header_timestamp = 0
        group.imported_metadata_header_path = None
        group.imported_metadata_col_names = None
        group.imported_metadata_dicts_timestamp = 0
        group.imported_metadata_dicts_path = None
        group.imported_metadata_dicts = None
        # A temporary variable used to store a compiled regex object
        group.regex_pattern = None
        group.can_remove = can_remove
        if can_remove:
            group.append(
                "remover",
                RemoveSettingButton(
                    "", "Remove this extraction method", self.extraction_methods, group
                ),
            )
        self.extraction_methods.append(group)

    @staticmethod
    def csv_path(group):
        return os.path.join(
            group.csv_location.get_absolute_path(), group.csv_filename.value
        )

    def get_csv_header(self, group):
        """Get the header line from the imported extraction group's csv file"""
        csv_path = self.csv_path(group)
        if csv_path == group.imported_metadata_header_path:
            if group.csv_location.is_url():
                return group.imported_metadata_col_names
            if os.path.isfile(csv_path):
                timestamp = os.stat(csv_path).st_mtime
                if timestamp <= group.imported_metadata_header_timestamp:
                    return group.imported_metadata_col_names
        group.imported_metadata_header_timestamp = time.time()
        group.imported_metadata_header_path = csv_path
        try:
            if group.csv_location.is_url():
                url = generate_presigned_url(csv_path)
                fd = urllib.request.urlopen(url)
            else:
                fd = open(csv_path, "r")
            rdr = csv.DictReader(fd)
            group.imported_metadata_col_names = rdr.fieldnames
            fd.close()
        except Exception as e:
            LOGGER.error(f"Error while decoding CSV - {csv_path}: {e.strerror}")
            return None
        return group.imported_metadata_col_names

    def refresh_group_joiner(self, group):
        """Refresh the metadata entries for a group's joiner"""
        if group.extraction_method != X_IMPORTED_EXTRACTION:
            return
        #
        # Get the key set.
        #
        possible_keys = set(DEFAULT_METADATA_TAGS)
        for extract_group in self.extraction_methods:
            if extract_group is group:
                # Only want keys made before this group.
                break
            if extract_group.extraction_method == X_MANUAL_EXTRACTION:
                self.compile_regex(group)
                possible_keys.update(group.regex_pattern.groupindex.keys())
        joiner = group.csv_joiner
        assert isinstance(joiner, Joiner)
        joiner.entities[self.IPD_JOIN_NAME] = list(possible_keys)
        header_keys = self.get_csv_header(group)
        if header_keys is None:
            header_keys = ["None"]
        joiner.entities[self.CSV_JOIN_NAME] = header_keys

    @staticmethod
    def compile_regex(group):
        # Compile regex string into and object
        if group.source == XM_FILE_NAME:
            group.regex_pattern = re.compile(group.file_regexp.value)
        else:
            group.regex_pattern = re.compile(group.folder_regexp.value)

    def settings(self):
        result = [
            self.wants_metadata,
            self.data_type_choice,
            self.data_types,
            self.extraction_method_count,
        ]

        for group in self.extraction_methods:
            result += [
                group.extraction_method,
                group.source,
                group.file_regexp,
                group.folder_regexp,
                group.filter_choice,
                group.filter,
                group.csv_location,
                group.csv_joiner,
                group.wants_case_insensitive,
                group.csv_filename,
            ]
        return result

    def visible_settings(self):
        result = [self.wants_metadata]
        if self.wants_metadata:
            for group in self.extraction_methods:
                if group.can_remove:
                    result += [group.divider]
                result += [group.extraction_method]
                if group.extraction_method == X_MANUAL_EXTRACTION:
                    result += [group.source]
                    if group.source == XM_FILE_NAME:
                        result += [group.file_regexp]
                    elif group.source == XM_FOLDER_NAME:
                        result += [group.folder_regexp]
                    result += [group.filter_choice]
                    if group.filter_choice == F_FILTERED_IMAGES:
                        result += [group.filter]
                elif group.extraction_method == X_IMPORTED_EXTRACTION:
                    result += [
                        group.csv_location,
                        group.csv_filename,
                        group.filter_choice,
                    ]
                    if group.filter_choice == F_FILTERED_IMAGES:
                        result += [group.filter]
                    result += [group.csv_joiner, group.wants_case_insensitive]
                if group.can_remove:
                    result += [group.remover]
            result += [self.add_extraction_method_button]
            try:
                has_keys = len(self.get_dt_metadata_keys()) > 0
            except Exception as e:
                has_keys = False
            if has_keys:
                result += [self.dtc_divider, self.data_type_choice]
                if self.data_type_choice == DTC_CHOOSE:
                    result.append(self.data_types)
            result += [self.table]
        return result

    def example_file_fn(self):
        """Get an example file name for the regexp editor"""
        if self.filtered_file_list is not None and self.filtered_file_list:
            return self.filtered_file_list[0].filename
        return "PLATE_A01_s1_w11C78E18A-356E-48EC-B204-3F4379DC43AB.tif"

    def example_directory_fn(self):
        """Get an example directory name for the regexp editor"""
        if self.filtered_file_list is not None and self.filtered_file_list:
            return self.filtered_file_list[0].dirname
        return "/images/2012_01_12"

    def change_causes_prepare_run(self, setting):
        """Return True if changing the setting passed changes the image sets

        setting - the setting that was changed
        """
        return setting in self.settings()

    @classmethod
    def is_input_module(cls):
        return True

    def prepare_run(self, workspace):
        """Initialize the pipeline's metadata"""
        if workspace.pipeline.in_batch_mode():
            return True

        pipeline = workspace.pipeline
        assert isinstance(pipeline, Pipeline)
        filtered_file_list = pipeline.get_filtered_file_list(workspace)

        self.run_extraction(filtered_file_list)

        key_set = sorted(self.get_metadata_keys())
        pipeline.set_image_plane_details(None, key_set, self)
        return True

    def run_extraction(self, file_objects):
        if not self.wants_metadata.value:
            return
        # We first run through the methods and pre-prepare objects we'll use repeatedly.
        for group in self.extraction_methods:
            if group.extraction_method == X_MANUAL_EXTRACTION:
                # Compile regular expressions into proper objects. You can call regex.search directly with
                # the pattern string, but this wastes time converting it into a regex object during each call.
                self.compile_regex(group)
            elif group.extraction_method == X_IMPORTED_EXTRACTION:
                # Import the csv metadata as a list of dictionaries.
                self.import_csv_dict(group)
        for file_object in file_objects:
            file_object.clear_metadata()
            for group in self.extraction_methods:
                if group.filter_choice == F_FILTERED_IMAGES:
                    modpath = file_object.modpath
                    if not group.filter.evaluate((FileCollectionDisplay.NODE_FILE, modpath, None,)):
                        # Doesn't pass file filter for this extraction method
                        continue
                if group.extraction_method == X_MANUAL_EXTRACTION:
                    # REGEX Mode
                    if group.source == XM_FILE_NAME:
                        target = file_object.filename
                    else:
                        target = file_object.dirname
                    matches = group.regex_pattern.search(target)
                    if matches:
                        file_metadata = self.apply_data_types(matches.groupdict())
                        file_object.add_metadata(file_metadata)
                elif group.extraction_method == X_IMPORTED_EXTRACTION:
                    # Imported from CSV. Test the image against available dicts.
                    joins = group.csv_joiner.parse()
                    image_meta = file_object.metadata
                    valid = True
                    if group.imported_metadata_dicts is None:
                        valid = False
                    for candidate_dict in group.imported_metadata_dicts or []:
                        if None in candidate_dict:
                            # Extra columns without header labels were present. Delete them.
                            del candidate_dict[None]
                        candidate_dict = self.apply_data_types(candidate_dict)
                        valid = True
                        for join_dict in joins:
                            csv_key = join_dict[CSV_JOIN_NAME]
                            image_key = join_dict[IPD_JOIN_NAME]
                            csv_val = candidate_dict[csv_key]
                            image_val = self.apply_data_type(image_meta.get(image_key, ""), self.get_data_type(image_key))
                            if group.wants_case_insensitive:
                                if isinstance(csv_val, str):
                                    csv_val = csv_val.lower()
                                if isinstance(image_val, str):
                                    image_val = image_val.lower()
                            if csv_val != image_val:
                                valid = False
                                break
                        if valid:
                            file_object.add_metadata(candidate_dict)
                            break
                    if not valid:
                        LOGGER.info(f"No matching metadata found for {file_object.filename}")
                        break
                    pass
                else:
                    raise NotImplementedError(f"Invalid extraction method '{group.extraction_method}'")
            if FTR_WELL not in file_object.metadata:
                row_key = next((x for x in ROW_KEYS if x in file_object.metadata), None)
                col_key = next((x for x in COL_KEYS if x in file_object.metadata), None)
                if row_key is not None and col_key is not None:
                    col = file_object.metadata[col_key]
                    row = file_object.metadata[row_key]
                    if row.isdigit():
                        row = int(row)
                    if isinstance(row, int):
                        if 0 < row <= 24:
                            row = string.ascii_uppercase[row - 1]
                    well = row + col
                    file_object.set_metadata(FTR_WELL, well)
        for group in self.extraction_methods:
            if group.extraction_method == X_IMPORTED_EXTRACTION:
                # Extraction is done, clear the dict list to save memory.
                del group.imported_metadata_dicts
                group.imported_metadata_dicts = None
                group.imported_metadata_dicts_timestamp = 0
                group.imported_metadata_dicts_path = None

    def import_csv_dict(self, group):
        csv_path = self.csv_path(group)
        if csv_path == group.imported_metadata_dicts_path and group.imported_metadata_dicts is not None:
            if group.csv_location.is_url():
                return group.imported_metadata_dicts
            if os.path.isfile(csv_path):
                timestamp = os.stat(csv_path).st_mtime
                if timestamp <= group.imported_metadata_dicts_timestamp:
                    return group.imported_metadata_dicts
        group.imported_metadata_dicts_timestamp = time.time()
        group.imported_metadata_dicts_path = csv_path
        try:
            if group.csv_location.is_url():
                url = generate_presigned_url(csv_path)
                fd = urllib.request.urlopen(url)
            else:
                fd = open(csv_path, "r")
            rdr = csv.DictReader(fd)
            group.imported_metadata_dicts = [x for x in rdr]
            fd.close()
        except Exception as e:
            print("Error while decoding CSV:", e)
            return None

    def run(self, workspace):
        pass
        
    def on_activated(self, workspace):
        self.workspace = workspace
        self.pipeline = workspace.pipeline
        self.filtered_file_list = workspace.pipeline.get_filtered_file_list(workspace)
        if self.pipeline.file_list_edited:
            self.pipeline.file_list_edited = False
        for group in self.extraction_methods:
            if group.extraction_method == X_IMPORTED_EXTRACTION:
                self.refresh_group_joiner(group)
        self.table.clear_rows()
        self.table.clear_columns()
        if not self.pipeline.file_list_edited:
            self.update_table()
        else:
            # File list has changed, reset 'needs metadata extraction' flag
            pass

    def on_setting_changed(self, setting, pipeline):
        """Update the imported extraction joiners on setting changes"""
        if not self.wants_metadata:
            print("Clearing metadata")
            for file_object in self.filtered_file_list:
                file_object.clear_metadata()
            return
        visible_settings = self.visible_settings()
        if setting == self.data_types or setting == self.data_type_choice:
            # The data types affect the joiner's matching
            setting_idx = len(self.visible_settings())
        elif setting in visible_settings:
            setting_idx = visible_settings.index(setting)
        else:
            setting_idx = len(self.visible_settings())
        for group in self.extraction_methods:
            if group.extraction_method == X_IMPORTED_EXTRACTION:
                idx = max(
                    *list(
                        map(
                            visible_settings.index,
                            [
                                group.csv_joiner,
                                group.csv_location,
                                group.wants_case_insensitive,
                            ],
                        )
                    )
                )
                if idx < setting_idx:
                    continue
                self.refresh_group_joiner(group)

    def update_table(self, *args):
        self.run_extraction(self.filtered_file_list)
        columns = set(self.get_metadata_keys())
        columns.discard(COL_SERIES)
        columns.discard(COL_INDEX)
        columns = [COL_PATH, COL_SERIES, COL_INDEX] + sorted(list(columns))
        self.table.clear_columns()
        self.table.clear_rows()
        data = []
        for file_object in self.filtered_file_list:
            md_keys = file_object.metadata
            row = [md_keys.get(k, None) for k in columns]
            data.append(row)

        for i, column in enumerate(columns):
            self.table.insert_column(i, column)

        self.table.add_rows(columns, data)
        report_progress("MetadataCount", None, "Found %d rows" % len(data))

    def on_deactivated(self):
        self.pipeline = None
        self.filtered_file_list = None

    def prepare_to_create_batch(self, workspace, fn_alter_path):
        """Alter internal paths for batch creation"""
        for group in self.extraction_methods:
            if group.extraction_method == X_IMPORTED_EXTRACTION:
                group.csv_location.alter_for_create_batch(fn_alter_path)

    def prepare_settings(self, setting_values):
        """Prepare the module to receive the settings"""
        #
        # Set the number of extraction methods based on the extraction method
        # count.
        #
        n_extraction_methods = int(setting_values[IDX_EXTRACTION_METHOD_COUNT])
        if len(self.extraction_methods) > n_extraction_methods:
            del self.extraction_methods[n_extraction_methods:]

        while len(self.extraction_methods) < n_extraction_methods:
            self.add_extraction_method()

    def validate_module(self, pipeline):
        """Validate the module settings

        pipeline - current pipeline

        Metadata throws an exception if any of the metadata tags collide with
        tags that can be automatically extracted.
        """
        for group in self.extraction_methods:
            if group.extraction_method == X_MANUAL_EXTRACTION:
                re_setting = (
                    group.file_regexp
                    if group.source == XM_FILE_NAME
                    else group.folder_regexp
                )
                for token in find_metadata_tokens(re_setting.value):
                    if token.upper() in [
                        reservedtag.upper() for reservedtag in RESERVED_METADATA_KEYS
                    ]:
                        raise ValidationError(
                            'The metadata tag, "%s", is reserved for use by CellProfiler.'
                            " Please use some other tag name." % token,
                            re_setting,
                        )

    def get_metadata_keys(self):
        """Return a collection of metadata keys to be associated with files"""
        keys = set([COL_URL])
        if not self.wants_metadata:
            return keys
        keys = set(DEFAULT_METADATA_TAGS)
        for extract_group in self.extraction_methods:
            if extract_group.extraction_method == X_MANUAL_EXTRACTION:
                self.compile_regex(extract_group)
                keys.update(extract_group.regex_pattern.groupindex.keys())
            elif extract_group.extraction_method == X_IMPORTED_EXTRACTION:
                keys.update(self.get_csv_header(extract_group) or '')
        if FTR_WELL not in keys and any(k in keys for k in ROW_KEYS) and any(k in keys for k in COL_KEYS):
            keys.add(FTR_WELL)
        return keys

    def get_dt_metadata_keys(self):
        """Get the metadata keys which can have flexible datatyping

        """
        return list(
            filter(
                (lambda k: k not in self.NUMERIC_DATA_TYPES), self.get_metadata_keys() or []
            )
        )

    NUMERIC_DATA_TYPES = (
        MD_SIZE_C,
        MD_SIZE_T,
        MD_SIZE_Z,
        MD_SIZE_X,
        MD_SIZE_Y,
        C_SERIES,
        C_FRAME,
        C_C,
        C_T,
        C_Z,
    )

    def get_data_type(self, key):
        """Get the data type for a particular metadata key"""
        if isinstance(key, str):
            return self.get_data_type([key]).get(key, COLTYPE_VARCHAR)
        result = {}
        if self.data_type_choice == DTC_CHOOSE:
            data_types = DataTypes.decode_data_types(self.data_types.value_text)
        for k in key:
            if k in self.NUMERIC_DATA_TYPES:
                result[k] = COLTYPE_INTEGER
            elif self.data_type_choice == DTC_CHOOSE:
                dt = data_types.get(k, DataTypes.DT_TEXT)
                if dt == DataTypes.DT_TEXT:
                    result[k] = COLTYPE_VARCHAR
                elif dt == DataTypes.DT_INTEGER:
                    result[k] = COLTYPE_INTEGER
                elif dt == DataTypes.DT_FLOAT:
                    result[k] = COLTYPE_FLOAT
            else:
                result[k] = COLTYPE_VARCHAR

        return result

    def apply_data_types(self, input_dict):
        return {k: self.apply_data_type(v, self.get_data_type(k)) for k, v, in input_dict.items()}

    @staticmethod
    def apply_data_type(value, data_type):
        if data_type == DataTypes.DT_TEXT:
            return str(value)
        elif data_type == DataTypes.DT_INTEGER:
            try:
                return int(value)
            except ValueError:
                LOGGER.warning(f"Metadata value {value} cannot be interpreted as an integer number.")
                return value
        elif data_type == DataTypes.DT_FLOAT:
            try:
                return float(value)
            except ValueError:
                LOGGER.warning(f"Metadata value {value} cannot be interpreted as a floating point number.")
                return value

        return value

    def wants_case_insensitive_matching(self, key):
        """Return True if the key should be matched using case-insensitive matching

        key - key to check.

        Currently, there is a case-insensitive matching flag in the
        imported metadata matcher. Perhaps this should be migrated into
        the data types control, but for now, we look for the key to be
        present in the joiner for any imported metadata matcher.
        """
        if not self.wants_metadata:
            return False
        for group in self.extraction_methods:
            if (
                    group.extraction_method == X_IMPORTED_EXTRACTION
                    and group.wants_case_insensitive
            ):
                joins = group.csv_joiner.parse()
                for join in joins:
                    if key in list(join.values()):
                        return True
        return False

    def get_measurement_columns(self, pipeline):
        """Get the metadata measurements collected by this module"""
        key_types = pipeline.get_available_metadata_keys()
        result = []
        for key, coltype in list(key_types.items()):
            if self.data_type_choice == DTC_CHOOSE:
                data_type = self.get_data_type(key)
                if data_type == DataTypes.DT_NONE:
                    continue
                elif data_type == DataTypes.DT_INTEGER:
                    data_type = COLTYPE_INTEGER
                elif data_type == DataTypes.DT_FLOAT:
                    data_type = COLTYPE_FLOAT
                else:
                    data_type = COLTYPE_VARCHAR_FILE_NAME
            else:
                data_type = COLTYPE_VARCHAR_FILE_NAME
            result.append(("Image", "_".join((C_METADATA, key)), data_type,))
        return result

    def get_categories(self, pipeline, object_name):
        """Return the measurement categories for a particular object"""
        if object_name == "Image" and len(self.get_metadata_keys()) > 0:
            return [C_METADATA]
        return []

    def get_measurements(self, pipeline, object_name, category):
        if object_name == "Image" and category == C_METADATA:
            keys = self.get_metadata_keys()
            return keys
        return []

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 1:
            n_groups = int(setting_values[IDX_EXTRACTION_METHOD_COUNT_V1])
            new_setting_values = setting_values[:IDX_EXTRACTION_METHOD_V1]
            for i in range(n_groups):
                new_setting_values += setting_values[
                                      (IDX_EXTRACTION_METHOD_V1 + LEN_EXTRACTION_METHOD_V1 * i): (
                                              IDX_EXTRACTION_METHOD_V1 + LEN_EXTRACTION_METHOD_V1 * (i + 1)
                                      )
                                      ]
                new_setting_values.append("No")
            setting_values = new_setting_values
            variable_revision_number = 2

        if variable_revision_number == 2:
            # Changed naming of extraction methods, metadata sources and filtering choices
            n_groups = int(setting_values[IDX_EXTRACTION_METHOD_COUNT_V2])
            new_setting_values = setting_values[:IDX_EXTRACTION_METHOD_V2]
            for i in range(n_groups):
                group = setting_values[
                        (IDX_EXTRACTION_METHOD_V2 + LEN_EXTRACTION_METHOD * i): (
                                IDX_EXTRACTION_METHOD_V2 + LEN_EXTRACTION_METHOD * (i + 1)
                        )
                        ]
                group[0] = (
                    X_AUTOMATIC_EXTRACTION
                    if group[0] == "Automatic"
                    else (
                        X_MANUAL_EXTRACTION
                        if group[0] == "Manual"
                        else X_IMPORTED_EXTRACTION
                    )
                )
                group[1] = (
                    XM_FILE_NAME if group[1] == "From file name" else XM_FOLDER_NAME
                )
                group[4] = (
                    F_FILTERED_IMAGES
                    if group[4] == "Images selected using a filter"
                    else F_ALL_IMAGES
                )
                new_setting_values += group
            setting_values = new_setting_values
            variable_revision_number = 3

        if variable_revision_number == 3:
            # Added data types
            setting_values = (
                    setting_values[:IDX_EXTRACTION_METHOD_COUNT_V3]
                    + [DTC_TEXT, "{}"]
                    + setting_values[IDX_EXTRACTION_METHOD_COUNT_V3:]
            )
            variable_revision_number = 4

        if variable_revision_number == 4:
            # Allow metadata CSVs to be loaded from default io directories.
            groups = []
            n_groups = int(setting_values[3])
            for group_idx in range(n_groups):
                # group offset: 4
                # no. group settings: 9
                group = setting_values[4 + (group_idx * 9): 4 + ((group_idx + 1) * 9)]

                csv_location = group[6]
                directory, filename = os.path.split(csv_location)
                if any(
                        [
                            csv_location.lower().startswith(scheme)
                            for scheme in ["http:", "https:", "ftp:"]
                        ]
                ):
                    directory_choice = URL_FOLDER_NAME
                else:
                    directory_choice = ABSOLUTE_FOLDER_NAME

                group[6] = "{}|{}".format(directory_choice, directory)
                group += [filename]
                groups += [group]

            setting_values[4:] = sum(groups, [])
            variable_revision_number = 5
        if variable_revision_number == 5:
            # Add record of group metadata storage.
            new_setting_values = setting_values
            groups = []
            n_groups = int(setting_values[3])
            for group_idx in range(n_groups):
                # group offset: 4
                # no. group settings: 10
                group = setting_values[
                        4 + (group_idx * 10): 4 + ((group_idx + 1) * 10)
                        ]
                group += ["No"]
                groups += [group]
            new_setting_values[4:] = sum(groups, [])
            setting_values = new_setting_values
            variable_revision_number = 6
        if variable_revision_number == 6:
            # Remove that record of group metadata storage. Remove file header scrapers.
            new_setting_values = setting_values[:4]
            n_groups = int(setting_values[3])
            new_n_groups = 0
            for group_idx in range(n_groups):
                # group offset: 4
                # no. group settings: 11
                group = setting_values[
                        4 + (group_idx * 11): 4 + ((group_idx + 1) * 11)
                        ]
                if group[0] == X_AUTOMATIC_EXTRACTION:
                    self.removed_automatic_extraction = True
                    continue
                new_setting_values += group[:-1]
                new_n_groups += 1
            new_setting_values[3] = str(new_n_groups)
            setting_values = new_setting_values
            variable_revision_number = 7

        return setting_values, variable_revision_number

    def volumetric(self):
        return True
