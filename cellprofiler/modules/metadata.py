# coding=utf-8

import cellprofiler.icons
import numpy as np
import logging

logger = logging.getLogger(__name__)
import csv
import re
import os
import time
import urllib
import urlparse
import _help

import cellprofiler.module as cpm
import cellprofiler.measurement as cpmeas
import cellprofiler.pipeline as cpp
import cellprofiler.setting as cps
from cellprofiler.setting import YES, NO
import javabridge as J
from cellprofiler.modules.images import FilePredicate
from cellprofiler.modules.images import ExtensionPredicate
from cellprofiler.modules.images import ImagePredicate
from cellprofiler.modules.images import DirectoryPredicate
from cellprofiler.modules.images import Images
from cellprofiler.modules.loadimages import \
    well_metadata_tokens, urlfilename, urlpathname
from cellprofiler.modules._help import FILTER_RULES_BUTTONS_HELP,PROTIP_RECOMEND_ICON
from cellprofiler.gui.help import content

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
-  In the image file itself.

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
""".format(**{
                "METADATA_DISPLAY_TABLE": _help.__image_resource('Metadata_ExampleDisplayTable.png')
            })

X_AUTOMATIC_EXTRACTION = "Extract from image file headers"
X_MANUAL_EXTRACTION = "Extract from file/folder names"
X_IMPORTED_EXTRACTION = "Import from file"
X_ALL_EXTRACTION_METHODS = [X_MANUAL_EXTRACTION,
                            X_IMPORTED_EXTRACTION,
                            X_AUTOMATIC_EXTRACTION]
XM_FILE_NAME = "File name"
XM_FOLDER_NAME = "Folder name"

DTC_TEXT = "Text"
DTC_CHOOSE = "Choose for each"
DTC_ALL = [DTC_TEXT, DTC_CHOOSE]

F_ALL_IMAGES = "All images"
F_FILTERED_IMAGES = "Images matching a rule"
COL_PATH = "Path / URL"
COL_SERIES = "Series"
COL_INDEX = "Frame"

'''Index of the extraction method count in the settings'''
IDX_EXTRACTION_METHOD_COUNT = 3
IDX_EXTRACTION_METHOD_COUNT_V1 = 1
IDX_EXTRACTION_METHOD_COUNT_V2 = 1
IDX_EXTRACTION_METHOD_COUNT_V3 = 1
'''Index of the first extraction method block in the settings'''
IDX_EXTRACTION_METHOD = 4
IDX_EXTRACTION_METHOD_V1 = 2
IDX_EXTRACTION_METHOD_V2 = 2
IDX_EXTRACTION_METHOD_V3 = 2
'''# of settings in an extraction method block'''
LEN_EXTRACTION_METHOD_V1 = 8
LEN_EXTRACTION_METHOD = 9


class Metadata(cpm.Module):
    variable_revision_number = 4
    module_name = "Metadata"
    category = "File Processing"

    CSV_JOIN_NAME = "CSV Metadata"
    IPD_JOIN_NAME = "Image Metadata"

    def create_settings(self):
        self.pipeline = None
        self.ipds = []
        module_explanation = [
            "The %s module optionally allows you to extract information" % self.module_name,
            "describing your images (i.e, metadata) which will be stored along",
            "with your measurements. This information can be contained in the",
            "file name and/or location, or in an external file."]
        self.set_notes([" ".join(module_explanation)])

        self.wants_metadata = cps.Binary(
            "Extract metadata?",
            False,
            doc="""\
Select "*{YES}*" if your file or path names or file headers contain
information (i.e., metadata) you would like to extract and store along
with your measurements. See the main module help for more details.
""".format(**{
                "YES": YES
            })
        )

        self.extraction_methods = []

        self.add_extraction_method(False)

        self.extraction_method_count = cps.HiddenCount(
            self.extraction_methods,
            "Extraction method count"
        )

        self.add_extraction_method_button = cps.DoSomething(
            "",
            "Add another extraction method",
            self.add_extraction_method
        )

        self.dtc_divider = cps.Divider()

        self.data_type_choice = cps.Choice(
            "Metadata data type",
            DTC_ALL,
            tooltips={
                DTC_TEXT: "Save all metadata as text",
                DTC_CHOOSE: "Choose the data type (text or numeric) for each metadata category"
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
""".format(**{
                "DTC_CHOOSE": DTC_CHOOSE,
                "DTC_TEXT": DTC_TEXT
            })
        )

        self.data_types = cps.DataTypes(
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
""".format(**{
                "DTC_CHOOSE": DTC_CHOOSE
            })
        )

        self.table = cps.Table(
            "",
            use_sash=True,
            corner_button=dict(
                fn_clicked=self.update_table,
                label="Update",
                tooltip="Update the metadata table"
            )
        )

    def add_extraction_method(self, can_remove=True):
        group = cps.SettingsGroup()

        self.extraction_methods.append(group)

        if can_remove:
            group.append("divider", cps.Divider())

        group.append(
            "extraction_method",
            cps.Choice(
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

   |image1|  *When would you want to
   use this option?* You have information curated in software that
   allows for export to a spreadsheet. This is commonly the case for
   laboratories that use data management systems that track samples and
   acquisition.
-  *{X_AUTOMATIC_EXTRACTION}*: This option retrieves information from
   the internal structure of the file format itself. Typically, image
   metadata is embedded in the image file as header information; this
   information includes the dimensions and color depth among other
   things. **If you select this method, you must press the “Update metadata” button
   (as opposed to the "Update" button beneath it) to extract the metadata.**
   Note that this extraction process can take a while for assays with lots
   of images since each one needs to read for extraction. Since the metadata
   is often image-format specific, this option will extract information that
   is common to most image types:

   -  *Series:* The series index of the image. This value is set to
      “None” if not applicable. Some image formats can store more than
      one stack in a single file; for those, the *Series* value for each
      stack in the file will be different
   -  *Frame:* The frame index of the image. This value is set to “None”
      if not applicable. For stack frames and movies, this is the frame
      number for an individual 2-D image slice.
   -  *ColorFormat:* Set to “Monochrome” for grayscale images, “RGB” for
      color.
   -  *SizeZ:* The number of image slices. Typically has a value > 1 for
      confocal stacks and the like.
   -  *SizeT:* The number of image frames. Typically has a value > 1 for
      movies.
   -  *SizeC:* The number of color channels. Typically has a value > 1
      for non-grayscale images and for confocal stacks containing
      channel images acquired using different filters and illumination
      sources.

   |image2| *When would you want to use this option?* You want to
   analyze images that are contained as file stacks, i.e., the images
   that are related to each other in some way, such as by time
   (temporal), space (spatial), or color (spectral).

Specifics on the metadata extraction options are described below. Any or
all of these options may be used at time; press the “Add another
extraction method” button to add more.

.. |image0| image:: {PROTIP_RECOMEND_ICON}
.. |image1| image:: {PROTIP_RECOMEND_ICON}
.. |image2| image:: {PROTIP_RECOMEND_ICON}
""".format(**{
                    "X_AUTOMATIC_EXTRACTION": X_AUTOMATIC_EXTRACTION,
                    "X_IMPORTED_EXTRACTION": X_IMPORTED_EXTRACTION,
                    "X_MANUAL_EXTRACTION": X_MANUAL_EXTRACTION,
                    "PROTIP_RECOMEND_ICON": PROTIP_RECOMEND_ICON
                })
            )
        )

        group.append(
            "source",
            cps.Choice(
                "Metadata source",
                [
                    XM_FILE_NAME,
                    XM_FOLDER_NAME
                ],
                doc="You can extract the metadata from the image's file name or from its folder name."
            )
        )

        group.append(
            "file_regexp",
            cps.RegexpText(
                "Regular expression to extract from file name",
                '^(?P<Plate>.*)_(?P<Well>[A-P][0-9]{2})_s(?P<Site>[0-9])_w(?P<ChannelNumber>[0-9])',
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
"""
            )
        )

        group.append(
            "folder_regexp",
            cps.RegexpText(
                "Regular expression to extract from folder name",
                '(?P<Date>[0-9]{4}_[0-9]{2}_[0-9]{2})$',
                get_example_fn=self.example_directory_fn,
                guess=cps.RegexpText.GUESS_FOLDER,
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
"""
            )
        )

        group.append(
            "filter_choice",
            cps.Choice(
                "Extract metadata from",
                [
                    F_ALL_IMAGES,
                    F_FILTERED_IMAGES
                ],
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
""".format(**{
                    "F_ALL_IMAGES": F_ALL_IMAGES,
                    "F_FILTERED_IMAGES": F_FILTERED_IMAGES
                })
            )
        )

        group.append(
            "filter",
            cps.Filter(
                "Select the filtering criteria",
                [
                    FilePredicate(),
                    DirectoryPredicate(),
                    ExtensionPredicate()
                ],
                'and (file does contain "")',
                doc="""\
Select "*{YES}*" to display and use rules to select files for metadata
extraction.

{FILTER_RULES_BUTTONS_HELP}
""".format(**{
                    "FILTER_RULES_BUTTONS_HELP": FILTER_RULES_BUTTONS_HELP,
                    "YES": YES
                })
            )
        )

        group.append(
            "csv_location",
            cps.PathnameOrURL(
                "Metadata file location",
                wildcard="Metadata files (*.csv)|*.csv|All files (*.*)|*.*",
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
"""
            )
        )

        group.append(
            "csv_joiner", cps.Joiner(
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
""".format(**{
                    "MODULE_ADD_BUTTON": content.MODULE_ADD_BUTTON
                })
            )
        )

        group.append(
            "wants_case_insensitive",
            cps.Binary(
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

.. |image0| image:: {PROTIP_RECOMEND_ICON}
""".format(**{
                    "NO": NO,
                    "YES": YES,
                    "PROTIP_RECOMEND_ICON": PROTIP_RECOMEND_ICON
                })
            )
        )

        group.append(
            "update_metadata",
            cps.DoSomething(
                "",
                "Update metadata",
                lambda: self.do_update_metadata(group),
                doc="Press this button to automatically extract metadata from your image files."
            )
        )

        group.imported_metadata_header_timestamp = 0
        group.imported_metadata_header_path = None
        group.imported_metadata_header_line = None
        group.can_remove = can_remove
        if can_remove:
            group.append("remover", cps.RemoveSettingButton(
                    '', 'Remove this extraction method',
                    self.extraction_methods, group))

    def get_group_header(self, group):
        '''Get the header line from the imported extraction group's csv file'''
        csv_path = group.csv_location.value
        if csv_path == group.imported_metadata_header_path:
            if group.csv_location.is_url():
                return group.imported_metadata_header_line
            if os.path.isfile(csv_path):
                timestamp = os.stat(csv_path).st_mtime
                if timestamp <= group.imported_metadata_header_timestamp:
                    return group.imported_metadata_header_line
        group.imported_metadata_header_timestamp = time.time()
        group.imported_metadata_header_path = csv_path
        try:
            if group.csv_location.is_url():
                fd = urllib.urlopen(csv_path)
            else:
                fd = open(csv_path, "rb")
            group.imported_metadata_header_line = fd.readline()
        except:
            return None
        return group.imported_metadata_header_line

    def build_imported_metadata_extractor(self, group, extractor,
                                          for_metadata_only):
        '''Build an extractor of imported metadata for this group

        group - a settings group to extract imported metadata

        extractor - the extractor as built up to the current point

        for_metadata_only - if true, only give the header to the
                 imported metadata extractor.
        '''
        key_pairs = []
        dt_numeric = (cpmeas.COLTYPE_FLOAT, cpmeas.COLTYPE_INTEGER)
        kp_cls = 'org/cellprofiler/imageset/MetadataKeyPair'
        kp_sig = '(Ljava/lang/String;Ljava/lang/String;)L%s;' % kp_cls
        for join_idx in group.csv_joiner.parse():
            csv_key = join_idx[self.CSV_JOIN_NAME]
            ipd_key = join_idx[self.IPD_JOIN_NAME]
            if self.get_data_type(csv_key) in dt_numeric and \
                            self.get_data_type(ipd_key) in dt_numeric:
                kp_method = "makeNumericKeyPair"
            elif group.wants_case_insensitive:
                kp_method = "makeCaseInsensitiveKeyPair"
            else:
                kp_method = "makeCaseSensitiveKeyPair"
            key_pair = J.static_call(
                    kp_cls, kp_method, kp_sig, csv_key, ipd_key)
            key_pairs.append(key_pair)
        key_pairs = J.get_nice_arg(
                key_pairs,
                "[L%s;" % kp_cls)

        if for_metadata_only:
            header = self.get_group_header(group)
            if header is None:
                return None
            rdr = J.make_instance(
                    "java/io/StringReader",
                    "(Ljava/lang/String;)V",
                    header)
        elif group.csv_location.is_url():
            jurl = J.make_instance(
                    "java/net/URL",
                    "(Ljava/lang/String;)V",
                    group.csv_location.value)
            stream = J.call(
                    jurl, "openStream",
                    "()Ljava/io/InputStream;")
            rdr = J.make_instance(
                    "java/io/InputStreamReader",
                    "(Ljava/io/InputStream;)V",
                    stream)
        else:
            stream = J.make_instance(
                    "java/io/FileInputStream",
                    "(Ljava/lang/String;)V",
                    group.csv_location.value)
            rdr = J.make_instance(
                    "java/io/InputStreamReader",
                    "(Ljava/io/InputStream;)V",
                    stream)
        return J.make_instance(
                "org/cellprofiler/imageset/ImportedMetadataExtractor",
                "(Ljava/io/Reader;[Lorg/cellprofiler/imageset/MetadataKeyPair;)V",
                rdr, key_pairs)

    def refresh_group_joiner(self, group):
        '''Refresh the metadata entries for a group's joiner'''
        if group.extraction_method != X_IMPORTED_EXTRACTION:
            return
        #
        # Build an extractor to this point, just for getting the metadata
        # keys.
        #
        extractor = self.build_extractor(group, True)
        #
        # Get the key set.
        #
        possible_keys = J.get_collection_wrapper(
                J.call(extractor, "getMetadataKeys", "()Ljava/util/List;"),
                J.to_string)
        joiner = group.csv_joiner
        assert isinstance(joiner, cps.Joiner)
        joiner.entities[self.IPD_JOIN_NAME] = list(possible_keys)
        header = self.get_group_header(group)
        if header is None:
            header_keys = ["None"]
        else:
            header_keys = J.get_collection_wrapper(
                    J.static_call(
                            "org/cellprofiler/imageset/ImportedMetadataExtractor",
                            "readHeader",
                            "(Ljava/lang/String;)Ljava/util/List;",
                            header), J.to_string)
        joiner.entities[self.CSV_JOIN_NAME] = list(header_keys)

    def settings(self):
        result = [self.wants_metadata, self.data_type_choice, self.data_types,
                  self.extraction_method_count]
        for group in self.extraction_methods:
            result += [
                group.extraction_method, group.source, group.file_regexp,
                group.folder_regexp, group.filter_choice, group.filter,
                group.csv_location, group.csv_joiner,
                group.wants_case_insensitive]
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
                    result += [group.csv_location, group.filter_choice]
                    if group.filter_choice == F_FILTERED_IMAGES:
                        result += [group.filter]
                    result += [group.csv_joiner, group.wants_case_insensitive]
                elif group.extraction_method == X_AUTOMATIC_EXTRACTION:
                    result += [group.filter_choice]
                    if group.filter_choice == F_FILTERED_IMAGES:
                        result += [group.filter]
                    result += [group.update_metadata]
                if group.can_remove:
                    result += [group.remover]
            result += [self.add_extraction_method_button]
            try:
                has_keys = len(self.get_dt_metadata_keys()) > 0
            except:
                has_keys = False
            if has_keys:
                result += [self.dtc_divider, self.data_type_choice]
                if self.data_type_choice == DTC_CHOOSE:
                    result.append(self.data_types)
            result += [self.table]
        return result

    def example_file_fn(self):
        '''Get an example file name for the regexp editor'''
        if self.pipeline is not None:
            if self.pipeline.has_cached_filtered_file_list():
                urls = self.pipeline.get_filtered_file_list(self.workspace)
                if len(urls) == 0:
                    urls = self.pipeline.file_list
            else:
                urls = self.pipeline.file_list
            if len(urls) > 0:
                return urlfilename(urls[0])
        return "PLATE_A01_s1_w11C78E18A-356E-48EC-B204-3F4379DC43AB.tif"

    def example_directory_fn(self):
        '''Get an example directory name for the regexp editor'''
        if self.pipeline is not None:
            if self.pipeline.has_cached_filtered_file_list():
                urls = self.pipeline.get_filtered_file_list(self.workspace)
                if len(urls) == 0:
                    urls = self.pipeline.file_list
            else:
                urls = self.pipeline.file_list
            if len(urls) > 0:
                return urlpathname(urls[0])
        return "/images/2012_01_12"

    def change_causes_prepare_run(self, setting):
        '''Return True if changing the setting passed changes the image sets

        setting - the setting that was changed
        '''
        return setting in self.settings()

    @classmethod
    def is_input_module(self):
        return True

    def prepare_run(self, workspace):
        '''Initialize the pipeline's metadata'''
        if workspace.pipeline.in_batch_mode():
            return True

        pipeline = workspace.pipeline
        assert isinstance(pipeline, cpp.Pipeline)
        filtered_file_list = pipeline.get_filtered_file_list(workspace)
        extractor = self.build_extractor()
        env = J.get_env()
        scls = env.find_class("java/lang/String")
        url_array = env.make_object_array(len(filtered_file_list), scls)
        metadata_array = env.make_object_array(len(filtered_file_list), scls)
        for i, url in enumerate(filtered_file_list):
            if isinstance(url, unicode):
                ourl = env.new_string(url)
            else:
                ourl = env.new_string_utf(url)
            env.set_object_array_element(url_array, i, ourl)
            xmlmetadata = workspace.file_list.get_metadata(url)
            if xmlmetadata is not None:
                xmlmetadata = env.new_string(xmlmetadata)
                env.set_object_array_element(metadata_array, i, xmlmetadata)
        key_set = J.make_instance("java/util/HashSet", "()V")
        jipds = J.call(
                extractor, "extract",
                "([Ljava/lang/String;[Ljava/lang/String;Ljava/util/Set;)"
                "[Lorg/cellprofiler/imageset/ImagePlaneDetails;",
                url_array, metadata_array, key_set)
        ipds = [cpp.ImagePlaneDetails(jipd)
                for jipd in env.get_object_array_elements(jipds)]
        keys = sorted(J.iterate_collection(key_set, J.to_string))
        pipeline.set_image_plane_details(ipds, keys, self)
        return True

    def build_extractor(self, end_group=None, for_metadata_only=False):
        '''Build a Java metadata extractor using the module settings

        end_group - stop building the extractor when you reach this group.
                    default is build all.
        for_metadata_only - only build an extractor to capture the header info
        '''
        #
        # Build a metadata extractor
        #
        extractor = J.make_instance(
                "org/cellprofiler/imageset/ImagePlaneMetadataExtractor",
                "()V")
        J.call(extractor, "addImagePlaneExtractor",
               "(Lorg/cellprofiler/imageset/MetadataExtractor;)V",
               J.make_instance(
                       "org/cellprofiler/imageset/URLSeriesIndexMetadataExtractor",
                       "()V"))
        if any([group.extraction_method == X_AUTOMATIC_EXTRACTION
                for group in self.extraction_methods]):
            for method_name, class_name in (
                    ("addImageFileExtractor", "OMEFileMetadataExtractor"),
                    ("addImageSeriesExtractor", "OMESeriesMetadataExtractor"),
                    ("addImagePlaneExtractor", "OMEPlaneMetadataExtractor")):
                class_name = "org/cellprofiler/imageset/" + class_name
                J.call(extractor, method_name,
                       "(Lorg/cellprofiler/imageset/MetadataExtractor;)V",
                       J.make_instance(class_name, "()V"))

        has_well_extractor = False
        for group in self.extraction_methods:
            if group == end_group:
                break
            if group.filter_choice == F_FILTERED_IMAGES:
                fltr = J.make_instance(
                        "org/cellprofiler/imageset/filter/Filter",
                        "(Ljava/lang/String;Ljava/lang/Class;)V",
                        group.filter.value_text,
                        J.class_for_name("org.cellprofiler.imageset.ImageFile"))
            else:
                fltr = None
            if group.extraction_method == X_MANUAL_EXTRACTION:
                if group.source == XM_FILE_NAME:
                    method = "addFileNameRegexp"
                    pattern = group.file_regexp.value
                elif group.source == XM_FOLDER_NAME:
                    method = "addPathNameRegexp"
                    pattern = group.folder_regexp.value
                # check for bad pattern before creating an extractor
                try:
                    re.search(pattern, "")
                except re.error:
                    continue
                J.call(extractor,
                       method,
                       "(Ljava/lang/String;Lorg/cellprofiler/imageset/filter/Filter;)V",
                       pattern, fltr)
            elif group.extraction_method == X_IMPORTED_EXTRACTION:
                imported_extractor = self.build_imported_metadata_extractor(
                        group, extractor, for_metadata_only)
                if imported_extractor is not None:
                    J.call(extractor,
                           "addImagePlaneDetailsExtractor",
                           "(Lorg/cellprofiler/imageset/MetadataExtractor;"
                           "Lorg/cellprofiler/imageset/filter/Filter;)V",
                           imported_extractor, fltr)
            #
            # Finally, we add the WellMetadataExtractor which has the inglorious
            # job of making a well name from row and column, if present,
            # but only if our existing metadata extractors have metadata that
            # might require it.
            #
            if not has_well_extractor:
                metadata_keys = J.call(
                        extractor, "getMetadataKeys", "()Ljava/util/List;")
                if J.static_call(
                        "org/cellprofiler/imageset/WellMetadataExtractor",
                        "maybeYouNeedThis", "(Ljava/util/List;)Z",
                        metadata_keys):
                    J.call(
                            extractor,
                            "addImagePlaneDetailsExtractor",
                            "(Lorg/cellprofiler/imageset/MetadataExtractor;)V",
                            J.make_instance(
                                    "org/cellprofiler/imageset/WellMetadataExtractor",
                                    "()V"))
                    has_well_extractor = True

        return extractor

    def run(self, workspace):
        pass

    def do_update_metadata(self, group):
        filelist = self.workspace.file_list
        urls = set(self.pipeline.get_filtered_file_list(self.workspace))
        if len(urls) == 0:
            return

        def msg(url):
            return "Processing %s" % url

        import wx
        from bioformats.formatreader import get_omexml_metadata
        from bioformats.omexml import OMEXML
        from cellprofiler.modules.loadimages import url2pathname
        with wx.ProgressDialog("Updating metadata",
                               msg(list(urls)[0]),
                               len(urls),
                               style=wx.PD_CAN_ABORT
                                       | wx.PD_APP_MODAL
                                       | wx.PD_ELAPSED_TIME
                                       | wx.PD_REMAINING_TIME) as dlg:
            for i, url in enumerate(urls):
                if i > 0:
                    keep_going, _ = dlg.Update(i, msg(url))
                    if not keep_going:
                        break
                if group.filter_choice == F_FILTERED_IMAGES:
                    match = group.filter.evaluate(
                            (cps.FileCollectionDisplay.NODE_IMAGE_PLANE,
                             Images.url_to_modpath(url), self))
                    if not match:
                        continue
                metadata = filelist.get_metadata(url)
                if metadata is None:
                    metadata = get_omexml_metadata(url=url)
                    filelist.add_metadata(url, metadata)

    def on_activated(self, workspace):
        self.workspace = workspace
        self.pipeline = workspace.pipeline
        for group in self.extraction_methods:
            if group.extraction_method == X_IMPORTED_EXTRACTION:
                self.refresh_group_joiner(group)
        self.table.clear_rows()
        self.table.clear_columns()
        if workspace.pipeline.has_cached_image_plane_details():
            self.update_table()

    def on_setting_changed(self, setting, pipeline):
        '''Update the imported extraction joiners on setting changes'''
        if not self.wants_metadata:
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
                idx = max(*map(visible_settings.index,
                               [group.csv_joiner, group.csv_location,
                                group.wants_case_insensitive]))
                if idx < setting_idx:
                    continue
                self.refresh_group_joiner(group)

    def update_table(self):
        columns = set(self.get_metadata_keys())
        columns.discard(COL_SERIES)
        columns.discard(COL_INDEX)
        columns = [COL_PATH, COL_SERIES, COL_INDEX] + \
                  sorted(list(columns))
        self.table.clear_columns()
        self.table.clear_rows()
        data = []
        md_keys = J.make_list(columns[3:])
        #
        # Use the low-level Javabridge interface to make things a bit faster
        #
        env = J.get_env()
        clsIPD = env.find_class("org/cellprofiler/imageset/ImagePlaneDetails")
        methodID = env.get_method_id(
                clsIPD, "getIPDFields", "(Ljava/util/List;)[Ljava/lang/String;")
        has_data = [False] * len(columns)
        for ipd in self.pipeline.get_image_plane_details(self.workspace):
            fields = env.call_method(ipd.jipd, methodID, md_keys.o)
            row = [None] * len(columns)
            for i, f in enumerate(env.get_object_array_elements(fields)):
                if f is not None:
                    has_data[i] = True
                    row[i] = J.to_string(f)
            data.append(row)
        columns = [c for c, h in zip(columns, has_data) if h]
        for i in range(len(data)):
            data[i] = [f for f, h in zip(data[i], has_data) if h]

        for i, column in enumerate(columns):
            self.table.insert_column(i, column)

        self.table.add_rows(columns, data)
        cellprofiler.preferences.report_progress(
                    "MetadataCount", None,
                    "Found %d rows" % len(data))

    def on_deactivated(self):
        self.pipeline = None

    def prepare_to_create_batch(self, workspace, fn_alter_path):
        '''Alter internal paths for batch creation'''
        for group in self.extraction_methods:
            if group.extraction_method == X_IMPORTED_EXTRACTION:
                group.csv_location.alter_for_create_batch(fn_alter_path)

    def prepare_settings(self, setting_values):
        '''Prepare the module to receive the settings'''
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
        '''Validate the module settings

        pipeline - current pipeline

        Metadata throws an exception if any of the metadata tags collide with
        tags that can be automatically extracted.
        '''
        for group in self.extraction_methods:
            if group.extraction_method == X_MANUAL_EXTRACTION:
                re_setting = (group.file_regexp if group.source == XM_FILE_NAME
                              else group.folder_regexp)
                for token in cpmeas.find_metadata_tokens(re_setting.value):
                    if token in cpmeas.RESERVED_METADATA_TAGS:
                        raise cps.ValidationError(
                                'The metadata tag, "%s", is reserved for use by CellProfiler. Please use some other tag name.' %
                                token, re_setting)

    def get_metadata_keys(self):
        '''Return a collection of metadata keys to be associated with files'''
        if not self.wants_metadata:
            return []
        extractor = self.build_extractor(for_metadata_only=True)
        keys = J.get_collection_wrapper(
                J.call(extractor,
                       "getMetadataKeys",
                       "()Ljava/util/List;"), J.to_string)
        return keys

    def get_dt_metadata_keys(self):
        '''Get the metadata keys which can have flexible datatyping

        '''
        return filter((lambda k: k not in self.NUMERIC_DATA_TYPES),
                      self.get_metadata_keys())

    NUMERIC_DATA_TYPES = (
        cpp.ImagePlaneDetails.MD_T, cpp.ImagePlaneDetails.MD_Z,
        cpp.ImagePlaneDetails.MD_SIZE_C, cpp.ImagePlaneDetails.MD_SIZE_T,
        cpp.ImagePlaneDetails.MD_SIZE_Z, cpp.ImagePlaneDetails.MD_SIZE_X,
        cpp.ImagePlaneDetails.MD_SIZE_Y, cpmeas.C_SERIES, cpmeas.C_FRAME)

    def get_data_type(self, key):
        '''Get the data type for a particular metadata key'''
        if isinstance(key, basestring):
            return self.get_data_type([key]).get(key, cpmeas.COLTYPE_VARCHAR)
        result = {}
        if self.data_type_choice == DTC_CHOOSE:
            data_types = cps.DataTypes.decode_data_types(
                    self.data_types.value_text)
        for k in key:
            if k in self.NUMERIC_DATA_TYPES:
                result[k] = cpmeas.COLTYPE_INTEGER
            elif self.data_type_choice == DTC_CHOOSE:
                dt = data_types.get(k, cps.DataTypes.DT_TEXT)
                if dt == cps.DataTypes.DT_TEXT:
                    result[k] = cpmeas.COLTYPE_VARCHAR
                elif dt == cps.DataTypes.DT_INTEGER:
                    result[k] = cpmeas.COLTYPE_INTEGER
                elif dt == cps.DataTypes.DT_FLOAT:
                    result[k] = cpmeas.COLTYPE_FLOAT
            else:
                result[k] = cpmeas.COLTYPE_VARCHAR

        return result

    def wants_case_insensitive_matching(self, key):
        '''Return True if the key should be matched using case-insensitive matching

        key - key to check.

        Currently, there is a case-insensitive matching flag in the
        imported metadata matcher. Perhaps this should be migrated into
        the data types control, but for now, we look for the key to be
        present in the joiner for any imported metadata matcher.
        '''
        if not self.wants_metadata:
            return False
        for group in self.extraction_methods:
            if group.extraction_method == X_IMPORTED_EXTRACTION and \
                    group.wants_case_insensitive:
                joins = group.csv_joiner.parse()
                for join in joins:
                    if key in join.values():
                        return True
        return False

    def get_measurement_columns(self, pipeline):
        '''Get the metadata measurements collected by this module'''
        key_types = pipeline.get_available_metadata_keys()
        result = []
        for key, coltype in key_types.iteritems():
            if self.data_type_choice == DTC_CHOOSE:
                data_type = self.get_data_type(key)
                if data_type == cps.DataTypes.DT_NONE:
                    continue
                elif data_type == cps.DataTypes.DT_INTEGER:
                    data_type = cpmeas.COLTYPE_INTEGER
                elif data_type == cps.DataTypes.DT_FLOAT:
                    data_type = cpmeas.COLTYPE_FLOAT
                else:
                    data_type = cpmeas.COLTYPE_VARCHAR_FILE_NAME
            else:
                data_type = cpmeas.COLTYPE_VARCHAR_FILE_NAME
            result.append((
                cpmeas.IMAGE, '_'.join((cpmeas.C_METADATA, key)), data_type))
        return result

    def get_categories(self, pipeline, object_name):
        '''Return the measurement categories for a particular object'''
        if object_name == cpmeas.IMAGE and len(self.get_metadata_keys()) > 0:
            return [cpmeas.C_METADATA]
        return []

    def get_measurements(self, pipeline, object_name, category):
        if object_name == cpmeas.IMAGE and category == cpmeas.C_METADATA:
            keys = self.get_metadata_keys()
            return keys
        return []

    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        if variable_revision_number == 1:
            n_groups = int(setting_values[IDX_EXTRACTION_METHOD_COUNT_V1])
            new_setting_values = setting_values[:IDX_EXTRACTION_METHOD_V1]
            for i in range(n_groups):
                new_setting_values += setting_values[
                                      (IDX_EXTRACTION_METHOD_V1 + LEN_EXTRACTION_METHOD_V1 * i):
                                      (IDX_EXTRACTION_METHOD_V1 + LEN_EXTRACTION_METHOD_V1 * (i + 1))]
                new_setting_values.append(cps.NO)
            setting_values = new_setting_values
            variable_revision_number = 2

        if variable_revision_number == 2:
            # Changed naming of extraction methods, metadata sources and filtering choices
            n_groups = int(setting_values[IDX_EXTRACTION_METHOD_COUNT_V2])
            new_setting_values = setting_values[:IDX_EXTRACTION_METHOD_V2]
            for i in range(n_groups):
                group = setting_values[
                        (IDX_EXTRACTION_METHOD_V2 + LEN_EXTRACTION_METHOD * i):
                        (IDX_EXTRACTION_METHOD_V2 + LEN_EXTRACTION_METHOD * (i + 1))]
                group[0] = X_AUTOMATIC_EXTRACTION if group[0] == "Automatic" \
                    else (X_MANUAL_EXTRACTION if group[0] == "Manual" \
                              else X_IMPORTED_EXTRACTION)
                group[1] = XM_FILE_NAME if group[1] == "From file name" \
                    else XM_FOLDER_NAME
                group[4] = F_FILTERED_IMAGES if group[4] == "Images selected using a filter" \
                    else F_ALL_IMAGES
                new_setting_values += group
            setting_values = new_setting_values
            variable_revision_number = 3

        if variable_revision_number == 3:
            # Added data types
            setting_values = setting_values[:IDX_EXTRACTION_METHOD_COUNT_V3] + \
                             [DTC_TEXT, "{}"] + setting_values[IDX_EXTRACTION_METHOD_COUNT_V3:]
            variable_revision_number = 4

        return setting_values, variable_revision_number, from_matlab

    def volumetric(self):
        return True
