__doc__ = """
The <b>Metadata</b> module associates information about the images (i.e., metadata)
with the images themselves. 
<hr>
The <b>Metadata</b> module allows you incorporate the metadata that is particular to the image 
format, assigned as part of the file name or location (by a vendor microscope, for example),
contained in a text file filled out by the user, or any of the above.

<h4>What is "metadata"?</h4>
The general term <i>metadata</i> refers to "data about data." For many assays, metadata
appears in the context of tagging images with various attributes, which 
can include (but is not limited to) items such as the following:
<ul>
<li>The height and width of an image, in pixels.</li>
<li>Is the image RGB, indexed or separate channels?</li>
<li>The number of timepoints or channels contained in the image file.</li>
<li>The experimental treatment applied to the well that the image was acquired from.</li>
<li>The row and column of the microtiter plate that the image was acquired from.</li>
<li>Etc.</li>
</ul>
It can be helpful to inform CellProfiler about certain types of metadata either contained
in the image files and/or specified elsewhere from another source, in order to define
a specific relationship between the images and the associated metadata. For instance:
<ul>
<li>You want images with a common identifier to be matched together so they are
processed together during the pipeline run;</li>
<li>You want certain information attached to the output measurements and filenames 
for annotation or sample-tracking purposes.</li>
</ul>

<p>The underlying assumption in matching metadata values to image sets is that there is an
exact pairing (i.e., a one-to-one match) for a given metadata tag combination. A common example is that for
a two-channel microtiter plate assay, each plate, well and site metadata from one channel
gets matched uniquely to the plate, well and site metadata from the other channel.</p>

<h4>What are the inputs?</h4>
The <b>Metadata</b> module receives the file list produced by the <b>Images</b> module. It then
tags (or attaches) information that can be obtained from several sources:
<ul>
<li>The metadata may be part of the image file name or location (e.g., as assigned by a vendor 
microscope). In this case, the user provides the text search pattern to obtain this information. </li>
<li>Alternately (or concurrently), the metadata may be contained in a text file created and 
filled out by the user or laboratory. If this is the case, the user will point the module to the 
location of this file.</li>
</ul>

<h4>What do the settings mean?</h4>
See below for help on the individual settings. In general, the settings serve in various forms of 
metadata extraction. You can extract metadata from all images from <b>Images</b> modules or a subset
of them by using rules to filter the list.

<h4>What do I get as output?</h4>
The <b>Metadata</b> module will take the metadata from the source(s) provided and attach them as additional
data for each image. If the metadata originates from an external source such as a CSV, there are some caveats
in the cases when metadata is either missing or duplicated for the referenced images; see the <b>NamesAndTypes</b>
for more details.

<p>If the metadata establishes how channels are related to one another, you can use them in <b>NamesAndTypes</b> 
to aid in creating an image set. You can also use <i>metadata tags</i> in your pipeline to reference the metadata 
values in later modules. Several modules are capable of using metadata tags for various purposes. Examples include:
<ul>
<li>You would like to create and apply an illumination correction function to all images from a particular
plate. You can use metadata tags to save each illumination correction function with a plate-specific
name in <b>SaveImages</b>, and then use <b>Images</b> to get files
with the name associated with your image's plate to be applied to your original images.</li>
<li>You have a set of experiments for which you would like to produce and save results
individually for each experiment but using only one analysis run. You can use metadata tags
in <b>ExportToSpreadsheet</b> or <b>ExportToDatabase</b> to save a spreadsheet for each experiment in 
a folder named according to the experiment.</li>
</ul>
In each case, the pre-defined metadata tag is used to name a file or folder. See the module setting help for additional
information on how to use them in the context of the specific module.</p>

<h4>Available measurements</h4>
<ul> 
<li><i>Metadata:</i> Each metadata identifier is prefixed by <i>Image_Metadata_</i> in the per-image table.</li>
</ul>
"""
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2013 Broad Institute
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org

import numpy as np
import logging
logger = logging.getLogger(__name__)
import csv
import re
import os

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.pipeline as cpp
import cellprofiler.settings as cps
import cellprofiler.utilities.jutil as J
from cellprofiler.modules.images import FilePredicate
from cellprofiler.modules.images import ExtensionPredicate
from cellprofiler.modules.images import ImagePredicate
from cellprofiler.modules.images import DirectoryPredicate
from cellprofiler.modules.images import Images, evaluate_url
from cellprofiler.modules.loadimages import needs_well_metadata
from cellprofiler.modules.loadimages import well_metadata_tokens
from cellprofiler.gui.help import FILTER_RULES_BUTTONS_HELP

X_AUTOMATIC_EXTRACTION = "Extract from image file headers"
X_MANUAL_EXTRACTION = "Extract from file/folder names"
X_IMPORTED_EXTRACTION = "Import from file"
X_ALL_EXTRACTION_METHODS = [X_AUTOMATIC_EXTRACTION, 
                            X_MANUAL_EXTRACTION,
                            X_IMPORTED_EXTRACTION]
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

class Metadata(cpm.CPModule):
    variable_revision_number = 4
    module_name = "Metadata"
    category = "File Processing"
    
    CSV_JOIN_NAME = "CSV Metadata"
    IPD_JOIN_NAME = "Image Metadata"

    def create_settings(self):
        self.pipeline = None
        self.ipds = []
        self.imported_metadata = []
        module_explanation = [
            "The %s module optionally allows you to extract information" %self.module_name,
            "describing your images (i.e, metadata) which will be stored along",
            "with your measurements. This information can be contained in the",
            "file name and/or location, or in an external file."]
        self.set_notes([" ".join(module_explanation)])
        
        self.wants_metadata = cps.Binary(
            "Extract metadata?", False,doc = """
            If your file or path names or file headers contain information 
            (i.e., metadata) you would like to extract and store along with your 
            measurements, then check this box. See the main module
            help for more details.""")
        
        self.extraction_methods = []
        self.add_extraction_method(False)
        
        self.extraction_method_count = cps.HiddenCount(
            self.extraction_methods, "Extraction method count")
        
        self.add_extraction_method_button = cps.DoSomething(
            "",
            "Add another extraction method", self.add_extraction_method)
        
        self.dtc_divider = cps.Divider()
        self.data_type_choice = cps.Choice(
            "Metadata data type", DTC_ALL,
            tooltips=dict(DTC_TEXT="Save all metadata as text",
                          DTC_CHOOSE="Choose the data type (text or numeric) for each metadata category"),
            doc="""
            Metadata can be saved as either a text or numeric measurement:
            <ul>
            <li><i>%(DTC_TEXT)s:</i> Save all metadata item as text.</li>
            <li><i>%(DTC_CHOOSE)s:</i> Choose the data type separately for each 
            metadata entry. An example of when this approach would be necessary 
            would be if a whole filename is captured as metadata but the file name is
            numeric, e.g., "0001101.tif". In this situation, if the file name needs to be used for an
            arithmetic calculation or index, the name would need to be converted to a number.
            On the other hand, if the file name has leading zeros which would 
            be removed if converted to a number, capturing the metdata values as text would be more
            appropriate.</li>
            </ul>
            """ % globals())
        
        self.data_types = cps.DataTypes(
            "Metadata types",
            name_fn = self.get_dt_metadata_keys, doc = """
            <i>(Used only when %(DTC_CHOOSE)s is selected for the metadata data type)</i><br>
            This setting determines the data type of each metadata field
            when stored as a measurement. 
            <ul>
            <li><i>Text:</i> Save the metadata as text.</li>
            <li><i>Integer:</i> Save the metadata as an integer.</li>
            <li><i>Float:</i> Save the metadata as a decimal number.</li>
            <li><i>None:</i> Do not save the metadata as a measurement.</li>
            </ul>"""%globals())
        
        self.table = cps.Table(
            "",use_sash=True, 
            corner_button = dict(fn_clicked = self.update_table,
                                 label = "Update",
                                 tooltip = "Update the metadata table"))
        
    def add_extraction_method(self, can_remove = True):
        group = cps.SettingsGroup()
        self.extraction_methods.append(group)
        if can_remove:
            group.append("divider", cps.Divider())
            
        group.append("extraction_method", cps.Choice(
            "Metadata extraction method", X_ALL_EXTRACTION_METHODS, doc="""
            <p>Metadata can be stored in either or both of two ways:
            <ul>
            <li><i>Internally:</i> This method is often through the file naming, directory structuring, 
            or the file header information.</li>
            <li><i>Externally:</i> This is through an external index, such as spreadsheet or
            database of some kind.</li>
            </ul>
            The <b>Metadata</b> module can extract internal or external metadata from the images 
            in any of three ways:
            <ul>
            <li><i>%(X_AUTOMATIC_EXTRACTION)s</i>: From the internal structure of the file format itself. 
            Typically, image information is embedded in the actual image file as header information;
            this information includes the dimensions and color depth among other things.
            If you select this method, press the "Update metadata" button
            below to extract the metadata. Since the metadata is often image-format specific, the
            module will extract information that is common to all images files:
            <ul>
            <li><i>Series:</i> The series index of the image. <i>None</i> if not applicable.</li>
            <li><i>Frame:</i> The frame index of the image. <i>None</i> if not applicable.</li>
            <li><i>ColorFormat:</i> <i>Monochrome</i> for grayscale images, <i>RGB</i> for color.</li>
            <li><i>SizeZ:</i> The number of image slices. Typically &gt; 1 for confocal stacks and the like.</li>
            <li><i>SizeT:</i> The number of image frames. Typically &gt; 1 for movies.</li>
            <li><i>SizeC:</i> The number of color channels. Typically &gt; 1 for non-grayscale images.</li>
            </ul>
            This extraction process can take a while for assays with lots of images
            since each one needs to read for extraction.</li>
            <li><i>%(X_MANUAL_EXTRACTION)s</i>: Specified based on the file nomenclature and/or location. This
            takes advantage of the fact that acquistion software often automatically assigns a regular 
            nomenclature to the files. Alternately, the researcher acquiring the images may also have
            a nomenclature in mind in order for bookkeeping purposes.</li>
            <li><i>%(X_IMPORTED_EXTRACTION)s</i>: From a comma-delimited list (csv) of information;
            provided as one type of metadata per column, and one row of metadata per image. This is a
            convenient way for you to add data from your own sources to the output generated by
            CellProfiler.</li>
            </ul>
            Additional extraction methods can be added by clicking the "Add" button below.</p>
            
            <p>For more details on how metadata is used downstream from this module, see the help for
            the <b>NamesAndTypes</b> or <b>Groups</b> modules.</p>"""%globals()))
        
        group.append("source", cps.Choice(
            "Metadata source", [XM_FILE_NAME, XM_FOLDER_NAME],doc = """
            You can extract the metadata from the image's file
            name or from its folder name."""))
        
        group.append("file_regexp", cps.RegexpText(
            "Regular expression", 
            '^(?P<Plate>.*)_(?P<Well>[A-P][0-9]{2})_s(?P<Site>[0-9])_w(?P<ChannelNumber>[0-9])',
            get_example_fn = self.example_file_fn,
            doc = """
            <a name='regular_expression'><i>(Used only if you want to extract 
            metadata from the file name)</i><br>
            The regular expression to extract the metadata from the file name 
            is entered here. Note that this field is available whether you have 
            selected <i>Text-Regular expressions</i> to load the files or not.
            Please see the general module help for more information on 
            construction of a regular expression.</a>
            <p>Clicking the magnifying glass icon to the right will bring up a
            tool for checking the accuracy of your regular expression. The 
            regular expression syntax can be used to name different parts of 
            your expression. The syntax <i>(?P&lt;fieldname&gt;expr)</i> will 
            extract whatever matches <i>expr</i> and assign it to the 
            measurement,<i>fieldname</i> for the image.
            <p>For instance, a researcher uses plate names composed of a string 
            of letters and numbers, followed by an underscore, then the well, 
            followed by another underscore, followed by an "s" and a digit
            representing the site taken within the well (e.g., <i>TE12345_A05_s1.tif</i>).
            The following regular expression will capture the plate, well, and 
            site in the fields "Plate", "Well", and "Site":<br><br>
            <table border = "1">
            <tr><td colspan = "2">^(?P&lt;Plate&gt;.*)_(?P&lt;Well&gt;[A-P][0-9]{1,2})_s(?P&lt;Site&gt;[0-9])</td></tr>
            <tr><td>^</td><td>Start only at beginning of the file name</td></tr>
            <tr><td>(?P&lt;Plate&gt;</td><td>Name the captured field <i>Plate</i></td></tr>
            <tr><td>.*</td><td>Capture as many characters as follow</td></tr>
            <tr><td>_</td><td>Discard the underbar separating plate from well</td></tr>
            <tr><td>(?P&lt;Well&gt;</td><td>Name the captured field <i>Well</i></td></tr>
            <tr><td>[A-P]</td><td>Capture exactly one letter between A and P</td></tr>
            <tr><td>[0-9]{1,2}</td><td>Capture one or two digits that follow</td></tr>
            <tr><td>_s</td><td>Discard the underbar followed by <i>s</i> separating well from site</td></tr>
            <tr><td>(?P&lt;Site&gt;</td><td>Name the captured field <i>Site</i></td></tr>
            <tr><td>[0-9]</td><td>Capture one digit following</td></tr>
            </table>
            
            <p>The regular expression can be typed in the upper text box, with 
            a sample file name given in the lower text box. Provided the syntax 
            is correct, the corresponding fields will be highlighted in the same
            color in the two boxes. Press <i>Submit</i> to enter the typed 
            regular expression.</p>
            
            <p>You can create metadata tags for any portion of the filename or path, but if you are
            specifying metadata for multiple images in a single <b>LoadImages</b> module, an image cycle can 
            only have one set of values for each metadata tag. This means that you can only 
            specify the metadata tags which have the same value across all images listed in the module. For example,
            in the example above, you might load two wavelengths of data, one named <i>TE12345_A05_s1_w1.tif</i>
            and the other <i>TE12345_A05_s1_w2.tif</i>, where the number following the <i>w</i> is the wavelength. 
            In this case, a "Wavelength" tag <i>should not</i> be included in the regular expression
            because while the "Plate", "Well" and "Site" metadata is identical for both images, the wavelength metadata is not.</p>
            
            <p>Note that if you use the special fieldnames <i>&lt;WellColumn&gt;</i> and 
            <i>&lt;WellRow&gt;</i> together, LoadImages will automatically create a <i>&lt;Well&gt;</i>
            metadata field by joining the two fieldname values together. For example, 
            if <i>&lt;WellRow&gt;</i> is "A" and <i>&lt;WellColumn&gt;</i> is "01", a field 
            <i>&lt;Well&gt;</i> will be "A01". This is useful if your well row and column names are
            separated from each other in the filename, but you want to retain the standard 
            well nomenclature.</p>"""))
   
        group.append("folder_regexp", cps.RegexpText(
            "Regular expression",
            '(?P<Date>[0-9]{4}_[0-9]{2}_[0-9]{2})$',
            get_example_fn = self.example_directory_fn,
            doc="""
            <i>(Used only if you want to extract metadata from the path)</i><br>
            Enter the regular expression for extracting the metadata from the 
            path. Note that this field is available whether you have selected 
            <i>Text-Regular expressions</i> to load the files or not.
            
            <p>Clicking the magnifying glass icon to the right will bring up a
            tool that will allow you to check the accuracy of your regular 
            expression. The regular expression syntax can be used to 
            name different parts of your expression. The syntax 
            <i>(?&lt;fieldname&gt;expr)</i> will extract whatever matches 
            <i>expr</i> and assign it to the image's <i>fieldname</i> measurement.
                        
            <p>For instance, a researcher uses folder names with the date and 
            subfolders containing the images with the run ID 
            (e.g., <i>./2009_10_02/1234/</i>) The following regular expression 
            will capture the plate, well, and site in the fields 
            <i>Date</i> and <i>Run</i>:<br>
            <table border = "1">
            <tr><td colspan = "2">.*[\\\/](?P&lt;Date&gt;.*)[\\\\/](?P&lt;Run&gt;.*)$</td></tr>
            <tr><td>.*[\\\\/]</td><td>Skip characters at the beginning of the pathname until either a slash (/) or
            backslash (\\) is encountered (depending on the operating system)</td></tr>
            <tr><td>(?P&lt;Date&gt;</td><td>Name the captured field <i>Date</i></td></tr>
            <tr><td>.*</td><td>Capture as many characters that follow</td></tr>
            <tr><td>[\\\\/]</td><td>Discard the slash/backslash character</td></tr>
            <tr><td>(?P&lt;Run&gt;</td><td>Name the captured field <i>Run</i></td></tr>
            <tr><td>.*</td><td>Capture as many characters as follow</td></tr>
            <tr><td>$</td><td>The <i>Run</i> field must be at the end of the path string, i.e., the
            last folder on the path. This also means that the Date field contains the parent
            folder of the Date folder.</td></tr>
            </table></p>"""))
 
        group.append("filter_choice", cps.Choice(
            "Extract metadata from",
            [F_ALL_IMAGES, F_FILTERED_IMAGES],doc = """
            Select whether you want to extract metadata from all of the images
            chosen by the <b>Images</b> module or a subset of the images.
            <p>This setting controls how different image types (e.g., an image
            of the GFP stain and a brightfield image) have different metadata
            extracted. There are two choices:<br>
            <ul>
            <li><i>%(F_ALL_IMAGES)s</i>: Extract metadata from all images specified in
            <b>Images</b>. This is the simplest choice and the appropriate one if you have
            only one kind of image (or only one image). CellProfiler will
            extract metadata from all images using the same method per iteration.</li>
            <li><i>%(F_FILTERED_IMAGES)s</i>: Extract metadata depending on specific file
            attributes. This is the appropriate choice if more than one image was taken of each 
            imaging site. You can specify distinctive criteria for each image subset with 
            matching metadata.</li>
            </ul></p>"""%globals()))
        
        group.append("filter", cps.Filter(
            "Select the filtering criteria", [FilePredicate(),
                 DirectoryPredicate(),
                 ExtensionPredicate()],
            'and (file does contain "")',doc = """
            Check this setting to display and use rules to select files for metadata extraction.
            <p>%(FILTER_RULES_BUTTONS_HELP)s</p>"""%globals()))
        
        group.append("csv_location", cps.Pathname(
            "Metadata file location",
            wildcard="Metadata files (*.csv)|*.csv|All files (*.*)|*.*"))
        
        group.append("csv_joiner", cps.Joiner(
            "Match file and image metadata", allow_none = False,doc="""
            Match columns in your .csv file to image metadata items
            <p>This setting controls how rows in your .csv file are matched to
            different images. The setting displays the columns in your
            .csv file in one of its columns and the metadata in your images
            in the other, including the metadata extracted by previous
            metadata extractors in this module.</p>
            """))
        
        group.append("wants_case_insensitive", cps.Binary(
            "Use case insensitive matching?", False, doc = """
            This setting controls whether row matching takes the
            metadata case into account when matching. If this setting is 
            not checked, metadata entries that only differ by case 
            (for instance, "A01" and "a01") will not match. If this setting
            is checked, then metadata entries that only differ by case
            will match. Check this setting if your CSV metadata is not being
            applied because the case does not match."""))
        
        group.append("update_metadata", cps.DoSomething(
            "", "Update metadata",
            lambda : self.do_update_metadata(group),doc = """
            Press this button to automatically extract metadata from
            your image files."""))
                 
        group.can_remove = can_remove
        if can_remove:
            group.append("remover", cps.RemoveSettingButton(
                'Remove above extraction method', 'Remove',
                self.extraction_methods, group))
            
    def get_dt_metadata_keys(self):
        '''Get the list of data-type metadata keys
        
        Get the metadata keys captured by file and folder metadata extraction
        and by metadata import.
        '''
        if not self.wants_metadata:
            return []
        keys = set()
        self.update_imported_metadata()
        for group in self.extraction_methods:
            if group.extraction_method == X_MANUAL_EXTRACTION:
                if group.source == XM_FILE_NAME:
                    regexp = group.file_regexp
                else:
                    regexp = group.folder_regexp
                keys.update(cpmeas.find_metadata_tokens(regexp.value))
            elif group.extraction_method == X_IMPORTED_EXTRACTION:
                imported_metadata = self.get_imported_metadata_for_group(group)
                if imported_metadata is None:
                    logger.warn("Unable to import metadata from %s" %
                                group.csv_location.value)
                else:
                    keys.update(imported_metadata.get_csv_metadata_keys())
        return sorted(keys)
    
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
            if len(self.get_dt_metadata_keys()) > 0:
                result += [self.dtc_divider, self.data_type_choice]
                if self.data_type_choice == DTC_CHOOSE:
                    result.append(self.data_types)
            result += [self.table]
        return result
    
    def example_file_fn(self):
        '''Get an example file name for the regexp editor'''
        if len(self.ipds) > 0:
            return os.path.split(self.ipds[0].path)[1]
        return "PLATE_A01_s1_w11C78E18A-356E-48EC-B204-3F4379DC43AB.tif"
            
    def example_directory_fn(self):
        '''Get an example directory name for the regexp editor'''
        if len(self.ipds) > 0:
            return os.path.split(self.ipds[0].path)[0]
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
        
        file_list = workspace.file_list
        pipeline = workspace.pipeline
        ipds = pipeline.get_filtered_image_plane_details(workspace)
        extractor = self.build_extractor()
        max_series = 0
        max_index = 0
        for ipd in ipds:
            if ipd.series is not None:
                max_series = max(max_series, ipd.series)
            if ipd.index is not None:
                max_index = max(max_index, ipd.index)
        if max_series > 0:
            series_digits = int(np.log10(max_series)) + 1
        else:
            series_digits = 1
        if max_index > 0:
            index_digits = int(np.log10(max_index)) + 1
        else:
            index_digits = 1
        if max_series > 0 or max_index > 0:
            script = """
            importPackage(Packages.org.cellprofiler.imageset);
            extractor.addImagePlaneExtractor(new SeriesIndexMetadataExtractor(
                seriesDigits, indexDigits));
            """
            J.run_script(script, dict(extractor = extractor,
                                      seriesDigits = series_digits,
                                      indexDigits = index_digits))
        env = J.get_env()
        entry_set_class = env.find_class("java/util/Map$Entry")
        get_key_id = env.get_method_id(entry_set_class, "getKey", "()Ljava/lang/Object;")
        get_value_id = env.get_method_id(entry_set_class, "getValue", "()Ljava/lang/Object;")
                
        def wrap_entry_set(o):
            return (env.get_string_utf(env.call_method(o, get_key_id)), 
                    env.get_string_utf(env.call_method(o, get_value_id)))
        #
        # Much of what appears below is optimized to avoid the cost of
        # "getting nice arguments" for the Java bridge. The IPDs should be
        # in alphabetical order which means that, for stacks, we can
        # save the results of OME-XML parsing in the Java ImageFile object.
        #
        extractor_class = env.find_class(
            "org/cellprofiler/imageset/ImagePlaneMetadataExtractor")
        extract_metadata_id = env.get_method_id(
            extractor_class,
            "extractMetadata",
            "(Ljava/lang/String;IILjava/lang/String;"
            "[Lorg/cellprofiler/imageset/filter/ImagePlaneDetails;"
            "[Lorg/cellprofiler/imageset/ImageFile;"
            ")Ljava/util/Iterator;")
        extract_metadata_if_id = env.get_method_id(
            extractor_class,
            "extractMetadata",
            "(Lorg/cellprofiler/imageset/ImageFile;II"
            "[Lorg/cellprofiler/imageset/filter/ImagePlaneDetails;"
            ")Ljava/util/Iterator;")
        ipd_class = env.find_class("org/cellprofiler/imageset/filter/ImagePlaneDetails")
        if_class = env.find_class("org/cellprofiler/imageset/ImageFile")
        clear_xml_document_id = env.get_method_id(
            if_class,
            "clearXMLDocument", "()V")
        pIPD = env.make_object_array(1, ipd_class)
        pIF = env.make_object_array(1, if_class)
        
        last_url = None
        last_if = None
        if_has_metadata = False
        for ipd in ipds:
            series, index = [x if x is not None else 0 
                             for x in ipd.series, ipd.index]
            if ipd.url != last_url:
                if if_has_metadata:
                    env.call_method(last_if, clear_xml_document_id)
                    x = env.exception_occurred()
                    if x is not None:
                        raise J.JavaException(x)
                    if_has_metadata = False
                xmlmetadata = file_list.get_metadata(ipd.url)
                if xmlmetadata is not None:
                    xmlmetadata = env.new_string(xmlmetadata)
                    if_has_metadata = True
                metadata = env.call_method(extractor, extract_metadata_id,
                                           env.new_string_utf(ipd.url),
                                           int(series), int(index),
                                           xmlmetadata, pIPD, pIF)
                x = env.exception_occurred()
                if x is not None:
                    raise J.JavaException(x)
                last_url = ipd.url
                last_if = env.get_object_array_elements(pIF)[0]
            else:
                metadata = env.call_method(
                    extractor, extract_metadata_if_id,
                    last_if, int(series), int(index), pIPD)
                x = env.exception_occurred()
                if x is not None:
                    raise J.JavaException(x)
            
            ipd.metadata.update(J.iterate_java(metadata, wrap_entry_set))
            ipd.jipd = env.get_object_array_elements(pIPD)[0]
        if if_has_metadata:
            env.call_method(last_if, clear_xml_document_id)
            x = env.exception_occurred()
            if x is not None:
                raise J.JavaException(x)
        return True
    
    def build_extractor(self):
        '''Build a Java metadata extractor using the module settings'''
        #
        # Build a metadata extractor
        #
        script = """
        importPackage(Packages.org.cellprofiler.imageset);
        importPackage(Packages.org.cellprofiler.imageset.filter);
        extractor = new ImagePlaneMetadataExtractor();
        extractor.addImagePlaneExtractor(new OMEMetadataExtractor());
        extractor;
        """
        extractor = J.run_script(script)
        for group in self.extraction_methods:
            if group.filter_choice == F_FILTERED_IMAGES:
                fltr = J.make_instance(
                    "org/cellprofiler/imageset/filter/Filter",
                    "(Ljava/lang/String;)V", group.filter.value_text)
            else:
                fltr = None
            if group.extraction_method == X_MANUAL_EXTRACTION:
                if group.source == XM_FILE_NAME:
                    method = "addFileNameRegexp"
                    pattern = group.file_regexp.value
                elif group.source == XM_FOLDER_NAME:
                    method = "addPathNameRegexp"
                    pattern = group.folder_regexp.value
                J.call(extractor,
                       method,
                       "(Ljava/lang/String;Lorg/cellprofiler/imageset/filter/Filter;)V",
                       pattern, fltr)
            elif group.extraction_method == X_IMPORTED_EXTRACTION:
                #
                # Create the array of key pairs for the join
                #
                key_pairs = []
                for join_idx in group.csv_joiner.parse():
                    key_pair = J.make_instance(
                        'org/cellprofiler/imageset/ImportedMetadataExtractor$KeyPair',
                        '(Ljava/lang/String;Ljava/lang/String;)V',
                        join_idx[self.CSV_JOIN_NAME], 
                        join_idx[self.IPD_JOIN_NAME])
                    key_pairs.append(key_pair)
                key_pairs = J.get_nice_arg(
                    key_pairs, 
                    "[Lorg/cellprofiler/imageset/ImportedMetadataExtractor$KeyPair;")
                #
                # Open the CSV file for reading, make an ImportedMetadataExtractor
                # and install it in the big extractor
                #
                script = """
                importPackage(Packages.org.cellprofiler.imageset);
                var inputStream = new java.io.FileInputStream(csv_path);
                var rdr = new java.io.InputStreamReader(inputStream);
                var iextractor = new ImportedMetadataExtractor(rdr, key_pairs, case_insensitive);
                extractor.addImagePlaneDetailsExtractor(iextractor, fltr);
                """
                J.run_script(script, dict(
                    csv_path=group.csv_location.value,
                    key_pairs=key_pairs,
                    case_insensitive = group.wants_case_insensitive.value,
                    extractor = extractor,
                    fltr = fltr))
        #
        # Finally, we add the WellMetadataExtractor which has the inglorious
        # job of making a well name from row and column, if present.
        #
        script = """
        importPackage(Packages.org.cellprofiler.imageset);
        extractor.addImagePlaneDetailsExtractor(new WellMetadataExtractor());
        """
        J.run_script(script, dict(extractor = extractor))
        
        return extractor
                    
        
    def run(self, workspace):
        pass
    
    def do_update_metadata(self, group):
        filelist = self.workspace.file_list
        urls = set([ipd.url for ipd in self.pipeline.get_filtered_image_plane_details(self.workspace)])
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
                               style = wx.PD_CAN_ABORT
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
                    metadata = get_omexml_metadata(url2pathname(url))
                    filelist.add_metadata(url, metadata)
                metadata = OMEXML(metadata)
                exemplar = cpp.ImagePlaneDetails(url, None, None, None)
                if not self.pipeline.find_image_plane_details(exemplar):
                    self.pipeline.add_image_plane_details([exemplar])
                self.pipeline.add_image_metadata(url, metadata)
            self.ipds = self.pipeline.get_filtered_image_plane_details(self.workspace)
            self.update_metadata_keys()
                
    def get_ipd_metadata(self, ipd):
        '''Get the metadata for an image plane details record'''
        assert isinstance(ipd, cpp.ImagePlaneDetails)
        m = ipd.metadata.copy()
        for group in self.extraction_methods:
            if group.filter_choice == F_FILTERED_IMAGES:
                if not evaluate_url(group.filter, ipd.url):
                    continue
            if group.extraction_method == X_MANUAL_EXTRACTION:
                m.update(self.manually_extract_metadata(group, ipd))
            elif group.extraction_method == X_AUTOMATIC_EXTRACTION:
                m.update(self.automatically_extract_metadata(group, ipd))
            elif group.extraction_method == X_IMPORTED_EXTRACTION:
                m.update(self.import_metadata(group, ipd, m))
        return m
                
    def manually_extract_metadata(self, group, ipd):
        if group.source == XM_FILE_NAME:
            text = os.path.split(ipd.path)[1]
            pattern = group.file_regexp.value
        elif group.source == XM_FOLDER_NAME:
            text = os.path.split(ipd.path)[0]
            pattern = group.folder_regexp.value
        else:
            return {}
        match = re.search(pattern, text)
        if match is None:
            return {}
        result = match.groupdict()
        tokens = result.keys()
        if needs_well_metadata(tokens):
            well_row_token, well_column_token = well_metadata_tokens(tokens)
            result[cpmeas.FTR_WELL] = \
                result[well_row_token] + result[well_column_token]
        return result
    
    def automatically_extract_metadata(self, group, ipd):
        return {}

    def get_imported_metadata_for_group(self, group):
        for imported_metadata in self.imported_metadata:
            assert isinstance(imported_metadata, self.ImportedMetadata)
            if imported_metadata.is_match(
                group.csv_location.value,
                group.csv_joiner,
                self.CSV_JOIN_NAME,
                self.IPD_JOIN_NAME):
                return imported_metadata
        return None
        
    def import_metadata(self, group, ipd, m):
        imported_metadata = self.get_imported_metadata_for_group(group)
        if imported_metadata is not None:
            return imported_metadata.get_ipd_metadata(
                m, group.wants_case_insensitive.value)
        return {}
    
    def on_activated(self, workspace):
        self.workspace = workspace
        self.pipeline = workspace.pipeline
        self.ipds = self.pipeline.get_filtered_image_plane_details(workspace)
        self.ipd_metadata_keys = []
        self.update_metadata_keys()
        self.update_imported_metadata()
        self.table.clear_rows()
        self.table.clear_columns()
        
    def on_setting_changed(self, setting, pipeline):
        self.update_imported_metadata()
        
    def update_imported_metadata(self):
        new_imported_metadata = []
        ipd_metadata_keys = set(getattr(self, "ipd_metadata_keys", []))
        for group in self.extraction_methods:
            if group.extraction_method == X_MANUAL_EXTRACTION:
                if group.source == XM_FILE_NAME:
                    regexp = group.file_regexp
                else:
                    regexp = group.folder_regexp
                ipd_metadata_keys.update(cpmeas.find_metadata_tokens(regexp.value))
            elif group.extraction_method == X_IMPORTED_EXTRACTION:
                joiner = group.csv_joiner
                csv_path = group.csv_location.value
                if not os.path.isfile(csv_path):
                    continue
                found = False
                best_match = None
                for i, imported_metadata in enumerate(self.imported_metadata):
                    assert isinstance(imported_metadata, self.ImportedMetadata)
                    if imported_metadata.is_match(csv_path, joiner,
                                                  self.CSV_JOIN_NAME,
                                                  self.IPD_JOIN_NAME):
                        new_imported_metadata.append(imported_metadata)
                        found = True
                        break
                    elif (best_match is None and 
                          imported_metadata.path == csv_path):
                        best_match = i
                if found:
                    del self.imported_metadata[i]
                else:
                    if best_match is not None:
                        imported_metadata = self.imported_metadata[i]
                        del self.imported_metadata[i]
                    else:
                        try:
                            imported_metadata = self.ImportedMetadata(csv_path)
                        except:
                            logger.debug("Failed to load csv file: %s" % csv_path)
                            continue
                    new_imported_metadata.append(imported_metadata)
                joiner.entities[self.CSV_JOIN_NAME] = \
                    imported_metadata.get_csv_metadata_keys()
                joiner.entities[self.IPD_JOIN_NAME] = \
                    list(ipd_metadata_keys)
                imported_metadata.set_joiner(joiner,
                                             self.CSV_JOIN_NAME,
                                             self.IPD_JOIN_NAME)
                ipd_metadata_keys.update(imported_metadata.get_csv_metadata_keys())
                    
        self.imported_metadata = new_imported_metadata            
        
    def update_table(self):
        columns = set()
        metadata = []
        for ipd in self.ipds:
            ipd_metadata = self.get_ipd_metadata(ipd)
            metadata.append(ipd_metadata)
            columns.update(ipd_metadata.keys())
        columns = [COL_PATH, COL_SERIES, COL_INDEX] + \
            sorted(list(columns))
        self.table.clear_columns()
        self.table.clear_rows()
        for i, column in enumerate(columns):
            self.table.insert_column(i, column)
            
        data = []
        for ipd, ipd_metadata in zip(self.ipds, metadata):
            row = [ipd.path, ipd.series, ipd.index]
            row += [ipd_metadata.get(column) for column in columns[3:]]
            data.append(row)
        self.table.add_rows(columns, data)
        
    def update_metadata_keys(self):
        self.ipd_metadata_keys = set(self.ipd_metadata_keys)
        for ipd in self.ipds:
            self.ipd_metadata_keys.update(ipd.metadata.keys())
        self.ipd_metadata_keys = sorted(self.ipd_metadata_keys)
        
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
        keys = set()
        self.update_imported_metadata()
        for group in self.extraction_methods:
            if group.extraction_method == X_MANUAL_EXTRACTION:
                if group.source == XM_FILE_NAME:
                    regexp = group.file_regexp
                else:
                    regexp = group.folder_regexp
                keys.update(cpmeas.find_metadata_tokens(regexp.value))
            elif group.extraction_method == X_IMPORTED_EXTRACTION:
                imported_metadata = self.get_imported_metadata_for_group(group)
                if imported_metadata is None:
                    logger.warn("Unable to import metadata from %s" %
                                group.csv_location.value)
                else:
                    keys.update(imported_metadata.metadata_keys)
            elif group.extraction_method == X_AUTOMATIC_EXTRACTION:
                # Assume that automatic extraction will populate T and Z
                keys.add(cpp.ImagePlaneDetails.MD_T)
                keys.add(cpp.ImagePlaneDetails.MD_Z)
        return list(keys)
    
    def get_measurement_columns(self, pipeline):
        '''Get the metadata measurements collected by this module'''
        keys = self.get_metadata_keys()
        data_types = self.data_types.get_data_types()
        data_types[cpp.ImagePlaneDetails.MD_T] = cps.DataTypes.DT_INTEGER
        data_types[cpp.ImagePlaneDetails.MD_Z] = cps.DataTypes.DT_INTEGER
        result = []
        for key in keys:
            if self.data_type_choice == DTC_CHOOSE:
                data_type = data_types.get(key, cps.DataTypes.DT_TEXT)
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
                cpmeas.IMAGE,  '_'.join((cpmeas.C_METADATA, key)), data_type))
        if needs_well_metadata(keys):
            result.append((cpmeas.IMAGE, cpmeas.M_WELL, 
                           cpmeas.COLTYPE_VARCHAR_FORMAT % 4))
        return result
    
    def get_categories(self, pipeline, object_name):
        '''Return the measurement categories for a particular object'''
        if object_name == cpmeas.IMAGE and len(self.get_metadata_keys()) > 0:
            return [cpmeas.C_METADATA]
        return []
            
    def get_measurements(self, pipeline, object_name, category):
        if object_name == cpmeas.IMAGE and category == cpmeas.C_METADATA:
            keys = self.get_metadata_keys()
            if needs_well_metadata(keys):
                keys = list(keys) + [cpmeas.FTR_WELL]
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
                    (IDX_EXTRACTION_METHOD_V1 + LEN_EXTRACTION_METHOD_V1 * (i+1))]
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
                    (IDX_EXTRACTION_METHOD_V2 + LEN_EXTRACTION_METHOD * (i+1))]
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
    
    class ImportedMetadata(object):
        '''A holder for the metadata from a csv file'''
        def __init__(self, path):
            self.joiner_initialized = False
            self.path = path
            self.path_timestamp = os.stat(path).st_mtime
            fd = open(path, "rb")
            rdr = csv.reader(fd)
            header = rdr.next()
            columns = [[] for  c in header]
            self.columns = dict([(c, l) for c,l in zip(header, columns)])
            for row in rdr:
                for i, field in enumerate(row):
                    columns[i].append(None if len(field) == 0 else field)
                if len(row) < len(columns):
                    for i in range(len(row), len(columns)):
                        columns[i].append(None)
                        
        def get_csv_metadata_keys(self):
            '''Get the metadata keys in the CSV header'''
            return sorted(self.columns.keys())
        
        def set_joiner(self, joiner, csv_name, ipd_name):
            '''Initialize to assign csv metadata to an image plane descriptor
            
            joiner - a joiner setting that describes the join between the
                     CSV metadata and the IPD metadata
            csv_name - the name assigned to the CSV file in the joiner
            ipd_name - the name assigned to the IPD in the joiner
            
            Creates a dictionary of keys from the CSV joining keys and
            records the keys that will be used to join in the IPD
            '''
            joins = joiner.parse()
            if len(joins) == 0:
                return
            if any([join.get(csv_name) not in self.columns.keys()
                    for join in joins]):
                return
            self.csv_keys = [join[csv_name] for join in joins]
            self.ipd_keys = [join[ipd_name] for join in joins]
            self.metadata_keys = set(self.columns.keys()).difference(self.csv_keys)
            self.d = {}
            self.d_lower = {}
            columns = [self.columns[key] for key in self.csv_keys]
            for i in range(len(columns[0])):
                key = tuple([column[i] for column in columns])
                self.d[key] = i
                key_lower = tuple([
                    k.lower() if isinstance(k, basestring) else k for k in key])
                self.d_lower[key_lower] = i
            self.joiner_initialized = True
            
        def get_ipd_metadata(self, ipd_metadata, case_insensitive=False):
            '''Get the matching metadata from the .csv for a given ipd
            
            ipd_metadata - the metadata dictionary for an IPD, possibly
            augmented by prior extraction
            '''
            if not self.joiner_initialized:
                return {}
            key = tuple([ipd_metadata.get(k) for k in self.ipd_keys])
            if case_insensitive:
                d = self.d_lower
                key = tuple([
                    k.lower() if isinstance(k, basestring) else k for k in key])
            else:
                d = self.d
            if not d.has_key(key):
                return {}
            return dict([(k, self.columns[k][d[key]]) 
                         for k in self.metadata_keys])
        
        def is_match(self, csv_path, joiner, csv_join_name, ipd_join_name):
            '''Check to see if this instance can handle the given csv and joiner
            
            csv_path - path to the CSV file to use
            
            joiner - the joiner to join to the ipd metadata
            
            csv_join_name - the join name in the joiner of the csv_file
            
            ipd_join_name - the join name in the joiner of the ipd metadata
            '''
            if csv_path != self.path:
                return False
            
            if os.stat(self.path).st_mtime != self.path_timestamp:
                return False
            
            if not self.joiner_initialized:
                self.set_joiner(joiner, csv_join_name, ipd_join_name)
                return True
            
            joins = joiner.parse()
            csv_keys = [join[csv_join_name] for join in joins]
            ipd_keys = [join[ipd_join_name] for join in joins]
            metadata_keys = set(self.columns.keys()).difference(self.csv_keys)
            for mine, yours in ((self.csv_keys, csv_keys),
                                (self.ipd_keys, ipd_keys),
                                (self.metadata_keys, metadata_keys)):
                if tuple(mine) != tuple(yours):
                    return False
            return True
            
            
