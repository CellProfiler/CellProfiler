'''<b>Export To Spreadsheet</b> exports measurements into one or more files that can be
opened in Excel or other spreadsheet programs
<hr>

This module will convert the measurements to a comma-, tab-, or other 
character-delimited text format and
save them to the hard drive in one or several files, as requested. 
<h2>Metadata tokens</h2>
<b>ExportToSpreadsheet</b> can write out separate files for groups of images based
on their metadata. This is controlled by the directory and file names
that you enter. For instance, you might have applied two treatments
to each of your samples and labeled them with the metadata names "Treatment1" 
and "Treatment2", and you might want to create separate files for each 
combination of treatments, storing all measurements with a given "Treatment1"
in separate directories. You can do this by specifying metadata tags for the folder name and
file name:
<ul>
<li>Choose <i>Default Input/Output Folder sub-folder</i> for the output file location.</li>
<li>Enter the sub-folder name "\g&lt;Treatment1&gt;"</li>
<li>Uncheck "Export all measurements?"</li>
<li>Uncheck <i>Use the object name for the file name?</i>.</li>
<li>Enter the file name "\g&lt;Treatment2&gt;.csv". </li>
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
'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Developed by the Broad Institute
# Copyright 2003-2011
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org
__version__="$Revision$"

import csv
import numpy as np
import os
import sys

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.pipeline as cpp
import cellprofiler.settings as cps
from cellprofiler.measurements import IMAGE, EXPERIMENT
from cellprofiler.preferences import get_absolute_path, get_output_file_name
from cellprofiler.preferences import ABSPATH_OUTPUT, ABSPATH_IMAGE
from cellprofiler.gui.help import USING_METADATA_TAGS_REF, USING_METADATA_HELP_REF
from cellprofiler.preferences import \
     standardize_default_folder_names, DEFAULT_INPUT_FOLDER_NAME, \
     DEFAULT_OUTPUT_FOLDER_NAME, ABSOLUTE_FOLDER_NAME, \
     DEFAULT_INPUT_SUBFOLDER_NAME, DEFAULT_OUTPUT_SUBFOLDER_NAME, \
     IO_FOLDER_CHOICE_HELP_TEXT, IO_WITH_METADATA_HELP_TEXT

MAX_EXCEL_COLUMNS = 256
MAX_EXCEL_ROWS = 65536

DELIMITER_TAB = "Tab"
DELIMITER_COMMA = 'Comma (",")'
DELIMITERS = (DELIMITER_COMMA,DELIMITER_TAB)

OBJECT_RELATIONSHIPS = "Object relationships"
RELATIONSHIPS = "Relationships"

"""Offset of the first object group in the settings"""
SETTING_OG_OFFSET = 15

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

DIR_CUSTOM = "Custom folder"
DIR_CUSTOM_WITH_METADATA = "Custom folder with metadata"

"""Options for GenePattern GCT file export"""
GP_NAME_FILENAME = "Image filename"
GP_NAME_METADATA = "Metadata"
GP_NAME_OPTIONS = [GP_NAME_METADATA, GP_NAME_FILENAME]

class ExportToSpreadsheet(cpm.CPModule):

    module_name = 'ExportToSpreadsheet'
    category = ["File Processing","Data Tools"]
    variable_revision_number = 7
    
    def create_settings(self):
        self.delimiter = cps.CustomChoice('Select or enter the column delimiter',DELIMITERS, doc = """
                            What delimiter do you want to use? This is the character that separates columns in a file. The
                            two default choices are tab and comma, but you can type in any single character delimiter you would prefer. Be sure that the delimiter you choose is not a character that is present within your data (for example, in file names).""")
        
        self.prepend_output_filename = cps.Binary("Prepend the output file name to the data file names?", True, doc = """
                            This can be useful if you want to run a pipeline multiple 
                            times without overwriting the old results.""")
        
        self.directory = cps.DirectoryPath(
            "Output file location",
            dir_choices = [
                DEFAULT_OUTPUT_FOLDER_NAME, DEFAULT_INPUT_FOLDER_NAME, 
                ABSOLUTE_FOLDER_NAME, DEFAULT_INPUT_SUBFOLDER_NAME,
                DEFAULT_OUTPUT_SUBFOLDER_NAME],
            doc="""This setting lets you choose the folder for the output
            files. %(IO_FOLDER_CHOICE_HELP_TEXT)s
            
            <p>%(IO_WITH_METADATA_HELP_TEXT)s %(USING_METADATA_TAGS_REF)s<br>
            For instance, if you have a metadata tag named 
            "Plate", you can create a per-plate folder by selecting one of the subfolder options
            and then specifying the subfolder name as "\g&lt;Plate&gt;". The module will 
            substitute the metadata values for the current image set for any metadata tags in the 
            folder name. %(USING_METADATA_HELP_REF)s.</p>"""%globals())
        
        self.add_metadata = cps.Binary("Add image metadata columns to your object data file?", False, doc = """"Image_Metadata_" columns are normally exported in the Image data file, but if you check this box they will also be exported with the Object data file(s).""")
        
        self.excel_limits = cps.Binary("Limit output to a size that is allowed in Excel?", False, doc = """
                            If your output has more than 256 columns, a window will open
                            which allows you to select the columns you'd like to export. If your output exceeds
                            65,000 rows, you can still open the .csv in Excel, but not all rows will be visible.""")
        
        self.pick_columns = cps.Binary("Select the columns of measurements to export?", False, doc = """
                            Checking this setting will open up a window that allows you to select the columns to export.""")
        
        self.columns = cps.MeasurementMultiChoice(
            "Press button to select measurements to export",
            doc = """<i>(Used only when selecting the columns of measurements to export)</i><br>This setting controls the columns to be exported. Press
            the button and check the measurements or categories to export""")
        
        self.wants_aggregate_means = cps.Binary("Calculate the per-image mean values for object measurements?", False, doc = """
                            <b>ExportToSpreadsheet</b> can calculate population statistics over all the 
                            objects in each image and save that value as an aggregate 
                            measurement in the Image file.  For instance, if you are measuring 
                            the area of the Nuclei objects and you check the box for this option, <b>ExportToSpreadsheet</b> will 
                            create a column in the Image file called "Mean_Nuclei_AreaShape_Area". 
                            <p>You may not want to use <b>ExportToSpreadsheet</b> to calculate these 
                            measurements if your pipeline generates a large number of per-object 
                            measurements; doing so might exceed Excel's limits on the number of columns (256). """)
        
        self.wants_aggregate_medians = cps.Binary("Calculate the per-image median values for object measurements?", False)
        
        self.wants_aggregate_std = cps.Binary("Calculate the per-image standard deviation values for object measurements?", False)
        
        self.wants_genepattern_file = cps.Binary("Create a GenePattern GCT file?", False, doc="""
            Create a GCT file compatible with <a href="http://www.broadinstitute.org/cancer/software/genepattern/">GenePattern</a>.
            The GCT file format is a tab-delimited text file format that describes a gene expression dataset; the specifics of the
            format are described <a href="http://www.broadinstitute.org/cancer/software/genepattern/tutorial/gp_fileformats.html#gct">here</a>.
            By converting your measurements into a GCT file, you can make use of GenePattern's data visualization and clustering methods.
            
            <p>Each row in the GCT file represents (ordinarily) a gene and each column represents a sample (in this case, a per-image set
            of measurements). In addition to any other spreadsheets desired, checking this box will produce a GCT file with the 
            extension .gct, prepended with the text selection above. If per-image aggregate measurements are requested above, those 
            measurements are included in the GCT file as well.</p>""")
        
        self.how_to_specify_gene_name = cps.Choice("Select source of sample row name", 
            GP_NAME_OPTIONS, GP_NAME_METADATA, doc = """
            <i>(Used only if a GenePattern file is requested)</i><br>
            The first column of the GCT file is the unique identifier for each sample, which is ordinarily the gene name. 
            This information may be specified in one of two ways:
            <ul>
            <li><i>Metadata:</i> If you used <b>LoadData</b> or <b>LoadImages</b> to input your images, you may use a per-image data measurement 
            (such as metadata) that corresponds to the identifier for this column. %(USING_METADATA_HELP_REF)s.</li>
            <li><i>Image filename:</i> If the gene name is not available, the image filename can be used as a surrogate identifier.</li>
            </ul>"""%globals())
        
        self.gene_name_column = cps.Measurement("Select the metadata to use as the identifier",
            lambda : cpmeas.IMAGE, doc = """
            <i>(Used only if a GenePattern file is requested and metadata is used to name each row)</i><br>
            Choose the measurement that corresponds to the identifier, such as metadata from <b>LoadData</b>'s input file. 
            %(USING_METADATA_HELP_REF)s."""%globals())
        
        self.use_which_image_for_gene_name = cps.ImageNameSubscriber("Select the image to use as the identifier","None", doc = """
            <i>(Used only if a GenePattern file is requested and image filename is used to name each row)</i><br>
            Select which image whose filename will be used to identify each sample row.""")
        
        self.wants_everything = cps.Binary(
            "Export all measurements, using default file names?", True,
            doc="""Check this setting to export every measurement.
            <b>ExportToSpreadsheet</b> will create one file per object type,
            as well as per-image, per-experiment and object relationships, 
            if relevant. 
            It will use the object name as the file name, 
            optionally prepending the output file name if
            specified above. Leave this box unchecked to specify which
            objects should be exported or to override the automatic names.""")
        
        self.object_groups = []
        self.add_object_group()
        self.add_button = cps.DoSomething("", "Add another data set",
                                           self.add_object_group)
    
    def add_object_group(self,can_remove = True):
        group = cps.SettingsGroup()
        group.append(
            "name", EEObjectNameSubscriber("Data to export",
            doc="""<i>(Used only when Export all measurements? is left unchecked)</i><br>
            Choose <i>Image</i>, <i>Experiment</i>, <i>Object relationships</i> 
            or an object name from the list. <b>ExportToSpreadsheet</b> will write out a
            file of measurements for the given category."""))
    
        group.append(
            "previous_file", cps.Binary(
                "Combine these object measurements with those of the previous object?",
                False,doc="""<i>(Used only when Export all measurements? is left unchecked)</i><br>Check this setting to create a file composed
                of measurements made on this object and the one directly
                above it. Leave the box unchecked to create separate
                files for this and the previous object."""))
        
        group.append("wants_automatic_file_name", cps.Binary(
            "Use the object name for the file name?", True,
            doc="""<i>(Used only when Export all measurements? is left unchecked)</i><br>Use the object name as selected above to generate a file
            name for the spreadsheet. For example, if you selected <i>Image</i>,
            above and have not checked the <i>Prepend output file name</i> option,
            your output file will be named "Image.csv". You can name
            the file yourself if you leave this box unchecked."""))
        
        group.append("file_name", 
                     cps.Text(
                         "File name", "DATA.csv",
                         metadata = True,
                         doc="""<i>(Used only when Export all measurements? is left unchecked)</i><br>Enter a file name for the named objects' 
                         measurements. <b>ExportToSpreadsheet</b> will
                         prepend the name of the measurements file to this
                         if you asked to do so above. If you have metadata 
                         associated with your images, this setting will also substitute
                         metadata tags if desired. %(USING_METADATA_TAGS_REF)s%(USING_METADATA_HELP_REF)s."""% globals()))
        
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
        while len(self.object_groups) > group_count:
            self.remove_object_group(self.object_groups[-1][OG_KEY])
        
        while len(self.object_groups) < group_count:
            self.add_object_group()

    def settings(self):
        """Return the settings in the order used when storing """
        result = [self.delimiter, self.prepend_output_filename,
                  self.add_metadata, self.excel_limits, self.pick_columns,
                  self.wants_aggregate_means, self.wants_aggregate_medians,
                  self.wants_aggregate_std, self.directory,
                  self.wants_genepattern_file, self.how_to_specify_gene_name, 
                  self.use_which_image_for_gene_name,self.gene_name_column,
                  self.wants_everything, self.columns]
        for group in self.object_groups:
            result += [group.name, group.previous_file, group.file_name,
                       group.wants_automatic_file_name]
        return result

    def visible_settings(self):
        """Return the settings as seen by the user"""
        result = [self.delimiter, self.prepend_output_filename,
                  self.directory]
        result += [ self.add_metadata, self.excel_limits, self.pick_columns]
        if self.pick_columns:
            result += [ self.columns]
        result += [ self.wants_aggregate_means, self.wants_aggregate_medians,
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
            result += [ self.add_button ]
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
                    raise cps.ValidationError("%s is not a defined metadata tag. Check the metadata specifications in your load modules" %
                                     undefined_tags[0], 
                                     group.file_name)

    def validate_module_warnings(self, pipeline):
        '''Warn user re: Test mode '''
        if pipeline.test_mode:
            raise cps.ValidationError("ExportToSpreadsheet will not produce output in Test Mode",
                                      self.directory)

    @property
    def delimiter_char(self):
        if self.delimiter == DELIMITER_TAB:
            return "\t"
        elif self.delimiter == DELIMITER_COMMA:
            return ","
        else:
            return self.delimiter.value
    
    def run(self, workspace):
        # all of the work is done in post_run()
        pass
    
    
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
        # Export all measurements if requested
        #
        if self.wants_everything:
            for object_name in workspace.measurements.get_object_names():
                self.run_objects([object_name], 
                                 "%s.csv" % object_name, workspace)
            return
        
        object_names = []
        #
        # Loop, collecting names of objects that get included in the same file
        #
        for i in range(len(self.object_groups)):
            group = self.object_groups[i]
            last_in_file = self.last_in_file(i)
            if len(object_names) == 0:
                if group.wants_automatic_file_name:
                    filename = "%s.csv" % group.name.value
                else:
                    filename = group.file_name.value
            object_names.append(group.name.value)
            if last_in_file:
                self.run_objects(object_names, filename, workspace)
                object_names = []

    def last_in_file(self, i):
        '''Return true if the group is the last to be included in a csv file
        
        i - the index of the group being considered.
        
        Objects can be collected together in one file. Return true if
        this is the last object in a collection.
        '''
    
        group = self.object_groups[i]
        return ((i == len(self.object_groups)-1) or
                (not is_object_group(group)) or
                (not is_object_group(self.object_groups[i+1])) or
                (not self.object_groups[i+1].previous_file.value))
        
    def should_stop_writing_measurements(self):
        '''All subsequent modules should not write measurements'''
        return True
    
    def run_objects(self, object_names, file_name, workspace):
        """Create a file (or files if there's metadata) based on the object names
        
        object_names - a sequence of object names (or Image or Experiment)
                       which tell us which objects get piled into each file
        file_name - a file name or file name with metadata tags to serve as the
                    output file.
        workspace - get the images from here.
        
        """
        if len(object_names) == 1 and object_names[0] == EXPERIMENT:
            self.make_experiment_file(file_name, workspace)
            return
        tags = cpmeas.find_metadata_tokens(file_name)
        if self.directory.is_custom_choice:
            tags += cpmeas.find_metadata_tokens(self.directory.custom_path)
        metadata_groups = workspace.measurements.group_by_metadata(tags)
        for metadata_group in metadata_groups:
            if len(object_names) == 1 and object_names[0] == IMAGE:
                self.make_image_file(file_name, 
                                     metadata_group.image_numbers, 
                                     workspace)
                if self.wants_genepattern_file.value:
                    self.make_gct_file(file_name, 
                                       metadata_group.image_numbers, 
                                       workspace)
            elif len(object_names) == 1 and object_names[0] == OBJECT_RELATIONSHIPS:
                self.make_relationships_file(file_name, 
                                             metadata_group.image_numbers, 
                                             workspace)
            else:
                self.make_object_file(object_names, file_name, 
                                      metadata_group.image_numbers, workspace)
    
    def make_full_filename(self, file_name, 
                           workspace = None, image_set_number = None):
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
        file_name = os.path.join(path_name, file_name)
        path, file = os.path.split(file_name)
        if not os.path.isdir(path):
            os.makedirs(path)
        if self.prepend_output_filename.value:
            file = os.path.splitext(get_output_file_name())[0] + '_' + file 
        return os.path.join(path,file)
    
    def make_experiment_file(self, file_name, workspace):
        """Make a file containing the experiment measurements
        
        file_name - create a file with this name
        workspace - the workspace that has the measurements
        """
        m = workspace.measurements
        feature_names = [ 
            feature_name for feature_name in m.get_feature_names(EXPERIMENT)
            if feature_name != cpp.EXIT_STATUS]
        if len(feature_names) == 0:
            return
        file_name = self.make_full_filename(file_name)
        fd = open(file_name,"wb")
        try:
            writer = csv.writer(fd,delimiter=self.delimiter_char)
            for feature_name in feature_names:
                v = m.get_all_measurements(EXPERIMENT, feature_name)
                writer.writerow((feature_name, unicode(v).encode('utf8')))
        finally:
            fd.close()
    
    def make_image_file(self, file_name, image_set_numbers, workspace):
        """Make a file containing image measurements
        
        file_name - create a file with this name
        image_set_numbers - the image sets whose data gets extracted
        workspace - workspace containing the measurements
        """
        m = workspace.measurements
        image_features = m.get_feature_names(IMAGE)
        image_features.insert(0, IMAGE_NUMBER)
        if not self.check_excel_limits(workspace, file_name,
                                       len(image_set_numbers),
                                       len(image_features)):
            return
        file_name = self.make_full_filename(file_name, workspace,
                                            image_set_numbers[0])
        fd = open(file_name,"wb")
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
                row = [ img_number
                       if feature_name == IMAGE_NUMBER
                       else agg_measurements[feature_name]
                       if agg_measurements.has_key(feature_name)
                       else m.get_measurement(IMAGE, feature_name, img_number)
                       for feature_name in image_features]
                row = ['' if x is None
                       else x if np.isscalar(x) 
                       else x[0] for x in row]
                row = [unicode(x).encode('utf8') for x in row]
                writer.writerow(row)
        finally:
            fd.close()

    def make_gct_file(self, file_name, image_set_numbers, workspace):
        """Make a GenePattern file containing image measurements
        Format specifications located at http://www.broadinstitute.org/cancer/software/genepattern/tutorial/gp_fileformats.html?gct
        
        file_name - create a file with this name
        image_set_numbers - the image sets whose data gets extracted
        workspace - workspace containing the measurements
        """
        from loaddata import is_path_name_feature, is_file_name_feature
        from loadimages import C_PATH_NAME, C_FILE_NAME
        from loadimages import C_MD5_DIGEST, C_SCALING, C_HEIGHT, C_WIDTH

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
        file_name = self.make_full_filename(file_name, workspace,
                                            image_set_numbers[0])
        
        # Use image name and append .gct extension
        path, name = os.path.splitext(file_name)
        file_name = os.path.join(path+'.gct')
        
        fd = open(file_name,"wb")
        try:
            writer = csv.writer(fd,delimiter="\t")
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
                            num_measures +=1
                    
                    writer.writerow(['#1.2'])
                    writer.writerow([len(image_set_numbers), num_measures])
                    
                    # Keep measurements only
                    measurement_feature_names = [x for x in image_features if not ignore_feature(x)]
                    
                    # The first headers need to be 'NAME' and 'Description'
                    written_image_names = ['NAME','Description'] + measurement_feature_names
                    writer.writerow(written_image_names)
                    
                    # Place the one of the paths and desired info column up front in image feature list
                    description_feature = [x for x in image_features if x.startswith(C_PATH_NAME+"_")]
                    if self.how_to_specify_gene_name == GP_NAME_METADATA:
                        name_feature = [self.gene_name_column.value]
                    elif self.how_to_specify_gene_name == GP_NAME_FILENAME:
                        name_feature = [x for x in image_features if x.startswith("_".join((C_FILE_NAME, self.use_which_image_for_gene_name.value)))]
                    image_features = [name_feature[0],description_feature[0]] + measurement_feature_names
                    
                # Output all measurements
                row = [ agg_measurements[feature_name]
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
        if self.excel_limits and workspace.frame is not None:
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
                import wx
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
                for agg, wants_it in (
                    (cpmeas.AGG_MEAN, self.wants_aggregate_means),
                    (cpmeas.AGG_MEDIAN, self.wants_aggregate_medians),
                    (cpmeas.AGG_STD_DEV, self.wants_aggregate_std)):
                    if not wants_it:
                        continue
                    for column in self.columns.selections:
                        if self.columns.get_measurement_object(column) not in (
                            cpmeas.IMAGE, cpmeas.EXPERIMENT, cpmeas.NEIGHBORS):
                            columns += [ cpmeas.get_agg_measurement_name(
                                agg, self.columns.get_measurement_object(column),
                                self.columns.get_measurement_feature(column))]
                                
            columns = set(columns)
            features = [x for x in features if x in columns]
        return features
        
    def make_object_file(self, object_names, file_name, 
                         image_set_numbers, workspace):
        """Make a file containing object measurements
        
        object_names - sequence of names of the objects whose measurements
                       will be included
        file_name - create a file with this name
        image_set_numbers -  the image sets whose data gets extracted
        workspace - workspace containing the measurements
        """
        m = workspace.measurements
        features = []
        features += [(IMAGE, IMAGE_NUMBER),
                     (object_names[0], OBJECT_NUMBER)]
        if self.add_metadata.value:
            mdfeatures = [(IMAGE, name) 
                          for name in m.get_feature_names(IMAGE)
                          if name.startswith("Metadata_")]
            mdfeatures.sort()
            features += mdfeatures
        for object_name in object_names:
            ofeatures = m.get_feature_names(object_name)
            ofeatures = self.filter_columns(ofeatures, object_name)
            ofeatures = [(object_name, feature_name)
                         for feature_name in ofeatures]
            ofeatures.sort()
            features += ofeatures
        file_name = self.make_full_filename(file_name, workspace,
                                            image_set_numbers[0])
        fd = open(file_name,"wb")
        if self.excel_limits:
            row_count = 1
            for img_number in image_set_numbers:
                object_count =\
                     np.max([m.get_measurement(IMAGE, "Count_%s"%name, img_number)
                             for name in object_names])
                row_count += int(object_count)
            if not self.check_excel_limits(workspace, file_name,
                                           row_count, len(features)):
                return

        try:
            writer = csv.writer(fd,delimiter=self.delimiter_char)
            #
            # We write the object names in the first row of headers if there are
            # multiple objects. Otherwise, we just write the feature names
            #
            for i in (0,1) if len(object_names) > 1 else (1,):
                writer.writerow([x[i] for x in features])
            for img_number in image_set_numbers:
                object_count =\
                     np.max([m.get_measurement(IMAGE, "Count_%s"%name, img_number)
                             for name in object_names])
                object_count = int(object_count)
                columns = [np.repeat(img_number, object_count)
                           if feature_name == IMAGE_NUMBER
                           else np.arange(1,object_count+1) 
                           if feature_name == OBJECT_NUMBER
                           else np.repeat(m.get_measurement(IMAGE, feature_name,
                                                            img_number), 
                                          object_count)
                           if object_name == IMAGE
                           else m.get_measurement(object_name, feature_name, 
                                                  img_number)
                           for object_name, feature_name in features]
                for obj_index in range(object_count):
                    row = [ column[obj_index] 
                            if (column is not None and 
                                obj_index < column.shape[0])
                            else np.NAN
                            for column in columns]
                    writer.writerow(row)
        finally:
            fd.close()
    
    def make_relationships_file(self, file_name, image_set_numbers, workspace):
        '''Create a CSV file documenting the relationships between objects'''
        
        file_name = self.make_full_filename(file_name, workspace,
                                            image_set_numbers[0])
        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        fd = open(file_name, "wb")
        image_number_map = {}
        module_map = {}
        group_numbers = set()
        for img_number in image_set_numbers:
            group_number = m.get_measurement(cpmeas.IMAGE, cpp.GROUP_NUMBER, img_number)
            group_index = m.get_measurement(cpmeas.IMAGE, cpp.GROUP_INDEX, img_number)
            image_number_map[(group_number, group_index)] = img_number
            group_numbers.add(group_number)
        for module in workspace.pipeline.modules():
            module_map[module.module_num] = module.module_name
            
        try:
            writer = csv.writer(fd,delimiter=self.delimiter_char)
            writer.writerow([
                "Module", "Module Number", "Relationship",
                "First Object Name", "First Image Number", "First Object Number",
                "Second Object Name", "Second Image Number", "Second Object Number"])
            for key in m.get_relationship_groups():
                r = m.get_relationships(
                    key.module_number, key.relationship, 
                    key.object_name1, key.object_name2,
                    key.group_number).view(np.recarray)
                if key.group_number not in group_numbers:
                    continue
                for i in range(len(r)):
                    image_number_1 = image_number_map[
                        (key.group_number, r.group_index1[i])]
                    image_number_2 = image_number_map[
                        (key.group_number, r.group_index2[i])]
                    module_name = module_map[key.module_number]
                    writer.writerow([
                        module_name, key.module_number, key.relationship,
                        key.object_name1, image_number_1, r.object_number1[i],
                        key.object_name2, image_number_2, r.object_number2[i]])
        finally:
            fd.close()
        
    def prepare_to_create_batch(self, pipeline, image_set_list, fn_alter_path):
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
            setting_values = [ DELIMITER_TAB, cps.YES, cps.NO, cps.NO, 
                              cps.NO, cps.NO ]
            for name in object_names:
                setting_values.extend([name, cps.NO, "%s.csv"%(name)])
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
                prefix = setting_values[9]+"_"
            else:
                prefix = ""
            object_names = [x for x in setting_values[:8]
                            if x != cps.DO_NOT_USE]
            setting_values = [ DELIMITER_TAB, cps.YES, cps.NO, cps.NO, 
                              cps.NO, cps.NO, cps.NO, cps.NO, cps.NO,
                              directory_choice, custom_directory ]
            for name in object_names:
                setting_values.extend([name, cps.NO, 
                                       "%s%s.csv"%(prefix,name)])
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
                new_setting_values += setting_values[i:i+3] + [cps.NO]

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
                    custom_directory = '.'+custom_directory[1:]
                else:
                    directory_choice = ABSOLUTE_FOLDER_NAME
            directory = cps.DirectoryPath.static_join_string(
                directory_choice, custom_directory)
            setting_values = (setting_values[:3] + 
                              setting_values[4:9] +
                              [directory] +
                              setting_values[11:])
            variable_revision_number = 6
            
        # Standardize input/output directory name references
        SLOT_DIRCHOICE = 8
        directory = setting_values[SLOT_DIRCHOICE]
        directory = cps.DirectoryPath.upgrade_setting(directory)
        setting_values = (setting_values[:SLOT_DIRCHOICE] +
                          [directory] + 
                          setting_values[SLOT_DIRCHOICE+1:])
        
        if variable_revision_number == 6 and not from_matlab:
            ''' Add GenePattern export options
            self.wants_genepattern_file, self.how_to_specify_gene_name, 
            self.use_which_image_for_gene_name,self.gene_name_column 
            '''
            setting_values = (setting_values[:9] +
                              [cps.NO,GP_NAME_METADATA,"None","None"] + 
                              setting_values[9:])
            variable_revision_number == 7
            
        return setting_values, variable_revision_number, from_matlab

def is_object_group(group):
    """True if the group's object name is not one of the static names"""
    return not group.name.value in (IMAGE, EXPERIMENT, OBJECT_RELATIONSHIPS)

class EEObjectNameSubscriber(cps.ObjectNameSubscriber):
    """ExportToExcel needs to prepend "Image" and "Experiment" to the list of objects
    
    """
    def get_choices(self, pipeline):
        choices = [ IMAGE, EXPERIMENT, OBJECT_RELATIONSHIPS ]
        choices += cps.ObjectNameSubscriber.get_choices(self, pipeline)
        return choices

ExportToExcel = ExportToSpreadsheet
