'''<b>Export To Spreadsheet</b> exports measurements into one or more files that can be
opened in Excel or other spreadsheet programs
<hr>

This module will convert the measurements to a comma-, tab-, or other 
character-delimited text format and
save them to the hard drive in one or several files, as requested. 
<h2>Metadata tokens</h2>
Export To Excel can write out separate files for groups of images based
on their metadata. This is controlled by the directory and file names
that you enter. For instance, you might have applied two treatments
to each of your samples and labeled them with the metadata name, "Treatment1" 
and "Treatment2" and you might want to create separate files for each 
combination of treatments, storing all measurements with a given "Treatment1"
in separate directories. You can do this by specifying metadata tags in
for the folder name and file name.
<br>
To do this, you would choose, "Custom folder with metadata", enter the
directory name, "\g&lt;Treatment1&gt;" and enter the file name, 
"\g&lt;Treatment2&gt;". Here's an example table of the files that
would be generated:<br><tt>
<table border="1"><tr><th>Treatment1</th><th>Treatment2</th><th>Path</th></tr>

<tr><td>1M_NaCl</td><td>20uM_DMSO</td><td>1M_NaCl/20uM_DMSO.csv</td></tr>
<tr><td>1M_NaCl</td><td>40uM_DMSO</td><td>1M_NaCl/40uM_DMSO.csv</td></tr>
<tr><td>2M_NaCl</td><td>20uM_DMSO</td><td>2M_NaCl/20uM_DMSO.csv</td></tr>
<tr><td>2M_NaCl</td><td>40uM_DMSO</td><td>2M_NaCl/40uM_DMSO.csv</td></tr>\

</table></tt>
'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Developed by the Broad Institute
# Copyright 2003-2010
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org
__version__="$Revision$"

import csv
import numpy as np
import os
import sys
import wx

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.settings as cps
from cellprofiler.measurements import IMAGE, EXPERIMENT
from cellprofiler.preferences import get_absolute_path, get_output_file_name
from cellprofiler.preferences import ABSPATH_OUTPUT, ABSPATH_IMAGE

DELIMITER_TAB = "Tab"
DELIMITER_COMMA = 'Comma (",")'
DELIMITERS = (DELIMITER_COMMA,DELIMITER_TAB)

"""Offset of the first object group in the settings"""
SETTING_OG_OFFSET = 12

"""Offset of the object name setting within an object group"""
SETTING_OBJECT_NAME_IDX = 0

"""Offset of the previous file flag setting within an object group"""
SETTING_PREVIOUS_FILE_IDX = 1

"""Offset of the file name setting within an object group"""
SETTING_FILE_NAME_IDX = 2

SETTING_AUTOMATIC_FILE_NAME_IDX = 3

"""# of settings within an object group"""
SETTING_OBJECT_GROUP_CT = 4

"""The caption for the image set index"""
IMAGE_NUMBER = "ImageNumber"

"""The caption for the object # within an image set"""
OBJECT_NUMBER = "ObjectNumber"

DIR_DEFAULT_IMAGE = "Default image folder"
DIR_DEFAULT_OUTPUT = "Default output folder"
DIR_CUSTOM = "Custom folder"
DIR_CUSTOM_WITH_METADATA = "Custom folder with metadata"

class ExportToSpreadsheet(cpm.CPModule):

    module_name = 'ExportToSpreadsheet'
    category = "Data Tools"
    variable_revision_number = 4
    
    def create_settings(self):
        self.delimiter = cps.CustomChoice('Select or enter the column delimiter',DELIMITERS, doc = """
                            What delimiter do you want to use? This is the character that separates columns in a file. The
                            two default choices are tab and comma, but you can type in any single character delimiter you would prefer. Be sure that the delimiter you choose is not a character that is present within your data (for example, in file names).""")
        
        self.prepend_output_filename = cps.Binary("Prepend the output file name to the data file names?", True, doc = """
                            This can be useful if you want to run a pipeline multiple 
                            times without overwriting the old results.""")
        self.directory_choice = cps.Choice(
            "Where do you want to save the files?",
            [DIR_DEFAULT_OUTPUT, DIR_DEFAULT_IMAGE, DIR_CUSTOM,
             DIR_CUSTOM_WITH_METADATA],
            doc="""This setting lets you choose the folder for the output
            files.<br><ul>
            <li><i>Default output folder</i>: saves the .csv files in the
            default output folder</li>
            <li><i>Default image folder</i>: saves the .csv files in the
            default image folder</li>
            <li><i>Custom folder</i>: lets you specify the folder name. Start
            the folder name with "." to name a sub-folder of the output folder
            (for instance, "./data"). Start the folder name with "&" to name
            a sub-folder of the image folder.</li>
            <li><i>Custom folder with metadata</i>: uses metadata substitution
            to name the folder and to group the image sets by metadata tag.
            For instance, if you have a metadata tag named, "Plate", you can
            create a folder per-plate using the metadata tag, "./&lt;Plate&gt;".
            </li></ul>""")
        self.custom_directory = cps.Text(
            "Folder name:", ".", doc="""This is the folder that will be used
            to store the output files. Start
            the folder name with "." to name a sub-folder of the output folder
            (for instance, "./data"). Start the folder name with "&" to name
            a sub-folder of the image folder.""")
        
        self.add_metadata = cps.Binary("Add image metadata columns to your object data file?", False, doc = """Image_Metadata_ columns are normally exported in the Image data file, but if you check this box, they will also be exported with the Object data file(s).""")
        
        self.add_indexes = cps.Binary("No longer used, always saved", True)
        
        self.excel_limits = cps.Binary("Limit output to a size that is allowed in Excel?", False, doc = """
                            If your output has more than 256 columns, a window will open
                            which allows you to select which columns you'd like to export. If your output exceeds
                            65,000 rows, you can still open the .csv in Excel, but not all rows will be visible.""")
        
        self.pick_columns = cps.Binary("Select the columns of measurements to export?", False, doc = """
                            Checking this setting will open up a window that allows you to select which columns to export.""")
        
        self.wants_aggregate_means = cps.Binary("Calculate the per-image mean values for object measurements?", False, doc = """
                            ExportToExcel can calculate population statistics over all the 
                            objects in each image and save that value as an aggregate 
                            measurement in the Image file.  For instance, if you are measuring 
                            the area of the Nuclei objects and you check the box for this option, ExportToExcel will 
                            create a column in the Image file called Mean_Nuclei_AreaShape_Area. 
                            <p>You may not want to use ExportToExcel to calculate these 
                            measurements if your pipeline generates a large number of per-object 
                            measurements; doing so might exceed Excel's limits on the number of columns (256). """)
        
        self.wants_aggregate_medians = cps.Binary("Calculate the per-image median values for object measurements?", False)
        
        self.wants_aggregate_std = cps.Binary("Calculate the per-image standard deviation values for object measurements?", False)
        
        self.wants_everything = cps.Binary(
            "Export all measurements?", True,
            doc="""Check this setting to export every measurement.
            <b>ExportToSpreadsheet</b> will create one file per object type,
            including image and experiment. It will use the object name as
            the file name, optionally prepending the output file name if
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
            doc="""Choose either "Image", "Experiment" or an object name
            from the list. <b>ExportToSpreadsheet</b> will write out a
            file of measurements for the given category."""))
    
        group.append(
            "previous_file", cps.Binary(
                "Combine these object measurements with those of the previous object?",
                False,doc="""Check this setting to create a file composed
                of measurements made on this object and the one directly
                above this one. Leave the box unchecked to create separate
                files for this and the previous object."""))
        
        group.append("wants_automatic_file_name", cps.Binary(
            "Use the object name for the file name?", True,
            doc="""Use the object name as selected above to generate a file
            name for the spreadsheet. For example, if you select, "Image",
            above and have not checked the "Prepend output file name" option,
            your output file will be named, "Image.csv". You can name
            the file yourself if you leave this box unchecked."""))
        
        group.append("file_name", 
                     cps.Text(
                         "File name:", "DATA.csv",
                         doc="""Enter a file name for the named objects' 
                         measurements. <b>ExportToSpreadsheet</b> will
                         prepend the name of the measurements file to this
                         if you asked to do so above. It will also substitute
                         metadata tokens if you asked to do that."""))
        
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
                  self.add_metadata, self.add_indexes,
                  self.excel_limits, self.pick_columns,
                  self.wants_aggregate_means, self.wants_aggregate_medians,
                  self.wants_aggregate_std, self.directory_choice,
                  self.custom_directory, self.wants_everything]
        for group in self.object_groups:
            result += [group.name, group.previous_file, group.file_name,
                       group.wants_automatic_file_name]
        return result

    def visible_settings(self):
        """Return the settings as seen by the user"""
        result = [self.delimiter, self.prepend_output_filename,
                  self.directory_choice]
        if self.directory_choice in (DIR_CUSTOM, DIR_CUSTOM_WITH_METADATA):
            result += [self.custom_directory]
        result += [ self.add_metadata, self.excel_limits, self.pick_columns,
                    self.wants_aggregate_means, self.wants_aggregate_medians,
                    self.wants_aggregate_std, self.wants_everything]
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
        '''Test the module settings to make sure they are internally consistent
        
        '''
        if (len(self.delimiter.value) != 1 and
            not self.delimiter.value in (DELIMITER_TAB, DELIMITER_COMMA)):
            raise cps.ValidationError("The CSV field delimiter must be a single character", self.delimiter)
        if pipeline.in_batch_mode() and self.pick_columns:
            raise cps.ValidationError("You can't chose columns in batch mode",
                                      self.pick_columns)

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
        if self.directory_choice == DIR_CUSTOM_WITH_METADATA:
            tags += cpmeas.find_metadata_tokens(self.custom_directory.value)
        metadata_groups = workspace.measurements.group_by_metadata(tags)
        for metadata_group in metadata_groups:
            if len(object_names) == 1 and object_names[0] == IMAGE:
                self.make_image_file(file_name, metadata_group.indexes, 
                                     workspace)
            else:
                self.make_object_file(object_names, file_name, 
                                      metadata_group.indexes, workspace)
    
    def make_full_filename(self, file_name, 
                           workspace = None, image_set_index = None):
        """Convert a file name into an absolute path
        
        We do a few things here:
        * apply metadata from an image set to the file name if an 
          image set is specified
        * change the relative path into an absolute one using the "." and "&"
          convention
        * Create any directories along the path
        """
        if not image_set_index is None:
            file_name = workspace.measurements.apply_metadata(file_name,
                                                              image_set_index)
        if self.directory_choice == DIR_DEFAULT_OUTPUT:
            file_name = get_absolute_path(file_name,
                                          abspath_mode = ABSPATH_OUTPUT)
        elif self.directory_choice == DIR_DEFAULT_IMAGE:
            file_name = get_absolute_path(file_name,
                                          abspath_mode = ABSPATH_IMAGE)
        else:
            path = self.custom_directory.value
            if (self.directory_choice == DIR_CUSTOM_WITH_METADATA and
                workspace is not None and image_set_index is not None):
                path = workspace.measurements.apply_metadata(path,
                                                             image_set_index)
            file_name = os.path.join(path, file_name)
            file_name = get_absolute_path(file_name,
                                          abspath_mode = ABSPATH_OUTPUT)
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
        file_name = self.make_full_filename(file_name)
        fd = open(file_name,"wb")
        try:
            writer = csv.writer(fd,delimiter=self.delimiter_char)
            m = workspace.measurements
            for feature_name in m.get_feature_names(EXPERIMENT):
                writer.writerow((feature_name, 
                                 m.get_all_measurements(EXPERIMENT, 
                                                        feature_name)))
        finally:
            fd.close()
    
    def make_image_file(self, file_name, image_set_indexes, workspace):
        """Make a file containing image measurements
        
        file_name - create a file with this name
        image_set_indexes - indexes of the image sets whose data gets
                            extracted
        workspace - workspace containing the measurements
        """
        file_name = self.make_full_filename(file_name, workspace,
                                            image_set_indexes[0])
        fd = open(file_name,"wb")
        try:
            writer = csv.writer(fd,delimiter=self.delimiter_char)
            m = workspace.measurements
            image_features = m.get_feature_names(IMAGE)
            image_features.insert(0, IMAGE_NUMBER)
            for index in image_set_indexes:
                aggs = []
                if self.wants_aggregate_means:
                    aggs.append(cpmeas.AGG_MEAN)
                if self.wants_aggregate_medians:
                    aggs.append(cpmeas.AGG_MEDIAN)
                if self.wants_aggregate_std:
                    aggs.append(cpmeas.AGG_STD_DEV)
                agg_measurements = m.compute_aggregate_measurements(index,
                                                                    aggs)
                if index == image_set_indexes[0]:
                    ordered_agg_names = list(agg_measurements.keys())
                    ordered_agg_names.sort()
                    image_features += ordered_agg_names
                    image_features.sort()
                    image_features = self.user_filter_columns(workspace.frame,
                                                              "Image CSV file columns",
                                                              image_features)
                    if image_features is None:
                        return
                    writer.writerow(image_features)
                row = [ index+1
                       if feature_name == IMAGE_NUMBER
                       else agg_measurements[feature_name]
                       if agg_measurements.has_key(feature_name)
                       else m.get_measurement(IMAGE, feature_name, index)
                       for feature_name in image_features]
                row = ['' if x is None
                       else x if np.isscalar(x) 
                       else x[0] for x in row]
                writer.writerow(row)
        finally:
            fd.close()
        
    def make_object_file(self, object_names, file_name, 
                         image_set_indexes, workspace):
        """Make a file containing object measurements
        
        object_names - sequence of names of the objects whose measurements
                       will be included
        file_name - create a file with this name
        image_set_indexes - indexes of the image sets whose data gets
                            extracted
        workspace - workspace containing the measurements
        """
        file_name = self.make_full_filename(file_name, workspace,
                                            image_set_indexes[0])
        fd = open(file_name,"wb")
        try:
            writer = csv.writer(fd,delimiter=self.delimiter_char)
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
                ofeatures = [(object_name, feature_name)
                             for feature_name in m.get_feature_names(object_name)]
                ofeatures.sort()
                features += ofeatures
            features = self.user_filter_columns(workspace.frame,
                                                "Select columns for %s"%(file_name),
                                                ["%s:%s"%x for x in features])
            features = [x.split(':') for x in features]
            #
            # We write the object names in the first row of headers if there are
            # multiple objects. Otherwise, we just write the feature names
            #
            for i in (0,1) if len(object_names) > 1 else (1,):
                writer.writerow([x[i] for x in features])
            for img_index in image_set_indexes:
                object_count =\
                     np.max([m.get_measurement(IMAGE, "Count_%s"%name, img_index)
                             for name in object_names])
                object_count = int(object_count)
                columns = [np.repeat(img_index+1, object_count)
                           if feature_name == IMAGE_NUMBER
                           else np.arange(1,object_count+1) 
                           if feature_name == OBJECT_NUMBER
                           else np.repeat(m.get_measurement(IMAGE, feature_name,
                                                            img_index), 
                                          object_count)
                           if object_name == IMAGE
                           else m.get_measurement(object_name, feature_name, 
                                                  img_index)
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
    
    def user_filter_columns(self, frame, title, columns):
        """Display a user interface for column selection"""
        if (frame is None or
            (self.pick_columns.value == False and
            (self.excel_limits.value == False or len(columns) < 256))):
            return columns
        
        dlg = wx.Dialog(frame,title = title,
                        style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        sizer = wx.BoxSizer(wx.VERTICAL)
        dlg.SetSizer(sizer)
        list_box = wx.CheckListBox(dlg, choices=columns)
        list_box.SetChecked(range(len(columns)))
        sizer.Add(list_box,1,wx.EXPAND|wx.ALL,3)
        sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(sub_sizer,0,wx.EXPAND)
        count_text = wx.StaticText(dlg,label="%d columns selected"%len(columns))
        sub_sizer.Add(count_text,0,wx.EXPAND|wx.ALL,3)
        select_all_button = wx.Button(dlg, label="All")
        sub_sizer.Add(select_all_button, 0, wx.ALIGN_LEFT|wx.ALL,3)
        select_none_button = wx.Button(dlg, label="None")
        sub_sizer.Add(select_none_button, 0, wx.ALIGN_LEFT|wx.ALL,3)
        def check_all(event):
            for i in range(len(columns)):
                list_box.Check(i, True)
            recount(event)
        def uncheck_all(event):
            for i in range(len(columns)):
                list_box.Check(i, False)
            recount(event)
        def recount(event):
            count = 0
            for i in range(len(columns)):
                if list_box.IsChecked(i):
                    count += 1
            count_text.Label = "%d columns selected"%(count)
        dlg.Bind(wx.EVT_BUTTON, check_all, select_all_button)
        dlg.Bind(wx.EVT_BUTTON, uncheck_all, select_none_button)
        dlg.Bind(wx.EVT_CHECKLISTBOX, recount, list_box)
        button_sizer = wx.StdDialogButtonSizer()
        button_sizer.AddButton(wx.Button(dlg,wx.ID_OK))
        button_sizer.AddButton(wx.Button(dlg,wx.ID_CANCEL))
        button_sizer.Realize()
        sizer.Add(button_sizer,0,wx.EXPAND|wx.ALL,3)
        if dlg.ShowModal() == wx.ID_OK:
            return [columns[i] for i in range(len(columns))
                    if list_box.IsChecked(i)] 
    
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
        if self.directory_choice == DIR_DEFAULT_OUTPUT:
            self.directory_choice.value = DIR_CUSTOM
            path = '.'
        elif self.directory_choice == DIR_DEFAULT_IMAGE:
            self.directory_choice.value = DIR_CUSTOM
            path = '&'
        elif self.directory_choice == DIR_CUSTOM_WITH_METADATA:
            # The patterns, "\g<...>" and "\(?", need to be protected
            # from backslashification.
            path = self.custom_directory.value
            end_new_style = path.find("\\g<")
            end_old_style = path.find("\(?")
            end = (end_new_style 
                   if (end_new_style != -1 and 
                       (end_old_style == -1 or end_old_style > end_new_style))
                   else end_old_style)
            if end != -1:
                pre_path = path[:end]
                pre_path = get_absolute_path(pre_path, 
                                             abspath_mode = ABSPATH_OUTPUT)
                pre_path = fn_alter_path(pre_path)
                path = pre_path + path[end:]
                self.custom_directory.value = path
                return True
        else:
            path = self.custom_directory.value
        path = get_absolute_path(path, abspath_mode = ABSPATH_OUTPUT)
        self.custom_directory.value = fn_alter_path(path)
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
                directory_choice = DIR_DEFAULT_OUTPUT
            elif setting_values[8] == '&':
                directory_choice = DIR_DEFAULT_IMAGE
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
            setting_values = (setting_values[:9] + [DIR_DEFAULT_OUTPUT, "."] +
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
            
        return setting_values, variable_revision_number, from_matlab

def is_object_group(group):
    """True if the group's object name is not one of the static names"""
    return not group.name.value in (IMAGE, EXPERIMENT)

class EEObjectNameSubscriber(cps.ObjectNameSubscriber):
    """ExportToExcel needs to prepend "Image" and "Experiment" to the list of objects
    
    """
    def get_choices(self, pipeline):
        choices = [ IMAGE, EXPERIMENT]
        choices += cps.ObjectNameSubscriber.get_choices(self, pipeline)
        return choices

ExportToExcel = ExportToSpreadsheet
