'''<b>RenameOrRenumberFiles</b> - Renames or renumbers files on the hard drive.
<hr>

This file renaming utility adjusts text within image file names. 
<i style='color:red'>Be very careful with this module because its purpose is 
to rename (= overwrite) files!!</i> You will have the opportunity to confirm 
the name change for the first cycle only. The folder containing the files must 
not contain subfolders or the subfolders and their contents will also be
renamed. The module will not rename the file in test mode. You should use
test mode to ensure that the module settings are correct.

You can use this module to standardize the number of characters in your
file names and to remove unwanted characters from your file names. This is
especially useful if you want file names that have numbers in them to appear
in numerical order when processed by LoadImages.<br>
<br>
Examples:<br>
<br>
Renumber:<br>
DrosDAPI_1.tif    -> DrosDAPI_001.tif<br>
DrosDAPI_10.tif   -> DrosDAPI_010.tif<br>
DrosDAPI_100.tif  -> DrosDAPI_100.tif<br>
(to accomplish this, retain 4 characters at the end, retain 9 characters
at the beginning, and use 3 numerical digits between).<br>
<br>
Renumbering is especially useful when numbers within image filenames do
not have a minimum number of digits and thus appear out of order when
listed in some Unix/Mac OSX systems. For example, on some systems, files
would appear like this and be measured out of expected sequence by
CellProfiler:<br>
DrosDAPI_1.tif<br>
DrosDAPI_10.tif<br>
DrosDAPI_2.tif<br>
DrosDAPI_3.tif<br>
DrosDAPI_4.tif<br>
...<br>
<br>
Rename:<br>
1DrosophilaDAPI_1.tif    -> 1DrosDP_1.tif<br>
2DrosophilaDAPI_10.tif   -> 2DrosDP_10.tif<br>
3DrosophilaDAPI_100.tif  -> 3DrosDP_100.tif<br>
(to accomplish this, retain 4 characters at the end, retain 5 characters
at the beginning, enter "DP" as text to place between, and leave
numerical digits as is).
'''

__version__="$Revision$"

import os

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps

A_RENUMBER = "Renumber"
A_DELETE = "Delete"

class RenameOrRenumberFiles(cpm.CPModule):
    
    module_name = "RenameOrRenumberFiles"
    category = "File Processing"
    variable_revision_number = 1
    
    def create_settings(self):
        '''Create the settings for the module's UI'''
        self.warning = cps.Divider(
            "This module allows you to rename (overwrite) your files. Please "
            "see the help for this module for warnings.")
        self.image_name = cps.FileImageNameSubscriber(
            'Image name:','None',
            doc="""This is the name of the image associated with the file
            you want to rename. It should be an image loaded by 
            <b>LoadImages</b>, <b>LoadData</b> or <b>LoadSingleImage</b>.
            Be very careful because you will be renaming this file!""")
        self.number_characters_prefix = cps.Integer(
            "Number of characters to retain at start of file name:", 6,
            minval=0,
            doc="""This is the number of characters at the start of the old
            file name that will be copied over to the new file name. For
            instance, if this setting is "6" and the file name is 
            "Image-734.tif", the output file name will also start with 
            "Image-".""")
        self.number_characters_suffix = cps.Integer(
            "Number of characters to retain at the end of file name:", 4,
            minval=0,
            doc="""This is the number of characters at the end of the old
            file name that will be copied over to the new file name. For
            instance, if this setting is "4" and the file name is
            "Image-734.tif", the output file name will also end with ".tif".""")
        self.action = cps.Choice(
            "What do you want to do with the remaining characters?",
            [A_RENUMBER, A_DELETE],
            doc ="""You can either treat the characters between the start and
            end as numbers or you can delete them. If you treat them as numbers,
            you will be given the opportunity to pad the numbers with zeros
            so that all of your file names will have a uniform length. For
            instance, if you were to renumber the highlighted portion of the
            file, "Image-<u>734</u>.tif", using four digits, the result would
            be, "Image-0734.tif".""")
        self.number_digits = cps.Integer(
            "How many numerical digits would you like to use?",
            4, minval=0,
            doc="""Use this setting to pad numbers with zeros so that they
            all have a uniform number of characters. For instance, padding
            with four digits has the following result:<br>
            <code><table>
            <tr><th>Original</th><th>Padded</th></tr>
            <tr><td>1</td><td>0001</td></tr>
            <tr><td>10</td><td>0010</td></tr>
            <tr><td>100</td><td>0100</td></tr>
            <tr><td>1000</td><td>1000</td></tr>
            </table></code>""")
        self.wants_text = cps.Binary(
            "Do you want to add text to the file name?", False,
            doc="""You can check this setting if you want to add text
            to the file name. If you chose "Renumber" above,
            <b>RenameOrRenumberFiles</b> will add the text after your number.
            If you chose "Delete", <b>RenameOrRenumberFiles</b> will replace
            the deleted text with the text you enter here.""")
        self.text_to_add = cps.Text(
            "Replacement text","",
            doc="""This is the text that appears either after your number or
            instead of the deleted text.""")
            
    def settings(self):
        '''Return settings in the order that they should appear in pipeline'''
        return [ self.image_name, self.number_characters_prefix,
                 self.number_characters_suffix, self.action, 
                 self.number_digits, self.wants_text, self.text_to_add]
    
    def visible_settings(self):
        '''Return the settings to display in the GUI'''
        result = [ self.warning, self.image_name, self.number_characters_prefix,
                   self.number_characters_suffix, self.action]
        if self.action == A_RENUMBER:
            result += [self.number_digits]
        result += [self.wants_text]
        if self.wants_text:
            result += [self.text_to_add]
        return result
    
    def is_interactive(self):
        '''Tell everyone that the GUI is not an interactive one'''
        return False
    
    def run(self, workspace):
        '''Run on an image set'''
        image_name = self.image_name.value
        m = workspace.measurements
        #
        # Get the path and file names from the measurements
        #
        path = m.get_current_image_measurement('PathName_%s' % image_name)
        file_name = m.get_current_image_measurement('FileName_%s' % image_name)
        #
        # Pull out the prefix, middle and suffix from the file name
        #
        prefix = file_name[:self.number_characters_prefix.value]
        if self.number_characters_suffix.value == 0:
            suffix = ""
            middle = file_name[self.number_characters_prefix.value:]
        else:
            suffix = file_name[-self.number_characters_suffix.value:]
            middle = file_name[self.number_characters_prefix.value:
                               -self.number_characters_suffix.value]
        #
        # Possibly apply the renumbering rule
        #
        if self.action == A_RENUMBER:
            if not middle.isdigit():
                raise ValueError(
                    ('The middle of the filename, "%s", is "%s".\n'
                     "It has non-numeric characters and can't be "
                     "converted to a number") %
                    ( file_name, middle ))
            format = '%0'+str(self.number_digits.value)+'d'
            middle = format % int(middle)
        elif self.action == A_DELETE:
            middle = ""
        else:
            raise NotImplementedError("Unknown action: %s" % self.action.value)
        #
        # Possibly apply the added text
        #
        if self.wants_text:
            middle += self.text_to_add.value
        new_file_name = prefix + middle + suffix
        if workspace.frame is not None:
            workspace.display_data.old_file_name = file_name
            workspace.display_data.new_file_name = new_file_name
        
        if workspace.pipeline.test_mode:
            return
        #
        # Perform the actual renaming
        #
        os.rename(os.path.join(path, file_name),
                  os.path.join(path, new_file_name))
        
    def display(self, workspace):
        '''Display the pathname conversion'''
        statistics = [('Old file name','New file name'),
                      (workspace.display_data.old_file_name,
                       workspace.display_data.new_file_name)]
        figure = workspace.create_or_find_figure(subplots=(1,1))
        figure.subplot_table(0,0,statistics, ratio = (.5, .5))

    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        '''Upgrade settings from previous pipeline versions
        
        setting_values - string values for each of the settings
        variable_revision_number - the revision number of the module at the
                                   time of saving
        module_name - the name of the module that saved the settings
        from_matlab - true if pipeline was saved by CP 1.0
        '''
        if from_matlab and variable_revision_number == 1:
            image_name, number_characters_prefix, number_characters_suffix,\
            text_to_add, number_digits = setting_values
            
            wants_text = text_to_add.lower() != cps.DO_NOT_USE.lower()
            wants_text = cps.YES if wants_text else cps.NO
            
            if number_digits.lower() == cps.DO_NOT_USE.lower():
                number_digits = 4
                action = A_DELETE
            else:
                action = A_RENUMBER
            
            setting_values = [image_name, number_characters_prefix, 
                              number_characters_suffix, action,
                              number_digits, wants_text, text_to_add]
            variable_revision_number = 1
            from_matlab = False
        return setting_values, variable_revision_number, from_matlab