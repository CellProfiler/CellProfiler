'''<b>Rename or Renumber Files</b> renames or renumbers files on the hard drive.
<hr>
This file-renaming utility adjusts text within image file names. 
<i><b>Be very careful with this module because its purpose is 
to rename (and overwrite) files!</b></i>  You will have the opportunity to confirm 
the name change for the first cycle only. If the folder containing the files 
contains subfolders, the subfolders and their contents will also be
renamed. The module will not rename the file in test mode, so you should use
test mode to ensure that the settings are correct.

You can use this module to standardize the number of characters in your
file names and to remove unwanted characters from your file names. This is
especially useful if you want file names that have numbers in them to appear
in numerical order when processed by <b>NamesAndTypes</b>.<br>

While this module performs basic renaming operations, if you can extract
metadata from your images using the <b>Metadata</b> module, you may find using
metadata substitution in <b>SaveImages</b> to be more flexible. Please
refer to metadata handling for those modules and in Help for more details.

<h3>Examples</h3>

<p>Renumbering can be useful when numbers within image filenames do
not have a minimum number of digits and thus appear out of order when
listed in some Unix/Mac OSX systems. For example, on some systems, files
would appear like this and thus be measured differently from the expected
sequence by CellProfiler:</p>
1DrosophilaDAPI_1.tif<br>
1DrosophilaDAPI_10.tif<br>
1DrosophilaDAPI_2.tif<br>
1DrosophilaDAPI_3.tif<br>
1DrosophilaDAPI_4.tif<br>
<p>To renumber the files in the expected order, the numeric digits need to be 
padded with zeros to the same length. In this case, you would want to:
<ul>
<li>Retain 16 characters at the beginning ("1DrosophilaDAPI_").</li>
<li>Retain 4 characters at the end (".tif").</li>
<li>"Renumber" as the handling method using 3 numerical digits.</li>
<li>Set the text addition box to "No".</li>
</ul>

<table border = "1">
<tr><td><i>Original name</i></td><td><i>New name</i></td></tr>
<tr><td>1DrosophilaDAPI_1.tif</td>  <td>1DrosophilaDAPI_001.tif</td></tr>
<tr><td>1DrosophilaDAPI_10.tif</td> <td>1DrosophilaDAPI_010.tif</td></tr>
<tr><td>1DrosophilaDAPI_100.tif</td><td>1DrosophilaDAPI_100.tif</td></tr>
</table>

<p>Renaming can be useful when file names are too long or have characters that 
interfere with other software or file systems. To accomplish the following, you would 
want to:
<ul>
<li>Retain 5 characters at the beginning ("1Dros")</li>
<li>Retain 8 characters at the end ("_<3-digit number>.tif", assuming they have already been renumbered)</li>
<li>Select "Delete" as the handling method</li>
<li>Check the text addition box and enter "DP" as text to place between the retained start and ending strings</li>
</ul>
<table border = "1">
<tr><td><i>Original name</i></td><td><i>New name</i></td></tr>
<tr><td>1DrosophilaDAPI_001.tif</td> <td>1DrosDP_001.tif</td></tr>
<tr><td>1DrosophilaDAPI_010.tif</td> <td>1DrosDP_010.tif</td></tr>
<tr><td>1DrosophilaDAPI_100.tif</td> <td>1DrosDP_100.tif</td></tr>
</table>

See also: <b>NamesAndTypes</b>, <b>SaveImages</b>
'''

# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2014 Broad Institute
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org

import os

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
from cellprofiler.settings import YES, NO

A_RENUMBER = "Renumber"
A_DELETE = "Delete"

class RenameOrRenumberFiles(cpm.CPModule):
    
    module_name = "RenameOrRenumberFiles"
    category = "File Processing"
    variable_revision_number = 2
    
    def create_settings(self):
        '''Create the settings for the module's UI'''
        self.warning = cps.Divider(
            "This module allows you to rename (overwrite) your files. Please "
            "see the help for this module for warnings.")
        
        self.image_name = cps.FileImageNameSubscriber(
            'Select the input image',cps.NONE,doc="""
            Select the images associated with the files
            you want to rename. This should be an image loaded by the
            <b>Input</b> modules.
            Be very careful because you will be renaming these files!""")
        
        self.number_characters_prefix = cps.Integer(
            "Number of characters to retain at start of file name", 6,
            minval=0, doc="""
            Number of characters at the start of the old
            file name that will be copied over verbatim to the new file name. For
            instance, if this setting is "6" and the file name is 
            "Image-734.tif", the output file name will also start with 
            "Image-".""")
        
        self.number_characters_suffix = cps.Integer(
            "Number of characters to retain at the end of file name", 4,
            minval=0,doc="""
            Number of characters at the end of the old
            file name that will be copied over verbatim to the new file name. For
            instance, if this setting is "4" and the file name is
            "Image-734.tif", the output file name will also end with ".tif".""")
        
        self.action = cps.Choice(
            "Handling of remaining characters",
            [A_RENUMBER, A_DELETE], doc ="""
            You can either treat the characters between the start and
            end as numbers or you can delete them. If you treat them as numbers,
            you will be given the opportunity to pad the numbers with zeros
            so that all of your file names will have a uniform length. For
            instance, if you were to renumber the highlighted portion of the
            file "Image-<u>734</u>.tif" using four digits, the result would
            be "Image-0734.tif".""")
        
        self.number_digits = cps.Integer(
            "Number of digits for numbers",
            4, minval=0, doc="""
            <i>(Used only if %(A_RENUMBER)s is selected)</i><br>
            Use this setting to pad numbers with zeros so that they
            all have a uniform number of characters. For instance, padding
            with four digits has the following result:<br>
            <code><table>
            <tr><th>Original</th><th>Padded</th></tr>
            <tr><td>1</td><td>0001</td></tr>
            <tr><td>10</td><td>0010</td></tr>
            <tr><td>100</td><td>0100</td></tr>
            <tr><td>1000</td><td>1000</td></tr>
            </table></code>"""%globals())
        
        self.wants_text = cps.Binary(
            "Add text to the file name?", False,doc="""
            Select <i>%(YES)s</i> if you want to add text
            to the file name. If you had chosen <i>%(A_RENUMBER)s</i> above,
            the module will add the text after your number.
            If you had chosen <i>%(A_DELETE)s</i>, the module will replace
            the deleted text with the text you enter here."""%globals())
        
        self.text_to_add = cps.Text(
            "Replacement text","",doc="""
            <i>(Used only if you chose to add text to the file name)</i><br>
            Enter the text that you want to add to each file name.""")
        
        self.wants_to_replace_spaces = cps.Binary(
            "Replace spaces?", False,doc = """
            Select <i>%(YES)s</i> to replace spaces in the final
            version of the file name with some other text. 
            <p>Select <i>%(NO)s</i> if the file name can have spaces 
            or if none of the file names have spaces.</p>"""%globals())
        
        self.space_replacement = cps.Text(
            "Space replacement", "_",doc = """
            This is the text that will be substituted for spaces
            in your file name.""")
            
    def settings(self):
        '''Return settings in the order that they should appear in pipeline'''
        return [ self.image_name, self.number_characters_prefix,
                 self.number_characters_suffix, self.action, 
                 self.number_digits, self.wants_text, self.text_to_add,
                 self.wants_to_replace_spaces, self.space_replacement]
    
    def visible_settings(self):
        '''Return the settings to display in the GUI'''
        result = [ self.warning, self.image_name, self.number_characters_prefix,
                   self.number_characters_suffix, self.action]
        if self.action == A_RENUMBER:
            result += [self.number_digits]
        result += [self.wants_text]
        if self.wants_text:
            result += [self.text_to_add]
        result += [self.wants_to_replace_spaces]
        if self.wants_to_replace_spaces:
            result += [self.space_replacement]
        return result
    
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
        if self.wants_to_replace_spaces:
            new_file_name = new_file_name.replace(
                " ", self.space_replacement.value)
        if self.show_window:
            workspace.display_data.old_file_name = file_name
            workspace.display_data.new_file_name = new_file_name
        
        if workspace.pipeline.test_mode:
            return
        #
        # Perform the actual renaming
        #
        os.rename(os.path.join(path, file_name),
                  os.path.join(path, new_file_name))
        
    def display(self, workspace, figure):
        '''Display the pathname conversion'''
        figure.set_subplots((1, 1))
        if workspace.pipeline.test_mode:
            figure.subplot_table(
                0, 0, [["Files not renamed in test mode"]])        
        else:
            statistics = [(workspace.display_data.old_file_name,
                                   workspace.display_data.new_file_name)]            
            figure.subplot_table(
                0, 0, statistics, col_labels = ('Old file name','New file name'))

    def validate_module_warnings(self, pipeline):
        '''Warn user re: Test mode '''
        if pipeline.test_mode:
            raise cps.ValidationError(
                "RenameOrRenumberFiles will not rename files in test mode",
                self.image_name)

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
        if (not from_matlab) and variable_revision_number == 1:
            #
            # Added wants_to_replace_spaces and space_replacement
            #
            setting_values = setting_values + [cps.NO, "_"]
            variable_revision_number = 2
        return setting_values, variable_revision_number, from_matlab
