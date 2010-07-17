"""<b>Load Single Image</b> loads a single image for use in all image cycles
<hr>

<p>This module tells CellProfiler where to retrieve a single image and gives the image a
meaningful name by which the other modules can access it. The module 
executes only the first time through the pipeline; thereafter the image
is accessible to all subsequent processing cycles. This is
particularly useful for loading an image like an illumination correction
image for use by the <b>CorrectIlluminationApply</b> module, when that single
image will be used to correct all images in the analysis run.</p>

<h3>Technical note</h3>

For most purposes, you will probably want to use the <b>LoadImages</b> module, not 
<b>LoadSingleImage</b>. The reason is that <b>LoadSingleImage</b> does not actually 
create image sets (or even a single image set). Instead, it adds the single image 
to every image cycle for an <i>already existing</i> image set. Hence 
<b>LoadSingleImage</b> should never be used as the only image-loading module in a 
pipeline; attempting to do so will display a warning message in the module settings. 
<p>If you have a single file to load in the pipeline (and only that file), you 
will want to use <b>LoadImages</b> or <b>LoadData</b> with a single, hardcoded file name. 

See also <b>LoadImages</b>,<b>LoadData</b>.

"""
__version__="$Revision$"

# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Developed by the Broad Institute
# Copyright 2003-2010
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org

import hashlib
import numpy as np
import re
import os

import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.preferences as cpprefs
import cellprofiler.settings as cps
from loadimages import LoadImagesImageProvider
from cellprofiler.gui.help import USING_METADATA_TAGS_REF, USING_METADATA_HELP_REF
from cellprofiler.preferences import standardize_default_folder_names, \
     DEFAULT_INPUT_FOLDER_NAME, DEFAULT_OUTPUT_FOLDER_NAME, \
     IO_FOLDER_CHOICE_HELP_TEXT, IO_WITH_METADATA_HELP_TEXT

DIR_CUSTOM_FOLDER = "Custom folder"
DIR_CUSTOM_WITH_METADATA = "Custom with metadata"

class LoadSingleImage(cpm.CPModule):

    module_name = "LoadSingleImage"
    category = "File Processing"
    variable_revision_number = 2
    def create_settings(self):
        """Create the settings during initialization
        
        """
        self.directory = cps.DirectoryPath(
            "Input image file location",
            doc = '''Select the folder containing the image(s) to be loaded. Generally, 
            it is best to store the image you want to load in either the Default Input or 
            Output Folder, so that the correct image is loaded into the pipeline 
            and typos are avoided. %(IO_FOLDER_CHOICE_HELP_TEXT)s
            
            <p>%(IO_WITH_METADATA_HELP_TEXT)s %(USING_METADATA_TAGS_REF)s. 
            For instance, if you have a "Plate" metadata tag, and your single files are 
            organized in subfolders named with the "Plate" tag, you can select one of the 
            subfolder options and then specify a subfolder name of "\g&lt;Plate&gt;" 
            to get the files from the subfolder associated with that image's plate. The module will 
            substitute the metadata values for the current image set for any metadata tags in the 
            folder name. %(USING_METADATA_HELP_REF)s.</p>'''%globals())
        
        self.file_settings = []
        self.add_file(can_remove = False)
        self.add_button = cps.DoSomething("", "Add another image", self.add_file)

    def add_file(self, can_remove = True):
        """Add settings for another file to the list"""
        group = cps.SettingsGroup()
        if can_remove:
            group.append("divider", cps.Divider(line=False))
        def get_directory_fn():
            return self.directory.get_absolute_path()
        
        group.append("file_name", cps.FilenameText(
            "Filename of the image to load (Include the extension, e.g., .tif)",
            "None",
            metadata=True,
            get_directory_fn = get_directory_fn,
            exts = [("Tagged image file (*.tif)","*.tif"),
                    ("Portable network graphics (*.png)", "*.png"),
                    ("JPEG file (*.jpg)", "*.jpg"),
                    ("Bitmap file (*.bmp)", "*.bmp"),
                    ("GIF file (*.gif)", "*.gif"),
                    ("Matlab image (*.mat)","*.mat"),
                    ("All files (*.*)", "*.*")],doc = """
                    The filename can be constructed in one of two ways:
                    <ul>
                    <li>As a fixed filename (e.g., <i>Exp1_D03f00d0.tif</i>). 
                    <li>Using the metadata associated with an image set in 
                    <b>LoadImages</b> or <b>LoadData</b>. This is especially useful 
                    if you want your output given a unique label according to the
                    metadata corresponding to an image group. The name of the metadata 
                    to substitute is included in a special tag format embedded 
                    in your file specification. %(USING_METADATA_TAGS_REF)s%(USING_METADATA_HELP_REF)s.</li>
                    </ul>
                    <p>Keep in mind that in either case, the image file extension, if any, must be included."""% globals() ))
        
        group.append("image_name", cps.FileImageNameProvider("Name the image that will be loaded", 
                    "OrigBlue", doc = '''What do you want to call the image you are loading? 
                    You can use this name to select the image in downstream modules.'''))
        if can_remove:
            group.append("remove", cps.RemoveSettingButton("", "Remove this image", self.file_settings, group))
        self.file_settings.append(group)

    def settings(self):
        """Return the settings in the order in which they appear in a pipeline file"""
        result = [self.directory]
        for file_setting in self.file_settings:
            result += [file_setting.file_name, file_setting.image_name]
        return result

    def visible_settings(self):
        result = [self.directory]
        for file_setting in self.file_settings:
            result += file_setting.visible_settings()
        result.append(self.add_button)
        return result 

    def prepare_settings(self, setting_values):
        """Adjust the file_settings depending on how many files there are"""
        count = (len(setting_values)-1)/2
        del self.file_settings[count:]
        while len(self.file_settings) < count:
            self.add_file()

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
        '''
        self.directory.alter_for_create_batch_files(fn_alter_path)
        return True

    def get_base_directory(self, workspace):
        return self.directory.get_absolute_path(workspace.measurements)
    
    def get_file_names(self, workspace):
        """Get the files for the current image set
        
        workspace - workspace for current image set
        
        returns a dictionary of image_name keys and file path values
        """
        result = {}
        for file_setting in self.file_settings:
            file_pattern = file_setting.file_name.value
            file_name = workspace.measurements.apply_metadata(file_pattern)
            result[file_setting.image_name.value] = file_name
                
        return result
            
    def run(self, workspace):
        dict = self.get_file_names(workspace)
        root = self.get_base_directory(workspace)
        statistics = [("Image name","File")]
        m = workspace.measurements
        for image_name in dict.keys():
            provider = LoadImagesImageProvider(image_name, root, 
                                               dict[image_name])
            workspace.image_set.providers.append(provider)
            #
            # Add measurements
            #
            m.add_measurement('Image','FileName_'+image_name, dict[image_name])
            m.add_measurement('Image','PathName_'+image_name, root)
            pixel_data = provider.provide_image(workspace.image_set).pixel_data
            digest = hashlib.md5()
            digest.update(np.ascontiguousarray(pixel_data).data)
            m.add_measurement('Image','MD5Digest_'+image_name, digest.hexdigest())
            statistics += [(image_name, dict[image_name])]
        if workspace.frame:
            title = "Load single image: image cycle # %d"%(workspace.measurements.image_set_number+1)
            figure = workspace.create_or_find_figure(title=title,
                                                     subplots=(1,1))
            figure.subplot_table(0,0, statistics)
    
    def get_measurement_columns(self, pipeline):
        columns = []
        for file_setting in self.file_settings:
            image_name = file_setting.image_name.value
            columns += [(cpmeas.IMAGE, '_'.join((feature, image_name)), coltype)
                        for feature, coltype in (
                            ('FileName', cpmeas.COLTYPE_VARCHAR_FILE_NAME),
                            ('PathName', cpmeas.COLTYPE_VARCHAR_PATH_NAME),
                            ('MD5Digest', cpmeas.COLTYPE_VARCHAR_FORMAT % 32))]
        return columns
    
    def validate_module(self, pipeline):
        '''Keep users from using LoadSingleImage to define image sets'''
        if not any([x.is_load_module() for x in pipeline.modules()]):
            raise cps.ValidationError(
                "LoadSingleImage cannot be used to run a pipeline on one "
                "image file. Please use LoadImages or LoadData instead.",
                self.directory)
        
    def upgrade_settings(self, setting_values, variable_revision_number, module_name, from_matlab):
        if from_matlab and variable_revision_number == 4:
            new_setting_values = list(setting_values)
            # The first setting was blank in Matlab. Now it contains
            # the directory choice
            if setting_values[1] == '.':
                new_setting_values[0] = cps.DEFAULT_INPUT_FOLDER_NAME
            elif setting_values[1] == '&':
                new_setting_values[0] = cps.DEFAULT_OUTPUT_FOLDER_NAME
            else:
                new_setting_values[0] = DIR_CUSTOM_FOLDER
            #
            # Remove "Do not use" images
            #
            for i in [8, 6, 4]:
                if new_setting_values[i+1] == cps.DO_NOT_USE:
                    del new_setting_values[i:i+2]
            setting_values = new_setting_values
            from_matlab = False
            variable_revision_number = 1
        #
        # Minor revision: default image folder -> default input folder
        #
        if variable_revision_number == 1 and not from_matlab:
            if setting_values[0].startswith("Default image"):
                dir_choice = cps.DEFAULT_INPUT_FOLDER_NAME
                custom_directory = setting_values[1]
            elif setting_values[0] in (DIR_CUSTOM_FOLDER, DIR_CUSTOM_WITH_METADATA):
                custom_directory = setting_values[1]
                if custom_directory[0] == ".":
                    dir_choice = cps.DEFAULT_INPUT_SUBFOLDER_NAME
                elif custom_directory[0] == "&":
                    dir_choice = cps.DEFAULT_OUTPUT_SUBFOLDER_NAME
                    custom_directory = "."+custom_directory[1:]
                else:
                    dir_choice = cps.ABSOLUTE_FOLDER_NAME
            else:
                dir_choice = setting_values[0]
                custom_directory = setting_values[1]
            directory = cps.DirectoryPath.static_join_string(
                dir_choice, custom_directory)
            setting_values = [directory] + setting_values[2:]
            variable_revision_number = 2
                
        # Standardize input/output directory name references
        SLOT_DIR = 0
        setting_values[SLOT_DIR] = cps.DirectoryPath.upgrade_setting(
            setting_values[SLOT_DIR])
        
        return setting_values, variable_revision_number, from_matlab

