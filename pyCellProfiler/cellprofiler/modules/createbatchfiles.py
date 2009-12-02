'''<b>Create Batch Files</b> produces files that allow individual batches of images to be processed
separately on a cluster of computers.
<hr>
This module creates a file that can be submitted in parallel to a
cluster for faster processing. This module should be placed at the end of
an image processing pipeline.

If your computer mounts the file system differently than the cluster computers,
<b>CreateBatchFiles</b> can replace the neccesary parts of the paths to the 
image and output files. For instance, at the Broad, our Windows machines may 
access files images by mounting the file system using a drive letter, like this:
<i>Z:\imaging_analysis</i>
and our cluster computers access the same file system like this:
<i>/imaging/analysis</i>
In this case, the local root path is <i>Z:\imaging_analysis</i> and the cluster 
root path is <i>/imaging/analysis</i>.
'''
#CellProfiler is distributed under the GNU General Public License.
#See the accompanying file LICENSE for details.
#
#Developed by the Broad Institute
#Copyright 2003-2009
#
#Please see the AUTHORS file for credits.
#
#Website: http://www.cellprofiler.org

__version__="$Revision$"

import numpy as np
import os
import re
import sys
import wx
import zlib

import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.pipeline as cpp
import cellprofiler.settings as cps
import cellprofiler.preferences as cpprefs

'''# of settings aside from the mappings'''
S_FIXED_COUNT = 6
'''# of settings per mapping'''
S_PER_MAPPING = 2

'''Name of the batch data file'''
F_BATCH_DATA = 'Batch_data.mat'

class CreateBatchFiles(cpm.CPModule):
    #
    # How it works:
    #
    # There are two hidden settings: batch_mode and pickled_image_set_list
    # batch_mode controls the mode: False means "save the pipeline" and
    #     True means "run the pipeline"
    # pickled_image_set_list holds the state of the image set list. If
    #     batch_mode is False, we save the state of the image set list in
    #     pickled_image_set_list. If batch_mode is True, we load the state
    #     from pickled_image_set_list.
    #
    module_name = "CreateBatchFiles"
    category = 'File Processing'
    variable_revision_number = 4
    
    #
    def create_settings(self):
        '''Create the module settings and name the module'''
        self.wants_default_output_directory = cps.Binary("Store batch files in default output directory", True,doc="""
                Do you want to store the batch files in the default output directory? 
                Check this box to store batch files in the Default Output directory. Uncheck
                the box to enter the path to the directory that will be used to store
                these files.""")
        
        self.custom_output_directory = cps.Text("Output directory path",
                                                cpprefs.get_default_output_directory(),doc="""
                                                What is the path to the output directory?""")
        
        # Worded this way not because I am windows-centric but because it's
        # easier than listing every other OS in the universe except for VMS
        self.remote_host_is_windows = cps.Binary("Are the cluster computers running Windows?",
                                                 False,doc="""
                Check this box if the cluster computers are running one of the Microsoft
                Windows operating systems. If you check this box, <b>CreateBatchFiles</b> will
                modify all paths to use the Windows file separator (backslash, &#92;). If you
                leave the box unchecked, <b>CreateBatchFiles</b> will modify all paths to use
                the Unix or Macintosh file separator (slash,&#47;).""")
        
        self.batch_mode = cps.Binary("Hidden: in batch mode", False)
        self.default_image_directory = cps.Setting("Hidden: default image directory at time of save",
                                                   cpprefs.get_default_image_directory())
        self.revision = cps.Integer("Hidden: SVN revision number", 0)
        self.mappings = []
        self.add_mapping()
        self.add_mapping_button = cps.DoSomething("Add another path?","Add",
                                                  self.add_mapping, doc="""
                Press this button to add another path mapping.""")
    
    def add_mapping(self):
        group = cps.SettingsGroup()
        group.append("local_directory",
                     cps.Text("Local root path",
                                cpprefs.get_default_image_directory(),doc="""
                                What is the path to files on this computer? 
                                This is the root path on the local machine (i.e., the computer setting up
                                the batch file). If <b>CreateBatchFiles</b> finds
                                any pathname that matches the local root path at the begining, it will replace the
                                start with the cluster root path.
                                <p>For example, if you have mapped the remote cluster machine as:<br><br>
                                <i>Z:\your_data\images</i> (on a Windows machine, for instance)<br><br>
                                and the cluster machine sees the same directory as:<br><br>
                                <i>/server_name/your_name/your_data/images</i><br><br>
                                you would want to put <i>Z:</i> here and <i>/server_name/your_name/</i> 
                                for the cluster path in the next setting."""))

        group.append("remote_directory",
                     cps.Text("Cluster root path",
                                cpprefs.get_default_image_directory(),doc="""
                                What is the path to files on the cluster? This is the cluster 
                                root path, i.e, how the cluster machine sees the
                                top-most directory where your input/output files are stored.
                                <p>For example, if you have mapped the remote cluster machine as:<br><br>
                                <i>Z:\your_data\images</i> (on a Windows machine, for instance)<br><br>
                                and the cluster machine sees the same directory as:<br><br>
                                <i>/server_name/your_name/your_data/images</i><br><br>
                                you would want to put <i>Z:</i> in the previous setting for the
                                local machine path and <i>/server_name/your_name/</i> here. """))
        group.append("remover",
                     cps.RemoveSettingButton("", "Remove above mapping", self.mappings, group))
        group.append("divider", cps.Divider(line=False))
        self.mappings.append(group)

    def settings(self):
        result = [self.wants_default_output_directory,
                  self.custom_output_directory, self.remote_host_is_windows,
                  self.batch_mode, 
                  self.default_image_directory, self.revision]
        for mapping in self.mappings:
            result += [mapping.local_directory, mapping.remote_directory]
        return result
    
    def prepare_settings(self, setting_values):
        if (len(setting_values) - S_FIXED_COUNT) % S_PER_MAPPING != 0:
            raise ValueError("# of mapping settings (%d) "
                             "is not a multiple of %d" %
                             (len(setting_values) - S_FIXED_COUNT, 
                             S_PER_MAPPING))
        mapping_count = (len(setting_values) - S_FIXED_COUNT) / S_PER_MAPPING
        while mapping_count < len(self.mappings):
            del self.mappings[-1]
        
        while mapping_count > len(self.mappings):
            self.add_mapping()
    
    def visible_settings(self):
        result = [self.wants_default_output_directory]
        if not self.wants_default_output_directory.value:
            result += [self.custom_output_directory]
        result += [self.remote_host_is_windows]
        for mapping in self.mappings:
            result += mapping.unpack_group()
        result += [self.add_mapping_button]
        return result
    
    def prepare_run(self, pipeline, image_set_list, frame):
        '''Invoke the image_set_list pickling mechanism and save the pipeline'''
        if self.batch_mode.value:
            self.enter_batch_mode(pipeline, image_set_list)
            return True
        else:
            self.save_pipeline(pipeline, image_set_list, frame)
            return False
    
    def run(self, workspace):
        # all the actual work is done in prepare_run
        pass

    def validate_module(self, pipeline):
        '''Make sure this is the last module in the pipeline'''
        if self.module_num != len(pipeline.modules()):
            raise cps.ValidationError("The CreateBatchFiles module must be "
                                      "the last in the pipeline.",
                                      self.wants_default_output_directory)
    
    def save_pipeline(self, pipeline, image_set_list, frame):
        '''Save the pipeline in Batch_data.mat
        
        Save the pickled image_set_list state in a setting and put this
        module in batch mode.
        '''
        from cellprofiler.utilities.get_revision import version
        assert isinstance(image_set_list, cpi.ImageSetList)
        assert isinstance(pipeline, cpp.Pipeline)
        pipeline = pipeline.copy()
        pipeline.prepare_to_create_batch(image_set_list, self.alter_path)
        bizarro_self = pipeline.module(self.module_num)
        assert isinstance(bizarro_self, CreateBatchFiles)
        state = image_set_list.save_state()
        state = zlib.compress(state)
        bizarro_self.revision.value = version
        bizarro_self.batch_state = np.fromstring(state, np.uint8)
        if self.wants_default_output_directory:
            bizarro_self.custom_output_directory.value = \
                        self.alter_path(cpprefs.get_default_output_directory())
        bizarro_self.default_image_directory.value = \
                    self.alter_path(cpprefs.get_default_image_directory())
        bizarro_self.batch_mode.value = True
        if self.wants_default_output_directory.value:
            path = cpprefs.get_default_output_directory()
        else:
            path = cpprefs.get_absolute_path(self.custom_output_directory.value)
        path = os.path.join(path, F_BATCH_DATA)
        if os.path.exists(path) and frame is not None:
            if (wx.MessageBox("%s already exists. Do you want to overwrite it?"%
                              path,
                              "Overwriting %s" % F_BATCH_DATA,
                              wx.YES_NO, frame) == wx.ID_NO):
                return
        
        pipeline.save(path)
    
    def in_batch_mode(self):
        '''Tell the system whether we are in batch mode on the cluster'''
        return self.batch_mode.value
    
    def enter_batch_mode(self, pipeline, image_set_list):
        '''Restore the image set list from its setting as we go into batch mode'''
        assert isinstance(image_set_list, cpi.ImageSetList)
        assert isinstance(pipeline, cpp.Pipeline)
        state = zlib.decompress(self.batch_state.tostring())
        image_set_list.load_state(state)
        cpprefs.set_default_output_directory(self.custom_output_directory.value)
        cpprefs.set_default_image_directory(self.default_image_directory.value)
    
    def turn_off_batch_mode(self):
        '''Remove any indications that we are in batch mode
        
        This call restores the module to an editable state.
        '''
        self.batch_mode.value = False
        self.batch_state = np.zeros((0,),np.uint8)
    
    def alter_path(self, path, **varargs):
        '''Modify the path passed so that it can be executed on the remote host
        
        path = path to modify
        regexp_substitution - if true, exclude \g<...> from substitution
        '''
        for mapping in self.mappings:
            if sys.platform.startswith('win'):
                # Windows is case-insentitve so do case-insensitve mapping
                if path.upper().startswith(mapping.local_directory.value.upper()):
                    path = (mapping.remote_directory.value +
                            path[len(mapping.local_directory.value):])
            else:
                if path.startswith(mapping.local_directory.value):
                    path = (mapping.remote_directory.value +
                            path[len(mapping.local_directory.value):])
        if self.remote_host_is_windows.value:
            path = path.replace('/','\\')
        elif (varargs.has_key("regexp_substitution") and
                 varargs["regexp_substitution"]):
            path = re.subn('\\\\\\\\','/',path)[0]
            path = re.subn('\\\\(?!g\\<[^>]*\\>)','/',path)[0]
        else:
            path = path.replace('\\','/')
        return path
        
            
    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        if from_matlab and variable_revision_number == 8:
            batch_save_path, old_pathname, new_pathname = setting_values[:3]
            if batch_save_path == '.':
                wants_default_output_directory = cps.YES
                batch_save_path = cpprefs.get_default_output_directory()
            else:
                wants_default_output_directory = cps.NO
            old_pathnames = old_pathname.split(',')
            new_pathnames = new_pathname.split(',')
            if len(old_pathnames) != len(new_pathnames):
                raise ValueError("Number of pathnames does not match. "
                                 "%d local pathnames, but %d remote pathnames" %
                                 (len(old_pathnames), len(new_pathnames)))
            setting_values = [wants_default_output_directory, batch_save_path,
                              cps.NO, cps.NO, ""]
            for old_pathname, new_pathname in zip(old_pathnames, new_pathnames):
                setting_values += [old_pathname, new_pathname]
            from_matlab = False
            variable_revision_number = 1
        if (not from_matlab) and variable_revision_number == 1:
            setting_values = (setting_values[:5] + 
                              [cpprefs.get_default_image_directory()] +
                              setting_values[5:])
            variable_revision_number = 2
        if (not from_matlab) and variable_revision_number == 2:
            from cellprofiler.utilities.get_revision import version
            
            setting_values = (setting_values[:6] + 
                              [version] +
                              setting_values[6:])
            variable_revision_number = 3
        if (not from_matlab) and variable_revision_number == 3:
            # Pickled image list is now the batch state
            self.batch_state = np.fromstring(zlib.compress(setting_values[4]),
                                             np.uint8)
            setting_values = setting_values[:4]+setting_values[5:]
            variable_revision_number = 4
        return setting_values, variable_revision_number, from_matlab
    
