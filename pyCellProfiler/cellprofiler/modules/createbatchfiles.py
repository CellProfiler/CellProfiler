'''createbatchfiles.py - implements the CreateBatchFiles module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version__="$Revision$"

import os
import wx
import uuid

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
    '''SHORT DESCRIPTION:
Produces files that allow individual batches of images to be processed
separately on a cluster of computers.
***********************************************************************
This module creates a set of files that can be submitted in parallel to a
cluster for faster processing. This module should be placed at the end of
an image processing pipeline.

CreateBatchFiles can rewrite the paths to image and output files if your
computer mounts the file system differently than the cluster computers.
For instance, at the Broad, our Windows machines access files images by
mounting the file system using a drive letter, like this:
Z:\imaging_analysis
and our cluster computers access the same file system like this:
/imaging/analysis
The local root path is "Z:\imaging_analysis" and the cluster root path is
"/imaging/analysis"

Settings:

* Do you want to store the batch files in the default output directory?
Check this box to store batch files in the default output directory. Uncheck
the box to enter the path to the directory that will be used to store
these files.

* Are the cluster computers running Windows?
Check this box if the cluster computers are running one of the Microsoft
Windows operating systems. If you check this box, CreateBatchFiles will
modify all paths to use the Windows file separator (backslash). If you
leave the box unchecked, CreateBatchFiles will modify all paths to use
the Unix or Macintosh file separator (slash).

* What is the path to files on this computer?
This is the local root path as described above. If CreateBatchFiles finds
any path that matches the local root path at the start, it will replace the
start with the cluster root path.

* What is the path to files on the cluster?
This is the cluster root path.

* Add another path?
Press this button to add another path mapping.
'''
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
    category = 'File Processing'
    variable_revision_number = 2
    
    def create_settings(self):
        '''Create the module settings and name the module'''
        self.module_name = "CreateBatchFiles"
        self.wants_default_output_directory = cps.Binary("Do you want to store the batch files in the default output directory?", True)
        self.custom_output_directory = cps.Text("What is the path to the output directory?",
                                                cpprefs.get_default_output_directory())
        # Worded this way not because I am windows-centric but because it's
        # easier than listing every other OS in the universe except for VMS
        self.remote_host_is_windows = cps.Binary("Are the cluster computers running Windows?",
                                                 False)
        self.batch_mode = cps.Binary("Hidden: in batch mode", False)
        self.pickled_image_set_list = cps.Setting("Hidden: contents of image set list","")
        self.default_image_directory = cps.Setting("Hidden: default image directory at time of save",
                                                   cpprefs.get_default_image_directory())
        self.mappings = []
        self.add_mapping()
        self.add_mapping_button = cps.DoSomething("Add another path?","Add",
                                                  self.add_mapping)
    
    def add_mapping(self):
        class Mapping(object):
            def __init__(self, mappings):
                self.key = uuid.uuid4()
                self.local_directory = cps.Text("What is the path to files on this computer?",
                                                cpprefs.get_default_image_directory())
                self.remote_directory = cps.Text("What is the path to files on the cluster?",
                                                 cpprefs.get_default_image_directory())
                def remove_fn(key = self.key, mappings = mappings):
                    index = [mapping.key for mapping in mappings].index(key)
                    del mappings[index]
                self.remove_button = cps.DoSomething("Remove the above directory mapping.",
                                                     "Remove",
                                                     remove_fn)
            
            def settings(self):
                return [self.local_directory, self.remote_directory]
            
            def visible_settings(self):
                return [self.local_directory, self.remote_directory,
                        self.remove_button]
        self.mappings.append(Mapping(self.mappings))
    
    def settings(self):
        result = [self.wants_default_output_directory,
                  self.custom_output_directory, self.remote_host_is_windows,
                  self.batch_mode, self.pickled_image_set_list,
                  self.default_image_directory]
        for mapping in self.mappings:
            result += mapping.settings()
        return result
    
    def prepare_to_set_values(self, setting_values):
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
            result += mapping.visible_settings()
        result += [self.add_mapping_button]
        return result
    
    def backwards_compatibilize(self, setting_values, variable_revision_number,
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
        return setting_values, variable_revision_number, from_matlab
    
    def prepare_run(self, pipeline, image_set_list, frame):
        '''Invoke the image_set_list pickling mechanism and save the pipeline'''
        if self.batch_mode.value:
            self.enter_batch_mode(pipeline, image_set_list)
            return True
        else:
            self.save_pipeline(pipeline, image_set_list, frame)
            return False
    
    def test_valid(self, pipeline):
        '''Make sure this is the last module in the pipeline'''
        if self.module_num != len(pipeline.modules()):
            raise cps.ValidationError("The CreateBatchFiles module must be "
                                      "the last in the pipeline.",
                                      self.wants_default_output_directory)
        return cpm.CPModule.test_valid(self, pipeline)
    
    def save_pipeline(self, pipeline, image_set_list, frame):
        '''Save the pipeline in Batch_data.mat
        
        Save the pickled image_set_list state in a setting and put this
        module in batch mode.
        '''
        assert isinstance(image_set_list, cpi.ImageSetList)
        assert isinstance(pipeline, cpp.Pipeline)
        pipeline = pipeline.copy()
        pipeline.prepare_to_create_batch(image_set_list, self.alter_path)
        bizarro_self = pipeline.module(self.module_num)
        assert isinstance(bizarro_self, CreateBatchFiles)
        bizarro_self.pickled_image_set_list.value = image_set_list.save_state()
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
        image_set_list.load_state(self.pickled_image_set_list.value)
        cpprefs.set_default_output_directory(self.custom_output_directory.value)
        cpprefs.set_default_image_directory(self.default_image_directory.value)
    
    def alter_path(self, path):
        '''Modify the path passed so that it can be executed on the remote host'''
        for mapping in self.mappings:
            if path.startswith(mapping.local_directory.value):
                path = (mapping.remote_directory.value +
                        path[len(mapping.local_directory.value):])
        if self.remote_host_is_windows.value:
            path = path.replace('/','\\')
        return path
        
            