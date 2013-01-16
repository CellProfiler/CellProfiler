"""<b>Groups</b> - organize image sets into groups
<hr>
TO DO: document module
"""
#CellProfiler is distributed under the GNU General Public License.
#See the accompanying file LICENSE for details.
#
#Copyright (c) 2003-2009 Massachusetts Institute of Technology
#Copyright (c) 2009-2013 Broad Institute
#All rights reserved.
#
#Please see the AUTHORS file for credits.
#
#Website: http://www.cellprofiler.org

import logging
logger = logging.getLogger(__name__)
import numpy as np
import os

import cellprofiler.cpmodule as cpm
import cellprofiler.pipeline as cpp
import cellprofiler.settings as cps
import cellprofiler.measurements as cpmeas

class Groups(cpm.CPModule):
    variable_revision_number = 2
    module_name = "Groups"
    category = "File Processing"
    
    IDX_GROUPING_METADATA_COUNT = 1
    
    def create_settings(self):
        self.pipeline = None
        self.metadata_keys = {}
        
        self.heading_text = cps.HTMLText(
            "", content="""
            This module allows you to break the images up into subsets
            (<a href="http://en.wikipedia.org/wiki/Groups">Groups</a>) of
            images which are processed independently (e.g. batches, plates,
            movies, etc.)""", size=(30, 3))
        self.wants_groups = cps.Binary(
            "Do you want to group your images?", False)
        self.grouping_text = cps.HTMLText(
            "", content="""
            Each unique metadata value (or combination of values) 
            will be defined as a group""", size=(30, 2))
        self.grouping_metadata = []
        self.grouping_metadata_count = cps.HiddenCount(
            self.grouping_metadata,
            "grouping metadata count")
        self.add_grouping_metadata(can_remove = False)
        self.add_grouping_metadata_button = cps.DoSomething(
            "Add another metadata item", "Add", self.add_grouping_metadata)

        self.grouping_list = cps.Table("Grouping list", min_size = (300, 100))
        
        self.image_set_list = cps.Table("Image sets")
        
    def add_grouping_metadata(self, can_remove = True):
        group = cps.SettingsGroup()
        self.grouping_metadata.append(group)
        def get_group_metadata_choices(pipeline):
            return self.get_metadata_choices(pipeline, group)
        group.append("metadata_choice", cps.Choice(
            "Metadata category", ["None"],
            choices_fn = get_group_metadata_choices))
        
        group.append("divider", cps.Divider())
        group.can_remove = can_remove
        if can_remove:
            group.append("remover", cps.RemoveSettingButton(
                "Remove the above metadata item", "Remove", 
                self.grouping_metadata, group))
        #
        # Has side effect of updating the metadata choices if the pipeline
        # is defined.
        #
        group.metadata_choice.test_valid(self.pipeline)
        
    def get_metadata_choices(self, pipeline, group):
        if self.pipeline is not None:
            return sorted(self.metadata_keys)
        #
        # Unfortunate - an expensive operation to find the possible metadata
        #               keys from one of the columns in an image set.
        # Just fake it into having something that will work
        #
        return [group.metadata_choice.value]
        
    def settings(self):
        result = [self.wants_groups, self.grouping_metadata_count]
        for group in self.grouping_metadata:
            result += [group.metadata_choice]
        return result
            
    def visible_settings(self):
        result = [self.heading_text, self.wants_groups]
        if self.wants_groups:
            result += [self.grouping_text]
            for group in self.grouping_metadata:
                result += [ group.metadata_choice]
                if group.can_remove:
                    result += [group.remover]
                result += [ group.divider ]
            result += [self.add_grouping_metadata_button, 
                       self.grouping_list, self.image_set_list]
        return result
    
    def prepare_settings(self, setting_values):
        nmetadata = int(setting_values[self.IDX_GROUPING_METADATA_COUNT])
        while len(self.grouping_metadata) > nmetadata:
            del self.grouping_metadata[-1]
            
        while len(self.grouping_metadata) < nmetadata:
            self.add_grouping_metadata()
            
    def on_activated(self, workspace):
        self.pipeline = workspace.pipeline
        self.workspace = workspace
        assert isinstance(self.pipeline, cpp.Pipeline)
        if self.wants_groups:
            self.image_sets_initialized = True
            workspace.refresh_image_set()
            self.metadata_keys = []
            m = workspace.measurements
            assert isinstance(m, cpmeas.Measurements)
            for feature_name in m.get_feature_names(cpmeas.IMAGE):
                if feature_name.startswith(cpmeas.C_METADATA):
                    self.metadata_keys.append(
                        feature_name[(len(cpmeas.C_METADATA)+1):])
            is_valid = True
            for group in self.grouping_metadata:
                try:
                    group.metadata_choice.test_valid(self.pipeline)
                except:
                    is_valid = False
            if is_valid:
                self.update_tables()
        else:
            self.image_sets_initialized = False
        
    def on_deactivated(self):
        self.pipeline = None
        
    def on_setting_changed(self, setting, pipeline):
        if (setting == self.wants_groups and self.wants_groups and
            not self.image_sets_initialized):
            workspace = self.workspace
            self.on_deactivated()
            self.on_activated(workspace)
            
        #
        # Unfortunately, test_valid has the side effect of getting
        # the choices set which is why it's called here
        #
        is_valid = True
        for group in self.grouping_metadata:
            try:
                group.metadata_choice.test_valid(pipeline)
            except:
                is_valid = False
        if is_valid:
            self.update_tables()
        
    def update_tables(self):
        if self.wants_groups:
            try:
                self.workspace.refresh_image_set()
            except:
                return
            m = self.workspace.measurements
            assert isinstance(m, cpmeas.Measurements)
            channel_descriptors = m.get_channel_descriptors()
            
            self.grouping_list.clear_columns()
            self.grouping_list.clear_rows()
            self.image_set_list.clear_columns()
            self.image_set_list.clear_rows()
            metadata_key_names = [group.metadata_choice.value
                                  for group in self.grouping_metadata]
            metadata_feature_names = ["_".join((cpmeas.C_METADATA, key))
                                      for key in metadata_key_names]
            metadata_key_names =  [
                x[(len(cpmeas.C_METADATA)+1):]
                for x in metadata_feature_names]
            image_set_feature_names = [
                cpmeas.GROUP_NUMBER, cpmeas.GROUP_INDEX] + metadata_feature_names
            self.image_set_list.insert_column(0, "Group number")
            self.image_set_list.insert_column(1, "Group index")
            
            for i, key in enumerate(metadata_key_names):
                for l, offset in ((self.grouping_list, 0),
                                  (self.image_set_list, 2)):
                    l.insert_column(i+offset, "Group: %s" % key)
                
            self.grouping_list.insert_column(len(metadata_key_names), "Count")
            
            image_numbers = m.get_image_numbers()
            group_indexes = m[cpmeas.IMAGE, 
                              cpmeas.GROUP_INDEX, 
                              image_numbers][:]
            group_numbers = m[cpmeas.IMAGE, 
                              cpmeas.GROUP_NUMBER, 
                              image_numbers][:]
            counts = np.bincount(group_numbers)
            first_indexes = np.argwhere(group_indexes == 1).flatten()
            group_keys = [
                m[cpmeas.IMAGE, feature, image_numbers]
                for feature in metadata_feature_names]
            k_count = sorted([(group_numbers[i], 
                               [x[i] for x in group_keys], 
                               counts[group_numbers[i]])
                              for i in first_indexes])
            for group_number, group_key_values, c in k_count:
                row = group_key_values + [c]
                self.grouping_list.data.append(row)

            for i, iscd in enumerate(channel_descriptors):
                assert isinstance(iscd, cpp.Pipeline.ImageSetChannelDescriptor)
                image_name = iscd.name
                idx = len(image_set_feature_names)
                self.image_set_list.insert_column(idx, "Path: %s" % image_name)
                self.image_set_list.insert_column(idx+1, "File: %s" % image_name)
                if iscd.channel_type == iscd.CT_OBJECTS:
                    image_set_feature_names.append(
                        cpmeas.C_OBJECTS_PATH_NAME + "_" + iscd.name)
                    image_set_feature_names.append(
                        cpmeas.C_OBJECTS_FILE_NAME + "_" + iscd.name)
                else:
                    image_set_feature_names.append(
                        cpmeas.C_PATH_NAME + "_" + iscd.name)
                    image_set_feature_names.append(
                        cpmeas.C_FILE_NAME + "_" + iscd.name)

            all_features = [m[cpmeas.IMAGE, ftr, image_numbers]
                            for ftr in image_set_feature_names]
            order = np.lexsort((group_indexes, group_numbers))
                
            for idx in order:
                row = [unicode(x[idx]) for x in all_features]
                self.image_set_list.data.append(row)
            
    def get_groupings(self, workspace):
        '''Return the image groupings of the image sets in an image set list
        
        returns a tuple of key_names and group_list:
        key_names - the names of the keys that identify the groupings
        group_list - a sequence composed of two-tuples.
                     the first element of the tuple has the values for
                     the key_names for this group.
                     the second element of the tuple is a sequence of
                     image numbers comprising the image sets of the group
        For instance, an experiment might have key_names of 'Metadata_Row'
        and 'Metadata_Column' and a group_list of:
        [ ({'Row':'A','Column':'01'), [0,96,192]),
          (('Row':'A','Column':'02'), [1,97,193]),... ]
        '''
        if not self.wants_groups:
            return
        key_list = ["_".join((cpmeas.C_METADATA, g.metadata_choice.value))
                    for g in self.grouping_metadata]
        m = workspace.measurements
        if any([key not in m.get_feature_names(cpmeas.IMAGE) for key in key_list]):
            # Premature execution of get_groupings if module is mis-configured
            return None
        return key_list, m.get_groupings(key_list)
    
    def change_causes_prepare_run(self, setting):
        '''Return True if changing the setting passed changes the image sets
        
        setting - the setting that was changed
        '''
        return setting in self.settings()
    
    def is_load_module(self):
        '''Marks this module as a module that affects the image sets
        
        Groups is a load module because it can reorder image sets, but only
        if grouping is turned on.
        '''
        return self.wants_groups.value
    
    def is_input_module(self):
        return True
            
    def prepare_run(self, workspace):
        '''Reorder the image sets and assign group number and index'''
        if workspace.pipeline.in_batch_mode():
            return True
        
        if not self.wants_groups:
            return True
        
        result = self.get_groupings(workspace)
        if result is None:
            return False
        key_list, groupings = result
        #
        # Sort the groupings by key
        #
        groupings = sorted(groupings)
        #
        # Create arrays of group number, group_index and image_number
        #
        group_numbers = np.hstack([
            np.ones(len(image_numbers), int) * (i + 1)
            for i, (keys, image_numbers) in enumerate(groupings)])
        group_indexes = np.hstack([
            np.arange(len(image_numbers)) + 1
            for keys, image_numbers in groupings])
        image_numbers = np.hstack([
            image_numbers for keys, image_numbers in groupings])
        order = np.lexsort((group_indexes, group_numbers ))
        group_numbers = group_numbers[order]
        group_indexes = group_indexes[order]
        
        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        #
        # Downstream processing requires that image sets be ordered by
        # increasing group number, then increasing group index.
        #
        new_image_numbers = np.zeros(np.max(image_numbers) + 1, int)
        new_image_numbers[image_numbers[order]] = np.arange(len(image_numbers))+1
        m.reorder_image_measurements(new_image_numbers)
        m.add_all_measurements(cpmeas.IMAGE, cpmeas.GROUP_NUMBER, group_numbers)
        m.add_all_measurements(cpmeas.IMAGE, cpmeas.GROUP_INDEX, group_indexes)
        return True
        
    def run(self, workspace):
        pass
    
    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        if variable_revision_number == 1:
            #
            # Remove the image name from the settings
            #
            new_setting_values = \
                setting_values[:(self.IDX_GROUPING_METADATA_COUNT+1)]
            for i in range(int(setting_values[self.IDX_GROUPING_METADATA_COUNT])):
                new_setting_values.append(
                    setting_values[self.IDX_GROUPING_METADATA_COUNT + 2 + i*2])
            setting_values = new_setting_values
            variable_revision_number = 2
        return setting_values, variable_revision_number, from_matlab
