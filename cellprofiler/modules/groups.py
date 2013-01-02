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
    variable_revision_number = 1
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
        group.append("image_name", cps.ImageNameSubscriber(
            "Image name", "DNA"))
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
            return sorted(self.metadata_keys.get(group.image_name.value, ["None"]))
        #
        # Unfortunate - an expensive operation to find the possible metadata
        #               keys from one of the columns in an image set.
        # Just fake it into having something that will work
        #
        return [group.metadata_choice.value]
        
    def settings(self):
        result = [self.wants_groups, self.grouping_metadata_count]
        for group in self.grouping_metadata:
            result += [group.image_name, group.metadata_choice]
        return result
            
    def visible_settings(self):
        result = [self.heading_text, self.wants_groups]
        if self.wants_groups:
            result += [self.grouping_text]
            for group in self.grouping_metadata:
                result += [ group.image_name, group.metadata_choice]
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
            self.image_set_channel_descriptors, \
                self.image_set_key_names, \
                self.image_sets = self.pipeline.get_image_sets(workspace, self)
            for i, iscd in enumerate(self.image_set_channel_descriptors):
                column_name = iscd.name
                metadata_keys = set()
                first = True
                for ipds in self.image_sets.values():
                    ipd = ipds[i]
                    if first:
                        metadata_keys = set(ipd.metadata.keys())
                        first = False
                    else:
                        metadata_keys.intersection_update(ipd.metadata.keys())
                self.metadata_keys[column_name] = list(metadata_keys)
            self.update_tables()
            for group in self.grouping_metadata:
                try:
                    group.metadata_choice.test_valid(self.pipeline)
                except:
                    pass # bad pipeline
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
        for group in self.grouping_metadata:
            try:
                group.metadata_choice.test_valid(pipeline)
            except:
                pass
        self.update_tables()
        
    def update_tables(self):
        if self.wants_groups:
            for group in self.grouping_metadata:
                image_name = group.image_name.value
                metadata_key = group.metadata_choice.value
                if image_name not in [
                    iscd.name for iscd in self.image_set_channel_descriptors]:
                    return # invalid image name
                if not self.metadata_keys.has_key(image_name):
                    return
                if metadata_key not in self.metadata_keys[image_name]:
                    return
            self.key_list,self.image_set_groupings = self.compute_groups(
                self.image_set_channel_descriptors, 
                self.image_set_key_names,
                self.image_sets)
            self.grouping_list.clear_columns()
            self.grouping_list.clear_rows()
            self.image_set_list.clear_columns()
            self.image_set_list.clear_rows()
            for i, (idx, key) in enumerate(self.key_list):
                image_name = self.image_set_channel_descriptors[idx].name
                for l in (self.grouping_list, self.image_set_list):
                    l.insert_column(i, "Group: %s:%s" % (image_name, key))
                
            self.grouping_list.insert_column(len(self.key_list), "Count")
            for i, iscd in enumerate(self.image_set_channel_descriptors):
                image_name = iscd.name
                idx = len(self.key_list) + i*2
                self.image_set_list.insert_column(idx, "Path: %s" % image_name)
                self.image_set_list.insert_column(idx+1, "File: %s" % image_name)
                
            for i, key_name in enumerate(self.image_set_key_names):
                idx = (len(self.key_list) + 
                       2 * len(self.image_set_channel_descriptors) + i)
                self.image_set_list.insert_column(idx, key_name)
            
            for j, grouping_keys in enumerate(
                sorted(self.image_set_groupings.keys())):
                count = len(self.image_set_groupings[grouping_keys])
                row = list(grouping_keys) + [str(count)]
                self.grouping_list.data.append(row)
                for image_set_keys in \
                    sorted(self.image_set_groupings[grouping_keys]):
                    ipds = self.image_sets[image_set_keys]
                    row = (list(grouping_keys) + 
                           sum([list(os.path.split(ipd.path)) 
                                for ipd in ipds], []) +
                           list(image_set_keys))
                    self.image_set_list.data.append(row)
            
    def compute_groups(self, image_set_channel_descriptors,
                       image_set_key_names, image_sets):
        '''Return the grouped image sets for the pipeline
        
        Returns a two-tuple. The first entry is a key list giving
        the metadata key and column index in the image set used to fetch
        the grouping metadata. If the entry has a length of zero, there
        is no grouping.
        
        The second entry is a dictionary of grouping key to list of
        image set keys.
        '''
        if not self.wants_groups:
            return [], { ():sorted(image_sets.keys()) }
        
        key_list = []
        image_set_groupings = {}
        column_dict = dict([(iscd.name, i) for i, iscd in 
                            enumerate(image_set_channel_descriptors)])
        for group in self.grouping_metadata:
            key_list.append((column_dict[group.image_name.value],
                             group.metadata_choice.value))
        for keys, ipds in image_sets.iteritems():
            grouping_keys = tuple([ipds[idx].metadata[k]
                                   for idx, k in key_list])
            if not image_set_groupings.has_key(grouping_keys):
                image_set_groupings[grouping_keys] = []
            image_set_groupings[grouping_keys].append(keys)
        for v in image_set_groupings.values():
            v.sort()
        return key_list, image_set_groupings

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
        
        key_list, groupings = self.get_groupings(workspace)
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
        order = np.lexsort((image_numbers, ))
        group_numbers = group_numbers[order]
        group_indexes = group_indexes[order]
        
        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        m.add_all_measurements(cpmeas.IMAGE, cpmeas.GROUP_NUMBER, group_numbers)
        m.add_all_measurements(cpmeas.IMAGE, cpmeas.GROUP_INDEX, group_indexes)
        #
        # Downstream processing requires that image sets be ordered by
        # increasing group number, then increasing group index.
        #
        new_image_numbers = np.zeros(np.max(image_numbers) + 1, int)
        new_image_numbers[image_numbers[order]] = np.arange(len(image_numbers))+1
        m.reorder_image_measurements(new_image_numbers)
        return True
        
    def run(self, workspace):
        pass
    
