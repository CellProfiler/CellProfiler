"""<b>Groups</b> - organize image sets into groups
<hr>
TO DO: document module
"""
#CellProfiler is distributed under the GNU General Public License.
#See the accompanying file LICENSE for details.
#
#Copyright (c) 2003-2009 Massachusetts Institute of Technology
#Copyright (c) 2009-2012 Broad Institute
#All rights reserved.
#
#Please see the AUTHORS file for credits.
#
#Website: http://www.cellprofiler.org

import logging
logger = logging.getLogger(__name__)

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps

class Groups(cpm.CPModule):
    def create_settings(self):
        self.pipeline = None
        self.metadata_keys = {}
        
        self.heading_text = cps.HTMLText(
            "", content="""
            This module allows you to break the images up into subsets
            (<a href="http://en.wikipedia.org/wiki/Groups">Groups</a>) of
            images which are processed independently (e.g. batches, plates,
            movies, etc.)""", size=(100, 3))
        self.wants_groups = cps.Binary(
            "Do you want to group your images?", False)
        self.heading_text = cps.HTMLText(
            "", content="""
            Each unique metadata value (or combination of values) 
            will be defined as a group""")
        self.grouping_metadata = []
        self.grouping_metadata_count = cps.HiddenCount(
            self.grouping_metadata,
            "grouping metadata count")
        self.add_grouping_metadata(can_remove = False)
        self.add_grouping_metadata_button = cps.DoSomething(
            "Add another metadata item", "Add", self.add_grouping_metadata)

        self.grouping_list = cps.Table("Grouping list")
        
        self.image_set_list = cps.Table("Image sets")
        
    def add_grouping_metadata(self, can_remove = True):
        group = cps.SettingsGroup()
        self.grouping_metadata.append(group)
        group.append("image_name", cps.ImageNameSubscriber(
            "Image name", "DNA"))
        group.append("metadata_choice", cps.Choice(
            "Metadata category", [],
            choices_fn = lambda group = group: self.get_metadata_choices(group)))
        
    def get_metadata_choices(self, group):
        if self.pipeline is not None:
            return self.metadata_keys.get(group.image_name.value, [])
        #
        # Unfortunate - an expensive operation to find the possible metadata
        #               keys from one of the columns in an image set.
        # Just fake it into having something that will work
        #
        return [group.metadata_choice.value]
        
    def settings(self):
        result = [self.wants_groups, self.grouping_metadata_count]
        for group in self.grouping_metadata:
            result += [group.image_name, group.metadata_key]
