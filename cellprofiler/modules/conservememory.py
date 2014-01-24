'''<b>Conserve Memory</b> speeds up CellProfiler by removing images from memory.
<hr>

This module removes images from memory, which can speed up processing and
prevent out-of-memory errors.

<i>Note:</i> CellProfiler 1.0's <b>SpeedUpCellProfiler</b> had an option that let you 
choose how often the output file of measurements was saved. This option is no longer neccessary since
the output file is automatically updated with each image set.
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


import gc
import numpy as np

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps

C_REMOVE_OLD = "Remove"
C_REMOVE = "Images to remove"
C_KEEP_OLD = "Keep"
C_KEEP = "Images to keep"

'''# of settings in a module independent of the image settings'''
S_NUMBER_OF_PER_MODULE_SETTINGS = 1
'''# of settings per image in the pipeline'''
S_NUMBER_OF_SETTINGS_PER_IMAGE = 1

class ConserveMemory(cpm.CPModule):

    module_name = "ConserveMemory"
    category = 'Other'
    variable_revision_number = 1
    
    def create_settings(self):
        self.how_to_remove = cps.Choice(
            "Specify which images?",
            [C_REMOVE, C_KEEP], doc="""
            You can select from the following options:
            <ul>
            <li><i>%(C_REMOVE)s:</i> Remove some images from memory and keep the rest.</li>
            <li><i>%(C_KEEP)s:</i> Keep some images and remove the rest.</li>
            </ul>"""%globals())
        
        self.spacer_top = cps.Divider(line=False)
        self.image_names = []
        self.add_image(can_remove = False)
        self.spacer_bottom = cps.Divider(line=False)
        self.add_image_button = cps.DoSomething("", "Add another image",
                                                self.add_image)
    

    def query(self):
        if self.how_to_remove == C_REMOVE:
            return "Select image to remove"
        else:
            return "Select image to keep"

    def add_image(self, can_remove = True):
        '''Add an image to the list of image names
        
        can_remove - set this to False to keep from showing the "remove"
                     button for images that must be present.
        '''
        group = cps.SettingsGroup()
        if can_remove:
            group.append("divider", cps.Divider(line=False))
        group.append("image_name", cps.ImageNameSubscriber(self.query(), cps.NONE))
        if can_remove:
            group.append("remover", cps.RemoveSettingButton("",
                                                        "Remove this image",
                                                        self.image_names,
                                                        group))
        self.image_names.append(group)
    
    def settings(self):
        return [self.how_to_remove] + [im.image_name for im in self.image_names]
    
    def prepare_settings(self, setting_values):
        image_count = ((len(setting_values) - S_NUMBER_OF_PER_MODULE_SETTINGS) /
                       S_NUMBER_OF_SETTINGS_PER_IMAGE)
        del self.image_names[image_count:]
        while image_count > len(self.image_names):
            self.add_image()
    
    def visible_settings(self):
        for image_setting in self.image_names:
            image_setting.image_name.text = self.query()

        result = [self.how_to_remove, self.spacer_top]

        for image_setting in self.image_names:
            result += image_setting.visible_settings()
        result += [self.add_image_button]
        return result
    
    def run(self, workspace):
        image_set = workspace.image_set
        image_names = [x.image_name.value for x in self.image_names]
        workspace.display_data.statistics = []
        if self.how_to_remove == C_KEEP:
            all_names = [x.name for x in image_set.providers]
            for name in set(all_names) - set(image_names):
                image_set.clear_image(name)
            for name in image_names:
                workspace.display_data.statistics.append(["Kept %s"%name])
        else:
            for name in image_names:
                image_set.clear_image(name)
                workspace.display_data.statistics.append(["Removed %s"%name])
        gc.collect()
    
    def display(self, workspace, figure):
        figure.set_subplots((1, 1))
        figure.subplot_table(0, 0, workspace.display_data.statistics)   
                
    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        if from_matlab and variable_revision_number == 5:
            new_setting_values = [ C_REMOVE ]
            for image_name in setting_values[2:]:
                if image_name.lower() != cps.DO_NOT_USE.lower():
                    new_setting_values.append(image_name)
            setting_values = new_setting_values
            variable_revision_number = 1
            from_matlab = False
        if (not from_matlab) and variable_revision_number == 1:
            # There was some skew in the capitalization of the first
            # setting.  We rewrite it, but we leave the revision
            # number at 1.
            remap = {'remove' : C_REMOVE, 
                     C_REMOVE_OLD: C_REMOVE,
                     'keep' : C_KEEP,
                     C_KEEP_OLD: C_KEEP
                     }
            if setting_values[0] in remap:
                setting_values[0] = remap[setting_values[0]]

        return setting_values, variable_revision_number, from_matlab
    
SpeedUpCellProfiler = ConserveMemory
