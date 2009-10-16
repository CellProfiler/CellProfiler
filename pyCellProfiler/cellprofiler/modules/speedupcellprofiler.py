'''<b>Speed Up CellProfiler</b> speeds up cellprofiler by removing images from memory
<hr>
This module removes images from memory which can speed up processing and
prevent memory errors.

Note: CellProfiler 1.0's SpeedUpCellProfiler had an option that let the user
choose how often the output file (DefaultOUT.mat) was saved. This option has been
moved to the preferences settings.
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

import gc
import numpy as np
import uuid

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps

C_REMOVE = "Remove"
C_KEEP = "Keep"
C_ALL = [C_REMOVE, C_KEEP]

'''# of settings in a module independent of the image settings'''
S_NUMBER_OF_PER_MODULE_SETTINGS = 1
'''# of settings per image in the pipeline'''
S_NUMBER_OF_SETTINGS_PER_IMAGE = 1

class SpeedUpCellProfiler(cpm.CPModule):

    module_name = "SpeedUpCellProfiler"
    category = 'Other'
    variable_revision_number = 2
    
    def create_settings(self):
        self.how_to_remove = cps.Choice("Do you want to choose the images to be removed or the images to keep?",
                                        C_ALL,doc="""
            Choose <i>%s</i> to remove some images from memory and keep the rest.
            Choose <i>%s</i> to keep some images and remove the rest."""%
                                (C_REMOVE, C_KEEP))
        
        self.image_names = []
        self.add_image(False)
        self.add_image_button = cps.DoSomething("Add another image",
                                                "Add",
                                                self.add_image)
    
    def add_image(self, can_delete=True):
        '''Add an image to the list of image names
        
        can_delete = True to add GUI elements that let the user delete the image
                   = False if the user should never delete it
        '''
        self.image_names.append(ImageSettings(self.image_names,
                                              self.how_to_remove,
                                              can_delete))
    
    def settings(self):
        result = [self.how_to_remove]
        for image_setting in self.image_names:
            result.extend(image_setting.settings())
        return result
    
    def prepare_to_set_values(self, setting_values):
        image_count = ((len(setting_values) - S_NUMBER_OF_PER_MODULE_SETTINGS) /
                       S_NUMBER_OF_SETTINGS_PER_IMAGE)
        while image_count < len(self.image_names):
            del self.image_names[-1]
        
        while image_count > len(self.image_names):
            self.add_image()
    
    def visible_settings(self):
        result = [self.how_to_remove]
        for image_setting in self.image_names:
            result.extend(image_setting.visible_settings())
        result.append(self.add_image_button)
        return result
    
    def run(self, workspace):
        image_set = workspace.image_set
        image_names = [x.image_name.value for x in self.image_names]
        if self.how_to_remove == C_KEEP:
            names = set([x.name for x in image_set.providers])
            names.difference_update(image_names)
            for name in names:
                image_set.clear_image(name)
        else:
            for name in image_names:
                image_set.clear_image(name)
        gc.collect()
    
    def test_valid(self, pipeline):
        for image_setting in self.image_names:
            image_setting.on_validate()

    def backwards_compatibilize(self, setting_values, variable_revision_number,
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
                setting_values[0] = 'Remove' if (setting_values[0] == 'remove') else 'Keep'
                variable_revision_number = 2
        return setting_values, variable_revision_number, from_matlab
    
class ImageSettings(object):
    def __init__(self, images, how_to_remove, can_delete):
        self.can_delete = can_delete
        self.how_to_remove = how_to_remove
        self.key = uuid.uuid4()
        self.image_name = cps.ImageNameSubscriber(self.image_text, "None")
        if can_delete:
            def remove(images=images, key = self.key):
                index = [x.key for x in images].index(key)
                del images[index]
            self.remove_button = cps.DoSomething("Remove image from list", 
                                                 "Remove",
                                                 remove)
    def settings(self):
        return [self.image_name]
    
    def visible_settings(self):
        if self.can_delete:
            return [self.image_name, self.remove_button]
        else:
            return [self.image_name]
    
    @property
    def image_text(self):
        if self.how_to_remove == C_REMOVE:
            return "What did you call the image that you want to remove from memory?"
        else:
            return "What did you call the image that you want to keep in memory"
    
    def on_validate(self):
        self.image_name.text = self.image_text
        
