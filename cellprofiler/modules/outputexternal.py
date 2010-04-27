'''OutputExternal.py - Let the user select which images they would like to make
available to external sources (eg: Java)
'''

# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Developed by the Broad Institute
# Copyright 2003-2010
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org

__version__="$Revision$"

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps

class OutputExternal(cpm.CPModule):
    variable_revision_number = 1
    module_name = 'OutputExternal'
    category = 'Other'
    
    def create_settings(self):
        self.image_names = [cps.ExternalImageNameSubscriber('Select an image a name to export')]
        self.add_button = cps.DoSomething('', 'Add another image name', self.add_image)
    
    def settings(self):
        return self.image_names
    
    def visible_settings(self):
        result = [self.image_names[0]]
        for group in self.image_names[1:]:
            result += group.visible_settings()
        result += [self.add_button]
        return result
    
    def add_image(self, can_remove=True):
        '''Add an image to the list of image names'''
        group = cps.SettingsGroup()
        group.append('divider', cps.Divider(line=False))
        group.append('image_name', cps.ExternalImageNameSubscriber('Select an image a name to export'))
        group.append('remover', cps.RemoveSettingButton('', 'Remove this image name', self.image_names, group))
        self.image_names.append(group)
        
    def run(self, workspace):
        pass