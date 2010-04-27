'''InputExternal.py - Let the user create image names that will be pulled from
external sources (eg: Java)
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

class InputExternal(cpm.CPModule):
    variable_revision_number = 1
    module_name = 'InputExternal'
    category = 'Other'
    
    def create_settings(self):
        self.image_names = [cps.ExternalImageNameProvider('Give this image a name')]
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
        group.append('image_name', cps.ExternalImageNameProvider('Give this image a name'))
        group.append('remover', cps.RemoveSettingButton('','Remove this image name', self.image_names, group))
        self.image_names.append(group)
        
    def run(self, workspace):
        pass