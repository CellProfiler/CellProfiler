'''<b>Input External</b> specifies the image names that will be pulled from
external sources (e.g., Java)
<hr>
<b>Input External</b> is a helper module for ImageJ. <b>Do not add it to a pipeline.</b>

See also <b>RunImageJ</b>'''

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
        self.image_names = []
        self.add_image(False)
        self.add_button = cps.DoSomething('', 'Add another image name', self.add_image)
    
    def settings(self):
        return [x.image_name for x in self.image_names]
    
    def visible_settings(self):
        result = []
        for group in self.image_names:
            result += group.visible_settings()
        result += [self.add_button]
        return result
    
    def add_image(self, can_remove=True):
        '''Add an image to the list of image names'''
        group = cps.SettingsGroup()
        group.append('divider', cps.Divider(line=False))
        group.append('image_name', cps.ExternalImageNameProvider('Give this image a name'))
        if can_remove:
            group.append('remover', cps.RemoveSettingButton('','Remove this image name', self.image_names, group))
        self.image_names.append(group)
        
    def prepare_settings(self, setting_values):
        while len(setting_values) < len(self.image_names):
            del self.image_names[-1]
        while len(setting_values) > len(self.image_names):
            self.add_image()
        
    def run(self, workspace):
        pass