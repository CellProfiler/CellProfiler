'''<b>Output External</b> specfies which images can be made
available to external sources (e.g., Java)
<hr>
<b>OutputExternal</b> is a helper module for ImageJ. <b>Do not add it to 
a pipeline.</b>

<p>The <b>InputExternal</b> and <b>OutputExternal</b> modules are 
placeholders if CellProfiler is run programatically. For example,
another program, e.g., a plugin to ImageJ, is provided with 
a CellProfiler pipeline. This program should then replace the input
modules with <b>InputExternal</b> modules and prompt the user what inputs 
should be supplied to the pipeline through <b>InputExternal</b>. The program 
should also specify which inputs are to be sent back to the source program 
via <b>OutputExternal</b>. The calling program would insert the images into 
the image set before running the pipeline and remove the images from the 
image set at the end.</p>

See also <b>RunImageJ</b>
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


import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps

class OutputExternal(cpm.CPModule):
    variable_revision_number = 1
    module_name = 'OutputExternal'
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
        group.append('image_name', cps.ExternalImageNameSubscriber('Select an image a name to export'))
        if can_remove:
            group.append('remover', cps.RemoveSettingButton('', 'Remove this image name', self.image_names, group))
        self.image_names.append(group)
        
    def prepare_settings(self, setting_values):
        while len(setting_values) < len(self.image_names):
            del self.image_names[-1]
        while len(setting_values) > len(self.image_names):
            self.add_image()
        
    def run(self, workspace):
        pass
