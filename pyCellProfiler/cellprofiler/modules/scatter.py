'''scatter.py - the ScatterPlot module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version__="$Revision$"

import numpy as np

import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps


class ScatterPlot(cpm.CPModule):
    '''
    SHORT DESCRIPTION:
    Plots stuff.  duh.
    '''
    module_name = "ScatterPlot"
    category = "Other"
    variable_revision_number = 1
    
    def get_x_object(self):
        return self.x_object.value

    def get_y_object(self):
        return self.y_object.value
    
    def create_settings(self):
        self.x_object = cps.ObjectNameSubscriber("From which object do you want to plot measurements on the x-axis?","None")
        self.x_axis = cps.Measurement('Which measurement do you want to plot on the x-axis?', self.get_x_object, 'None')
        self.divider = cps.Divider()
        self.y_object = cps.ObjectNameSubscriber("From which object do you want to plot measurements on the y-axis?","None")
        self.y_axis = cps.Measurement('Which measurement do you want to plot on the y-axis?', self.get_y_object, 'None')
        
    def settings(self):
        return [self.x_object, self.x_axis, self.divider, 
                self.y_object, self.y_axis]

    def visible_settings(self):
        return self.settings()

    def run(self, workspace):
        m = workspace.get_measurements()
        x = m.get_current_measurement(self.get_x_object(), self.x_axis.value)
        y = m.get_current_measurement(self.get_y_object(), self.y_axis.value)
        
        data = []
        for xx, yy in zip(x,y):
            data += [[xx,yy]]
        
        if workspace.frame:
            figure = workspace.create_or_find_figure(subplots=(1,1))
            figure.subplot_scatter(0, 0, data,
                                   xlabel=self.x_axis.value,
                                   ylabel=self.y_axis.value)
                
    
    def backwards_compatibilize(self, setting_values, variable_revision_number, 
                                module_name, from_matlab):
        return setting_values, variable_revision_number, from_matlab
