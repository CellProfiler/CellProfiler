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
        # XXX: Need docs
        self.x_object = cps.ObjectNameSubscriber(
            'From which object do you want to plot measurements on the x-axis?',
            'None', 
            doc=''' ''')
        self.x_axis = cps.Measurement(
            'Which measurement do you want to plot on the x-axis?', 
            self.get_x_object, 'None', 
            doc=''' ''')
        self.y_object = cps.ObjectNameSubscriber(
            'From which object do you want to plot measurements on the y-axis?',
            'None', 
            doc=''' ''')
        self.y_axis = cps.Measurement(
            'Which measurement do you want to plot on the y-axis?', 
            self.get_y_object, 'None',
            doc=''' ''')
        self.xscale = cps.Choice(
            'How should the X axis be scaled?', ['linear', 'log'], None,
            doc=''' ''')
        self.yscale = cps.Choice(
            'How should the Y axis be scaled?', ['linear', 'log'], None,
            doc=''' ''')
        self.title = cps.Text(
            'Optionally enter a title for this plot.', '',
            doc=''' ''')
        
    def settings(self):
        return [self.x_object, self.x_axis, self.y_object, self.y_axis,
                self.xscale, self.yscale, self.title]

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
                                   ylabel=self.y_axis.value,
                                   xscale=self.xscale.value,
                                   yscale=self.yscale.value,
                                   title='%s (cycle %s)'%(self.title.value, workspace.image_set.number+1))

    def run_as_data_tool(self, workspace):
        self.run(workspace)
    
    def backwards_compatibilize(self, setting_values, variable_revision_number, 
                                module_name, from_matlab):
        return setting_values, variable_revision_number, from_matlab
