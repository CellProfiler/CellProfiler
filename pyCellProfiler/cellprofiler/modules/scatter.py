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
import cellprofiler.measurements as cpmeas

SOURCE_CHOICE = [cpmeas.IMAGE, "Object"]
SCALE_CHOICE = ['linear', 'log']


class ScatterPlot(cpm.CPModule):
    '''
    SHORT DESCRIPTION:
    '''
    module_name = "ScatterPlot"
    category = "Other"
    variable_revision_number = 1
    
    def create_settings(self):
        # XXX: Need docs
        self.source = cps.Choice("Plot an image or object measurement?", SOURCE_CHOICE)
        self.x_object = cps.ObjectNameSubscriber(
            'From which object do you want to plot measurements on the x-axis?',
            'None')
        self.x_axis = cps.Measurement(
            'Which measurement do you want to plot on the x-axis?', 
            self.get_x_object, 'None')
        self.y_object = cps.ObjectNameSubscriber(
            'From which object do you want to plot measurements on the y-axis?',
            'None')
        self.y_axis = cps.Measurement(
            'Which measurement do you want to plot on the y-axis?', 
            self.get_y_object, 'None')
        self.xscale = cps.Choice(
            'How should the X axis be scaled?', SCALE_CHOICE, None)
        self.yscale = cps.Choice(
            'How should the Y axis be scaled?', SCALE_CHOICE, None)
        self.title = cps.Text(
            'Optionally enter a title for this plot.', '')

    def get_x_object(self):
        if self.source.value == cpmeas.IMAGE:
            return cpmeas.IMAGE
        return self.x_object.value
        
    def get_y_object(self):
        if self.source.value == cpmeas.IMAGE:
            return cpmeas.IMAGE
        return self.x_object.value
        
    def settings(self):
        retval = [self.source]
        if self.source.value != cpmeas.IMAGE:
            retval += [self.x_object, self.x_axis, self.y_object]
        else:
            retval += [self.x_axis]
        retval += [self.y_axis, self.xscale, self.yscale, self.title]
        return retval

    def visible_settings(self):
        return self.settings()

    def run(self, workspace):
        m = workspace.get_measurements()
        if self.source.value == cpmeas.IMAGE:
            x = m.get_all_measurements(cpmeas.IMAGE, self.y_axis.value)
            y = m.get_all_measurements(cpmeas.IMAGE, self.y_axis.value)
        else:
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
