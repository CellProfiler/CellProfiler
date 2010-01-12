'''histogram.py - the Histogram module

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


class Histogram(cpm.CPModule):
    '''
    SHORT DESCRIPTION:
    Plots stuff.  duh.
    '''
    module_name = "Histogram"
    category = "Other"
    variable_revision_number = 2
    
    def get_object(self):
        return self.object.value
    
    def create_settings(self):
        # XXX: Need docs
        self.object = cps.ObjectNameSubscriber(
            'From which object do you want to plot measurements?','None',
            doc=''' ''')
        self.x_axis = cps.Measurement(
            'Which measurement do you want to plot?', self.get_object, 'None',
            doc=''' ''')
        self.bins = cps.Integer(
            'How many bins do you want?', 100, 1, 1000,
            doc=''' ''')
        self.xscale = cps.Choice(
            'Transform the data?', ['no', 'log'], None,
            doc=''' ''')
        self.yscale = cps.Choice(
            'How should the Y axis be scaled?', ['linear', 'log'], None,
            doc=''' ''')
        self.title = cps.Text(
            'Optionally enter a title for this plot.', '',
            doc=''' ''')
        
    def settings(self):
        return [self.object, self.x_axis, self.bins, self.xscale, self.yscale,
                self.title]

    def visible_settings(self):
        return self.settings()

    def run(self, workspace):
        if workspace.frame:
            m = workspace.get_measurements()
            x = m.get_current_measurement(self.get_object(), self.x_axis.value)
            figure = workspace.create_or_find_figure(subplots=(1,1))
            figure.subplot_histogram(0, 0, x, 
                                     bins=self.bins.value,
                                     xlabel=self.x_axis.value,
                                     xscale=self.xscale.value,
                                     yscale=self.yscale.value,
                                     title='%s (cycle %s)'%(self.title.value, workspace.image_set.number+1))
            
    
    def backwards_compatibilize(self, setting_values, variable_revision_number, 
                                module_name, from_matlab):
        if variable_revision_number==1:
            # Add bins
            setting_values = setting_values + [10]
            variable_revision_number = 2
        return setting_values, variable_revision_number, from_matlab

