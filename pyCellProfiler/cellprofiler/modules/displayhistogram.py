'''<b>Display Histogram </b> plots a histogram of the desired measurement.
<hr>
A histogram is a plot of tabulated data frequencies (each of which is
shown as a bar), created by binning the measurement data. You can specify the 
number of bins to use, as well as whether the data is to be transformed
beforehand.

<p>The module shows the values generated for the current cycle. However, 
this module can also be run as a Data Tool, in which you will first be asked
for the output file produced by the analysis run. The resultant plot is 
created from all the measurements collected during the run.</p>

See also <b>DisplayDensityPlot</b>, <b>DisplayScatterPlot</b>
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

import numpy as np

import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps

class DisplayHistogram(cpm.CPModule):
    
    module_name = "DisplayHistogram"
    category = "Data Tools"
    variable_revision_number = 2
    
    def get_object(self):
        return self.object.value
    
    def create_settings(self):
        self.object = cps.ObjectNameSubscriber(
                            'Select the object','None',
                            doc='''
                            Choose the name of objects identified by some previous 
                            module (such as <b>IdentifyPrimaryObjects</b> or 
                            <b>IdentifySecondaryObjects</b>) to be displayed.''')
        
        self.x_axis = cps.Measurement(
                            'Select the measurement to plot', self.get_object, 'None',
                            doc='''
                            Choose the object measurement made by a previous 
                            module to plot.''')
        
        self.bins = cps.Integer(
                            'Number of bins', 100, 1, 1000,
                            doc='''
                            Enter the number of equally-spaced bins that you want 
                            used on the x-axis.''')
        
        self.xscale = cps.Choice(
                            'Transform the data?', ['no', 'log'], None,
                            doc='''
                            The measurement data can be scaled either with a 
                            linear scale (<i>No</i>) or with a <i>log</i> (base 10) 
                            scaling.
                            <p>Using a log scaling is useful when one of the 
                            measurements being plotted covers a large range of 
                            values; a log scale can bring out features in the 
                            measurements that would not easily be seen if the 
                            measurement is plotted linearly.<p>''')
        
        self.yscale = cps.Choice(
                            'How should the Y axis be scaled?', ['linear', 'log'], None,
                            doc='''
                            The Y-axis can be scaled either with a <i>linear</i> 
                            scale or with a <i>log</i> (base 10) scaling. 
                            <p>Using a log scaling is useful when one of the 
                            measurements being plotted covers a large range of 
                            values; a log scale can bring out features in the 
                            measurements that would not easily be seen if the 
                            measurement is plotted linearly.</p>''')
        
        self.title = cps.Text(
                            'Optionally enter a title for this plot.', '',
                            doc = '''
                            Enter a title for the plot. If no title is desired,
                            leave this setting blank and the title will default 
                            to <i>(cycle N)</i> where <i>N</i> is the current image 
                            cycle being executed.''')
        
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
            
    def run_as_data_tool(self, workspace):
        self.run(workspace)

    def backwards_compatibilize(self, setting_values, variable_revision_number, 
                                module_name, from_matlab):
        if variable_revision_number==1:
            # Add bins
            setting_values = setting_values + [10]
            variable_revision_number = 2
        return setting_values, variable_revision_number, from_matlab

