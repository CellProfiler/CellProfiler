'''<b>Display Scatter Plot </b> plots the values for two measurements.
<hr>
A scatter plot displays the relationship between two measurements as a 
collection of points, one on each axis. You can specify the type of scaling used
for each axis.

<p>The module shows the values generated for the current cycle. However, 
this module can also be run as a Data Tool, in which you will first be asked
for the output file produced by the analysis run. The resultant plot is 
created from all the measurements collected during the run.</p>

See also <b>DisplayDensitylot</b>, <b>DisplayHistogram</b>
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
import cellprofiler.measurements as cpmeas

SOURCE_CHOICE = [cpmeas.IMAGE, "Object"]
SCALE_CHOICE = ['linear', 'log']

class DisplayScatterPlot(cpm.CPModule):
    
    module_name = "DisplayScatterPlot"
    category = "Data Tools"
    variable_revision_number = 1
    
    def create_settings(self):
        self.source = cps.Choice("Type of measurement to plot", SOURCE_CHOICE,doc = '''
                            You can plot two types of measurements:
                            <ul>
                            <li><i>Image:</i> Per-image measurements are produced from
                            <b>MeasureImage</b>, or modules which generate per-image 
                            measurements (e.g., object counts in <b>IdentifyObject</b> 
                            modules). For these type of measurements, one 
                            measurement is produced for each image analyzed.</li>
                            <li><i>Object:</i> Per-object measurements are produced from
                            <b>MeasureObject<b> modules. For these type of measurements, 
                            one measureement is produced for each identified object.</li>
                            </ul>''')
        
        self.x_object = cps.ObjectNameSubscriber(
                            'Select the object to plot on the x-axis',
                            'None',doc = '''
                            Choose the name of objects identified by some previous 
                            module (such as <b>IdentifyPrimaryObjects</b> or 
                            <b>IdentifySecondaryObjects</b>) to be displayed on the x-axis.''')
        
        self.x_axis = cps.Measurement(
                            'Select the measurement to plot on the x-axis', 
                            self.get_x_object, 'None',doc = '''
                            Choose the image or object measurement made by a previous 
                            module to plot on the x-axis.''')
        
        self.y_object = cps.ObjectNameSubscriber(
                            'Select the object to plot on the y-axis',
                            'None',doc = '''
                            Choose the name of objects identified by some previous 
                            module (such as <b>IdentifyPrimaryObjects</b> or 
                            <b>IdentifySecondaryObjects</b>) to be displayed on the x-axis.''')
        
        self.y_axis = cps.Measurement(
                            'Which measurement do you want to plot on the y-axis?', 
                            self.get_y_object, 'None', doc = '''
                            Choose the image or object measurement made by a previous 
                            module to plot on the y-axis.''')
        
        self.xscale = cps.Choice(
                            'How should the X axis be scaled?', SCALE_CHOICE, None, doc='''
                            The X-axis can be scaled either with a <i>linear</i> 
                            scale or with a <i>log</i> (base 10) scaling. 
                            <p>Using a log scaling is useful when one of the 
                            measurements being plotted covers a large range of 
                            values; a log scale can bring out features in the 
                            measurements that would not easily be seen if the 
                            measurement is plotted linearly.</p>''')
        
        self.yscale = cps.Choice(
                            'How should the Y axis be scaled?', SCALE_CHOICE, None, doc='''
                            The Y-axis can be scaled either with a <i>linear</i> 
                            scale or with a <i>log</i> (base 10) scaling. 
                            <p>Using a log scaling is useful when one of the 
                            measurements being plotted covers a large range of 
                            values; a log scale can bring out features in the 
                            measurements that would not easily be seen if the 
                            measurement is plotted linearly.</p>''')
        
        self.title = cps.Text(
                            'Optionally enter a title for this plot.', '',doc = '''
                            Enter a title for the plot. If no title is desired,
                            leave this setting blank and the title will default 
                            to <i>(cycle N)</i> where <i>N</i> is the current image 
                            cycle being executed.''')

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
            xvals = m.get_all_measurements(cpmeas.IMAGE, self.x_axis.value)
            yvals = m.get_all_measurements(cpmeas.IMAGE, self.y_axis.value)
            title = '%s'%(self.title.value)
        else:
            xvals = m.get_current_measurement(self.get_x_object(), self.x_axis.value)
            yvals = m.get_current_measurement(self.get_y_object(), self.y_axis.value)
            title = '%s (cycle %d)'%(self.title.value, workspace.image_set.number+1)
        
        if workspace.frame:
            figure = workspace.create_or_find_figure(subplots=(1,1))
            figure.subplot_scatter(0, 0, xvals, yvals,
                                   xlabel=self.x_axis.value,
                                   ylabel=self.y_axis.value,
                                   xscale=self.xscale.value,
                                   yscale=self.yscale.value,
                                   title=title)

    def run_as_data_tool(self, workspace):
        self.run(workspace)
    
    def backwards_compatibilize(self, setting_values, variable_revision_number, 
                                module_name, from_matlab):
        return setting_values, variable_revision_number, from_matlab
