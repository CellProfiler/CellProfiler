'''<b>Display Scatter Plot </b> plots the values for two measurements
<hr>

A scatter plot displays the relationship between two measurements (that is, features) as a 
collection of points.  If there are too many data points on the plot, you should consider 
using <b>DisplayDensityPlot</b> instead.

<p>The module will show a plot shows the values generated for the current cycle. However, 
this module can also be run as a Data Tool, in which you will first be asked
for the output file produced by the analysis run. The resultant plot is 
created from all the measurements collected during the run.</p>

See also <b>DisplayDensityPlot</b>, <b>DisplayHistogram</b>.
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
                            <li><i>Image:</i> For a per-image measurement, one numerical value is 
                            recorded for each image analyzed.
                            Per-image measurements are produced by
                            many modules. Many have <b>MeasureImage</b> in the name but others do not
                            (e.g., the number of objects in each image is a per-image 
                            measurement made by <b>IdentifyObject</b> 
                            modules).</li>
                            <li><i>Object:</i> For a per-object measurement, each identified 
                            object is measured, so there may be none or many 
                            numerical values recorded for each image analyzed. These are usually produced by
                            modules with <b>MeasureObject</b> in the name.</li>
                            </ul>''')
        
        self.x_object = cps.ObjectNameSubscriber(
                            'Select the object to plot on the X-axis',
                            'None',doc = '''<i>(Used only when plotting objects)</i><br>
                            Choose the name of objects identified by some previous 
                            module (such as <b>IdentifyPrimaryObjects</b> or 
                            <b>IdentifySecondaryObjects</b>) whose measurements are to be displayed on the X-axis.''')
        
        self.x_axis = cps.Measurement(
                            'Select the measurement to plot on the X-axis', 
                            self.get_x_object, 'None',doc = '''
                            Choose the measurement (made by a previous 
                            module) to plot on the X-axis.''')
        
        self.y_object = cps.ObjectNameSubscriber(
                            'Select the object to plot on the Y-axis',
                            'None',doc = '''<i>(Used only when plotting objects)</i><br>
                            Choose the name of objects identified by some previous 
                            module (such as <b>IdentifyPrimaryObjects</b> or 
                            <b>IdentifySecondaryObjects</b>) whose measurements are to be displayed on the Y-axis.''')
        
        self.y_axis = cps.Measurement(
                            'Select the measurement to plot on the Y-axis', 
                            self.get_y_object, 'None', doc = '''
                            Choose the measurement (made by a previous 
                            module) to plot on the Y-axis.''')
        
        self.xscale = cps.Choice(
                            'How should the X-axis be scaled?', SCALE_CHOICE, None, doc='''
                            The X-axis can be scaled with either a <i>linear</i> 
                            scale or a <i>log</i> (base 10) scaling. 
                            <p>Log scaling is useful when one of the 
                            measurements being plotted covers a large range of 
                            values; a log scale can bring out features in the 
                            measurements that would not easily be seen if the 
                            measurement is plotted linearly.</p>''')
        
        self.yscale = cps.Choice(
                            'How should the Y-axis be scaled?', SCALE_CHOICE, None, doc='''
                            The Y-axis can be scaled with either a <i>linear</i> 
                            scale or with a <i>log</i> (base 10) scaling. 
                            <p>Log scaling is useful when one of the 
                            measurements being plotted covers a large range of 
                            values; a log scale can bring out features in the 
                            measurements that would not easily be seen if the 
                            measurement is plotted linearly.</p>''')
        
        self.title = cps.Text(
                            'Enter a title for the plot, if desired', '',doc = '''
                            Enter a title for the plot. If you leave this blank,
                            the title will default 
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
            xvals, yvals = np.array([
                (x if np.isscalar(x) else x[0], y if np.isscalar(y) else y[0]) 
                for x,y in zip(xvals, yvals)
                if (x is not None) and (y is not None)]).transpose()
            title = '%s'%(self.title.value)
        else:
            xvals = m.get_current_measurement(self.get_x_object(), self.x_axis.value)
            yvals = m.get_current_measurement(self.get_y_object(), self.y_axis.value)
            title = '%s (cycle %d)'%(self.title.value, workspace.measurements.image_set_number)
        
        if workspace.frame:
            figure = workspace.create_or_find_figure(title="DisplayScatterplot', image cycle #%d"%(
                workspace.measurements.image_set_number),subplots=(1,1))
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
