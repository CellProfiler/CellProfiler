'''<b>Display Scatter Plot </b> plots the values for two measurements.
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

import numpy as np

import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.settings as cps

SOURCE_IM = cpmeas.IMAGE
SOURCE_OBJ = "Object"
SOURCE_CHOICE = [SOURCE_IM, SOURCE_OBJ]
SCALE_CHOICE = ['linear', 'log']

class DisplayScatterPlot(cpm.CPModule):

    module_name = "DisplayScatterPlot"
    category = "Data Tools"
    variable_revision_number = 2

    def create_settings(self):
        self.x_source = cps.Choice(
            "Type of measurement to plot on X-axis", SOURCE_CHOICE,doc = '''
            You can plot two types of measurements:
            <ul>
            <li><i>%(SOURCE_IM)s:</i> For a per-image measurement, one numerical value is
            recorded for each image analyzed.
            Per-image measurements are produced by
            many modules. Many have <b>MeasureImage</b> in the name but others do not
            (e.g., the number of objects in each image is a per-image
            measurement made by the <b>IdentifyObject</b>
            modules).</li>
            <li><i>%(SOURCE_OBJ)s:</i> For a per-object measurement, each identified
            object is measured, so there may be none or many
            numerical values recorded for each image analyzed. These are usually produced by
            modules with <b>MeasureObject</b> in the name.</li>
            </ul>'''%globals())

        self.x_object = cps.ObjectNameSubscriber(
            'Select the object to plot on the X-axis',
            cps.NONE,doc = '''<i>(Used only when plotting objects)</i><br>
            Choose the name of objects identified by some previous
            module (such as <b>IdentifyPrimaryObjects</b> or
            <b>IdentifySecondaryObjects</b>) whose measurements are to be displayed on the X-axis.''')

        self.x_axis = cps.Measurement(
            'Select the measurement to plot on the X-axis',
            self.get_x_object, cps.NONE,doc = '''
            Choose the measurement (made by a previous
            module) to plot on the X-axis.''')

        self.y_source = cps.Choice("Type of measurement to plot on Y-axis", SOURCE_CHOICE,doc = '''
            You can plot two types of measurements:
            <ul>
            <li><i>%(SOURCE_IM)s:</i> For a per-image measurement, one numerical value is
            recorded for each image analyzed.
            Per-image measurements are produced by
            many modules. Many have <b>MeasureImage</b> in the name but others do not
            (e.g., the number of objects in each image is a per-image
            measurement made by <b>IdentifyObject</b>
            modules).</li>
            <li><i>%(SOURCE_OBJ)s:</i> For a per-object measurement, each identified
            object is measured, so there may be none or many
            numerical values recorded for each image analyzed. These are usually produced by
            modules with <b>MeasureObject</b> in the name.</li>
            </ul>'''%globals())

        self.y_object = cps.ObjectNameSubscriber(
            'Select the object to plot on the Y-axis',
            cps.NONE,doc = '''<i>(Used only when plotting objects)</i><br>
            Choose the name of objects identified by some previous
            module (such as <b>IdentifyPrimaryObjects</b> or
            <b>IdentifySecondaryObjects</b>) whose measurements are to be displayed on the Y-axis.''')

        self.y_axis = cps.Measurement(
            'Select the measurement to plot on the Y-axis',
            self.get_y_object, cps.NONE, doc = '''
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
        if self.x_source.value == cpmeas.IMAGE:
            return cpmeas.IMAGE
        return self.x_object.value

    def get_y_object(self):
        if self.y_source.value == cpmeas.IMAGE:
            return cpmeas.IMAGE
        return self.x_object.value

    def settings(self):
        result = [self.x_source, self.x_object, self.x_axis]
        result += [self.y_source, self.y_object, self.y_axis ]
        result += [self.xscale, self.yscale, self.title]
        return result

    def visible_settings(self):
        result = [self.x_source]
        if self.x_source.value != cpmeas.IMAGE:
            result += [self.x_object, self.x_axis]
        else:
            result += [self.x_axis]
        result += [self.y_source]
        if self.y_source.value != cpmeas.IMAGE:
            result += [self.y_object, self.y_axis ]
        else:
            result += [self.y_axis]
        result += [self.xscale, self.yscale, self.title]
        return result

    def run(self, workspace):
        m = workspace.get_measurements()
        if self.x_source.value == self.y_source.value:
            if self.x_source.value == cpmeas.IMAGE:
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
        else:
            if self.x_source.value == cpmeas.IMAGE:
                xvals = m.get_all_measurements(cpmeas.IMAGE, self.x_axis.value)
                yvals = m.get_current_measurement(self.get_y_object(), self.y_axis.value)
                xvals = np.array([xvals[0]]*len(yvals))
            else:
                xvals = m.get_current_measurement(self.get_x_object(), self.x_axis.value)
                yvals = m.get_all_measurements(cpmeas.IMAGE, self.y_axis.value)
                yvals = np.array([yvals[0]]*len(xvals))
            xvals, yvals = np.array([
                (x if np.isscalar(x) else x[0], y if np.isscalar(y) else y[0])
                for x,y in zip(xvals, yvals)
                if (x is not None) and (y is not None)]).transpose()

        if self.show_window:
            workspace.display_data.xvals = xvals
            workspace.display_data.yvals = yvals

    def display(self, workspace, figure):
        xvals = workspace.display_data.xvals
        yvals = workspace.display_data.yvals
        title = '%s'%(self.title.value)
        figure.set_subplots((1, 1))
        figure.subplot_scatter(0, 0, xvals, yvals,
                               xlabel=self.x_axis.value,
                               ylabel=self.y_axis.value,
                               xscale=self.xscale.value,
                               yscale=self.yscale.value,
                               title=title)

    def run_as_data_tool(self, workspace):
        self.run(workspace)

    def upgrade_settings(self, setting_values, variable_revision_number,
                                module_name, from_matlab):
        """Adjust the setting_values to upgrade from a previous version"""
        if not from_matlab and variable_revision_number == 1:
            if setting_values[0] == cpmeas.IMAGE:
                # self.source, self.x_axis, "Image", self.y_axis, self.xscale, self.yscale, self.title
                new_setting_values = [setting_values[0], cps.NONE, setting_values[1], cpmeas.IMAGE, cps.NONE] + setting_values[2:]
            else:
                # self.source, self.x_object, self.x_axis, self.y_object, self.y_axis, self.xscale, self.yscale, self.title
                new_setting_values = setting_values[:3] + [SOURCE_OBJ] + setting_values[3:]
            setting_values = new_setting_values

            variable_revision_number = 2

        return setting_values, variable_revision_number, from_matlab
