# coding=utf-8

"""
**Display Histogram** plots a histogram of the desired measurement.

--------------

A histogram is a plot of tabulated data frequencies (each of which is
shown as a bar) created by binning measurement data for a set of
objects. A two-dimensional histogram can be created using the
**DisplayDensityPlot** module.

The module shows the values generated for the current cycle. However,
this module can also be run as a Data Tool, in which you will first be
asked for the output file produced by the analysis run. The resultant
plot is created from all the measurements collected during the run.

See also **DisplayDensityPlot**, **DisplayScatterPlot**.
"""

import numpy as np

import cellprofiler.image as cpi
import cellprofiler.module as cpm
import cellprofiler.setting as cps
from cellprofiler.setting import YES, NO


class DisplayHistogram(cpm.Module):
    module_name = "DisplayHistogram"
    category = "Data Tools"
    variable_revision_number = 3

    def get_object(self):
        return self.object.value

    def create_settings(self):
        """Create the module settings

        create_settings is called at the end of initialization.
        """
        self.object = cps.ObjectNameSubscriber(
                'Select the object whose measurements will be displayed',
                cps.NONE, doc='''
            Choose the name of objects identified by some previous
            module (such as <b>IdentifyPrimaryObjects</b> or
            <b>IdentifySecondaryObjects</b>) whose measurements are to be displayed.''')

        self.x_axis = cps.Measurement(
                'Select the object measurement to plot',
                self.get_object, cps.NONE, doc='''
            Choose the object measurement made by a previous
            module to plot.''')

        self.bins = cps.Integer(
                'Number of bins', 100, 1, 1000, doc='''
            Enter the number of equally-spaced bins that you want
            used on the X-axis.''')

        self.xscale = cps.Choice(
                'Transform the data prior to plotting along the X-axis?',
                ['no', 'log'], None, doc='''
            The measurement data can be scaled with either a
            linear scale (<i>No</i>) or a <i>log</i> (base 10)
            scaling.
            <p>Log scaling is useful when one of the
            measurements being plotted covers a large range of
            values; a log scale can bring out features in the
            measurements that would not easily be seen if the
            measurement is plotted linearly.<p>''')

        self.yscale = cps.Choice(
                'How should the Y-axis be scaled?',
                ['linear', 'log'], None, doc='''
            The Y-axis can be scaled either with either a <i>linear</i>
            scale or a <i>log</i> (base 10) scaling.
            <p>Log scaling is useful when one of the
            measurements being plotted covers a large range of
            values; a log scale can bring out features in the
            measurements that would not easily be seen if the
            measurement is plotted linearly.</p>''')

        self.title = cps.Text(
                'Enter a title for the plot, if desired', '', doc='''
            Enter a title for the plot. If you leave this blank,
            the title will default
            to <i>(cycle N)</i> where <i>N</i> is the current image
            cycle being executed.''')

        self.wants_xbounds = cps.Binary(
                'Specify min/max bounds for the X-axis?',
                False, doc='''
            Select <i>%(YES)s</i> to specify minimum and maximum values for the
            plot on the X-axis. This is helpful if an outlier bin skews the
            plot such that the bins of interest are no longer visible.''' % globals())

        self.xbounds = cps.FloatRange(
                'Minimum/maximum values for the X-axis')

    def settings(self):
        """Return the settings to be loaded or saved to/from the pipeline

        These are the settings (from cellprofiler.settings) that are
        either read from the strings in the pipeline or written out
        to the pipeline. The settings should appear in a consistent
        order so they can be matched to the strings in the pipeline.
        """
        return [self.object, self.x_axis, self.bins, self.xscale, self.yscale,
                self.title, self.wants_xbounds, self.xbounds]

    def visible_settings(self):
        """The settings that are visible in the UI
        """
        result = [self.object, self.x_axis, self.bins, self.xscale, self.yscale,
                  self.title, self.wants_xbounds]
        if self.wants_xbounds:
            result += [self.xbounds]
        return result

    def run(self, workspace):
        """Run the module
        """
        if self.show_window:
            m = workspace.get_measurements()
            x = m.get_current_measurement(self.get_object(), self.x_axis.value)
            if self.wants_xbounds:
                x = x[x > self.xbounds.min]
                x = x[x < self.xbounds.max]
            workspace.display_data.x = x
            workspace.display_data.title = '%s (cycle %s)' % (self.title.value, workspace.measurements.image_set_number)

    def run_as_data_tool(self, workspace):
        self.run(workspace)

    def display(self, workspace, figure):
        if self.show_window:
            figure.set_subplots((1, 1))
            figure.subplot_histogram(0, 0, workspace.display_data.x,
                                     bins=self.bins.value,
                                     xlabel=self.x_axis.value,
                                     xscale=self.xscale.value,
                                     yscale=self.yscale.value,
                                     title=workspace.display_data.title)

    def backwards_compatibilize(self, setting_values, variable_revision_number,
                                module_name, from_matlab):
        if variable_revision_number == 1:
            # Add bins=100 to second position
            setting_values.insert(2, 100)
            variable_revision_number = 2
        if variable_revision_number == 2:
            # add wants_xbounds=False and xbounds=(0,1)
            setting_values = setting_values + [False, (0, 1)]
            variable_revision_number = 3
        return setting_values, variable_revision_number, from_matlab
