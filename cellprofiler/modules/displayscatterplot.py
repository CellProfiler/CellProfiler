"""
DisplayScatterPlot
==================

**DisplayScatterPlot** plots the values for two measurements.

A scatter plot displays the relationship between two measurements (that
is, features) as a collection of points. If there are too many data
points on the plot, you should consider using **DisplayDensityPlot**
instead.

The module will show a plot of the values generated for the current
cycle. However, this module can also be run as a Data Tool, in which you
will first be asked for the output file produced by the analysis run.
The resulting plot is created from all the measurements collected during
the run.

At this time, the display produced when **DisplayScatterPlot** is run as a
module cannot be saved in the pipeline (e.g., by using **SaveImages**). The
display can be saved manually by selecting the window produced by the
module and clicking the Save icon in its menu bar or by choosing *File
> Save* from CellProfiler's main menu bar.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           NO
============ ============ ===============

See also
^^^^^^^^

See also **DisplayDensityPlot**, **DisplayHistogram**.
"""

import numpy
from cellprofiler_core.constants.measurement import IMAGE

from cellprofiler_core.module import Module
from cellprofiler_core.setting import Measurement
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.subscriber import LabelSubscriber
from cellprofiler_core.setting.text import Text

SOURCE_IM = IMAGE
SOURCE_OBJ = "Object"
SOURCE_CHOICE = [SOURCE_IM, SOURCE_OBJ]
SCALE_CHOICE = ["linear", "log"]


class DisplayScatterPlot(Module):
    module_name = "DisplayScatterPlot"
    category = "Data Tools"
    variable_revision_number = 2

    def create_settings(self):
        self.x_source = Choice(
            "Type of measurement to plot on X-axis",
            SOURCE_CHOICE,
            doc="""\
You can plot two types of measurements:

-  *%(SOURCE_IM)s:* For a per-image measurement, one numerical value is
   recorded for each image analyzed. Per-image measurements are produced
   by many modules. Many have **MeasureImage** in the name but others do
   not (e.g., the number of objects in each image is a per-image
   measurement made by the **Identify** modules).
-  *%(SOURCE_OBJ)s:* For a per-object measurement, each identified
   object is measured, so there may be none or many numerical values
   recorded for each image analyzed. These are usually produced by
   modules with **MeasureObject** in the name.
"""
            % globals(),
        )

        self.x_object = LabelSubscriber(
            "Select the object to plot on the X-axis",
            "None",
            doc="""\
*(Used only when plotting objects)*

Choose the name of objects identified by some previous module (such as
**IdentifyPrimaryObjects** or **IdentifySecondaryObjects**) whose
measurements are to be displayed on the X-axis.
""",
        )

        self.x_axis = Measurement(
            "Select the measurement to plot on the X-axis",
            self.get_x_object,
            "None",
            doc="""Choose the measurement (made by a previous module) to plot on the X-axis.""",
        )

        self.y_source = Choice(
            "Type of measurement to plot on Y-axis",
            SOURCE_CHOICE,
            doc="""\
You can plot two types of measurements:

-  *%(SOURCE_IM)s:* For a per-image measurement, one numerical value is
   recorded for each image analyzed. Per-image measurements are produced
   by many modules. Many have **MeasureImage** in the name but others do
   not (e.g., the number of objects in each image is a per-image
   measurement made by **Identify** modules).
-  *%(SOURCE_OBJ)s:* For a per-object measurement, each identified
   object is measured, so there may be none or many numerical values
   recorded for each image analyzed. These are usually produced by
   modules with **MeasureObject** in the name.
"""
            % globals(),
        )

        self.y_object = LabelSubscriber(
            "Select the object to plot on the Y-axis",
            "None",
            doc="""\
*(Used only when plotting objects)*

Choose the name of objects identified by some previous module (such as
**IdentifyPrimaryObjects** or **IdentifySecondaryObjects**) whose
measurements are to be displayed on the Y-axis.
""",
        )

        self.y_axis = Measurement(
            "Select the measurement to plot on the Y-axis",
            self.get_y_object,
            "None",
            doc="""Choose the measurement (made by a previous module) to plot on the Y-axis.""",
        )

        self.xscale = Choice(
            "How should the X-axis be scaled?",
            SCALE_CHOICE,
            None,
            doc="""\
The X-axis can be scaled with either a *linear* scale or a *log* (base
10) scaling.

Log scaling is useful when one of the measurements being plotted covers
a large range of values; a log scale can bring out features in the
measurements that would not easily be seen if the measurement is plotted
linearly.
""",
        )

        self.yscale = Choice(
            "How should the Y-axis be scaled?",
            SCALE_CHOICE,
            None,
            doc="""\
The Y-axis can be scaled with either a *linear* scale or with a *log*
(base 10) scaling.

Log scaling is useful when one of the measurements being plotted covers
a large range of values; a log scale can bring out features in the
measurements that would not easily be seen if the measurement is plotted
linearly.
""",
        )

        self.title = Text(
            "Enter a title for the plot, if desired",
            "",
            doc="""\
Enter a title for the plot. If you leave this blank, the title will
default to *(cycle N)* where *N* is the current image cycle being
executed.
""",
        )

    def get_x_object(self):
        if self.x_source.value == IMAGE:
            return IMAGE
        return self.x_object.value

    def get_y_object(self):
        if self.y_source.value == IMAGE:
            return IMAGE
        return self.y_object.value

    def settings(self):
        result = [self.x_source, self.x_object, self.x_axis]
        result += [self.y_source, self.y_object, self.y_axis]
        result += [self.xscale, self.yscale, self.title]
        return result

    def visible_settings(self):
        result = [self.x_source]
        if self.x_source.value != IMAGE:
            result += [self.x_object, self.x_axis]
        else:
            result += [self.x_axis]
        result += [self.y_source]
        if self.y_source.value != IMAGE:
            result += [self.y_object, self.y_axis]
        else:
            result += [self.y_axis]
        result += [self.xscale, self.yscale, self.title]
        return result

    def run(self, workspace):
        m = workspace.get_measurements()
        if self.x_source.value == self.y_source.value:
            if self.x_source.value == IMAGE:
                xvals = m.get_all_measurements(IMAGE, self.x_axis.value)
                yvals = m.get_all_measurements(IMAGE, self.y_axis.value)
                xvals, yvals = numpy.array(
                    [
                        (
                            x if numpy.isscalar(x) else x[0],
                            y if numpy.isscalar(y) else y[0],
                        )
                        for x, y in zip(xvals, yvals)
                        if (x is not None) and (y is not None)
                    ]
                ).transpose()
                title = "%s" % self.title.value
            else:
                xvals = m.get_current_measurement(
                    self.get_x_object(), self.x_axis.value
                )
                yvals = m.get_current_measurement(
                    self.get_y_object(), self.y_axis.value
                )
                title = "%s (cycle %d)" % (
                    self.title.value,
                    workspace.measurements.image_set_number,
                )
        else:
            if self.x_source.value == IMAGE:
                xvals = m.get_all_measurements(IMAGE, self.x_axis.value)
                yvals = m.get_current_measurement(
                    self.get_y_object(), self.y_axis.value
                )
                xvals = numpy.array([xvals[0]] * len(yvals))
            else:
                xvals = m.get_current_measurement(
                    self.get_x_object(), self.x_axis.value
                )
                yvals = m.get_all_measurements(IMAGE, self.y_axis.value)
                yvals = numpy.array([yvals[0]] * len(xvals))
            xvals, yvals = numpy.array(
                [
                    (x if numpy.isscalar(x) else x[0], y if numpy.isscalar(y) else y[0])
                    for x, y in zip(xvals, yvals)
                    if (x is not None) and (y is not None)
                ]
            ).transpose()

        if self.show_window:
            workspace.display_data.xvals = xvals
            workspace.display_data.yvals = yvals

    def display(self, workspace, figure):
        xvals = workspace.display_data.xvals
        yvals = workspace.display_data.yvals
        title = "%s" % self.title.value
        figure.set_subplots((1, 1))
        figure.subplot_scatter(
            0,
            0,
            xvals,
            yvals,
            xlabel=self.x_axis.value,
            ylabel=self.y_axis.value,
            xscale=self.xscale.value,
            yscale=self.yscale.value,
            title=title,
        )

    def run_as_data_tool(self, workspace):
        self.run(workspace)

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        """Adjust the setting_values to upgrade from a previous version"""
        if variable_revision_number == 1:
            if setting_values[0] == IMAGE:
                # self.source, self.x_axis, "Image", self.y_axis, self.xscale, self.yscale, self.title
                new_setting_values = [
                    setting_values[0],
                    "None",
                    setting_values[1],
                    IMAGE,
                    "None",
                ] + setting_values[2:]
            else:
                # self.source, self.x_object, self.x_axis, self.y_object, self.y_axis, self.xscale, self.yscale, self.title
                new_setting_values = (
                    setting_values[:3] + [SOURCE_OBJ] + setting_values[3:]
                )
            setting_values = new_setting_values

            variable_revision_number = 2

        return setting_values, variable_revision_number
