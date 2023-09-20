"""
DisplayDensityPlot
==================

**DisplayDensityPlot** plots measurements as a two-dimensional density
plot.

A density plot displays the relationship between two measurements (that
is, features) but instead of showing each data point as a dot, as in a
scatter plot, the data points are binned into an equally-spaced grid of
points, where the color of each point in the grid represents the
tabulated frequency of the measurements within that region of the grid.
A density plot is also known as a 2-D histogram; in a conventional
histogram the height of a bar indicates how many data points fall in
that region. By contrast, in a density plot (2-D histogram), the color
of a portion of the plot indicates the number of data points in that
region.

The module shows the values generated for the current cycle. However,
this module can also be run as a Data Tool, in which case you will first
be asked for the output file produced by the analysis run. The
resulting plot is created from all the measurements collected during
the run.

At this time, the display produced when **DisplayDensityPlot** is run as a
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

See also **DisplayScatterPlot**, **DisplayHistogram**.
"""

import matplotlib.cm

from cellprofiler_core.module import Module
from cellprofiler_core.setting import Measurement
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.subscriber import LabelSubscriber
from cellprofiler_core.setting.text import Integer
from cellprofiler_core.setting.text import Text


class DisplayDensityPlot(Module):
    module_name = "DisplayDensityPlot"
    category = "Data Tools"
    variable_revision_number = 1

    def get_x_object(self):
        return self.x_object.value

    def get_y_object(self):
        return self.y_object.value

    def create_settings(self):
        self.x_object = LabelSubscriber(
            "Select the object to display on the X-axis",
            "None",
            doc="""\
Choose the name of objects identified by some previous module (such as
**IdentifyPrimaryObjects** or **IdentifySecondaryObjects**) whose
measurements are to be displayed on the X-axis.
""",
        )

        self.x_axis = Measurement(
            "Select the object measurement to plot on the X-axis",
            self.get_x_object,
            "None",
            doc="""Choose the object measurement made by a previous module to display on the X-axis.""",
        )

        self.y_object = LabelSubscriber(
            "Select the object to display on the Y-axis",
            "None",
            doc="""\
Choose the name of objects identified by some previous module (such as
**IdentifyPrimaryObjects** or **IdentifySecondaryObjects**) whose
measurements are to be displayed on the Y-axis.
""",
        )

        self.y_axis = Measurement(
            "Select the object measurement to plot on the Y-axis",
            self.get_y_object,
            "None",
            doc="""Choose the object measurement made by a previous module to display on the Y-axis.""",
        )

        self.gridsize = Integer(
            "Select the grid size",
            100,
            1,
            1000,
            doc="""\
Enter the number of grid regions you want used on each
axis. Increasing the number of grid regions increases the
resolution of the plot.""",
        )

        self.xscale = Choice(
            "How should the X-axis be scaled?",
            ["linear", "log"],
            None,
            doc="""\
The X-axis can be scaled either with a *linear* scale or with a *log*
(base 10) scaling.

Using a log scaling is useful when one of the measurements being plotted
covers a large range of values; a log scale can bring out features in
the measurements that would not easily be seen if the measurement is
plotted linearly.
""",
        )

        self.yscale = Choice(
            "How should the Y-axis be scaled?",
            ["linear", "log"],
            None,
            doc="""\
The Y-axis can be scaled either with a *linear* scale or with a *log*
(base 10) scaling.

Using a log scaling is useful when one of the measurements being plotted
covers a large range of values; a log scale can bring out features in
the measurements that would not easily be seen if the measurement is
plotted linearly.
""",
        )

        self.bins = Choice(
            "How should the colorbar be scaled?",
            ["linear", "log"],
            None,
            doc="""\
The colorbar can be scaled either with a *linear* scale or with a *log*
(base 10) scaling.

Using a log scaling is useful when one of the measurements being plotted
covers a large range of values; a log scale can bring out features in
the measurements that would not easily be seen if the measurement is
plotted linearly.
""",
        )

        maps = [m for m in list(matplotlib.cm.datad.keys()) if not m.endswith("_r")]
        maps.sort()

        self.colormap = Choice(
            "Select the color map",
            maps,
            "jet",
            doc="""\
Select the color map for the density plot. See `this page`_ for pictures
of the available colormaps.

.. _this page: http://matplotlib.org/users/colormaps.html
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

    def settings(self):
        return [
            self.x_object,
            self.x_axis,
            self.y_object,
            self.y_axis,
            self.gridsize,
            self.xscale,
            self.yscale,
            self.bins,
            self.colormap,
            self.title,
        ]

    def visible_settings(self):
        return self.settings()

    def run(self, workspace):
        m = workspace.get_measurements()
        x = m.get_current_measurement(self.get_x_object(), self.x_axis.value)
        y = m.get_current_measurement(self.get_y_object(), self.y_axis.value)

        data = []
        for xx, yy in zip(x, y):
            data += [[xx, yy]]

        bins = None
        if self.bins.value != "linear":
            bins = self.bins.value

        if self.show_window:
            workspace.display_data.data = data
            workspace.display_data.bins = bins

    def display(self, workspace, figure):
        data = workspace.display_data.data
        bins = workspace.display_data.bins
        figure.set_subplots((1, 1))
        figure.subplot_density(
            0,
            0,
            data,
            gridsize=self.gridsize.value,
            xlabel=self.x_axis.value,
            ylabel=self.y_axis.value,
            xscale=self.xscale.value,
            yscale=self.yscale.value,
            bins=bins,
            cmap=self.colormap.value,
            title="%s (cycle %s)"
            % (self.title.value, workspace.measurements.image_set_number),
        )

    def run_as_data_tool(self, workspace):
        self.run(workspace)
