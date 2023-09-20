"""
DefineGrid
==========

**DefineGrid** produces a grid of desired specifications either
manually, or automatically based on previously identified objects.

This module defines the location of a grid that can be used by modules
downstream. You can use it in combination with **IdentifyObjectsInGrid**
to measure the size, shape, intensity and texture of each object or
location in a grid. The grid is defined by the location of marker spots
(control spots), which are either indicated manually or found
automatically using previous modules in the pipeline. You can then use
the grid to make measurements (using **IdentifyObjectsInGrid**). If you are using images of
plastic plates, it may be useful to precede this module with an
**IdentifyPrimaryObjects** module to find the plastic plate, followed by
a **Crop** module to remove the plastic edges of the plate, so that the
grid can be defined within the smooth portion of the plate only. If the
plates are not centered in exactly the same position from one image to
the next, this allows the plates to be identified automatically and then
cropped so that the interior of the plates, upon which the grids will be
defined, are always in precise alignment with each other.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           NO
============ ============ ===============

See also
^^^^^^^^

See also **IdentifyObjectsInGrid**.

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  *Rows, Columns*: The number of rows and columns in the grid.
-  *XSpacing, YSpacing:* The spacing in X and Y of the grid elements.
-  *XLocationOfLowestXSpot:* The X coordinate location of the lowest
   spot on the X-axis.
-  *YLocationOfLowestYSpot:* The Y coordinate location of the lowest
   spot on the Y-axis.
"""

import logging

import centrosome.cpmorphology
import numpy
from cellprofiler_core.constants.measurement import COLTYPE_FLOAT
from cellprofiler_core.constants.measurement import COLTYPE_INTEGER
from cellprofiler_core.constants.measurement import IMAGE
from cellprofiler_core.image import Image
from cellprofiler_core.module import Module
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting import Coordinates
from cellprofiler_core.setting import ValidationError
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.subscriber import LabelSubscriber
from cellprofiler_core.setting.text import GridName
from cellprofiler_core.setting.text import ImageName
from cellprofiler_core.setting.text import Integer

from cellprofiler_core.utilities.grid import Grid

LOGGER = logging.getLogger(__name__)

NUM_TOP_LEFT = "Top left"
NUM_BOTTOM_LEFT = "Bottom left"
NUM_TOP_RIGHT = "Top right"
NUM_BOTTOM_RIGHT = "Bottom right"
NUM_BY_ROWS = "Rows"
NUM_BY_COLUMNS = "Columns"

EO_EACH = "Each cycle"
EO_ONCE = "Once"

AM_AUTOMATIC = "Automatic"
AM_MANUAL = "Manual"

MAN_MOUSE = "Mouse"
MAN_COORDINATES = "Coordinates"

FAIL_NO = "No"
FAIL_ANY_PREVIOUS = "Use any previous grid"
FAIL_FIRST = "Use the first cycle's grid"

"""The module dictionary keyword of the first or most recent good gridding"""
GOOD_GRIDDING = "GoodGridding"

"""Measurement category for this module"""
M_CATEGORY = "DefinedGrid"
"""Feature name of top left spot X coordinate"""
F_X_LOCATION_OF_LOWEST_X_SPOT = "XLocationOfLowestXSpot"
"""Feature name of top left spot Y coordinate"""
F_Y_LOCATION_OF_LOWEST_Y_SPOT = "YLocationOfLowestYSpot"
"""Feature name of x distance between spots"""
F_X_SPACING = "XSpacing"
"""Feature name of y distance between spots"""
F_Y_SPACING = "YSpacing"
"""Feature name of # of rows in grid"""
F_ROWS = "Rows"
"""Feature name of # of columns in grid"""
F_COLUMNS = "Columns"


class DefineGrid(Module):
    module_name = "DefineGrid"
    variable_revision_number = 1
    category = "Other"

    def create_settings(self):
        """Create your settings by subclassing this function

        create_settings is called at the end of initialization.
        """
        self.grid_image = GridName(
            "Name the grid",
            doc="""\
This is the name of the grid. You can use this name to
retrieve the grid in subsequent modules.""",
        )

        self.grid_rows = Integer(
            "Number of rows",
            8,
            1,
            doc="""Along the height of the grid, define the number of rows.""",
        )

        self.grid_columns = Integer(
            "Number of columns",
            12,
            1,
            doc="""Along the width of the grid, define the number of columns.""",
        )

        self.origin = Choice(
            "Location of the first spot",
            [NUM_TOP_LEFT, NUM_BOTTOM_LEFT, NUM_TOP_RIGHT, NUM_BOTTOM_RIGHT],
            doc="""\
Grid cells are numbered consecutively; this option identifies the
origin for the numbering system and the direction for numbering.
For instance, if you choose "*%(NUM_TOP_LEFT)s*", the top left cell is
cell #1 and cells to the right and bottom are indexed with
larger numbers."""
            % globals(),
        )

        self.ordering = Choice(
            "Order of the spots",
            [NUM_BY_ROWS, NUM_BY_COLUMNS],
            doc="""\
Grid cells can either be numbered by rows, then columns or by columns,
then rows. For instance, if you asked to start numbering a 96-well
plate at the top left (by specifying the location of the first spot), then:

-  *%(NUM_BY_ROWS)s:* this option will give well A01 the index 1, B01
   the index 2, and so on up to H01 which receives the index 8. Well A02
   will be assigned the index 9.
-  *%(NUM_BY_COLUMNS)s:* with this option, the well A02 will be
   assigned 2, well A12 will be assigned 12 and well B01 will be
   assigned 13.
"""
            % globals(),
        )

        self.each_or_once = Choice(
            "Define a grid for which cycle?",
            [EO_EACH, EO_ONCE],
            doc="""\
The setting allows you choose when you want to define a new grid:

-  *%(EO_ONCE)s:* If all of your images are perfectly aligned with each
   other (due to very consistent image acquisition, consistent grid
   location within the plate, and/or automatic cropping precisely within
   each plate), you can define the location of the marker spots once for
   all of the image cycles.
-  *%(EO_EACH)s:* If the location of the grid will vary from one image
   cycle to the next then you should define the location of the marker
   spots for each cycle independently.
"""
            % globals(),
        )

        self.auto_or_manual = Choice(
            "Select the method to define the grid",
            [AM_AUTOMATIC, AM_MANUAL],
            doc="""\
Select whether you would like to define the grid automatically (based on
objects you have identified in a previous module) or manually. This
setting controls how the grid is defined:

-  *%(AM_MANUAL)s:* In manual mode, you manually indicate known
   locations of marker spots in the grid and have the rest of the
   positions calculated from those marks, no matter what the image
   itself looks like. You can define the grid either by clicking on the
   image with a mouse or by entering coordinates.
-  *%(AM_AUTOMATIC)s:* If you would like the grid to be defined
   automatically, an **IdentifyPrimaryObjects** module must be run prior
   to this module to identify the objects that will be used to define
   the grid. The left-most, right-most, top-most, and bottom-most object
   will be used to define the edges of the grid, and the rows and
   columns will be evenly spaced between these edges. Note that
   Automatic mode requires that the incoming objects are nicely defined:
   for example, if there is an object at the edge of the images that is
   not really an object that ought to be in the grid, a skewed grid will
   result. You might wish to use a **FilterObjects** module to clean up
   badly identified objects prior to defining the grid. If the spots are
   slightly out of alignment with each other from one image cycle to the
   next, this allows the identification to be a bit flexible and adapt
   to the real location of the spots.
"""
            % globals(),
        )

        self.object_name = LabelSubscriber(
            "Select the previously identified objects",
            "None",
            doc="""\
*(Used only if you selected "%(AM_AUTOMATIC)s" to define the grid)*

Select the previously identified objects you want to use to define the
grid. Use this setting to specify the name of the objects that will be
used to define the grid.
"""
            % globals(),
        )

        self.manual_choice = Choice(
            "Select the method to define the grid manually",
            [MAN_MOUSE, MAN_COORDINATES],
            doc="""\
*(Used only if you selected "%(AM_MANUAL)s" to define the grid)*

Specify whether you want to define the grid using the mouse or by
entering the coordinates of the cells.

-  *%(MAN_MOUSE)s:* The user interface displays the image you specify.
   You will be asked to click in the center of two of the grid cells and
   specify the row and column for each. The grid coordinates will be
   computed from this information.
-  *%(MAN_COORDINATES)s:* Enter the X and Y coordinates of the grid
   cells directly. You can display an image of your grid to find the
   locations of the centers of the cells, then enter the X and Y
   position and cell coordinates for each of two cells.
"""
            % globals(),
        )

        self.manual_image = ImageSubscriber(
            "Select the image to display when drawing",
            "None",
            doc="""\
*(Used only if you selected "%(AM_MANUAL)s" and "%(MAN_MOUSE)s" to define
the grid)*

Specify the image you want to display when defining the grid. This
setting lets you choose the image to display in the grid definition user
interface.
"""
            % globals(),
        )

        self.first_spot_coordinates = Coordinates(
            "Coordinates of the first cell",
            (0, 0),
            doc="""\
*(Used only if you selected "%(AM_MANUAL)s" and "%(MAN_COORDINATES)s" to
define the grid)*

Enter the coordinates of the first cell on your grid. This setting
defines the location of the first of two cells in your grid. You should
enter the coordinates of the center of the cell. You can display an
image of your grid and use the pixel coordinate display to determine the
coordinates of the center of your cell.
"""
            % globals(),
        )

        self.first_spot_row = Integer(
            "Row number of the first cell",
            1,
            minval=1,
            doc="""\
*(Used only if you selected "%(AM_MANUAL)s" and "%(MAN_COORDINATES)s" to
define the grid)*

Enter the row index for the first cell here. Rows are numbered starting
at the origin. For instance, if you chose "*%(NUM_TOP_LEFT)s*" as your
origin, well A01 will be row number 1 and H01 will be row number 8. If
you chose "*%(NUM_BOTTOM_LEFT)s*", A01 will be row number 8 and H01 will
be row number 12.
"""
            % globals(),
        )

        self.first_spot_col = Integer(
            "Column number of the first cell",
            1,
            minval=1,
            doc="""\
*(Used only if you selected "%(AM_MANUAL)s" and "%(MAN_COORDINATES)s" to
define the grid)*

Enter the column index for the first cell here. Columns are numbered
starting at the origin. For instance, if you chose "*%(NUM_TOP_LEFT)s*"
as your origin, well A01 will be column number *1* and A12 will be
column number *12*. If you chose "*%(NUM_TOP_RIGHT)s*", A01 and A12 will
be *12* and *1*, respectively.
"""
            % globals(),
        )

        self.second_spot_coordinates = Coordinates(
            "Coordinates of the second cell",
            (0, 0),
            doc="""\
*(Used only if you selected "%(AM_MANUAL)s" and "%(MAN_COORDINATES)s" to
define the grid)*

This setting defines the location of the second of two cells in your
grid. You should enter the coordinates of the center of the cell. You
can display an image of your grid and use the pixel coordinate
display to determine the coordinates (X,Y) of the center of your cell.
"""
            % globals(),
        )

        self.second_spot_row = Integer(
            "Row number of the second cell",
            1,
            minval=1,
            doc="""\
*(Used only if you selected "%(AM_MANUAL)s" and "%(MAN_COORDINATES)s" to
define the grid)*

Enter the row index for the second cell here. Rows are numbered starting
at the origin. For instance, if you chose "*%(NUM_TOP_LEFT)s*" as your
origin, well A01 will be row number 1 and H01 will be row number 8. If
you chose "*%(NUM_BOTTOM_LEFT)s*", A01 will be row number 8 and H01 will
be row number 12.
"""
            % globals(),
        )

        self.second_spot_col = Integer(
            "Column number of the second cell",
            1,
            minval=1,
            doc="""\
*(Used only if you selected "%(AM_MANUAL)s" and "%(MAN_COORDINATES)s" to
define the grid)*

Enter the column index for the second cell here. Columns are numbered
starting at the origin. For instance, if you chose "*%(NUM_TOP_LEFT)s*"
as your origin, well A01 will be column number 1 and A12 will be column
number 12. If you chose "*%(NUM_TOP_RIGHT)s*", A01 and A12 will be 12
and 1, respectively.
"""
            % globals(),
        )

        self.wants_image = Binary(
            "Retain an image of the grid?",
            False,
            doc="""\
Select "*Yes*" to retain an image of the grid for use later in the
pipeline. This module can create an annotated image of the grid that can
be saved using the **SaveImages** module.
"""
            % globals(),
        )

        self.display_image_name = ImageSubscriber(
            "Select the image on which to display the grid",
            "Leave blank",
            can_be_blank=True,
            doc="""\
*(Used only if saving an image of the grid)*

Enter the name of the image that should be used as the background for
annotations (grid lines and grid indexes). This image will be used for
the figure and for the saved image.
""",
        )

        self.save_image_name = ImageName(
            "Name the output image",
            "Grid",
            doc="""\
*(Used only if retaining an image of the grid for use later in the
pipeline)*

Enter the name you want to use for the output image. You can save this
image using the **SaveImages** module.
""",
        )

        self.failed_grid_choice = Choice(
            "Use a previous grid if gridding fails?",
            [FAIL_NO, FAIL_ANY_PREVIOUS, FAIL_FIRST],
            doc="""\
If the gridding fails, this setting allows you to control how the module
responds to the error:

-  *%(FAIL_NO)s:* The module will stop the pipeline if gridding fails.
-  *%(FAIL_ANY_PREVIOUS)s:* The module will use the the most recent
   successful gridding.
-  *%(FAIL_FIRST)s:* The module will use the first gridding.

Note that the pipeline will stop in all cases if gridding fails on the
first image.
"""
            % globals(),
        )

    def settings(self):
        """Return the settings to be loaded or saved to/from the pipeline

        These are the settings (from cellprofiler_core.settings) that are
        either read from the strings in the pipeline or written out
        to the pipeline. The settings should appear in a consistent
        order so they can be matched to the strings in the pipeline.
        """
        return [
            self.grid_image,
            self.grid_rows,
            self.grid_columns,
            self.origin,
            self.ordering,
            self.each_or_once,
            self.auto_or_manual,
            self.object_name,
            self.manual_choice,
            self.manual_image,
            self.first_spot_coordinates,
            self.first_spot_row,
            self.first_spot_col,
            self.second_spot_coordinates,
            self.second_spot_row,
            self.second_spot_col,
            self.wants_image,
            self.save_image_name,
            self.display_image_name,
            self.failed_grid_choice,
        ]

    def visible_settings(self):
        """The settings that are visible in the UI
        """
        result = [
            self.grid_image,
            self.grid_rows,
            self.grid_columns,
            self.origin,
            self.ordering,
            self.each_or_once,
            self.auto_or_manual,
        ]
        if self.auto_or_manual == AM_AUTOMATIC:
            result += [self.object_name, self.failed_grid_choice]
        elif self.auto_or_manual == AM_MANUAL:
            result += [self.manual_choice]
            if self.manual_choice == MAN_MOUSE:
                result += [self.manual_image]
            elif self.manual_choice == MAN_COORDINATES:
                result += [
                    self.first_spot_coordinates,
                    self.first_spot_row,
                    self.first_spot_col,
                    self.second_spot_coordinates,
                    self.second_spot_row,
                    self.second_spot_col,
                ]
            else:
                raise NotImplementedError(
                    "Unknown manual choice: %s" % self.manual_choice.value
                )
        else:
            raise NotImplementedError(
                "Unknown automatic / manual choice: %s" % self.auto_or_manual.value
            )
        result += [self.wants_image]
        if self.wants_image:
            result += [self.save_image_name]
        result += [self.display_image_name]
        return result

    def run(self, workspace):
        """Run the module

        workspace    - The workspace contains
            pipeline     - instance of cpp for this run
            image_set    - the images in the image set being processed
            object_set   - the objects (labeled masks) in this image set
            measurements - the measurements for this run
            frame        - the parent frame to whatever frame is created. None means don't draw.
        """
        background_image = self.get_background_image(workspace, None)

        if (
            self.each_or_once == EO_ONCE
            and self.get_good_gridding(workspace) is not None
        ):
            gridding = self.get_good_gridding(workspace)
        if self.auto_or_manual == AM_AUTOMATIC:
            gridding = self.run_automatic(workspace)
        elif self.manual_choice == MAN_COORDINATES:
            gridding = self.run_coordinates(workspace)
        elif self.manual_choice == MAN_MOUSE:
            gridding = workspace.interaction_request(
                self, background_image, workspace.measurements.image_set_number
            )
        self.set_good_gridding(workspace, gridding)
        workspace.set_grid(self.grid_image.value, gridding)
        #
        # Save measurements
        #
        self.add_measurement(
            workspace,
            F_X_LOCATION_OF_LOWEST_X_SPOT,
            gridding.x_location_of_lowest_x_spot,
        )
        self.add_measurement(
            workspace,
            F_Y_LOCATION_OF_LOWEST_Y_SPOT,
            gridding.y_location_of_lowest_y_spot,
        )
        self.add_measurement(workspace, F_ROWS, gridding.rows)
        self.add_measurement(workspace, F_COLUMNS, gridding.columns)
        self.add_measurement(workspace, F_X_SPACING, gridding.x_spacing)
        self.add_measurement(workspace, F_Y_SPACING, gridding.y_spacing)

        # update background image
        background_image = self.get_background_image(workspace, gridding)

        workspace.display_data.gridding = gridding.serialize()
        workspace.display_data.background_image = background_image
        workspace.display_data.image_set_number = (
            workspace.measurements.image_set_number
        )

        if self.wants_image:
            import matplotlib.transforms
            import matplotlib.figure
            import matplotlib.backends.backend_agg
            from cellprofiler.gui.tools import figure_to_image

            figure = matplotlib.figure.Figure()
            canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(figure)
            ax = figure.add_subplot(1, 1, 1)
            self.display_grid(
                background_image, gridding, workspace.measurements.image_set_number, ax
            )
            #
            # This is the recipe for just showing the axis
            #
            figure.set_frameon(False)
            ax.set_axis_off()
            figure.subplots_adjust(0, 0, 1, 1, 0, 0)
            ai = ax.images[0]
            shape = ai.get_size()
            dpi = figure.dpi
            width = float(shape[1]) / dpi
            height = float(shape[0]) / dpi
            figure.set_figheight(height)
            figure.set_figwidth(width)
            bbox = matplotlib.transforms.Bbox(
                numpy.array([[0.0, 0.0], [width, height]])
            )
            transform = matplotlib.transforms.Affine2D(
                numpy.array([[dpi, 0, 0], [0, dpi, 0], [0, 0, 1]])
            )
            figure.bbox = matplotlib.transforms.TransformedBbox(bbox, transform)
            image_pixels = figure_to_image(figure, dpi=dpi)
            image = Image(image_pixels)

            workspace.image_set.add(self.save_image_name.value, image)

    def get_background_image(self, workspace, gridding):
        if (
            self.auto_or_manual == AM_MANUAL
            and self.manual_choice == MAN_MOUSE
            and gridding is None
        ):
            image = workspace.image_set.get_image(self.manual_image.value).pixel_data
        elif self.display_image_name.value == "Leave blank":
            if gridding is None:
                return None
            image = numpy.zeros(
                (
                    int(
                        gridding.total_height
                        + (
                            gridding.y_location_of_lowest_y_spot
                            - gridding.y_spacing / 2
                        )
                        * 2
                    )
                    + 2,
                    int(
                        gridding.total_width
                        + (
                            gridding.x_location_of_lowest_x_spot
                            - gridding.x_spacing / 2
                        )
                        * 2
                    )
                    + 2,
                    3,
                )
            )
        else:
            image = workspace.image_set.get_image(
                self.display_image_name.value
            ).pixel_data
            if image.ndim == 2:
                image = numpy.dstack((image, image, image))
        return image

    def run_automatic(self, workspace):
        """Automatically define a grid based on objects

        Returns a CPGridInfo object
        """
        objects = workspace.object_set.get_objects(self.object_name.value)
        centroids = centrosome.cpmorphology.centers_of_labels(objects.segmented)
        try:
            if centroids.shape[1] < 2:
                #
                # Failed if too few objects
                #
                raise RuntimeError("%s has too few grid cells" % self.object_name.value)
            #
            # Artificially swap these to match the user's orientation
            #
            first_row, second_row = (1, self.grid_rows.value)
            if self.origin in (NUM_BOTTOM_LEFT, NUM_BOTTOM_RIGHT):
                first_row, second_row = (second_row, first_row)
            first_column, second_column = (1, self.grid_columns.value)
            if self.origin in (NUM_TOP_RIGHT, NUM_BOTTOM_RIGHT):
                first_column, second_column = (second_column, first_column)
            first_x = numpy.min(centroids[1, :])
            first_y = numpy.min(centroids[0, :])
            second_x = numpy.max(centroids[1, :])
            second_y = numpy.max(centroids[0, :])
            result = self.build_grid_info(
                first_x,
                first_y,
                first_row,
                first_column,
                second_x,
                second_y,
                second_row,
                second_column,
                objects.segmented.shape,
            )
        except Exception:
            if self.failed_grid_choice != FAIL_NO:
                result = self.get_good_gridding(workspace)
                if result is None:
                    raise RuntimeError(
                        "%s has too few grid cells and there is no previous successful grid"
                        % self.object_name.value
                    )
            raise
        return result

    def run_coordinates(self, workspace):
        """Define a grid based on the coordinates of two points

        Returns a CPGridInfo object
        """
        if self.display_image_name.value in workspace.image_set.names:
            image = workspace.image_set.get_image(self.display_image_name.value)
            shape = image.pixel_data.shape[:2]
        else:
            shape = None
        return self.build_grid_info(
            self.first_spot_coordinates.x,
            self.first_spot_coordinates.y,
            self.first_spot_row.value,
            self.first_spot_col.value,
            self.second_spot_coordinates.x,
            self.second_spot_coordinates.y,
            self.second_spot_row.value,
            self.second_spot_col.value,
            shape,
        )

    def handle_interaction(self, background_image, image_set_number):
        return self.run_mouse(background_image, image_set_number)

    def run_mouse(self, background_image, image_set_number):
        """Define a grid by running the UI

        Returns a CPGridInfo object
        """
        import matplotlib
        import matplotlib.backends.backend_wxagg as backend
        import wx
        from wx.lib.intctrl import IntCtrl

        #
        # Make up a dialog box. It has the following structure:
        #
        # Dialog:
        #    top_sizer:
        #        Canvas
        #            Figure
        #               Axis
        #        control_sizer
        #            first_sizer
        #               first_row
        #               first_col
        #            second_sizer
        #               second_row
        #               second_col
        #            button_sizer
        #               Redisplay
        #               OK
        #               cancel
        #    status bar
        #
        figure = matplotlib.figure.Figure()
        frame = wx.Dialog(
            wx.GetApp().TopWindow,
            title="Select grid cells, image cycle #%d:" % (image_set_number),
        )
        top_sizer = wx.BoxSizer(wx.VERTICAL)
        frame.SetSizer(top_sizer)
        canvas = backend.FigureCanvasWxAgg(frame, -1, figure)
        top_sizer.Add(canvas, 1, wx.EXPAND)
        top_sizer.Add(
            wx.StaticText(
                frame,
                -1,
                "Select the center of a grid cell with the left mouse button.\n",
            ),
            0,
            wx.EXPAND | wx.ALL,
            5,
        )
        control_sizer = wx.BoxSizer(wx.HORIZONTAL)
        top_sizer.Add(control_sizer, 0, wx.EXPAND | wx.ALL, 5)
        FIRST_CELL = "First cell"
        SECOND_CELL = "Second cell"
        cell_choice = wx.RadioBox(
            frame,
            label="Choose current cell",
            choices=[FIRST_CELL, SECOND_CELL],
            style=wx.RA_VERTICAL,
        )
        control_sizer.Add(cell_choice)
        #
        # Text boxes for the first cell's row and column
        #
        first_sizer = wx.GridBagSizer(2, 2)
        control_sizer.Add(first_sizer, 1, wx.EXPAND | wx.ALL, 5)
        first_sizer.Add(
            wx.StaticText(frame, -1, "First cell column:"),
            wx.GBPosition(0, 0),
            flag=wx.EXPAND,
        )
        first_column = IntCtrl(frame, -1, 1, min=1, max=self.grid_columns.value)
        first_sizer.Add(first_column, wx.GBPosition(0, 1), flag=wx.EXPAND)
        first_sizer.Add(
            wx.StaticText(frame, -1, "First cell row:"),
            wx.GBPosition(1, 0),
            flag=wx.EXPAND,
        )
        first_row = IntCtrl(frame, -1, 1, min=1, max=self.grid_rows.value)
        first_sizer.Add(first_row, wx.GBPosition(1, 1), flag=wx.EXPAND)
        first_sizer.Add(wx.StaticText(frame, -1, "X:"), wx.GBPosition(0, 2))
        first_x = IntCtrl(frame, -1, 100, min=1)
        first_sizer.Add(first_x, wx.GBPosition(0, 3))
        first_sizer.Add(wx.StaticText(frame, -1, "Y:"), wx.GBPosition(1, 2))
        first_y = IntCtrl(frame, -1, 100, min=1)
        first_sizer.Add(first_y, wx.GBPosition(1, 3))
        #
        # Text boxes for the second cell's row and column
        #
        second_sizer = wx.GridBagSizer(2, 2)
        control_sizer.Add(second_sizer, 1, wx.EXPAND | wx.ALL, 5)
        second_sizer.Add(
            wx.StaticText(frame, -1, "Second cell column:"),
            wx.GBPosition(0, 0),
            flag=wx.EXPAND,
        )
        second_column = IntCtrl(
            frame, -1, self.grid_columns.value, min=1, max=self.grid_columns.value
        )
        second_sizer.Add(second_column, wx.GBPosition(0, 1), flag=wx.EXPAND)
        second_sizer.Add(
            wx.StaticText(frame, -1, "Second cell row:"),
            wx.GBPosition(1, 0),
            flag=wx.EXPAND,
        )
        second_row = IntCtrl(
            frame, -1, self.grid_rows.value, min=1, max=self.grid_rows.value
        )
        second_sizer.Add(second_row, wx.GBPosition(1, 1), flag=wx.EXPAND)
        second_sizer.Add(wx.StaticText(frame, -1, "X:"), wx.GBPosition(0, 2))
        second_x = IntCtrl(frame, -1, 200, min=1)
        second_sizer.Add(second_x, wx.GBPosition(0, 3))
        second_sizer.Add(wx.StaticText(frame, -1, "Y:"), wx.GBPosition(1, 2))
        second_y = IntCtrl(frame, -1, 200, min=1)
        second_sizer.Add(second_y, wx.GBPosition(1, 3))
        #
        # Buttons
        #
        button_sizer = wx.BoxSizer(wx.VERTICAL)
        control_sizer.Add(button_sizer, 0, wx.EXPAND | wx.ALL, 5)
        redisplay_button = wx.Button(frame, -1, "Redisplay")
        button_sizer.Add(redisplay_button)
        button_sizer.Add(wx.Button(frame, wx.OK, "OK"))
        button_sizer.Add(wx.Button(frame, wx.CANCEL, "Cancel"))
        #
        # Status bar
        #
        status_bar = wx.StatusBar(frame, style=0)
        top_sizer.Add(status_bar, 0, wx.EXPAND)
        status_bar.SetFieldsCount(1)
        SELECT_FIRST_CELL = "Select the center of the first cell"
        SELECT_SECOND_CELL = "Select the center of the second cell"
        status_bar.SetStatusText(SELECT_FIRST_CELL)
        status = [wx.OK]
        gridding = [None]
        if self.display_image_name == "Leave blank":
            image_shape = None
        else:
            image_shape = background_image.shape[:2]

        def redisplay(event):
            figure.clf()
            axes = figure.add_subplot(1, 1, 1)

            if (event is not None) or (gridding[0] is None):
                do_gridding(
                    first_x.GetValue(),
                    first_y.GetValue(),
                    second_x.GetValue(),
                    second_y.GetValue(),
                )
            self.display_grid(background_image, gridding[0], image_set_number, axes)
            canvas.draw()

        def cancel(event):
            status[0] = wx.CANCEL
            frame.SetReturnCode(wx.CANCEL)
            frame.Close(True)

        def ok(event):
            status[0] = wx.OK
            frame.SetReturnCode(wx.OK)
            frame.Close(True)

        def on_cell_selection(event):
            if cell_choice.GetSelection() == 0:
                status_bar.SetStatusText(SELECT_FIRST_CELL)
            else:
                status_bar.SetStatusText(SELECT_SECOND_CELL)

        def do_gridding(x1, y1, x2, y2):
            try:
                gridding[0] = self.build_grid_info(
                    int(x1),
                    int(y1),
                    int(first_row.GetValue()),
                    int(first_column.GetValue()),
                    int(x2),
                    int(y2),
                    int(second_row.GetValue()),
                    int(second_column.GetValue()),
                    image_shape,
                )
            except Exception as e:
                LOGGER.error(e, exc_info=True)
                status_bar.SetStatusText(str(e))
                return False
            return True

        def button_release(event):
            if event.inaxes == figure.axes[0]:
                if cell_choice.GetSelection() == 0:
                    new_first_x = str(int(event.xdata))
                    new_first_y = str(int(event.ydata))
                    if do_gridding(
                        new_first_x,
                        new_first_y,
                        second_x.GetValue(),
                        second_y.GetValue(),
                    ):
                        first_x.SetValue(new_first_x)
                        first_y.SetValue(new_first_y)
                        cell_choice.SetSelection(1)
                        status_bar.SetStatusText(SELECT_SECOND_CELL)
                else:
                    new_second_x = str(int(event.xdata))
                    new_second_y = str(int(event.ydata))
                    if do_gridding(
                        first_x.GetValue(),
                        first_y.GetValue(),
                        new_second_x,
                        new_second_y,
                    ):
                        second_x.SetValue(new_second_x)
                        second_y.SetValue(new_second_y)
                        cell_choice.SetSelection(0)
                        status_bar.SetStatusText(SELECT_FIRST_CELL)
                redisplay(None)

        redisplay(None)
        frame.Fit()
        frame.Bind(wx.EVT_BUTTON, redisplay, redisplay_button)
        frame.Bind(wx.EVT_BUTTON, cancel, id=wx.CANCEL)
        frame.Bind(wx.EVT_BUTTON, ok, id=wx.OK)
        frame.Bind(wx.EVT_RADIOBOX, on_cell_selection, cell_choice)
        canvas.mpl_connect("button_release_event", button_release)
        frame.ShowModal()
        do_gridding(
            first_x.GetValue(),
            first_y.GetValue(),
            second_x.GetValue(),
            second_y.GetValue(),
        )
        frame.Destroy()
        if status[0] != wx.OK:
            raise RuntimeError("Pipeline aborted during grid editing")
        return gridding[0]

    def get_feature_name(self, feature):
        return "_".join((M_CATEGORY, self.grid_image.value, feature))

    def add_measurement(self, workspace, feature, value):
        """Add an image measurement using our category and grid

        feature - the feature name of the measurement to add
        value - the value for the measurement
        """
        feature_name = self.get_feature_name(feature)
        workspace.measurements.add_image_measurement(feature_name, value)

    def build_grid_info(
        self,
        first_x,
        first_y,
        first_row,
        first_col,
        second_x,
        second_y,
        second_row,
        second_col,
        image_shape=None,
    ):
        """Populate and return a CPGridInfo based on two cell locations"""
        first_row, first_col = self.canonical_row_and_column(first_row, first_col)
        second_row, second_col = self.canonical_row_and_column(second_row, second_col)
        gridding = Grid()
        gridding.x_spacing = float(first_x - second_x) / float(first_col - second_col)
        gridding.y_spacing = float(first_y - second_y) / float(first_row - second_row)
        gridding.x_location_of_lowest_x_spot = int(
            first_x - first_col * gridding.x_spacing
        )
        gridding.y_location_of_lowest_y_spot = int(
            first_y - first_row * gridding.y_spacing
        )
        gridding.rows = self.grid_rows.value
        gridding.columns = self.grid_columns.value
        gridding.left_to_right = self.origin in (NUM_TOP_LEFT, NUM_BOTTOM_LEFT)
        gridding.top_to_bottom = self.origin in (NUM_TOP_LEFT, NUM_TOP_RIGHT)
        gridding.total_width = int(gridding.x_spacing * gridding.columns)
        gridding.total_height = int(gridding.y_spacing * gridding.rows)

        line_left_x = int(gridding.x_location_of_lowest_x_spot - gridding.x_spacing / 2)
        line_top_y = int(gridding.y_location_of_lowest_y_spot - gridding.y_spacing / 2)
        #
        # Make a 2 x columns array of x-coordinates of vertical lines (x0=x1)
        #
        gridding.vert_lines_x = numpy.tile(
            (numpy.arange(gridding.columns + 1) * gridding.x_spacing + line_left_x),
            (2, 1),
        ).astype(int)
        #
        # Make a 2 x rows array of y-coordinates of horizontal lines (y0=y1)
        #
        gridding.horiz_lines_y = numpy.tile(
            (numpy.arange(gridding.rows + 1) * gridding.y_spacing + line_top_y), (2, 1)
        ).astype(int)
        #
        # Make a 2x columns array of y-coordinates of vertical lines
        # all of which are from line_top_y to the bottom
        #
        gridding.vert_lines_y = numpy.transpose(
            numpy.tile(
                (line_top_y, line_top_y + gridding.total_height),
                (gridding.columns + 1, 1),
            )
        ).astype(int)
        gridding.horiz_lines_x = numpy.transpose(
            numpy.tile(
                (line_left_x, line_left_x + gridding.total_width),
                (gridding.rows + 1, 1),
            )
        ).astype(int)
        gridding.x_locations = (
            gridding.x_location_of_lowest_x_spot
            + numpy.arange(gridding.columns) * gridding.x_spacing
        ).astype(int)
        gridding.y_locations = (
            gridding.y_location_of_lowest_y_spot
            + numpy.arange(gridding.rows) * gridding.y_spacing
        ).astype(int)
        #
        # The spot table has the numbering for each spot in the grid
        #
        gridding.spot_table = numpy.arange(gridding.rows * gridding.columns) + 1
        if self.ordering == NUM_BY_COLUMNS:
            gridding.spot_table.shape = (gridding.rows, gridding.columns)
        else:
            gridding.spot_table.shape = (gridding.columns, gridding.rows)
            gridding.spot_table = numpy.transpose(gridding.spot_table)
        if self.origin in (NUM_BOTTOM_LEFT, NUM_BOTTOM_RIGHT):
            # Flip top and bottom
            gridding.spot_table = gridding.spot_table[::-1, :]
        if self.origin in (NUM_TOP_RIGHT, NUM_BOTTOM_RIGHT):
            # Flip left and right
            gridding.spot_table = gridding.spot_table[:, ::-1]
        if image_shape is not None:
            gridding.image_height = image_shape[0]
            gridding.image_width = image_shape[1]
        else:
            # guess the image shape by adding the same border to the right
            # and bottom that we have on the left and top
            top_edge = int(
                gridding.y_location_of_lowest_y_spot - gridding.y_spacing / 2
            )
            right_edge = int(
                gridding.x_location_of_lowest_x_spot - gridding.x_spacing / 2
            )
            gridding.image_height = top_edge * 2 + gridding.y_spacing * gridding.rows
            gridding.image_width = (
                right_edge * 2 + gridding.x_spacing * gridding.columns
            )
        return gridding

    def canonical_row_and_column(self, row, column):
        """Convert a row and column as entered by the user to canonical form

        The user might select something other than the bottom left as the
        origin of their coordinate space. This method returns a row and
        column using a numbering where the top left corner is 0,0
        """
        if self.origin in (NUM_BOTTOM_LEFT, NUM_BOTTOM_RIGHT):
            row = self.grid_rows.value - row
        else:
            row -= 1
        if self.origin in (NUM_TOP_RIGHT, NUM_BOTTOM_RIGHT):
            column = self.grid_columns.value - column
        else:
            column -= 1
        return row, column

    def display(self, workspace, figure):
        if self.show_window:
            figure.set_subplots((1, 1))
            figure.clf()
            ax = figure.subplot(0, 0)
            gridding = Grid()
            gridding.deserialize(workspace.display_data.gridding)
            self.display_grid(
                workspace.display_data.background_image,
                gridding,
                workspace.display_data.image_set_number,
                ax,
            )

    def display_grid(self, background_image, gridding, image_set_number, axes):
        """Display the grid in a figure"""
        import matplotlib

        axes.cla()
        assert isinstance(axes, matplotlib.axes.Axes)
        assert isinstance(gridding, Grid)
        #
        # draw the image on the figure
        #
        if background_image is None:
            background_image = self.get_background_image(None, gridding)
        axes.imshow(background_image)
        #
        # Draw lines
        #
        for xc, yc in (
            (gridding.horiz_lines_x, gridding.horiz_lines_y),
            (gridding.vert_lines_x, gridding.vert_lines_y),
        ):
            for i in range(xc.shape[1]):
                line = matplotlib.lines.Line2D(xc[:, i], yc[:, i], color="red")
                axes.add_line(line)
        #
        # Draw labels in corners
        #
        for row in (0, gridding.rows - 1):
            for column in (0, gridding.columns - 1):
                label = str(gridding.spot_table[row, column])
                x = gridding.x_locations[column]
                y = gridding.y_locations[row]
                text = matplotlib.text.Text(
                    x,
                    y,
                    label,
                    horizontalalignment="center",
                    verticalalignment="center",
                    size="smaller",
                    color="black",
                    bbox=dict(facecolor="white", alpha=0.5, edgecolor="black"),
                )
                axes.add_artist(text)
        axes.axis("image")

    def get_good_gridding(self, workspace):
        """Get either the first gridding or the most recent successful gridding"""
        d = self.get_dictionary()
        if not GOOD_GRIDDING in d:
            return None
        return d[GOOD_GRIDDING]

    def set_good_gridding(self, workspace, gridding):
        """Set the gridding to use upon failure"""
        d = self.get_dictionary()
        if self.failed_grid_choice == FAIL_ANY_PREVIOUS or GOOD_GRIDDING not in d:
            d[GOOD_GRIDDING] = gridding

    def validate_module(self, pipeline):
        """Make sure that the row and column are different"""
        if self.auto_or_manual == AM_MANUAL and self.manual_choice == MAN_COORDINATES:
            if self.first_spot_row.value == self.second_spot_row.value:
                raise ValidationError(
                    "The first and second row numbers must be different in "
                    "order to calculate the distance between rows.",
                    self.second_spot_row,
                )
            if self.first_spot_col.value == self.second_spot_col.value:
                raise ValidationError(
                    "The first and second column numbers must be different "
                    "in order to calculate the distance between columns.",
                    self.second_spot_col,
                )

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        """Adjust setting values if they came from a previous revision

        setting_values - a sequence of strings representing the settings
                         for the module as stored in the pipeline
        variable_revision_number - the variable revision number of the
                         module at the time the pipeline was saved. Use this
                         to determine how the incoming setting values map
                         to those of the current module version.
        module_name - the name of the module that did the saving. This can be
                      used to import the settings from another module if
                      that module was merged into the current module
        """
        if variable_revision_number == 1:
            #
            # Some of the wording changed for the failed grid choice
            #
            if setting_values[-1] == "Any Previous":
                setting_values = setting_values[:-1] + [FAIL_ANY_PREVIOUS]
            elif setting_values[-1] == "The First":
                setting_values = setting_values[:-1] + [FAIL_FIRST]
        return setting_values, variable_revision_number

    def get_measurement_columns(self, pipeline):
        """Return a sequence describing the measurement columns needed by this module

        This call should return one element per image or object measurement
        made by the module during image set analysis. The element itself
        is a 3-tuple:
        first entry: either one of the predefined measurement categories,
                     {"Image", "Experiment" or "Neighbors" or the name of one
                     of the objects.}
        second entry: the measurement name (as would be used in a call
                      to add_measurement)
        third entry: the column data type (for instance, "varchar(255)" or
                     "float")
        """
        return [
            (IMAGE, self.get_feature_name(F_ROWS), COLTYPE_INTEGER),
            (IMAGE, self.get_feature_name(F_COLUMNS), COLTYPE_INTEGER),
            (IMAGE, self.get_feature_name(F_X_SPACING), COLTYPE_FLOAT),
            (IMAGE, self.get_feature_name(F_Y_SPACING), COLTYPE_FLOAT),
            (
                IMAGE,
                self.get_feature_name(F_X_LOCATION_OF_LOWEST_X_SPOT),
                COLTYPE_FLOAT,
            ),
            (
                IMAGE,
                self.get_feature_name(F_Y_LOCATION_OF_LOWEST_Y_SPOT),
                COLTYPE_FLOAT,
            ),
        ]

    def get_categories(self, pipeline, object_name):
        """Return the categories of measurements that this module produces

        object_name - return measurements made on this object (or 'Image' for image measurements)
        """
        if object_name == IMAGE:
            return [M_CATEGORY]
        return []

    def get_measurements(self, pipeline, object_name, category):
        if object_name == IMAGE and category == M_CATEGORY:
            return [
                "_".join((self.grid_image.value, feature))
                for feature in (
                    F_ROWS,
                    F_COLUMNS,
                    F_X_SPACING,
                    F_Y_SPACING,
                    F_X_LOCATION_OF_LOWEST_X_SPOT,
                    F_Y_LOCATION_OF_LOWEST_Y_SPOT,
                )
            ]
        return []
