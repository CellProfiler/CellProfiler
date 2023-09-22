import csv
import logging
import os
import sys
import textwrap
import uuid

import centrosome.cpmorphology
import centrosome.outline
import matplotlib
import matplotlib.axes
import matplotlib.backend_bases
import matplotlib.backends.backend_wxagg
import matplotlib.cm
import matplotlib.collections
import matplotlib.colorbar
import matplotlib.colors
import matplotlib.gridspec
import matplotlib.image
import matplotlib.patches
import matplotlib.pyplot
import matplotlib.transforms
import matplotlib.widgets
import numpy
import numpy.ma
import scipy.ndimage
import scipy.sparse
import skimage.exposure
import wx
import wx.grid
import wx.lib.intctrl
import wx.lib.masked
from cellprofiler_core.preferences import INTENSITY_MODE_GAMMA
from cellprofiler_core.preferences import INTENSITY_MODE_LOG
from cellprofiler_core.preferences import INTENSITY_MODE_RAW
from cellprofiler_core.preferences import get_default_colormap
from cellprofiler_core.preferences import get_intensity_mode
from cellprofiler_core.preferences import get_next_cpfigure_position
from cellprofiler_core.preferences import get_normalization_factor
from cellprofiler_core.preferences import get_primary_outline_color
from cellprofiler_core.preferences import get_secondary_outline_color
from cellprofiler_core.preferences import get_tertiary_outline_color
from cellprofiler_core.preferences import get_title_font_name
from cellprofiler_core.preferences import get_title_font_size
from cellprofiler_core.utilities.core.object import overlay_labels

from ._navigation_toolbar import NavigationToolbar
from ._outline_artist import OutlineArtist
from ..artist import CPImageArtist
from ..constants.figure import COLOR_NAMES
from ..constants.figure import CPLDM_ALPHA
from ..constants.figure import CPLDM_LINES
from ..constants.figure import CPLDM_NONE
from ..constants.figure import CPLDM_OUTLINES
from ..constants.figure import CPLD_ALPHA_COLORMAP
from ..constants.figure import CPLD_ALPHA_VALUE
from ..constants.figure import CPLD_LABELS
from ..constants.figure import CPLD_LINE_WIDTH
from ..constants.figure import CPLD_MODE
from ..constants.figure import CPLD_NAME
from ..constants.figure import CPLD_OUTLINE_COLOR
from ..constants.figure import CPLD_SHOW
from ..constants.figure import EVT_NAV_MODE_CHANGE
from ..constants.figure import MATPLOTLIB_FILETYPES
from ..constants.figure import MATPLOTLIB_UNSUPPORTED_FILETYPES
from ..constants.figure import MENU_CLOSE_ALL
from ..constants.figure import MENU_CLOSE_WINDOW
from ..constants.figure import MENU_FILE_SAVE
from ..constants.figure import MENU_FILE_SAVE_TABLE
from ..constants.figure import MENU_LABELS_ALPHA
from ..constants.figure import MENU_LABELS_LINES
from ..constants.figure import MENU_LABELS_OFF
from ..constants.figure import MENU_LABELS_OUTLINE
from ..constants.figure import MENU_LABELS_OVERLAY
from ..constants.figure import MENU_RGB_CHANNELS
from ..constants.figure import MENU_SAVE_SUBPLOT
from ..constants.figure import MENU_TOOLS_MEASURE_LENGTH
from ..constants.figure import MODE_MEASURE_LENGTH
from ..constants.figure import MODE_NONE
from ..constants.figure import NAV_MODE_NONE
from ..constants.figure import WINDOW_IDS
from ..constants.figure import MENU_INTERPOLATION_NEAREST
from ..constants.figure import MENU_INTERPOLATION_BILINEAR
from ..constants.figure import MENU_INTERPOLATION_BICUBIC
from ..help import make_help_menu
from ..help.content import FIGURE_HELP
from ..tools import renumber_labels_for_display
from ..utilities.figure import allow_sharexy
from ..utilities.figure import close_all
from ..utilities.figure import create_or_find
from ..utilities.figure import find_fig
from ..utilities.figure import format_plate_data_as_array
from ..utilities.figure import get_matplotlib_interpolation_preference
from ..utilities.figure import get_menu_id
from ..utilities.figure import match_rgbmask_to_image
from ..utilities.figure import wraparound
from ..utilities.icon import get_cp_icon

LOGGER = logging.getLogger(__name__)

for filetype in matplotlib.backends.backend_wxagg.FigureCanvasWxAgg.filetypes:
    if filetype not in MATPLOTLIB_FILETYPES:
        MATPLOTLIB_UNSUPPORTED_FILETYPES.append(filetype)

for filetype in MATPLOTLIB_UNSUPPORTED_FILETYPES:
    del matplotlib.backends.backend_wxagg.FigureCanvasWxAgg.filetypes[filetype]


class Figure(wx.Frame):
    """A wx.Frame with a figure inside"""

    def __init__(
        self,
        parent=None,
        identifier=-1,
        title="",
        pos=wx.DefaultPosition,
        size=wx.DefaultSize,
        style=wx.DEFAULT_FRAME_STYLE,
        name=wx.FrameNameStr,
        subplots=None,
        on_close=None,
        secret_panel_class=None,
        help_menu_items=FIGURE_HELP,
    ):
        """Initialize the frame:

        parent   - parent window to this one, typically CPFrame
        id       - window ID
        title    - title in title bar
        pos      - 2-tuple position on screen in pixels
        size     - 2-tuple size of frame in pixels
        style    - window style
        name     - searchable window name
        subplots - 2-tuple indicating the layout of subplots inside the window
        on_close - a function to run when the window closes
        secret_panel_class - class to use to construct the secret panel
        help_menu_items - menu items to place in the help menu
        """
        if pos == wx.DefaultPosition:
            pos = get_next_cpfigure_position()
        super(Figure, self).__init__(parent, identifier, title, pos, size, style, name)
        self.close_fn = on_close
        self.mouse_mode = MODE_NONE
        self.length_arrow = None
        self.table = None
        self.current_plane = 0
        self.images = {}
        self.colorbar = {}
        self.volumetric = False
        self.subplot_params = {}
        self.subplot_user_params = {}
        self.event_bindings = {}
        self.popup_menus = {}
        self.subplot_menus = {}
        self.widgets = []
        self.mouse_down = None
        self.many_plots = False
        self.remove_menu = []
        self.figure = matplotlib.pyplot.Figure(constrained_layout=True)
        self.figure.set_constrained_layout_pads(
            w_pad=0.1, h_pad=0.05, wspace=0, hspace=0
        )
        self.panel = matplotlib.backends.backend_wxagg.FigureCanvasWxAgg(
            self, -1, self.figure
        )
        self.__gridspec = None
        self.dimensions = 2
        if secret_panel_class is None:
            secret_panel_class = wx.Panel
        self.secret_panel = secret_panel_class(self)
        self.secret_panel.Hide()
        self.status_bar = self.CreateStatusBar()
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.Bind(wx.EVT_SIZE, self.on_size)
        if subplots:
            self.subplots = numpy.zeros(subplots, dtype=object)
        self.create_menu(help_menu_items)
        self.create_toolbar()
        self.figure.canvas.mpl_connect("button_press_event", self.on_button_press)
        self.figure.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.figure.canvas.mpl_connect("button_release_event", self.on_button_release)
        self.figure.canvas.mpl_connect("resize_event", self.on_resize)
        try:
            self.SetIcon(get_cp_icon())
        except:
            pass
        if size == wx.DefaultSize:
            self.panel.SetInitialSize(wx.Size(640, 480))
            self.panel.SetMinSize(wx.Size(320, 240))
            self.Fit()
        else:
            self.Layout()
        self.Show()
        if sys.platform.lower().startswith("win"):
            try:
                parent_menu_bar = parent.MenuBar
            except:
                # when testing, there may be no parent
                parent_menu_bar = None
            if parent_menu_bar is not None and isinstance(parent_menu_bar, wx.MenuBar):
                for menu, label in parent_menu_bar.GetMenus():
                    if "Window" in label:
                        menu_ids = [menu_item.Id for menu_item in menu.MenuItems]
                        for window_id in WINDOW_IDS + [None]:
                            if window_id not in menu_ids:
                                break
                        if window_id is None:
                            window_id = wx.NewId()
                            WINDOW_IDS.append(window_id)
                        assert isinstance(menu, wx.Menu)
                        menu.Append(window_id, title)

                        def on_menu_command(event):
                            self.Raise()

                        parent.Bind(wx.EVT_MENU, on_menu_command, id=window_id)
                        self.remove_menu.append([menu, window_id])

    def create_menu(self, figure_help=FIGURE_HELP):
        self.MenuBar = wx.MenuBar()
        self.__menu_file = wx.Menu()
        self.__menu_file.Append(MENU_FILE_SAVE, "&Save")
        self.__menu_file.Append(MENU_FILE_SAVE_TABLE, "&Save table")
        self.__menu_file.Enable(MENU_FILE_SAVE_TABLE, False)
        self.Bind(wx.EVT_MENU, self.on_file_save, id=MENU_FILE_SAVE)
        self.Bind(wx.EVT_MENU, self.on_file_save_table, id=MENU_FILE_SAVE_TABLE)
        self.MenuBar.Append(self.__menu_file, "&File")

        self.menu_subplots = wx.Menu()
        self.MenuBar.Append(self.menu_subplots, "Subplots")

        self.Bind(wx.EVT_MENU, self.on_measure_length, id=MENU_TOOLS_MEASURE_LENGTH)

        # work around mac window menu losing bindings
        if wx.Platform == "__WXMAC__":
            hidden_menu = wx.Menu()
            hidden_menu.Append(MENU_CLOSE_ALL, "&L")
            self.Bind(
                wx.EVT_MENU, lambda evt: close_all(self.Parent), id=MENU_CLOSE_ALL
            )
            accelerators = wx.AcceleratorTable(
                [
                    (wx.ACCEL_CMD, ord("W"), MENU_CLOSE_WINDOW),
                    (wx.ACCEL_CMD, ord("L"), MENU_CLOSE_ALL),
                ]
            )
        else:
            accelerators = wx.AcceleratorTable(
                [(wx.ACCEL_CMD, ord("W"), MENU_CLOSE_WINDOW)]
            )

        self.SetAcceleratorTable(accelerators)
        self.Bind(wx.EVT_MENU, self.on_close, id=MENU_CLOSE_WINDOW)
        self.MenuBar.Append(make_help_menu(figure_help, self), "&Help")

    def create_toolbar(self):
        self.navtoolbar = NavigationToolbar(self.figure.canvas, want_measure=True)
        self.navtoolbar.Bind(EVT_NAV_MODE_CHANGE, self.on_navtool_changed)

    def clf(self):
        """Clear the figure window, resetting the display"""
        self.figure.clf()
        if hasattr(self, "subplots"):
            self.subplots[:, :] = None
        # Remove the subplot menus
        for (x, y) in self.subplot_menus:
            self.menu_subplots.Remove(self.subplot_menus[(x, y)])
        for (x, y) in self.event_bindings:
            [self.figure.canvas.mpl_disconnect(b) for b in self.event_bindings[(x, y)]]
        self.subplot_menus = {}
        self.subplot_params = {}
        self.subplot_user_params = {}
        self.colorbar = {}
        self.images = {}
        for x, y, width, height, halign, valign, ctrl in self.widgets:
            ctrl.Destroy()
        self.widgets = []

    def on_resize(self, event):
        """Handle mpl_connect('resize_event')"""
        assert isinstance(event, matplotlib.backend_bases.ResizeEvent)
        for x, y, width, height, halign, valign, ctrl in self.widgets:
            self.align_widget(
                ctrl, x, y, width, height, halign, valign, event.width, event.height
            )
            ctrl.ForceRefresh()  # I don't know why, but it seems to be needed.

    @staticmethod
    def align_widget(
        ctrl, x, y, width, height, halign, valign, canvas_width, canvas_height
    ):
        """Align a widget within the canvas

        ctrl - the widget to be aligned

        x, y - the fractional position (0 <= {x,y} <= 1) of the top-left of the
               allotted space for the widget

        width, height - the fractional width and height of the allotted space

        halign, valign - alignment of the widget if its best size is smaller
                         than the space (wx.ALIGN_xx or wx.EXPAND)

        canvas_width, canvas_height - the width and height of the canvas parent
        """
        assert isinstance(ctrl, wx.Window)
        x = x * canvas_width
        y = y * canvas_height
        width = width * canvas_width
        height = height * canvas_height

        best_width, best_height = ctrl.GetBestSize()
        vscroll_x = wx.SystemSettings.GetMetric(wx.SYS_VSCROLL_X)
        hscroll_y = wx.SystemSettings.GetMetric(wx.SYS_HSCROLL_Y)
        if height < best_height:
            #
            # If the control's ideal height is less than what's allowed
            # then we have to account for the scroll bars
            #
            best_width += vscroll_x
        if width < best_width:
            best_height += hscroll_y

        if height > best_height and valign != wx.EXPAND:
            if valign == wx.ALIGN_BOTTOM:
                y = y + height - best_height
                height = best_height
            elif valign in (wx.ALIGN_CENTER, wx.ALIGN_CENTER_VERTICAL):
                y += (height - best_height) / 2
            height = best_height
        if width > best_width:
            if halign == wx.ALIGN_RIGHT:
                x = x + width - best_width
            elif halign in (wx.ALIGN_CENTER, wx.ALIGN_CENTER_VERTICAL):
                x += (width - best_width) / 2
            width = best_width
        ctrl.SetPosition(wx.Point(x, y))
        ctrl.SetSize(wx.Size(width, height))

    def on_size(self, event):
        """Handle resizing of canvas, bars and secret panel

        Sizers have proven to be too unpredictable and useless. So
        we do it manually here. Reinventing the wheel is so much quicker
        and works much better.
        """
        if any([not hasattr(self, bar) for bar in ("navtoolbar", "status_bar")]):
            return
        available_width, available_height = self.GetClientSize()
        nbheight = self.navtoolbar.GetSize()[1]

        # On some operating systems (specifically Ubuntu), the matplotlib toolbar
        # may not render with an appropriate height. This is the result of creating
        # the navtoolbar without an appropriate slider. Until this is refactored
        # to make it size properly, this check ensures the navtoolbar at least appears.
        # https://github.com/CellProfiler/CellProfiler/issues/2679
        if nbheight < 40:
            nbheight = 40

        self.navtoolbar.SetPosition((0, 0))
        self.navtoolbar.SetSize((available_width, nbheight))

        if self.secret_panel.IsShown():
            sp_width = self.secret_panel.GetVirtualSize()[0]
            canvas_width = min(
                max(available_width - sp_width, 250), available_width - 100
            )
            sp_width = available_width - canvas_width
            self.secret_panel.SetPosition((canvas_width, nbheight))
            self.secret_panel.SetSize((sp_width, available_height - nbheight))
            self.secret_panel.Layout()
            self.secret_panel.SetupScrolling()
            self.secret_panel.ClearBackground()
            self.secret_panel.Refresh()
            for kid in self.secret_panel.GetChildren():
                kid.Refresh()
                kid.Update()
        else:
            canvas_width = available_width

        self.panel.SetPosition((0, nbheight))
        self.panel.SetSize((canvas_width, available_height - nbheight))
        self.ClearBackground()

    def on_close(self, event):
        if self.close_fn is not None:
            self.close_fn(event)

        self.clf()  # Free memory allocated by imshow

        for menu, menu_id in self.remove_menu:
            self.Parent.Unbind(wx.EVT_MENU, id=menu_id)

            menu.Delete(menu_id)

        self.Destroy()

    def on_navtool_changed(self, event):
        if event.EventObject.mode == "measure":
            self.on_measure_length(event)
        elif self.mouse_mode == MODE_MEASURE_LENGTH:
            self.mouse_mode = MODE_NONE


    def on_measure_length(self, event):
        """Measure length menu item selected."""
        if self.mouse_mode == MODE_NONE:
            self.mouse_mode = MODE_MEASURE_LENGTH
        else:
            self.mouse_mode = MODE_NONE
            self.navtoolbar.cancel_mode()
            self.Layout()

    def on_button_press(self, event):
        if not hasattr(self, "subplots"):
            return
        if event.inaxes in self.subplots.flatten() or self.dimensions == 3:
            if event.xdata is not None:
                self.mouse_down = (event.xdata, event.ydata)
            else:
                self.mouse_down = None
            if self.mouse_mode == MODE_MEASURE_LENGTH:
                self.on_measure_length_mouse_down(event)

    def on_measure_length_mouse_down(self, event):
        pass

    def on_mouse_move(self, evt):
        if self.mouse_down is None:
            x0 = evt.xdata
            x1 = evt.xdata
            y0 = evt.ydata
            y1 = evt.ydata
        else:
            x0 = min(self.mouse_down[0], evt.xdata or self.mouse_down[0])
            x1 = max(self.mouse_down[0], evt.xdata or self.mouse_down[0])
            y0 = min(self.mouse_down[1], evt.ydata or self.mouse_down[1])
            y1 = max(self.mouse_down[1], evt.ydata or self.mouse_down[1])

        if self.mouse_mode == MODE_MEASURE_LENGTH:
            self.on_mouse_move_measure_length(evt, x0, y0, x1, y1)
        else:
            self.on_mouse_move_show_pixel_data(evt, x0, y0, x1, y1)

    def get_pixel_data_fields_for_status_bar(self, image, xi, yi):
        fields = []
        is_float = True

        x, y = [int(round(xy)) for xy in (xi, yi)]

        if not self.in_bounds(image, x, y):
            return fields

        if numpy.issubdtype(image.dtype, numpy.integer):
            is_float = False

        if image.dtype.type == numpy.uint8:
            image = image.astype(numpy.float32) / 255.0
        if image.ndim == 2:
            if is_float:
                fields += ["Intensity: {:.4f}".format(image[y, x] or 0)]
            # This is to allow intensity values to be displayed more intuitively
            else:
                fields += ["Intensity: {:d}".format(image[y, x] or 0)]

        elif image.ndim == 3 and image.shape[2] == 3:
            fields += [
                "Red: %.4f" % (image[y, x, 0]),
                "Green: %.4f" % (image[y, x, 1]),
                "Blue: %.4f" % (image[y, x, 2]),
            ]
        elif image.ndim == 3:
            fields += [
                "Channel %d: %.4f" % (idx + 1, image[y, x, idx])
                for idx in range(image.shape[2])
            ]

        return fields

    @staticmethod
    def in_bounds(image, xi, yi):
        """Return false if xi or yi are outside of the bounds of the image"""
        return not (
            image is None
            or xi >= image.shape[1]
            or yi >= image.shape[0]
            or xi < 0
            or yi < 0
        )

    def on_mouse_move_measure_length(self, event, x0, y0, x1, y1):
        # Get the fields later populated in the statusbar
        fields = None
        if event.xdata and event.ydata:
            xi = int(event.xdata + 0.5)
            yi = int(event.ydata + 0.5)
            fields = self.get_fields(event, yi, xi, x1)
        else:
            # Mouse has moved off the plot, stop updating.
            return

        # Calculate the length field if mouse is down
        if self.mouse_down is not None:
            x0 = min(self.mouse_down[0], event.xdata)
            x1 = max(self.mouse_down[0], event.xdata)
            y0 = min(self.mouse_down[1], event.ydata)
            y1 = max(self.mouse_down[1], event.ydata)

            length = numpy.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
            fields.append("Length: %.1f" % length)
            xinterval = event.inaxes.xaxis.get_view_interval()
            yinterval = event.inaxes.yaxis.get_view_interval()
            diagonal = numpy.sqrt(
                (xinterval[1] - xinterval[0]) ** 2 + (yinterval[1] - yinterval[0]) ** 2
            )
            mutation_scale = max(min(int(length * 100 / diagonal), 20), 1)
            if self.length_arrow is not None:
                self.length_arrow.set_positions(
                    (self.mouse_down[0], self.mouse_down[1]), (event.xdata, event.ydata)
                )
                self.length_arrow.set_mutation_scale(mutation_scale)
            else:
                self.length_arrow = matplotlib.patches.FancyArrowPatch(
                    (self.mouse_down[0], self.mouse_down[1]),
                    (event.xdata, event.ydata),
                    edgecolor="blue",
                    arrowstyle="<->",
                    mutation_scale=mutation_scale,
                )
                try:
                    event.inaxes.add_patch(self.length_arrow)
                except:
                    self.length_arrow = None
            self.figure.canvas.draw()
            self.Refresh()

        # Update the statusbar
        if fields:
            self.status_bar.SetFieldsCount(len(fields))

            for idx, field in enumerate(fields):
                self.status_bar.SetStatusText(field, i=idx)
        else:
            self.status_bar.SetFieldsCount(1)
            self.status_bar.SetStatusText("")

    def get_fields(self, event, yi, xi, x1):
        """Get the standard fields at the cursor location"""
        if event.inaxes:
            fields = ["X: %d" % xi, "Y: %d" % yi]

            if self.dimensions >= 2:
                im = self.find_image_for_axes(event.inaxes)
            else:  # self.dimensions == 3
                axes = event.inaxes

                axes_image_list = axes.get_images()

                if len(axes_image_list):
                    axes_image = axes_image_list[0]

                    fields += ["Z: {}".format(axes.get_label())]

                    im = axes_image.get_array().data
                else:
                    im = None

            if im is not None:
                fields += self.get_pixel_data_fields_for_status_bar(im, x1, yi)
            elif isinstance(event.inaxes, matplotlib.axes.Axes):
                for artist in event.inaxes.artists:
                    if isinstance(artist, CPImageArtist):
                        fields += [
                            "%s: %.4f" % (k, v)
                            for k, v in list(artist.get_channel_values(xi, yi).items())
                        ]
        else:
            fields = []

        return fields

    def on_mouse_move_show_pixel_data(self, event, x0, y0, x1, y1):
        # Get the fields later populated in the statusbar
        fields = None
        if event.xdata and event.ydata:
            xi = int(event.xdata + 0.5)
            yi = int(event.ydata + 0.5)
            fields = self.get_fields(event, yi, xi, x1)

        # Update the statusbar
        if fields:
            self.status_bar.SetFieldsCount(len(fields))

            for idx, field in enumerate(fields):
                self.status_bar.SetStatusText(field, i=idx)
        else:
            self.status_bar.SetFieldsCount(1)
            self.status_bar.SetStatusText("")

    def find_image_for_axes(self, axes):
        for i, sl in enumerate(self.subplots):
            for j, slax in enumerate(sl):
                if axes == slax:
                    rtnimg = self.images.get((i, j), None)
                    if self.dimensions == 3 and rtnimg is not None:
                        rtnimg = rtnimg[self.current_plane, :, :]
                    return rtnimg
        return None

    def on_button_release(self, event):
        if not hasattr(self, "subplots"):
            return

        if (
            event.inaxes in self.subplots.flatten()
            and self.mouse_down
            and event.xdata is not None
        ):
            x0 = min(self.mouse_down[0], event.xdata)
            x1 = max(self.mouse_down[0], event.xdata)
            y0 = min(self.mouse_down[1], event.ydata)
            y1 = max(self.mouse_down[1], event.ydata)

            if self.mouse_mode == MODE_MEASURE_LENGTH:
                self.on_measure_length_done(event, x0, y0, x1, y1)
        elif self.mouse_down:
            if self.mouse_mode == MODE_MEASURE_LENGTH:
                self.on_measure_length_canceled(event)

        self.mouse_down = None

    def on_measure_length_done(self, event, x0, y0, x1, y1):
        self.on_measure_length_canceled(event)

    def on_measure_length_canceled(self, event):
        if self.length_arrow is not None:
            self.length_arrow.remove()

            self.length_arrow = None

        self.figure.canvas.draw()

        self.Refresh()

    def on_file_save(self, event):
        with wx.FileDialog(
            self,
            "Save figure",
            wildcard="PDF file (*.pdf)|*.pdf|PNG image (*.png)|*.png",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        ) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                path = dlg.GetPath()

                if dlg.FilterIndex == 1:
                    file_format = "png"
                elif dlg.FilterIndex == 0:
                    file_format = "pdf"
                elif dlg.FilterIndex == 2:
                    file_format = "tif"
                elif dlg.FilterIndex == 3:
                    file_format = "jpg"
                else:
                    file_format = "pdf"

                if "." not in os.path.split(path)[1]:
                    path += "." + file_format

                self.figure.savefig(path, format=file_format)

    def on_file_save_table(self, event):
        if self.table is None:
            return

        with wx.FileDialog(
            self,
            "Save table",
            wildcard="Excel file (*.csv)|*.csv",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        ) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                path = dlg.GetPath()

                with open(path, "w", newline="") as fd:
                    csv.writer(fd).writerows(self.table)

    def on_file_save_subplot(self, event, x, y):
        """Save just the contents of a subplot w/o decorations

        event - event generating the request

        x, y - the placement of the subplot
        """
        #
        # Thank you Joe Kington
        # http://stackoverflow.com/questions/4325733/save-a-subplot-in-matplotlib
        #
        ax = self.subplots[x, y]

        extent = ax.get_window_extent().transformed(
            self.figure.dpi_scale_trans.inverted()
        )

        with wx.FileDialog(
            self,
            "Save axes",
            wildcard="PDF file (*.pdf)|*.pdf|Png image (*.png)|*.png|Postscript file (*.ps)|*.ps",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        ) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                path = dlg.GetPath()

                if dlg.FilterIndex == 1:
                    file_format = "png"
                elif dlg.FilterIndex == 0:
                    file_format = "pdf"
                elif dlg.FilterIndex == 2:
                    file_format = "ps"
                else:
                    file_format = "pdf"

                self.figure.savefig(path, format=file_format, bbox_inches=extent)

    def set_subplots(self, subplots, dimensions=2):
        self.clf()  # get rid of any existing subplots, menus, etc.

        self.dimensions = dimensions

        if max(subplots) > 3:
            self.many_plots = True

        if subplots is None:
            if hasattr(self, "subplots"):
                delattr(self, "subplots")
        else:
            if dimensions >= 2:
                self.subplots = numpy.zeros(subplots, dtype=object)
                self.__gridspec = matplotlib.gridspec.GridSpec(
                    subplots[1], subplots[0], figure=self.figure
                )
            else:
                self.set_grids(subplots)

    @allow_sharexy
    def subplot(self, x, y, sharex=None, sharey=None):
        """Return the indexed subplot

        x - column
        y - row
        sharex - If creating a new subplot, you can specify a subplot instance
                 here to share the X axis with. eg: for zooming, panning
        sharey - If creating a new subplot, you can specify a subplot instance
                 here to share the Y axis with. eg: for zooming, panning
        """
        if self.dimensions == 4:
            return None
        if not self.subplots[x, y]:
            if self.__gridspec:
                # Add the plot to a premade subplot layout
                plot = self.figure.add_subplot(
                    self.__gridspec[y, x], sharex=sharex, sharey=sharey,
                )
            else:
                rows, cols = self.subplots.shape
                plot = self.figure.add_subplot(
                    cols, rows, x + y * rows + 1, sharex=sharex, sharey=sharey
                )

            self.subplots[x, y] = plot

        return self.subplots[x, y]

    def set_subplot_title(self, title, x, y):
        """Set a subplot's title in the standard format

        title - title for subplot
        x - subplot's column
        y - subplot's row
        """
        fontname = get_title_font_name()
        if self.many_plots:
            fontsize = 8
        else:
            fontsize = get_title_font_size()
        self.subplot(x, y).set_title(
            textwrap.fill(title, 30 if fontsize > 10 else 50), fontname=fontname, fontsize=fontsize,
        )

    def clear_subplot(self, x, y):
        """Clear a subplot of its gui junk. Noop if no subplot exists at x,y

        x - subplot's column
        y - subplot's row
        """
        if not self.subplots[x, y]:
            return

        axes = self.subplot(x, y)

        try:
            del self.images[(x, y)]

            del self.popup_menus[(x, y)]
        except:
            pass

        axes.clear()

    def show_imshow_popup_menu(self, pos, subplot_xy):
        popup = self.get_imshow_menu(subplot_xy)
        self.PopupMenu(popup, pos)

    def get_imshow_menu(self, coordinates):
        """returns a menu corresponding to the specified subplot with items to:
        - launch the image in a new cpfigure window
        - Show image histogram
        - Change contrast stretching
        - Toggle channels on/off
        Note: Each item is bound to a handler.
        """
        (x, y) = coordinates

        params = self.subplot_params[(x, y)]

        # If no popup has been built for this subplot yet, then create one
        popup = wx.Menu()
        self.popup_menus[(x, y)] = popup
        has_image = params["vmax"] is not None
        open_in_new_figure_item = wx.MenuItem(popup, -1, "Open image in new window")
        popup.Append(open_in_new_figure_item)
        if has_image:
            contrast_item = wx.MenuItem(popup, -1, "Adjust Contrast")
            popup.Append(contrast_item)
            show_hist_item = wx.MenuItem(popup, -1, "Show image histogram")
            popup.Append(show_hist_item)

        submenu = wx.Menu()
        item_nearest = submenu.Append(
            MENU_INTERPOLATION_NEAREST,
            "Nearest neighbor",
            "Use the intensity of the nearest image pixel when displaying "
            "screen pixels at sub-pixel resolution. This produces a blocky "
            "image, but the image accurately reflects the data.",
            wx.ITEM_RADIO,
        )
        item_bilinear = submenu.Append(
            MENU_INTERPOLATION_BILINEAR,
            "Linear",
            "Use the weighted average of the four nearest image pixels when "
            "displaying screen pixels at sub-pixel resolution. This produces "
            "a smoother, more visually appealing image, but makes it more "
            "difficult to find pixel borders",
            wx.ITEM_RADIO,
        )
        item_bicubic = submenu.Append(
            MENU_INTERPOLATION_BICUBIC,
            "Cubic",
            "Perform a bicubic interpolation of the nearby image pixels when "
            "displaying screen pixels at sub-pixel resolution. This produces "
            "the most visually appealing image but is the least faithful to "
            "the image pixel values.",
            wx.ITEM_RADIO,
        )
        popup.Append(-1, "Interpolation", submenu)
        save_subplot_id = get_menu_id(MENU_SAVE_SUBPLOT, (x, y))
        popup.Append(
            save_subplot_id,
            "Save subplot",
            "Save just the display portion of this subplot",
        )

        if params["interpolation"] == "bilinear":
            item_bilinear.Check()
        elif params["interpolation"] == "bicubic":
            item_bicubic.Check()
        else:
            item_nearest.Check()

        def open_image_in_new_figure(evt):
            """Callback for "Open image in new window" popup menu item """
            # Store current zoom limits
            xlims = self.subplot(x, y).get_xlim()
            ylims = self.subplot(x, y).get_ylim()
            new_title = self.subplot(x, y).get_title()
            fig = create_or_find(
                self, -1, new_title, subplots=(1, 1), name=str(uuid.uuid4())
            )
            fig.dimensions = self.dimensions
            fig.subplot_imshow(0, 0, self.images[(x, y)], **params)

            # XXX: Cheat here so the home button works.
            # This needs to be fixed so it copies the view history for the
            # launched subplot to the new figure.
            fig.navtoolbar.push_current()

            # Set current zoom
            fig.subplot(0, 0).set_xlim(xlims[0], xlims[1])
            fig.subplot(0, 0).set_ylim(ylims[0], ylims[1])
            fig.figure.canvas.draw()

        def show_hist(evt):
            """Callback for "Show image histogram" popup menu item"""
            new_title = "%s %s image histogram" % (self.Title, (x, y))
            fig = create_or_find(self, -1, new_title, subplots=(1, 1), name=new_title)
            fig.subplot_histogram(
                0, 0, self.images[(x, y)].flatten(), bins=200, xlabel="pixel intensity"
            )
            fig.figure.canvas.draw()

        def open_contrast_dialog(evt):
            nonlocal params
            orig_params = params.copy()
            id_dict = {
                False: "Raw",
                True: "Normalized",
                "log": "Log Normalized",
                "gamma": "Gamma Normalized",
            }
            if params["normalize"]:
                maxval = 1
                minval = 0
            else:
                imgmax = orig_params["vmax"]
                imgmin = orig_params["vmin"]
                maxval = max(1, imgmax)
                minval = min(0, imgmin)
            start_min = int((params["vmin"] / minval) * 255 if minval != 0 else 0)
            start_max = int((params["vmax"] / maxval) * 255)
            axes = self.subplot(x, y)
            background = self.figure.canvas.copy_from_bbox(axes.bbox)
            size = self.images[(x, y)].shape[:2]
            axesdata = axes.plot([0, 0], list(size), "k")[0]
            if sys.platform == "win32":
                slider_flags = wx.SL_HORIZONTAL | wx.SL_MIN_MAX_LABELS
            else:
                slider_flags = wx.SL_HORIZONTAL

            with wx.Dialog(
                self, title="Adjust Contrast", size=wx.Size(250, 350)
            ) as dlg:
                dlg.Sizer = wx.BoxSizer(wx.VERTICAL)

                sizer = wx.BoxSizer(wx.VERTICAL)
                dlg.Sizer.Add(sizer, 1, wx.EXPAND | wx.ALL, border=5)
                sizer.Add(
                    wx.StaticText(dlg, label="Normalization Mode"), 0, wx.ALIGN_LEFT,
                )

                method_select = wx.ComboBox(
                    dlg,
                    id=2,
                    choices=list(id_dict.values()),
                    value=id_dict[params["normalize"]],
                    style=wx.CB_READONLY,
                )
                sizer.Add(method_select, flag=wx.ALL | wx.EXPAND, border=3)

                sizer.Add(
                    wx.StaticText(dlg, label="Minimum Brightness"), 0, wx.ALIGN_LEFT,
                )
                slidermin = wx.Slider(
                    dlg,
                    id=0,
                    value=start_min,
                    minValue=0,
                    maxValue=255,
                    style=slider_flags,
                    name="Minimum Intensity",
                )
                sliderminbox = wx.lib.intctrl.IntCtrl(
                    dlg,
                    id=0,
                    value=start_min,
                    min=-999,
                    max=9999,
                    limited=True,
                    size=wx.Size(35, 22),
                    style=wx.TE_CENTRE,
                )
                minbright_sizer = wx.BoxSizer()
                minbright_sizer.Add(slidermin, 1, wx.EXPAND)
                minbright_sizer.AddSpacer(4)
                minbright_sizer.Add(sliderminbox)
                sizer.Add(minbright_sizer, 1, wx.EXPAND)

                sizer.Add(
                    wx.StaticText(dlg, label="Maximum Brightness"), 0, wx.ALIGN_LEFT,
                )
                slidermax = wx.Slider(
                    dlg,
                    id=1,
                    value=start_max,
                    minValue=0,
                    maxValue=255,
                    style=slider_flags,
                    name="Maximum Intensity",
                )
                slidermaxbox = wx.lib.intctrl.IntCtrl(
                    dlg,
                    id=1,
                    value=start_max,
                    min=-999,
                    max=9999,
                    limited=True,
                    size=wx.Size(35, 22),
                    style=wx.TE_CENTRE,
                )
                maxbright_sizer = wx.BoxSizer()
                maxbright_sizer.Add(slidermax, 1, wx.EXPAND)
                maxbright_sizer.AddSpacer(4)
                maxbright_sizer.Add(slidermaxbox)
                sizer.Add(maxbright_sizer, 1, wx.EXPAND)
                normtext = wx.StaticText(dlg, label="Normalization Factor")
                sizer.Add(
                    normtext, 0, wx.ALIGN_LEFT,
                )
                slidernorm = wx.Slider(
                    dlg,
                    id=2,
                    value=float(get_normalization_factor()),
                    minValue=0,
                    maxValue=20,
                    style=slider_flags,
                    name="Normalization Factor",
                )
                slidernormbox = wx.lib.masked.NumCtrl(
                    dlg,
                    id=2,
                    value=float(get_normalization_factor()),
                    min=0,
                    max=1000,
                    limited=True,
                    size=wx.Size(35, 22),
                    integerWidth=4,
                    fractionWidth=1,
                    autoSize=False,
                    style=wx.TE_CENTRE,
                )
                norm_sizer = wx.BoxSizer()
                norm_sizer.Add(slidernorm, 1, wx.EXPAND)
                norm_sizer.AddSpacer(4)
                norm_sizer.Add(slidernormbox)
                sizer.Add(norm_sizer, 1, wx.EXPAND)

                if params["normalize"] not in ("log", "gamma"):
                    normtext.Enable(False)
                    slidernorm.Enable(False)
                    slidernormbox.Enable(False)
                if self.subplots.shape != (1, 1):
                    sizer.Add(
                        wx.Button(dlg, wx.ID_APPLY, label="Apply to all"),
                        0,
                        wx.ALIGN_CENTER_HORIZONTAL,
                    )
                button_sizer = wx.StdDialogButtonSizer()
                button_sizer.AddButton(wx.Button(dlg, wx.ID_OK))
                button_sizer.AddButton(wx.Button(dlg, wx.ID_CANCEL))
                dlg.Sizer.Add(button_sizer, 0, wx.ALIGN_CENTER_HORIZONTAL)
                button_sizer.Realize()

                def on_slider(event):
                    if event.Id == 0:
                        sliderminbox.ChangeValue(slidermin.GetValue())
                        sliderminbox.Update()
                    elif event.Id == 1:
                        slidermaxbox.ChangeValue(slidermax.GetValue())
                        slidermaxbox.Update()
                    elif event.Id == 2:
                        slidernormbox.ChangeValue(slidernorm.GetValue())
                        slidernormbox.Update()
                    event.Skip()

                def on_int_entry(event):
                    if event.Id == 0:
                        slidermin.SetValue(sliderminbox.GetValue())
                    elif event.Id == 1:
                        slidermax.SetValue(slidermaxbox.GetValue())
                    elif event.Id == 2:
                        slidernorm.SetValue(slidernormbox.GetValue())
                    apply_contrast(event)

                def apply_contrast(event):
                    current_max = slidermaxbox.GetValue()
                    current_min = sliderminbox.GetValue()
                    if event.Id == 0:
                        if current_max <= current_min:
                            slidermax.SetValue(current_min)
                            slidermaxbox.SetValue(current_min)
                    elif event.Id == 1:
                        if current_min >= current_max:
                            slidermin.SetValue(current_max)
                            sliderminbox.SetValue(current_max)
                    elif event.Id == 2:
                        if params["normalize"] == "log":
                            params["normalize_args"] = {
                                "gain": float(slidernormbox.GetValue())
                            }
                        elif params["normalize"] == "gamma":
                            params["normalize_args"] = {
                                "gamma": float(slidernormbox.GetValue())
                            }
                    params["vmin"] = (sliderminbox.GetValue() / 255) * maxval
                    params["vmax"] = (slidermaxbox.GetValue() / 255) * maxval
                    refresh_figure(axes, background, axesdata)

                def change_contrast_mode(event):
                    newvalue = next(
                        paramvalue
                        for paramvalue, listvalue in id_dict.items()
                        if listvalue == method_select.GetValue()
                    )
                    if newvalue == params["normalize"]:
                        return
                    params["normalize"] = newvalue
                    params["vmin"] = 0
                    if not newvalue:
                        nonlocal imgmax, maxval, orig_params
                        imgmax = orig_params["vmax"]
                        maxval = max(1, imgmax)
                        params["vmax"] = max(1, self.images[(x, y)].max())
                    else:
                        params["vmax"] = 1
                        if newvalue == "log":
                            params["normalize_args"] = {"gain": 1.0}
                        elif newvalue == "gamma":
                            params["normalize_args"] = {"gamma": 1.0}
                    slidermin.SetValue(0)
                    sliderminbox.SetValue(0)
                    slidermax.SetValue(255)
                    slidermaxbox.SetValue(255)
                    slidernorm.SetValue(float(get_normalization_factor()))
                    slidernormbox.SetValue(float(get_normalization_factor()))
                    want_norm = params["normalize"] in ("log", "gamma")
                    normtext.Enable(want_norm)
                    slidernorm.Enable(want_norm)
                    slidernormbox.Enable(want_norm)
                    refresh_figure(axes, background, axesdata)

                def apply_to_all(event):
                    if event.Id == wx.ID_APPLY:
                        numx, numy = self.subplots.shape
                        for xcoord in range(numx):
                            for ycoord in range(numy):
                                subplot_item = self.subplot(xcoord, ycoord)
                                if hasattr(subplot_item, "displayed"):
                                    plot_params = self.subplot_params[(xcoord, ycoord)]
                                    plot_params["vmin"] = params["vmin"]
                                    plot_params["vmax"] = params["vmax"]
                                    plot_params["normalize"] = params["normalize"]
                                    plot_params["normalize_args"] = params[
                                        "normalize_args"
                                    ]
                                    image = self.images[(xcoord, ycoord)]
                                    if not isinstance(image, numpy.ma.MaskedArray):
                                        img_data = self.normalize_image(
                                            self.images[(xcoord, ycoord)], **plot_params
                                        )
                                        subplot_item.displayed.set_data(img_data)
                                else:
                                    # Should be a table, make sure the invisible subplot stays hidden.
                                    subplot_item.axis("off")
                                    if (
                                        not hasattr(subplot_item, "deleted")
                                        and self.dimensions == 3
                                    ):
                                        self.figure.delaxes(subplot_item)
                                        subplot_item.deleted = True

                        self.figure.canvas.draw()
                    else:
                        event.Skip()

                # For small images we can draw fast enough for a live preview.
                # For large images, we draw when the slider is released.
                if size[0] * size[1] > 1048576 or self.dimensions > 2:  # 1024x1024
                    dlg.Bind(wx.EVT_SCROLL_THUMBRELEASE, apply_contrast)
                    dlg.Bind(wx.EVT_SCROLL_CHANGED, apply_contrast)
                else:
                    dlg.Bind(wx.EVT_SLIDER, apply_contrast)
                dlg.Bind(wx.EVT_COMBOBOX, change_contrast_mode)
                dlg.Bind(wx.EVT_SLIDER, on_slider)
                dlg.Bind(wx.EVT_TEXT, on_int_entry)
                dlg.Bind(wx.lib.masked.EVT_NUM, on_int_entry)
                dlg.Bind(wx.EVT_BUTTON, apply_to_all)
                dlg.Layout()
                if dlg.ShowModal() == wx.ID_OK:
                    return
                else:
                    params = orig_params
                    refresh_figure()

        def refresh_figure(axes=None, background=None, axesdata=None):
            subplot = self.subplot(x, y)
            img_data = self.normalize_image(self.images[(x, y)], **params)
            subplot.displayed.set_data(img_data)
            if axes is not None:
                self.figure.canvas.restore_region(background)
                axes.draw_artist(axesdata)
                axes.draw_artist(subplot.displayed)
                self.figure.canvas.blit(axes.bbox)
            else:
                self.figure.canvas.draw()

        def change_interpolation(evt):
            if evt.Id == MENU_INTERPOLATION_NEAREST:
                params["interpolation"] = "nearest"
            elif evt.Id == MENU_INTERPOLATION_BILINEAR:
                params["interpolation"] = "bilinear"
            elif evt.Id == MENU_INTERPOLATION_BICUBIC:
                params["interpolation"] = "bicubic"
            axes = self.subplot(x, y)
            if hasattr(axes, 'displayed') and hasattr(axes.displayed, "_interpolation"):
                # Directly update interpolation if the subplot is an image object
                axes.displayed._interpolation = params["interpolation"]
            for artist in axes.artists:
                if isinstance(artist, CPImageArtist):
                    artist.interpolation = params["interpolation"]
                    artist.kwargs["interpolation"] = params["interpolation"]
                    self.figure.canvas.draw()
                    return
            else:
                refresh_figure()

        def on_adjust_labels_alpha(labels):
            with wx.Dialog(self, title="Adjust labels transparency") as dlg:
                name = labels.get(CPLD_NAME, "Objects")
                orig_alpha = int(labels[CPLD_ALPHA_VALUE] * 100 + 0.5)
                dlg.Sizer = wx.BoxSizer(wx.VERTICAL)
                sizer = wx.BoxSizer(wx.VERTICAL)
                dlg.Sizer.Add(sizer, 1, wx.EXPAND | wx.ALL, 8)
                sizer.Add(
                    wx.StaticText(dlg, label="%s transparency"),
                    0,
                    wx.ALIGN_CENTER_HORIZONTAL,
                )
                sizer.AddSpacer(4)
                slider = wx.Slider(
                    dlg,
                    value=orig_alpha,
                    minValue=0,
                    maxValue=100,
                    style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS,
                )
                sizer.Add(slider, 1, wx.EXPAND)
                button_sizer = wx.StdDialogButtonSizer()
                button_sizer.AddButton(wx.Button(dlg, wx.ID_OK))
                button_sizer.AddButton(wx.Button(dlg, wx.ID_CANCEL))
                dlg.Sizer.Add(button_sizer)
                button_sizer.Realize()

                dlg.Layout()
                if dlg.ShowModal() != wx.ID_OK:
                    slider.SetValue(orig_alpha)
                else:
                    labels[CPLD_ALPHA_VALUE] = float(slider.GetValue()) / 100.0
                    refresh_figure()

        if self.is_color_image(self.images[x, y]):
            submenu = wx.Menu()
            rgb_mask = match_rgbmask_to_image(params["rgb_mask"], self.images[x, y])
            ids = [
                get_menu_id(MENU_RGB_CHANNELS, (x, y, i)) for i in range(len(rgb_mask))
            ]
            for name, value, identifier in zip(wraparound(COLOR_NAMES), rgb_mask, ids):
                item = submenu.Append(
                    identifier, name, "Show/Hide the %s channel" % name, wx.ITEM_CHECK
                )
                if value != 0:
                    item.Check()
            popup.Append(-1, "Channels", submenu)

            def toggle_channels(evt):
                """Callback for channel menu items."""
                if "rgb_mask" not in params:
                    params["rgb_mask"] = list(rgb_mask)
                else:
                    # copy to prevent modifying shared values
                    params["rgb_mask"] = list(params["rgb_mask"])
                for idx, identifier in enumerate(ids):
                    if identifier == evt.Id:
                        params["rgb_mask"][idx] = not params["rgb_mask"][idx]
                refresh_figure()

            for identifier in ids:
                self.Bind(wx.EVT_MENU, toggle_channels, id=identifier)

        if params["cplabels"] is not None and len(params["cplabels"]) > 0:
            for i, cplabels in enumerate(params["cplabels"]):
                submenu = wx.Menu()
                name = cplabels.get(CPLD_NAME, "Objects #%d" % i)
                for mode, menud, mlabel, mhelp in (
                    (
                        CPLDM_OUTLINES,
                        MENU_LABELS_OUTLINE,
                        "Outlines",
                        "Display outlines of objects",
                    ),
                    (
                        CPLDM_ALPHA,
                        MENU_LABELS_OVERLAY,
                        "Overlay",
                        "Display objects as an alpha-overlay",
                    ),
                    (
                        CPLDM_LINES,
                        MENU_LABELS_LINES,
                        "Lines",
                        "Draw lines around objects",
                    ),
                    (CPLDM_NONE, MENU_LABELS_OFF, "Off", "Turn object labels off"),
                ):
                    menu_id = get_menu_id(menud, (x, y, i))
                    item = submenu.AppendRadioItem(menu_id, mlabel, mhelp)
                    if cplabels[CPLD_MODE] == mode:
                        item.Check()

                    def select_mode(event, cplabels=cplabels, mode=mode):
                        cplabels[CPLD_MODE] = mode
                        self.update_line_labels(self.subplot(x, y), params)
                        refresh_figure()

                    self.Bind(wx.EVT_MENU, select_mode, id=menu_id)
                if cplabels[CPLD_MODE] == CPLDM_ALPHA:
                    menu_id = get_menu_id(MENU_LABELS_ALPHA, (x, y, i))
                    item = submenu.Append(
                        menu_id,
                        "Adjust transparency",
                        "Change the alpha-blend for the labels overlay to make it more or less transparent",
                    )
                    self.Bind(
                        wx.EVT_MENU,
                        lambda event, cplabels=cplabels: on_adjust_labels_alpha(
                            cplabels
                        ),
                        id=menu_id,
                    )
                popup.Append(-1, name, submenu)

        self.Bind(wx.EVT_MENU, open_image_in_new_figure, open_in_new_figure_item)
        if has_image:
            self.Bind(wx.EVT_MENU, open_contrast_dialog, contrast_item)
            self.Bind(wx.EVT_MENU, show_hist, show_hist_item)
        self.Bind(wx.EVT_MENU, change_interpolation, id=MENU_INTERPOLATION_NEAREST)
        self.Bind(wx.EVT_MENU, change_interpolation, id=MENU_INTERPOLATION_BICUBIC)
        self.Bind(wx.EVT_MENU, change_interpolation, id=MENU_INTERPOLATION_BILINEAR)
        self.Bind(
            wx.EVT_MENU,
            lambda event: self.on_file_save_subplot(event, x, y),
            id=MENU_SAVE_SUBPLOT[(x, y)],
        )
        return popup

    def set_grids(self, shape):
        self.__gridspec = matplotlib.gridspec.GridSpec(*shape[::-1])
        self.figure.set_constrained_layout(False)

    def gridshow(
        self, x, y, image, title=None, colormap="gray", colorbar=False, normalize=True
    ):
        gx, gy = self.__gridspec.get_geometry()

        gridspec = matplotlib.gridspec.GridSpecFromSubplotSpec(
            3, 3, subplot_spec=self.__gridspec[gy * y + x], wspace=0.1, hspace=0.1
        )

        z = image.shape[0]

        vmin = min(image[position * (z - 1) // 8].min() for position in range(9))

        vmax = max(image[position * (z - 1) // 8].max() for position in range(9))

        cmap = colormap

        if isinstance(cmap, matplotlib.cm.ScalarMappable):
            cmap = cmap.cmap

        axes = []

        for position in range(9):
            ax = matplotlib.pyplot.Subplot(self.figure, gridspec[position])

            if position == 1 and title is not None:
                ax.set_title(title)

            if position / 3 != 2:
                ax.set_xticklabels([])

            if position % 3 != 0:
                ax.set_yticklabels([])

            norm = (
                matplotlib.colors.SymLogNorm(
                    linthresh=0.03, linscale=0.03, vmin=vmin, vmax=vmax
                )
                if normalize
                else None
            )

            ax.imshow(image[position * (z - 1) // 8], cmap=cmap, norm=norm)

            ax.set_xlabel("Z: {:d}".format(position * (z - 1) // 8))

            self.figure.add_subplot(ax)

            axes += [ax]

        if colorbar:
            colormap.set_array(image)

            colormap.autoscale()

            self.figure.colorbar(colormap, ax=axes)

        matplotlib.pyplot.show()

    @allow_sharexy
    def subplot_imshow(
        self,
        x,
        y,
        image,
        title=None,
        clear=True,
        colormap=None,
        colorbar=False,
        normalize=None,
        vmin=0,
        vmax=1,
        rgb_mask=(1, 1, 1),
        sharex=None,
        sharey=None,
        use_imshow=False,
        interpolation=None,
        cplabels=None,
        normalize_args=None,
    ):
        """Show an image in a subplot

        x, y  - show image in this subplot
        image - image to show
        title - add this title to the subplot
        clear - clear the subplot axes before display if true
        colormap - for a grayscale or labels image, use this colormap
                   to assign colors to the image
        colorbar - display a colorbar if true
        normalize - whether or not to normalize the image. If True, vmin, vmax
                    are ignored.
        vmin, vmax - Used to scale a luminance image to 0-1. If either is None,
                     the min and max of the luminance values will be used.
                     If normalize is True, vmin and vmax will be ignored.
        rgb_mask - 3-element list to be multiplied to all pixel values in the
                   image. Used to show/hide individual channels in color images.
        sharex, sharey - specify a subplot to link axes with (for zooming and
                         panning). Specify a subplot using CPFigure.subplot(x,y)
        use_imshow - True to use Axes.imshow to paint images, False to fill
                     the image into the axes after painting.
        cplabels - a list of dictionaries of labels properties. Each dictionary
                   describes a set of labels. See the documentation of
                   the CPLD_* constants for details.
        """
        if normalize_args is None:
            normalize_args = {}

        orig_vmin = vmin
        orig_vmax = vmax

        if interpolation is None:
            interpolation = get_matplotlib_interpolation_preference()

        if normalize is None:
            normalize = get_intensity_mode()

            if normalize == INTENSITY_MODE_RAW:
                normalize = False
            elif normalize == False:
                normalize = False
            elif normalize == INTENSITY_MODE_LOG:
                normalize = "log"
                normalize_args["gain"] = float(get_normalization_factor())
            elif normalize == INTENSITY_MODE_GAMMA:
                normalize = "gamma"
                normalize_args["gamma"] = float(get_normalization_factor())
            else:
                normalize = True

        if cplabels is None:
            cplabels = []
        else:
            use_imshow = False

            new_cplabels = []

            for i, d in enumerate(cplabels):
                d = d.copy()

                if CPLD_OUTLINE_COLOR not in d:
                    if i == 0:
                        d[CPLD_OUTLINE_COLOR] = get_primary_outline_color()
                    elif i == 1:
                        d[CPLD_OUTLINE_COLOR] = get_secondary_outline_color()
                    elif i == 2:
                        d[CPLD_OUTLINE_COLOR] = get_tertiary_outline_color()
                    else:
                        d[CPLD_OUTLINE_COLOR] = wx.Colour(255, 255, 255)

                if CPLD_MODE not in d:
                    d[CPLD_MODE] = CPLDM_OUTLINES

                if CPLD_LINE_WIDTH not in d:
                    d[CPLD_LINE_WIDTH] = 1

                if CPLD_ALPHA_COLORMAP not in d:
                    d[CPLD_ALPHA_COLORMAP] = get_default_colormap()

                if CPLD_ALPHA_VALUE not in d:
                    d[CPLD_ALPHA_VALUE] = 0.25

                new_cplabels.append(d)

            cplabels = new_cplabels

        # NOTE: self.subplot_user_params is used to store changes that are made
        #    to the display through GUI interactions (eg: hiding a channel).
        #    Once a subplot that uses this mechanism has been drawn, it will
        #    continually load defaults from self.subplot_user_params instead of
        #    the default values specified in the function definition.
        kwargs = {
            "title": title,
            "clear": False,
            "colormap": colormap,
            "colorbar": colorbar,
            "normalize": normalize,
            "vmin": vmin,
            "vmax": vmax,
            "rgb_mask": rgb_mask,
            "use_imshow": use_imshow,
            "interpolation": interpolation,
            "cplabels": cplabels,
            "normalize_args": normalize_args,
        }

        if (x, y) not in self.subplot_user_params:
            self.subplot_user_params[(x, y)] = {}

        if (x, y) not in self.subplot_params:
            self.subplot_params[(x, y)] = {}

        # overwrite keyword arguments with user-set values
        kwargs.update(self.subplot_user_params[(x, y)])

        self.subplot_params[(x, y)].update(kwargs)

        if kwargs["colormap"] is None:
            kwargs["colormap"] = get_default_colormap()

        # and fetch back out
        title = kwargs["title"]
        colormap = kwargs["colormap"]
        colorbar = kwargs["colorbar"]
        normalize = kwargs["normalize"]
        vmin = kwargs["vmin"]
        vmax = kwargs["vmax"]
        rgb_mask = kwargs["rgb_mask"]
        interpolation = kwargs["interpolation"]

        # Note: if we do not do this, then passing in vmin,vmax without setting
        # normalize=False will cause the normalized image to be stretched
        # further which makes no sense.
        # ??? - We may want to change the normalize vs vmin,vmax behavior so if
        # vmin,vmax are passed in, then normalize is ignored.
        if normalize:
            vmin, vmax = 0, 1

        if clear:
            self.clear_subplot(x, y)

        # Store the raw image keyed by it's subplot location
        self.images[(x, y)] = image

        # Draw (actual image drawing in on_redraw() below)
        imshape = image.shape if self.dimensions == 2 else image.shape[1:]
        subplot = self.subplot(x, y, sharex=sharex, sharey=sharey)
        subplot.set_adjustable("box", True)
        subplot.plot([0, 0], list(imshape[:2]), "k")
        subplot.set_xlim([0, imshape[1]])
        subplot.set_ylim([imshape[0] - 0.5, -0.5])
        subplot.set_aspect("equal")
        if self.many_plots:
            subplot.tick_params(labelsize=6)

        # Set title
        if title is not None:
            self.set_subplot_title(title, x, y)

        # Update colorbar
        if orig_vmin is not None:
            tick_vmin = orig_vmin
        elif normalize == "log":
            tick_vmin = image[image > 0].min()
        else:
            tick_vmin = image.min()

        if orig_vmax is not None:
            tick_vmax = orig_vmax
        else:
            tick_vmax = image.max()

        if isinstance(colormap, str):
            colormap = matplotlib.cm.ScalarMappable(cmap=colormap)

        # NOTE: We bind this event each time imshow is called to a new closure
        #    of on_release so that each function will be called when a
        #    button_release_event is fired.  It might be cleaner to bind the
        #    event outside of subplot_imshow, and define a handler that iterates
        #    through each subplot to determine what kind of action should be
        #    taken. In this case each subplot_xxx call would have to append
        #    an action response to a dictionary keyed by subplot.
        if (x, y) in self.event_bindings:
            [self.figure.canvas.mpl_disconnect(b) for b in self.event_bindings[(x, y)]]

        def on_release(evt):
            if evt.inaxes == subplot:
                if evt.button != 1:
                    self.show_imshow_popup_menu(
                        (evt.x, self.figure.canvas.GetSize()[1] - evt.y), (x, y)
                    )

        self.event_bindings[(x, y)] = [
            self.figure.canvas.mpl_connect("button_release_event", on_release)
        ]

        if colorbar and not self.is_color_image(image):
            colormap.set_array(self.images[(x, y)])

            colormap.autoscale()

        if self.dimensions == 3:
            self.navtoolbar.set_volumetric()
            z = image.shape[0]
            self.current_plane = min(z // 2,self.navtoolbar.slider.GetValue())

        image = self.normalize_image(self.images[(x, y)], **kwargs)

        self.subplot(x, y).displayed = subplot.imshow(image, interpolation=interpolation)

        self.update_line_labels(subplot, kwargs)

        #
        # Colorbar support
        #
        if colorbar and not self.is_color_image(image):
            if subplot not in self.colorbar:
                cax = matplotlib.colorbar.make_axes(subplot)[0]

                bar = subplot.figure.colorbar(
                    colormap, cax, subplot, use_gridspec=False
                )

                self.colorbar[subplot] = (cax, bar)
            else:
                cax, bar = self.colorbar[subplot]

                bar.set_array(self.images[(x, y)])

                bar.update_normal(colormap)

                bar.update_ticks()

        # Also add this menu to the main menu
        if (x, y) in self.subplot_menus:
            # First trash the existing menu if there is one
            self.menu_subplots.Remove(self.subplot_menus[(x, y)])

        menu_pos = 0

        for yy in range(y + 1):
            if yy == y:
                cols = x
            else:
                cols = self.subplots.shape[0]

            for xx in range(cols):
                if (xx, yy) in self.images:
                    menu_pos += 1

        self.subplot_menus[(x, y)] = self.menu_subplots.Insert(
            menu_pos,
            -1,
            (title or "Subplot (%s,%s)" % (x, y)),
            self.get_imshow_menu((x, y)),
        )

        # Attempt to update histogram plot if one was created
        hist_fig = find_fig(self, name="%s %s image histogram" % (self.Name, (x, y)))

        if hist_fig:
            hist_fig.subplot_histogram(
                0, 0, self.images[(x, y)].flatten(), bins=200, xlabel="pixel intensity",
            )

            hist_fig.figure.canvas.draw()

        if self.dimensions == 3:
            self.navtoolbar.set_volumetric()
            self.navtoolbar.slider.SetValue(self.current_plane)
            self.navtoolbar.planetext.SetValue(self.current_plane)
            self.navtoolbar.slider.SetMax(z - 1)
            self.navtoolbar.planetext.SetMax(z - 1)

            def next_plane(event):
                if (val := self.navtoolbar.slider.GetValue()) < z - 1:
                    change_plane(val + 1)

            def prev_plane(event):
                if (val := self.navtoolbar.slider.GetValue()) > 0:
                    change_plane(val - 1)

            def change_plane(newplane):
                if newplane == self.current_plane:
                    return
                else:
                    self.current_plane = newplane
                    self.navtoolbar.slider.SetValue(newplane)
                    self.navtoolbar.planetext.ChangeValue(newplane)
                    self.navtoolbar.planetext.Update()

                    display_plane()

            def change_slider(event):
                change_plane(event.GetInt())

            def change_text(event):
                if event.String != '':
                    change_plane(int(event.String))

            def display_plane():
                numx, numy = self.subplots.shape
                for xcoord in range(numx):
                    for ycoord in range(numy):
                        subplot_item = self.subplot(xcoord, ycoord)
                        if hasattr(subplot_item, "displayed"):
                            params = self.subplot_params[(xcoord, ycoord)]
                            img_data = self.normalize_image(
                                self.images[(xcoord, ycoord)], **params
                            )
                            subplot_item.displayed.set_data(img_data)
                        else:
                            # Should be a table, we don't need the axis.
                            if not hasattr(subplot_item, "deleted"):
                                self.figure.delaxes(subplot_item)
                                subplot_item.deleted = True
                self.figure.canvas.draw()

            self.navtoolbar.nextplane.Bind(wx.EVT_BUTTON, next_plane)
            self.navtoolbar.prevplane.Bind(wx.EVT_BUTTON, prev_plane)
            self.navtoolbar.slider.Bind(wx.EVT_SLIDER, change_slider)
            self.navtoolbar.planetext.Bind(wx.EVT_TEXT, change_text)

        return subplot

    @staticmethod
    def update_line_labels(subplot, kwargs):
        outlines = [x for x in subplot.collections if isinstance(x, OutlineArtist)]

        for outline in outlines:
            outline.remove()

        for cplabels in kwargs["cplabels"]:
            if not cplabels.get(CPLD_SHOW, True):
                continue

            if cplabels[CPLD_MODE] == CPLDM_LINES:
                subplot.add_collection(
                    OutlineArtist(
                        cplabels[CPLD_NAME],
                        cplabels[CPLD_LABELS],
                        linewidth=cplabels[CPLD_LINE_WIDTH],
                        colors=numpy.array(cplabels[CPLD_OUTLINE_COLOR], float) / 255.0,
                    )
                )

    @allow_sharexy
    def subplot_imshow_color(
        self, x, y, image, title=None, normalize=False, rgb_mask=None, volumetric=False, **kwargs,
    ):
        if rgb_mask is None:
            rgb_mask = [1, 1, 1]

        if volumetric:
            chan_index = 3
        else:
            chan_index = 2

        # Truncate multichannel data that is not RGB (4+ channel data) and display it as RGB.
        if image.shape[chan_index] > 3:
            LOGGER.warning(
                "Multichannel display is only supported for RGB (3-channel) data."
                " Input image has {:d} channels. The first 3 channels are displayed as RGB.".format(
                    image.shape[chan_index]
                )
            )

            if not volumetric:

                return self.subplot_imshow(
                    x,
                    y,
                    image[:, :, :3],
                    title,
                    normalize=normalize,
                    rgb_mask=rgb_mask,
                    **kwargs,
                )
            
            else:

                return self.subplot_imshow(
                    x,
                    y,
                    image[:, :, :, :3],
                    title,
                    normalize=normalize,
                    rgb_mask=rgb_mask,
                    **kwargs,
                )

        return self.subplot_imshow(
            x, y, image, title, normalize=normalize, rgb_mask=rgb_mask, **kwargs
        )

    @allow_sharexy
    def subplot_imshow_labels(
        self,
        x,
        y,
        image,
        title=None,
        clear=True,
        sharex=None,
        sharey=None,
        use_imshow=False,
        background_image=None,
        max_label=None,
        seed=None,
        colormap=None,
    ):
        """
        Show a labels matrix using a custom colormap which better showcases the individual label values

        :param x: the subplot's row coordinate
        :param y: the subplot's column coordinate
        :param image: the segmentation to display
        :param title: the caption for the image
        :param clear: clear the axis before showing
        :param sharex: the row coordinate of the subplot that dictates panning and zooming, if any
        :param sharey: the column coordinate of the subplot that dictates panning and zooming, if any
        :param use_imshow: use matplotlib's imshow to display, instead of creating our own artist
        :param dimensions: dimensions of the data to display (2 or 3)
        :param background_image: a base image to overlay label data on, or None for blank
        :param max_label: set the maximum label in the segmentation, or None to use the segmentation's maximum label
                          (useful for generating consistent label colors in displays with varying # of objects)
        :param seed: shuffle label colors with this seed, or None for completely random (useful for generating
                     consistent label colors in multiple displays)
        :param colormap: load a shared cmap if provided
        :return:
        """
        if background_image is not None:
            opacity = 0.7
            label_image = overlay_labels(
                labels=image,
                opacity=opacity,
                pixel_data=background_image,
                max_label=max_label,
                seed=seed,
            )
            colormap = None
        else:
            # Mask the original labels
            label_image = numpy.ma.masked_where(image == 0, image)
            if not colormap:
                colormap = self.return_cmap(
                    numpy.max(image) if numpy.max(image) > 255 else None
                )
            else:
                colormap = colormap

        return self.subplot_imshow(
            x,
            y,
            label_image,
            title,
            clear,
            normalize=False,
            vmin=None,
            vmax=None,
            sharex=sharex,
            sharey=sharey,
            use_imshow=use_imshow,
            colormap=colormap,
        )

    def return_cmap(self, nindexes=None):
        if nindexes is not None:
            nindexes = max(nindexes, 255)
        # Get the colormap from the user preferences
        colormap = matplotlib.cm.get_cmap(get_default_colormap(), lut=nindexes,)
        # Initialize the colormap so we have access to the LUT
        colormap._init()
        # N is the number of "entries" in the LUT. `_lut` goes a little bit beyond that,
        # I think because there are "under" and "over" values. Regardless, we only one this
        # part of the LUT
        n = colormap.N
        # Get the LUT (only the part we care about)
        lut = colormap._lut[:n].copy()
        # Shuffle the colors so adjacently labeled objects are different colors
        numpy.random.shuffle(lut)
        # Set the LUT
        colormap._lut[:n] = lut
        # Make sure the background is black
        colormap.set_bad(color="black")
        return colormap

    @allow_sharexy
    def subplot_imshow_ijv(
        self,
        x,
        y,
        ijv,
        shape=None,
        title=None,
        clear=True,
        renumber=True,
        sharex=None,
        sharey=None,
        use_imshow=False,
    ):
        """Show an ijv-style labeling using the default color map

        x,y - the subplot's coordinates
        ijv - a pixel-by-pixel labeling where ijv[:,0] is the i coordinate,
              ijv[:,1] is the j coordinate and ijv[:,2] is the label
        shape - the shape of the final image. If "none", we try to infer
                from the maximum I and J
        title - the caption for the image
        clear - clear the axis before showing
        sharex, sharey - the coordinates of the subplot that dictates
                panning and zooming, if any
        use_imshow - Use matplotlib's imshow to display instead of creating
                     our own artist.
        """
        if shape is None:
            if len(ijv) == 0:
                shape = [1, 1]
            else:
                shape = [numpy.max(ijv[:, 0]) + 1, numpy.max(ijv[:, 1]) + 1]

        image = numpy.zeros(list(shape) + [3], float)

        if len(ijv) > 0:
            cm = matplotlib.cm.get_cmap(get_default_colormap())

            max_label = numpy.max(ijv[:, 2])

            if renumber:
                numpy.random.seed(0)
                order = numpy.random.permutation(max_label)
            else:
                order = numpy.arange(max_label)

            order = numpy.hstack(([0], order + 1))

            colors = matplotlib.cm.ScalarMappable(cmap=cm).to_rgba(order)

            r, g, b, a = [
                scipy.sparse.coo_matrix(
                    (colors[ijv[:, 2], i], (ijv[:, 0], ijv[:, 1])), shape=shape
                ).toarray()
                for i in range(4)
            ]

            for i, plane in enumerate((r, g, b)):
                image[a != 0, i] = plane[a != 0] / a[a != 0]

        return self.subplot_imshow(
            x,
            y,
            image,
            title,
            clear,
            normalize=False,
            vmin=None,
            vmax=None,
            sharex=sharex,
            sharey=sharey,
            use_imshow=use_imshow,
        )

    @allow_sharexy
    def subplot_imshow_grayscale(self, x, y, image, title=None, **kwargs):
        """Show an intensity image in shades of gray

        x,y - the subplot's coordinates
        image - the binary image to show
        title - the caption for the image
        clear - clear the axis before showing
        colorbar - show a colorbar relating intensity to color
        normalize - True to normalize to all shades of gray, False to
                    map grays between vmin and vmax
        vmin, vmax - the minimum and maximum intensities
        sharex, sharey - the coordinates of the subplot that dictates
                panning and zooming, if any
        use_imshow - Use matplotlib's imshow to display instead of creating
                     our own artist.
        """
        if image.dtype.type == numpy.float64:
            image = image.astype(numpy.float32)
        kwargs = kwargs.copy()
        kwargs["colormap"] = "Greys_r"
        return self.subplot_imshow(x, y, image, title=title, **kwargs)

    @allow_sharexy
    def subplot_imshow_bw(self, x, y, image, title=None, **kwargs):
        """Show a binary image in black and white

        x,y - the subplot's coordinates
        image - the binary image to show
        title - the caption for the image
        clear - clear the axis before showing
        sharex, sharey - the coordinates of the subplot that dictates
                panning and zooming, if any
        use_imshow - Use matplotlib's imshow to display instead of creating
                     our own artist.
        """
        kwargs = kwargs.copy()
        kwargs["colormap"] = "binary_r"
        return self.subplot_imshow(x, y, image, title=title, **kwargs)

    def normalize_image(self, image, **kwargs):
        """Produce a color image normalized according to user spec"""
        if 0 in image.shape:
            # No normalization to perform for images with an empty dimension.
            # Return the image.
            # https://github.com/CellProfiler/CellProfiler/issues/3330
            return image
        colormap = kwargs["colormap"]
        normalize = kwargs["normalize"]
        vmin = kwargs["vmin"]
        vmax = kwargs["vmax"]
        rgb_mask = kwargs["rgb_mask"]
        in_range = (image.min(), image.max())
        image = image.astype(numpy.float32)
        if self.dimensions == 3:
            orig_image_max = image.max()
            if self.current_plane >= image.shape[0]:
                image = image[image.shape[0]-1, :, :]
            else:
                image = image[self.current_plane, :, :]
        if isinstance(colormap, matplotlib.cm.ScalarMappable):
            colormap = colormap.cmap

        # Perform normalization
        if normalize == "log":
            if self.is_color_image(image):
                image = [
                    skimage.exposure.adjust_log(
                        image[:, :, ch], **kwargs["normalize_args"]
                    )
                    for ch in range(image.shape[2])
                ]

                image = numpy.dstack(image)
            else:
                image = skimage.exposure.adjust_log(image, **kwargs["normalize_args"])
        elif normalize == "gamma":
            if self.is_color_image(image):
                image = [
                    skimage.exposure.adjust_gamma(
                        image[:, :, ch], **kwargs["normalize_args"]
                    )
                    for ch in range(image.shape[2])
                ]

                image = numpy.dstack(image)
            else:
                image = skimage.exposure.adjust_gamma(image, **kwargs["normalize_args"])
        elif normalize:
            if self.is_color_image(image):
                image = [
                    skimage.exposure.rescale_intensity(image[:, :, ch])
                    for ch in range(image.shape[2])
                ]

                image = numpy.dstack(image)
            else:
                if in_range[0].dtype == bool:
                    image = skimage.exposure.rescale_intensity(image)
                else:
                    image = skimage.exposure.rescale_intensity(image, in_range=in_range)

        # Apply rgb mask to hide/show channels
        if self.is_color_image(image):
            rgb_mask = match_rgbmask_to_image(rgb_mask, image)
            image *= rgb_mask
            if image.shape[2] == 2:
                image = numpy.dstack(
                    [
                        image[:, :, 0],
                        image[:, :, 1],
                        numpy.zeros(image.shape[:2], image.dtype),
                    ]
                )
            # Apply display bounds
            if vmin is not None and vmax is not None:
                image = skimage.exposure.rescale_intensity(image, in_range=(vmin, vmax))
        
        if not self.is_color_image(image):
            if not normalize:
                if self.dimensions == 3:
                    norm = matplotlib.colors.Normalize(vmin=0, vmax=orig_image_max)
                else:
                    if image.max() < 255:
                        norm = matplotlib.colors.Normalize(vmin=0, vmax=255)
                    else:
                        norm = matplotlib.colors.Normalize(vmin=0, vmax=image.max())
            else:
                norm = None
            mappable = matplotlib.cm.ScalarMappable(cmap=colormap, norm=norm)
            mappable.set_clim(vmin, vmax)
            image = mappable.to_rgba(image)[:, :, :3]
        #
        # add the segmentations
        #
        for cplabel in kwargs["cplabels"]:
            if cplabel[CPLD_MODE] == CPLDM_NONE or not cplabel.get(CPLD_SHOW, True):
                continue
            loffset = 0
            ltotal = sum([numpy.max(labels) for labels in cplabel[CPLD_LABELS]])
            if ltotal == 0:
                continue
            for labels in cplabel[CPLD_LABELS]:
                if cplabel[CPLD_MODE] == CPLDM_OUTLINES:
                    oc = numpy.array(cplabel[CPLD_OUTLINE_COLOR], float)[:3] / 255
                    lm = centrosome.outline.outline(labels) != 0
                    lo = lm.astype(float)
                    lw = float(cplabel[CPLD_LINE_WIDTH])
                    if lw > 1:
                        # Alpha-blend for distances beyond 1
                        hw = lw / 2
                        d = scipy.ndimage.distance_transform_edt(~lm)
                        dti, dtj = numpy.where((d < hw + 0.5) & ~lm)
                        lo[dti, dtj] = numpy.minimum(1, hw + 0.5 - d[dti, dtj])
                    image = (
                        image * (1 - lo[:, :, numpy.newaxis])
                        + lo[:, :, numpy.newaxis] * oc[numpy.newaxis, numpy.newaxis, :]
                    )
                elif cplabel[CPLD_MODE] == CPLDM_ALPHA:
                    #
                    # For alpha overlays, renumber
                    lnumbers = renumber_labels_for_display(labels) + loffset
                    mappable = matplotlib.cm.ScalarMappable(
                        cmap=cplabel[CPLD_ALPHA_COLORMAP]
                    )
                    mappable.set_clim(1, ltotal)
                    limage = mappable.to_rgba(lnumbers[labels != 0])[:, :3]
                    alpha = cplabel[CPLD_ALPHA_VALUE]
                    image[labels != 0, :] *= 1 - alpha
                    image[labels != 0, :] += limage * alpha
                loffset += numpy.max(labels)
        return image

    def subplot_table(
        self,
        x,
        y,
        statistics,
        col_labels=None,
        row_labels=None,
        n_cols=1,
        n_rows=1,
        title=None,
        **kwargs,
    ):
        """Put a table into a subplot

        x,y - subplot's column and row
        statistics - a sequence of sequences that form the values to
                     go into the table
        col_labels - labels for the column header

        row_labels - labels for the row header

        **kwargs - for backwards compatibility, old argument values
        """

        if self.dimensions == 2:
            nx, ny = self.subplots.shape
        else:
            ny, nx = self.__gridspec.get_geometry()

        xstart = float(x) / float(nx)
        ystart = float(y) / float(ny)
        width = float(n_cols) / float(nx)
        height = float(n_rows) / float(ny)
        cw, ch = self.figure.canvas.GetSize()
        ctrl = wx.grid.Grid(self.figure.canvas)

        if title is not None:
            if title == "default":
                title = (
                    "Per-image means, use an Export module for per-object measurements"
                )
            elif title == "short":
                title = "Per-image means"
            ystart += 0.1
            height -= 0.1
            axes = self.subplot(x, y)
            if not self.figure.get_constrained_layout():
                self.figure.tight_layout()
            axes.axis("off")
            axes.annotate(title, xy=(0.5, 1.0), ha="center", va="top", fontsize=9)

        self.widgets.append(
            (xstart, ystart, width, height, wx.ALIGN_CENTER, wx.ALIGN_CENTER, ctrl)
        )
        nrows = len(statistics)
        ncols = 0 if nrows == 0 else len(statistics[0])
        ctrl.CreateGrid(nrows, ncols)
        if col_labels is not None:
            for i, value in enumerate(col_labels):
                ctrl.SetColLabelValue(i, str(value))
        else:
            ctrl.SetColLabelSize(0)
        if row_labels is not None:
            ctrl.GetGridRowLabelWindow().Font = ctrl.GetLabelFont()
            ctrl.SetRowLabelAlignment(wx.ALIGN_LEFT, wx.ALIGN_CENTER)
            max_width = 0
            for i, value in enumerate(row_labels):
                value = str(value)
                ctrl.SetRowLabelValue(i, value)
                max_width = max(
                    max_width,
                    ctrl.GetGridRowLabelWindow().GetTextExtent(value + "M")[0],
                )
            ctrl.SetRowLabelSize(max_width)
        else:
            ctrl.SetRowLabelSize(0)

        for i, row in enumerate(statistics):
            for j, value in enumerate(row):
                ctrl.SetCellValue(i, j, str(value))
                ctrl.SetReadOnly(i, j, True)
        ctrl.AutoSize()
        ctrl.Show()
        self.align_widget(
            ctrl,
            xstart,
            ystart,
            width,
            height,
            wx.ALIGN_CENTER,
            wx.ALIGN_CENTER,
            cw,
            ch,
        )
        self.table = []
        if col_labels is not None:
            if row_labels is not None:
                # Need a blank corner header if both col and row labels
                col_labels = [""] + list(col_labels)
            self.table.append(col_labels)
        if row_labels is not None:
            self.table += [[a] + list(b) for a, b in zip(row_labels, statistics)]
        else:
            self.table += statistics
        self.__menu_file.Enable(MENU_FILE_SAVE_TABLE, True)

    def subplot_scatter(
        self,
        x,
        y,
        xvals,
        yvals,
        xlabel="",
        ylabel="",
        xscale="linear",
        yscale="linear",
        title="",
        color="b",
        cmap=None,
        clear=True,
    ):
        """Put a scatterplot into a subplot

        x, y - subplot's column and row
        xvals, yvals - values to scatter
        xlabel - string label for x axis
        ylabel - string label for y axis
        xscale - scaling of the x axis (e.g., 'log' or 'linear')
        yscale - scaling of the y axis (e.g., 'log' or 'linear')
        title  - string title for the plot
        color  - color, sequence, or sequence of color for the plotted points
        cmap   - matplotlib Colormap
        """
        xvals = numpy.array(xvals).flatten()
        yvals = numpy.array(yvals).flatten()
        if self.dimensions == 2:
            if clear:
                self.clear_subplot(x, y)

            self.figure.set_facecolor((1, 1, 1))
            self.figure.set_edgecolor((1, 1, 1))

            axes = self.subplot(x, y)
        else:
            gx, gy = self.__gridspec.get_geometry()

            gridspec = matplotlib.gridspec.GridSpecFromSubplotSpec(
                ncols=1, nrows=1, subplot_spec=self.__gridspec[gy * y + x]
            )

            axes = matplotlib.pyplot.Subplot(self.figure, gridspec[0])

        plot = axes.scatter(
            xvals,
            yvals,
            facecolor=(0.0, 0.62, 1.0),
            edgecolor="none",
            c=color,
            cmap=cmap,
            alpha=0.75,
        )
        axes.set_title(title)
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)

        if self.dimensions == 3:
            self.figure.add_subplot(axes)

        return plot

    def subplot_histogram(
        self,
        x,
        y,
        values,
        bins=20,
        xlabel="",
        xscale=None,
        yscale="linear",
        title="",
        clear=True,
    ):
        """Put a histogram into a subplot

        x,y - subplot's column and row
        values - values to plot
        bins - number of bins to aggregate data in
        xlabel - string label for x axis
        xscale - 'log' to log-transform the data
        yscale - scaling of the y axis (e.g., 'log')
        title  - string title for the plot
        """
        if clear:
            self.clear_subplot(x, y)
        axes = self.subplot(x, y)
        self.figure.set_facecolor((1, 1, 1))
        self.figure.set_edgecolor((1, 1, 1))
        values = numpy.array(values).flatten()
        if xscale == "log":
            values = numpy.log(values[values > 0])
            xlabel = "Log(%s)" % (xlabel or "?")
        # hist apparently doesn't like nans, need to preen them out first
        # (infinities are not much better)
        values = values[numpy.isfinite(values)]
        # nothing to plot?
        if values.shape[0] == 0:
            axes = self.subplot(x, y)
            plot = axes.text(0.1, 0.5, "No valid values to plot.")
            axes.set_xlabel(xlabel)
            axes.set_title(title)
            return plot

        axes = self.subplot(x, y)
        plot = axes.hist(
            values,
            bins,
            facecolor=(0.0, 0.62, 1.0),
            edgecolor="none",
            log=(yscale == "log"),
            alpha=0.75,
        )
        axes.set_xlabel(xlabel)
        axes.set_title(title)

        return plot

    def subplot_density(
        self,
        x,
        y,
        points,
        gridsize=100,
        xlabel="",
        ylabel="",
        xscale="linear",
        yscale="linear",
        bins=None,
        cmap="jet",
        title="",
        clear=True,
    ):
        """Put a histogram into a subplot

        x,y - subplot's column and row
        points - values to plot
        gridsize - x & y bin size for data aggregation
        xlabel - string label for x axis
        ylabel - string label for y axis
        xscale - scaling of the x axis (e.g., 'log' or 'linear')
        yscale - scaling of the y axis (e.g., 'log' or 'linear')
        bins - scaling of the color map (e.g., None or 'log', see mpl.hexbin)
        title  - string title for the plot
        """
        if clear:
            self.clear_subplot(x, y)
        axes = self.subplot(x, y)
        self.figure.set_facecolor((1, 1, 1))
        self.figure.set_edgecolor((1, 1, 1))

        points = numpy.array(points)

        # Clip to positives if in log space
        if xscale == "log":
            points = points[(points[:, 0] > 0)]
        if yscale == "log":
            points = points[(points[:, 1] > 0)]

        # nothing to plot?
        if len(points) == 0 or points == [[]]:
            return

        plot = axes.hexbin(
            points[:, 0],
            points[:, 1],
            gridsize=gridsize,
            xscale=xscale,
            yscale=yscale,
            bins=bins,
            cmap=matplotlib.cm.get_cmap(cmap),
        )
        cb = self.figure.colorbar(plot)
        if bins == "log":
            cb.set_label("log10(N)")

        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_title(title)

        xmin = numpy.nanmin(points[:, 0])
        xmax = numpy.nanmax(points[:, 0])
        ymin = numpy.nanmin(points[:, 1])
        ymax = numpy.nanmax(points[:, 1])

        # Pad all sides
        if xscale == "log":
            xmin /= 1.5
            xmax *= 1.5
        else:
            xmin = xmin - (xmax - xmin) / 20.0
            xmax = xmax + (xmax - xmin) / 20.0

        if yscale == "log":
            ymin /= 1.5
            ymax *= 1.5
        else:
            ymin = ymin - (ymax - ymin) / 20.0
            ymax = ymax + (ymax - ymin) / 20.0

        axes.axis([xmin, xmax, ymin, ymax])

        return plot

    def subplot_platemap(
        self, x, y, plates_dict, plate_type, cmap="jet", colorbar=True, title=""
    ):
        """Draws a basic plate map (as an image).
        x, y       - subplot's column and row (should be 0,0)
        plates_dict - dict of the form: d[plate][well] --> numeric value
                     well must be in the form "A01"
        plate_type - '96' or '384'
        cmap       - a colormap from matplotlib.cm
                     Warning: gray is currently used for NaN values)
        title      - name for this subplot
        clear      - clear the subplot axes before display if True
        """
        plate_names = sorted(plates_dict.keys())

        if "plate_choice" not in self.__dict__:
            platemap_plate = plate_names[0]
            # Add plate selection choice
            #
            # Make the text transparent so the gradient shows.
            # Intercept paint to paint the foreground only
            # Intercept erase background to do nothing
            # Intercept size to make sure we redraw
            #
            plate_static_text = wx.StaticText(
                self.navtoolbar, -1, "Plate: ", style=wx.TRANSPARENT_WINDOW
            )

            def on_paint_text(event):
                dc = wx.PaintDC(plate_static_text)
                dc.SetFont(plate_static_text.GetFont())
                dc.DrawText(plate_static_text.GetLabel(), 0, 0)

            def on_size(event):
                plate_static_text.Refresh()
                event.Skip()

            plate_static_text.Bind(wx.EVT_ERASE_BACKGROUND, lambda event: None)
            plate_static_text.Bind(wx.EVT_PAINT, on_paint_text)
            plate_static_text.Bind(wx.EVT_SIZE, on_size)

            self.plate_choice = wx.Choice(self.navtoolbar, -1, choices=plate_names)

            def on_plate_selected(event):
                self.draw_platemap()

            self.plate_choice.Bind(wx.EVT_CHOICE, on_plate_selected)
            self.plate_choice.SetSelection(0)
            self.navtoolbar.AddControl(plate_static_text)
            self.navtoolbar.AddControl(self.plate_choice)
            self.navtoolbar.Realize()
            self.plate_choice.plates_dict = plates_dict
            self.plate_choice.plate_type = plate_type
            self.plate_choice.x = x
            self.plate_choice.y = y
            self.plate_choice.cmap = matplotlib.cm.get_cmap(cmap)
            self.plate_choice.axis_title = title
            self.plate_choice.colorbar = colorbar
        else:
            selection = self.plate_choice.GetStringSelection()
            self.plate_choice.SetItems(plate_names)
            if selection in plate_names:
                self.plate_choice.SetStringSelection(selection)
            else:
                self.plate_choice.SetSelection(0)
            dest = self.plate_choice.plates_dict
            for key in plates_dict:
                if key not in dest:
                    dest[key] = plates_dict[key]
                else:
                    destplate = dest[key]
                    srcplate = plates_dict[key]
                    for subkey in srcplate:
                        if subkey not in destplate:
                            destplate[subkey] = srcplate[subkey]
                        elif not numpy.isnan(srcplate[subkey]):
                            destplate[subkey] = srcplate[subkey]

        return self.draw_platemap()

    def draw_platemap(self):
        alphabet = "ABCDEFGHIJKLMNOP"  # enough letters for a 384 well plate
        x = self.plate_choice.x
        y = self.plate_choice.y
        axes = self.subplot(x, y)
        platemap_plate = self.plate_choice.GetStringSelection()
        plates_dict = self.plate_choice.plates_dict
        plate_type = self.plate_choice.plate_type
        title = self.plate_choice.axis_title
        cmap = self.plate_choice.cmap
        colorbar = self.plate_choice.colorbar
        data = format_plate_data_as_array(plates_dict[platemap_plate], plate_type)

        nrows, ncols = data.shape

        # Draw NaNs as gray
        # XXX: What if colormap with gray in it?
        cmap.set_bad("gray", 1.0)
        clean_data = numpy.ma.array(data, mask=numpy.isnan(data))
        plot = axes.imshow(
            clean_data, cmap=cmap, interpolation="nearest", shape=data.shape
        )
        axes.set_title(title)
        axes.set_xticks(list(range(ncols)))
        axes.set_yticks(list(range(nrows)))
        axes.set_xticklabels(list(range(1, ncols + 1)), minor=False)
        axes.set_yticklabels(alphabet[:nrows], minor=False)
        axes.axis("image")

        if colorbar and not numpy.all(numpy.isnan(data)):
            if axes in self.colorbar:
                cb = self.colorbar[axes]
                cb.set_clim(numpy.min(clean_data), numpy.max(clean_data))
                cb.update_normal(clean_data)
            else:
                self.colorbar[axes] = self.figure.colorbar(plot, ax=axes)

        def format_coord(x, y):
            col = int(x + 0.5)
            row = int(y + 0.5)
            if (0 <= col < ncols) and (0 <= row < nrows):
                val = data[row, col]
                res = "%s%02d - %1.4f" % (alphabet[row], int(col + 1), val)
            else:
                res = "%s%02d" % (alphabet[row], int(col + 1))
                # TODO:
            #            hint = wx.TipWindow(self, res)
            #            wx.FutureCall(500, hint.Close)
            return res

        axes.format_coord = format_coord
        axes.figure.canvas.draw()
        return plot

    def is_color_image(self, image):
        if self.dimensions < image.ndim or len(image.shape) > image.ndim:
            return image.shape[-1] >= 2
        else:
            return False
