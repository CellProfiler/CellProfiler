# coding=utf-8
""" figure.py - provides a frame with a figure inside
"""

import cellprofiler.gui
import cellprofiler.gui.artist
import cellprofiler.gui.help
import cellprofiler.preferences
import cellprofiler.preferences
import centrosome.cpmorphology
import centrosome.outline
import tools
import csv
import functools
import javabridge
import logging
import matplotlib
import matplotlib.backends.backend_wxagg
import matplotlib.backends.backend_wxagg
import matplotlib.cm
import matplotlib.colorbar
import matplotlib.gridspec
import matplotlib.patches
import matplotlib.pyplot
import numpy
import numpy.ma
import os
import scipy.ndimage
import scipy.sparse
import sys
import uuid
import wx

im = None

logger = logging.getLogger(__name__)

#
# Monkey-patch the backend canvas to only report the truly supported filetypes
#
mpl_filetypes = ["png", "pdf"]
mpl_unsupported_filetypes = [
    ft for ft in matplotlib.backends.backend_wxagg.FigureCanvasWxAgg.filetypes
    if ft not in mpl_filetypes]
for ft in mpl_unsupported_filetypes:
    del matplotlib.backends.backend_wxagg.FigureCanvasWxAgg.filetypes[ft]

g_use_imshow = False


def log_transform(im):
    """returns log(image) scaled to the interval [0,1]"""
    orig = im
    try:
        im = im.copy()
        im[numpy.isnan(im)] = 0
        (minimum, maximum) = (im[im > 0].min(), im[numpy.isfinite(im)].max())
        if (maximum > minimum) and (maximum > 0):
            return (numpy.log(im.clip(minimum, maximum)) - numpy.log(minimum)) / (numpy.log(maximum) - numpy.log(minimum))
    except:
        pass
    return orig


def auto_contrast(im):
    """returns image scaled to the interval [0,1]"""
    im = im.copy()
    if numpy.prod(im.shape) == 0:
        return im
    (minimum, maximum) = (im.min(), im.max())
    # Check that the image isn't binary
    if numpy.any((im > minimum) & (im < maximum)):
        im -= im.min()
        if im.max() > 0:
            im /= im.max()
    return im


def is_color_image(im):
    return im.ndim == 3 and im.shape[2] >= 2


COLOR_NAMES = ['Red', 'Green', 'Blue', 'Yellow', 'Cyan', 'Magenta', 'White']
COLOR_VALS = [[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
              [1, 1, 0],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]]

"""subplot_imshow cplabels dictionary key: segmentation labels image"""
CPLD_LABELS = "labels"
"""subplot_imshow cplabels dictionary key: objects name"""
CPLD_NAME = "name"
"""subplot_imshow cplabels dictionary key: color to use for outlines"""
CPLD_OUTLINE_COLOR = "outline_color"
"""subplot_imshow cplabels dictionary key: display mode - outlines or alpha"""
CPLD_MODE = "mode"
"""subplot_imshow cplabels mode value: show outlines of objects"""
CPLDM_OUTLINES = "outlines"
"""subplot_imshow cplabels mode value: show objects as an alpha-transparent color overlay"""
CPLDM_ALPHA = "alpha"
"""subplot_imshow cplabels mode value: draw outlines using matplotlib plot"""
CPLDM_LINES = "lines"
"""subplot_imshow cplabels mode value: don't show these objects"""
CPLDM_NONE = "none"
"""subplot_imshow cplabels dictionary key: line width of outlines"""
CPLD_LINE_WIDTH = "line_width"
"""subplot_imshow cplabels dictionary key: color map to use in alpha mode"""
CPLD_ALPHA_COLORMAP = "alpha_colormap"
"""subplot_imshow cplabels dictionary key: alpha value to use in overlay mode"""
CPLD_ALPHA_VALUE = "alpha_value"
"""subplot_imshow cplabels dictionary key: show (TRUE) or hide (False)"""
CPLD_SHOW = "show"


def wraparound(sequence):
    while True:
        for l in sequence:
            yield l


def make_1_or_3_channels(im):
    if im.ndim == 2 or im.shape[2] == 1:
        return im.astype(numpy.float32)
    if im.shape[2] == 3:
        return (im * 255).clip(0, 255).astype(numpy.uint8)
    out = numpy.zeros((im.shape[0], im.shape[1], 3), numpy.float32)
    for chanidx, weights in zip(range(im.shape[2]), wraparound(COLOR_VALS)):
        for idx, v in enumerate(weights):
            out[:, :, idx] += v * im[:, :, chanidx]
    return (out * 255).clip(0, 255).astype(numpy.uint8)


def make_3_channels_float(im):
    if im.ndim == 3 and im.shape[2] == 1:
        im = im[:, :, 0]
    if im.ndim == 2:
        return numpy.dstack((im, im, im)).astype(numpy.double).clip(0, 1)
    out = numpy.zeros((im.shape[0], im.shape[1], 3), numpy.double)
    for chanidx, weights in zip(range(im.shape[2]), wraparound(COLOR_VALS)):
        for idx, v in enumerate(weights):
            out[:, :, idx] += v * im[:, :, chanidx]
    return out.clip(0, 1)


def getbitmap(im):
    if im.ndim == 2:
        im = (255 * numpy.dstack((im, im, im))).astype(numpy.uint8)
    h, w, _ = im.shape
    outim = wx.EmptyImage(w, h)
    b = buffer(im)  # make sure buffer exists through the remainder of function
    outim.SetDataBuffer(b)
    return outim.ConvertToBitmap()


def match_rgbmask_to_image(rgb_mask, image):
    rgb_mask = list(rgb_mask)  # copy
    nchannels = image.shape[2]
    del rgb_mask[nchannels:]
    if len(rgb_mask) < nchannels:
        rgb_mask = rgb_mask + [1] * (nchannels - len(rgb_mask))
    return rgb_mask


window_ids = []


def window_name(module):
    """Return a module's figure window name"""
    return "CellProfiler:%s:%s" % (module.module_name, module.module_num)


def find_fig(parent=None, title="", name=wx.FrameNameStr, subplots=None):
    """Find a figure frame window. Returns the window or None"""
    for w in wx.GetTopLevelWindows():
        if w.GetName() == name:
            return w


def create_or_find(parent=None, identifier=-1, title="",
                   pos=wx.DefaultPosition, size=wx.DefaultSize,
                   style=wx.DEFAULT_FRAME_STYLE, name=wx.FrameNameStr,
                   subplots=None,
                   on_close=None):
    """Create or find a figure frame window"""
    win = find_fig(parent, title, name, subplots)
    return win or Figure(parent, identifier, title, pos, size, style, name,
                         subplots, on_close)


def close_all(parent):
    windows = [x for x in parent.GetChildren()
               if isinstance(x, wx.Frame)]

    for window in windows:
        if isinstance(window, Figure):
            window.on_close(None)
        else:
            window.Close()

    cellprofiler.preferences.reset_cpfigure_position()


def allow_sharexy(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if 'sharexy' in kwargs:
            assert (not 'sharex' in kwargs) and (not 'sharey' in kwargs), \
                "Cannot specify sharexy with sharex or sharey"
            kwargs['sharex'] = kwargs['sharey'] = kwargs.pop('sharexy')
        return fn(*args, **kwargs)

    if wrapper.__doc__ is not None:
        wrapper.__doc__ += \
            '\n        sharexy=ax can be used to specify sharex=ax, sharey=ax'
    return wrapper


MENU_FILE_SAVE = wx.NewId()
MENU_FILE_SAVE_TABLE = wx.NewId()
MENU_CLOSE_WINDOW = wx.NewId()
MENU_TOOLS_MEASURE_LENGTH = wx.NewId()
MENU_CLOSE_ALL = wx.NewId()
MENU_LABELS_OUTLINE = {}
MENU_LABELS_OVERLAY = {}
MENU_LABELS_LINES = {}
MENU_LABELS_OFF = {}
MENU_LABELS_ALPHA = {}
MENU_SAVE_SUBPLOT = {}
MENU_RGB_CHANNELS = {}


def get_menu_id(d, idx):
    if idx not in d:
        d[idx] = wx.NewId()
    return d[idx]


'''mouse tool mode - do nothing'''
MODE_NONE = 0

'''mouse tool mode - show pixel data'''
MODE_MEASURE_LENGTH = 2


class Figure(wx.Frame):
    """A wx.Frame with a figure inside"""

    def __init__(self, parent=None, identifier=-1, title="",
                 pos=wx.DefaultPosition, size=wx.DefaultSize,
                 style=wx.DEFAULT_FRAME_STYLE, name=wx.FrameNameStr,
                 subplots=None, on_close=None,
                 secret_panel_class=None,
                 help_menu_items=cellprofiler.gui.help.FIGURE_HELP):
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
        global window_ids
        if pos == wx.DefaultPosition:
            pos = cellprofiler.preferences.get_next_cpfigure_position()
        super(Figure, self).__init__(parent, identifier, title, pos, size, style, name)
        self.close_fn = on_close
        self.mouse_mode = MODE_NONE
        self.length_arrow = None
        self.table = None
        self.images = {}
        self.colorbar = {}
        self.subplot_params = {}
        self.subplot_user_params = {}
        self.event_bindings = {}
        self.popup_menus = {}
        self.subplot_menus = {}
        self.widgets = []
        self.mouse_down = None
        self.remove_menu = []
        self.figure = matplotlib.pyplot.Figure()
        self.panel = matplotlib.backends.backend_wxagg.FigureCanvasWxAgg(self, -1, self.figure)
        self.__gridspec = None
        self.__grid_config = None
        if secret_panel_class is None:
            secret_panel_class = wx.Panel
        self.secret_panel = secret_panel_class(self)
        self.secret_panel.Hide()
        self.status_bar = self.CreateStatusBar()
        wx.EVT_CLOSE(self, self.on_close)
        self.Bind(wx.EVT_SIZE, self.on_size)
        if subplots:
            self.subplots = numpy.zeros(subplots, dtype=object)
        self.create_menu(help_menu_items)
        self.create_toolbar()
        self.figure.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.figure.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.figure.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.figure.canvas.mpl_connect('resize_event', self.on_resize)
        try:
            self.SetIcon(cellprofiler.gui.get_cp_icon())
        except:
            pass
        if size == wx.DefaultSize:
            self.panel.SetInitialSize(wx.Size(640, 480))
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
            if (parent_menu_bar is not None and
                    isinstance(parent_menu_bar, wx.MenuBar)):
                for menu, label in parent_menu_bar.GetMenus():
                    if label == "&Window":
                        menu_ids = [menu_item.Id
                                    for menu_item in menu.MenuItems]
                        for window_id in window_ids + [None]:
                            if window_id not in menu_ids:
                                break
                        if window_id is None:
                            window_id = wx.NewId()
                            window_ids.append(window_id)
                        assert isinstance(menu, wx.Menu)
                        menu.Append(window_id, title)

                        def on_menu_command(event):
                            self.Raise()

                        wx.EVT_MENU(parent, window_id, on_menu_command)
                        self.remove_menu.append([menu, window_id])

    def create_menu(self, figure_help=cellprofiler.gui.help.FIGURE_HELP):
        self.MenuBar = wx.MenuBar()
        self.__menu_file = wx.Menu()
        self.__menu_file.Append(MENU_FILE_SAVE, "&Save")
        self.__menu_file.Append(MENU_FILE_SAVE_TABLE, "&Save table")
        self.__menu_file.Enable(MENU_FILE_SAVE_TABLE, False)
        wx.EVT_MENU(self, MENU_FILE_SAVE, self.on_file_save)
        wx.EVT_MENU(self, MENU_FILE_SAVE_TABLE, self.on_file_save_table)
        self.MenuBar.Append(self.__menu_file, "&File")

        self.__menu_tools = wx.Menu()
        self.__menu_item_measure_length = \
            self.__menu_tools.AppendCheckItem(MENU_TOOLS_MEASURE_LENGTH,
                                              "Measure &length")
        self.MenuBar.Append(self.__menu_tools, "&Tools")

        self.menu_subplots = wx.Menu()
        self.MenuBar.Append(self.menu_subplots, 'Subplots')

        wx.EVT_MENU(self, MENU_TOOLS_MEASURE_LENGTH, self.on_measure_length)

        # work around mac window menu losing bindings
        if wx.Platform == '__WXMAC__':
            hidden_menu = wx.Menu()
            hidden_menu.Append(MENU_CLOSE_ALL, "&L")
            self.Bind(wx.EVT_MENU, lambda evt: close_all(self.Parent), id=MENU_CLOSE_ALL)
            accelerators = wx.AcceleratorTable(
                    [(wx.ACCEL_CMD, ord('W'), MENU_CLOSE_WINDOW),
                     (wx.ACCEL_CMD, ord('L'), MENU_CLOSE_ALL)])
        else:
            accelerators = wx.AcceleratorTable(
                    [(wx.ACCEL_CMD, ord('W'), MENU_CLOSE_WINDOW)])

        self.SetAcceleratorTable(accelerators)
        wx.EVT_MENU(self, MENU_CLOSE_WINDOW, self.on_close)
        self.MenuBar.Append(cellprofiler.gui.help.make_help_menu(figure_help, self), "&Help")

    def create_toolbar(self):
        self.navtoolbar = CPNavigationToolbar(self.figure.canvas)
        if wx.VERSION < (2, 9, 1, 1, ''):
            # avoid crash on latest wx 2.9
            self.navtoolbar.DeleteToolByPos(6)
        self.navtoolbar.Bind(EVT_NAV_MODE_CHANGE, self.on_navtool_changed)

    def clf(self):
        """Clear the figure window, resetting the display"""
        self.figure.clf()
        if hasattr(self, "subplots"):
            self.subplots[:, :] = None
        # Remove the subplot menus
        for (x, y) in self.subplot_menus:
            self.menu_subplots.RemoveItem(self.subplot_menus[(x, y)])
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
            self.align_widget(ctrl, x, y, width, height, halign, valign,
                              event.width, event.height)
            ctrl.ForceRefresh()  # I don't know why, but it seems to be needed.

    @staticmethod
    def align_widget(ctrl, x, y, width, height,
                     halign, valign, canvas_width, canvas_height):
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

        best_width, best_height = ctrl.GetBestSizeTuple()
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
        if any([not hasattr(self, bar) for bar in "navtoolbar", "status_bar"]):
            return
        available_width, available_height = self.GetClientSize()
        nbheight = self.navtoolbar.GetSize()[1]
        self.navtoolbar.SetPosition((0, 0))
        self.navtoolbar.SetSize((available_width, nbheight))
        if self.secret_panel.IsShown():
            sp_width = self.secret_panel.GetVirtualSize()[0]
            canvas_width = min(max(available_width - sp_width, 250),
                               available_width - 100)
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
        if event.EventObject.mode != NAV_MODE_NONE and \
                        self.mouse_mode == MODE_MEASURE_LENGTH:
            self.mouse_mode = MODE_NONE
            self.__menu_item_measure_length.Check(False)

    def on_measure_length(self, event):
        """Measure length menu item selected."""
        if self.__menu_item_measure_length.IsChecked():
            self.mouse_mode = MODE_MEASURE_LENGTH
            self.navtoolbar.cancel_mode()
            self.Layout()
        elif self.mouse_mode == MODE_MEASURE_LENGTH:
            self.mouse_mode = MODE_NONE

    def on_button_press(self, event):
        if not hasattr(self, "subplots"):
            return
        if event.inaxes in self.subplots.flatten():
            self.mouse_down = (event.xdata, event.ydata)
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
            x0 = min(self.mouse_down[0], evt.xdata)
            x1 = max(self.mouse_down[0], evt.xdata)
            y0 = min(self.mouse_down[1], evt.ydata)
            y1 = max(self.mouse_down[1], evt.ydata)
        if self.mouse_mode == MODE_MEASURE_LENGTH:
            self.on_mouse_move_measure_length(evt, x0, y0, x1, y1)
        elif not self.mouse_mode == MODE_NONE:
            self.on_mouse_move_show_pixel_data(evt, x0, y0, x1, y1)

    def get_pixel_data_fields_for_status_bar(self, im, xi, yi):
        fields = []
        x, y = [int(round(xy)) for xy in xi, yi]
        if not self.in_bounds(im, x, y):
            return fields
        if im.dtype.type == numpy.uint8:
            im = im.astype(numpy.float32) / 255.0
        if im.ndim == 2:
            fields += ["Intensity: %.4f" % (im[y, x])]
        elif im.ndim == 3 and im.shape[2] == 3:
            fields += ["Red: %.4f" % (im[y, x, 0]),
                       "Green: %.4f" % (im[y, x, 1]),
                       "Blue: %.4f" % (im[y, x, 2])]
        elif im.ndim == 3:
            fields += ["Channel %d: %.4f" % (idx + 1, im[y, x, idx]) for idx in range(im.shape[2])]
        return fields

    @staticmethod
    def in_bounds(im, xi, yi):
        """Return false if xi or yi are outside of the bounds of the image"""
        return not (im is None or xi >= im.shape[1] or yi >= im.shape[0]
                    or xi < 0 or yi < 0)

    def on_mouse_move_measure_length(self, event, x0, y0, x1, y1):
        if event.xdata is None or event.ydata is None:
            return
        xi = int(event.xdata + .5)
        yi = int(event.ydata + .5)
        im = None
        fields = self.get_fields(event, yi, xi, x1)

        if self.mouse_down is not None:
            x0 = min(self.mouse_down[0], event.xdata)
            x1 = max(self.mouse_down[0], event.xdata)
            y0 = min(self.mouse_down[1], event.ydata)
            y1 = max(self.mouse_down[1], event.ydata)

            length = numpy.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
            fields.append("Length: %.1f" % length)
            xinterval = event.inaxes.xaxis.get_view_interval()
            yinterval = event.inaxes.yaxis.get_view_interval()
            diagonal = numpy.sqrt((xinterval[1] - xinterval[0]) ** 2 +
                                  (yinterval[1] - yinterval[0]) ** 2)
            mutation_scale = min(int(length * 100 / diagonal), 20)
            if self.length_arrow is not None:
                self.length_arrow.set_positions((self.mouse_down[0],
                                                 self.mouse_down[1]),
                                                (event.xdata,
                                                 event.ydata))
            else:
                self.length_arrow = \
                    matplotlib.patches.FancyArrowPatch((self.mouse_down[0],
                                                        self.mouse_down[1]),
                                                       (event.xdata,
                                                        event.ydata),
                                                       edgecolor='red',
                                                       arrowstyle='<->',
                                                       mutation_scale=mutation_scale)
                try:
                    event.inaxes.add_patch(self.length_arrow)
                except:
                    self.length_arrow = None
            self.figure.canvas.draw()
            self.Refresh()
        self.status_bar.SetFields(fields)

    def get_fields(self, event, yi, xi, x1):
        """Get the standard fields at the cursor location"""
        if event.inaxes:
            fields = ["X: %d" % xi, "Y: %d" % yi]
            self.find_image_for_axes(event.inaxes)
            if im is not None:
                fields += self.get_pixel_data_fields_for_status_bar(im, x1, yi)
            elif isinstance(event.inaxes, matplotlib.axes.Axes):
                for artist in event.inaxes.artists:
                    if isinstance(
                            artist, cellprofiler.gui.artist.CPImageArtist):
                        fields += ["%s: %.4f" % (k, v) for k, v in
                                   artist.get_channel_values(xi, yi).items()]
        else:
            fields = []
        return fields

    def on_mouse_move_show_pixel_data(self, event, x0, y0, x1, y1):
        if event.xdata is None or event.ydata is None:
            return
        xi = int(event.xdata + .5)
        yi = int(event.ydata + .5)
        fields = self.get_fields(event, yi, xi, x1)
        if len(fields) > 0:
            self.status_bar.SetFields(fields)

    def find_image_for_axes(self, axes):
        value = None

        for i, sl in enumerate(self.subplots):
            for j, slax in enumerate(sl):
                if axes == slax:
                    value = self.images.get((i, j), None)

        return value

    def on_button_release(self, event):
        if not hasattr(self, "subplots"):
            return
        if event.inaxes in self.subplots.flatten() and self.mouse_down:
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
        with wx.FileDialog(self, "Save figure",
                           wildcard=("PDF file (*.pdf)|*.pdf|"
                                     "PNG image (*.png)|*.png"),
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
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
        with wx.FileDialog(self, "Save table",
                           wildcard="Excel file (*.csv)|*.csv",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                path = dlg.GetPath()
                with open(path, "wb") as fd:
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
                self.figure.dpi_scale_trans.inverted())
        with wx.FileDialog(self, "Save axes",
                           wildcard=("PDF file (*.pdf)|*.pdf|"
                                     "Png image (*.png)|*.png|"
                                     "Postscript file (*.ps)|*.ps"),
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
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
                self.figure.savefig(path,
                                    format=file_format,
                                    bbox_inches=extent)

    def set_subplots(self, subplots):
        self.clf()  # get rid of any existing subplots, menus, etc.
        if subplots is None:
            if hasattr(self, 'subplots'):
                delattr(self, 'subplots')
        else:
            self.subplots = numpy.zeros(subplots, dtype=object)

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
        if not self.subplots[x, y]:
            rows, cols = self.subplots.shape
            plot = self.figure.add_subplot(cols, rows, x + y * rows + 1,
                                           sharex=sharex, sharey=sharey)
            self.subplots[x, y] = plot
        return self.subplots[x, y]

    def set_subplot_title(self, title, x, y):
        """Set a subplot's title in the standard format

        title - title for subplot
        x - subplot's column
        y - subplot's row
        """
        fontname = fontname = cellprofiler.preferences.get_title_font_name()

        self.subplot(x, y).set_title(title,
                                     fontname=fontname,
                                     fontsize=cellprofiler.preferences.get_title_font_size())

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

    def get_imshow_menu(self, (x, y)):
        """returns a menu corresponding to the specified subplot with items to:
        - launch the image in a new cpfigure window
        - Show image histogram
        - Change contrast stretching
        - Toggle channels on/off
        Note: Each item is bound to a handler.
        """
        MENU_CONTRAST_RAW = wx.NewId()
        MENU_CONTRAST_NORMALIZED = wx.NewId()
        MENU_CONTRAST_LOG = wx.NewId()
        MENU_INTERPOLATION_NEAREST = wx.NewId()
        MENU_INTERPOLATION_BILINEAR = wx.NewId()
        MENU_INTERPOLATION_BICUBIC = wx.NewId()

        params = self.subplot_params[(x, y)]

        # If no popup has been built for this subplot yet, then create one
        popup = wx.Menu()
        self.popup_menus[(x, y)] = popup
        open_in_new_figure_item = wx.MenuItem(popup, -1,
                                              'Open image in new window')
        popup.AppendItem(open_in_new_figure_item)
        show_hist_item = wx.MenuItem(popup, -1, 'Show image histogram')
        popup.AppendItem(show_hist_item)

        submenu = wx.Menu()
        item_raw = submenu.Append(MENU_CONTRAST_RAW, 'Raw',
                                  'Do not transform pixel intensities',
                                  wx.ITEM_RADIO)
        item_normalized = submenu.Append(MENU_CONTRAST_NORMALIZED,
                                         'Normalized',
                                         'Stretch pixel intensities to fit '
                                         'the interval [0,1]',
                                         wx.ITEM_RADIO)
        item_log = submenu.Append(MENU_CONTRAST_LOG, 'Log normalized',
                                  'Log transform pixel intensities, then '
                                  'stretch them to fit the interval [0,1]',
                                  wx.ITEM_RADIO)

        if params['normalize'] == 'log':
            item_log.Check()
        elif params['normalize']:
            item_normalized.Check()
        else:
            item_raw.Check()
        popup.AppendMenu(-1, 'Image contrast', submenu)

        submenu = wx.Menu()
        item_nearest = submenu.Append(
                MENU_INTERPOLATION_NEAREST,
                "Nearest neighbor",
                "Use the intensity of the nearest image pixel when displaying "
                "screen pixels at sub-pixel resolution. This produces a blocky "
                "image, but the image accurately reflects the data.",
                wx.ITEM_RADIO)
        item_bilinear = submenu.Append(
                MENU_INTERPOLATION_BILINEAR,
                "Linear",
                "Use the weighted average of the four nearest image pixels when "
                "displaying screen pixels at sub-pixel resolution. This produces "
                "a smoother, more visually appealing image, but makes it more "
                "difficult to find pixel borders", wx.ITEM_RADIO)
        item_bicubic = submenu.Append(
                MENU_INTERPOLATION_BICUBIC,
                "Cubic",
                "Perform a bicubic interpolation of the nearby image pixels when "
                "displaying screen pixels at sub-pixel resolution. This produces "
                "the most visually appealing image but is the least faithful to "
                "the image pixel values.", wx.ITEM_RADIO)
        popup.AppendMenu(-1, "Interpolation", submenu)
        save_subplot_id = get_menu_id(MENU_SAVE_SUBPLOT, (x, y))
        popup.Append(save_subplot_id,
                     "Save subplot",
                     "Save just the display portion of this subplot")

        if params['interpolation'] == matplotlib.image.BILINEAR:
            item_bilinear.Check()
        elif params['interpolation'] == matplotlib.image.BICUBIC:
            item_bicubic.Check()
        else:
            item_nearest.Check()

        def open_image_in_new_figure(evt):
            """Callback for "Open image in new window" popup menu item """
            # Store current zoom limits
            xlims = self.subplot(x, y).get_xlim()
            ylims = self.subplot(x, y).get_ylim()
            new_title = self.subplot(x, y).get_title()
            fig = create_or_find(self, -1, new_title, subplots=(1, 1),
                                 name=str(uuid.uuid4()))
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
            new_title = '%s %s image histogram' % (self.Title, (x, y))
            fig = create_or_find(self, -1, new_title, subplots=(1, 1), name=new_title)
            fig.subplot_histogram(0, 0, self.images[(x, y)].flatten(), bins=200, xlabel='pixel intensity')
            fig.figure.canvas.draw()

        def change_contrast(evt):
            """Callback for Image contrast menu items"""
            # Store zoom limits
            xlims = self.subplot(x, y).get_xlim()
            ylims = self.subplot(x, y).get_ylim()
            axes = self.subplot(x, y)
            if evt.Id == MENU_CONTRAST_RAW:
                params['normalize'] = False
            elif evt.Id == MENU_CONTRAST_NORMALIZED:
                params['normalize'] = True
            elif evt.Id == MENU_CONTRAST_LOG:
                params['normalize'] = 'log'
            for artist in axes.artists:
                if isinstance(artist, CPImageArtist):
                    artist.kwargs["normalize"] = params['normalize']
                    self.figure.canvas.draw()
                    return
            else:
                self.subplot_imshow(x, y, self.images[(x, y)], **params)
                # Restore plot zoom
                self.subplot(x, y).set_xlim(xlims[0], xlims[1])
                self.subplot(x, y).set_ylim(ylims[0], ylims[1])
                self.figure.canvas.draw()

        def change_interpolation(evt):
            if evt.Id == MENU_INTERPOLATION_NEAREST:
                params['interpolation'] = matplotlib.image.NEAREST
            elif evt.Id == MENU_INTERPOLATION_BILINEAR:
                params['interpolation'] = matplotlib.image.BILINEAR
            elif evt.Id == MENU_INTERPOLATION_BICUBIC:
                params['interpolation'] = matplotlib.image.BICUBIC
            axes = self.subplot(x, y)
            for artist in axes.artists:
                if isinstance(artist, CPImageArtist):
                    artist.interpolation = params['interpolation']
                    artist.kwargs["interpolation"] = params["interpolation"]
                    self.figure.canvas.draw()
                    return
            else:
                xlims = self.subplot(x, y).get_xlim()
                ylims = self.subplot(x, y).get_ylim()
                self.subplot_imshow(x, y, self.images[(x, y)], **params)
                # Restore plot zoom
                self.subplot(x, y).set_xlim(xlims[0], xlims[1])
                self.subplot(x, y).set_ylim(ylims[0], ylims[1])
                self.figure.canvas.draw()

        if is_color_image(self.images[x, y]):
            submenu = wx.Menu()
            rgb_mask = match_rgbmask_to_image(params['rgb_mask'], self.images[x, y])
            ids = [get_menu_id(MENU_RGB_CHANNELS, (x, y, i))
                   for i in range(len(rgb_mask))]
            for name, value, identifier in zip(wraparound(COLOR_NAMES), rgb_mask, ids):
                item = submenu.Append(identifier, name, 'Show/Hide the %s channel' % name, wx.ITEM_CHECK)
                if value != 0:
                    item.Check()
            popup.AppendMenu(-1, 'Channels', submenu)

            def toggle_channels(evt):
                """Callback for channel menu items."""
                # Store zoom limits
                xlims = self.subplot(x, y).get_xlim()
                ylims = self.subplot(x, y).get_ylim()
                if 'rgb_mask' not in params:
                    params['rgb_mask'] = list(rgb_mask)
                else:
                    # copy to prevent modifying shared values
                    params['rgb_mask'] = list(params['rgb_mask'])
                for idx, identifier in enumerate(ids):
                    if identifier == evt.Id:
                        params['rgb_mask'][idx] = not params['rgb_mask'][idx]
                self.subplot_imshow(x, y, self.images[(x, y)], **params)
                # Restore plot zoom
                self.subplot(x, y).set_xlim(xlims[0], xlims[1])
                self.subplot(x, y).set_ylim(ylims[0], ylims[1])
                self.figure.canvas.draw()

            for identifier in ids:
                self.Bind(wx.EVT_MENU, toggle_channels, id=identifier)

        if params['cplabels'] is not None and len(params['cplabels']) > 0:
            for i, cplabels in enumerate(params['cplabels']):
                submenu = wx.Menu()
                name = cplabels.get(CPLD_NAME, "Objects #%d" % i)
                for mode, menud, mlabel, mhelp in (
                        (CPLDM_OUTLINES, MENU_LABELS_OUTLINE,
                         "Outlines", "Display outlines of objects"),
                        (CPLDM_ALPHA, MENU_LABELS_OVERLAY,
                         "Overlay", "Display objects as an alpha-overlay"),
                        (CPLDM_LINES, MENU_LABELS_LINES,
                         "Lines", "Draw lines around objects"),
                        (CPLDM_NONE, MENU_LABELS_OFF,
                         "Off", "Turn object labels off")):
                    menu_id = get_menu_id(menud, (x, y, i))
                    item = submenu.AppendRadioItem(menu_id, mlabel, mhelp)
                    if cplabels[CPLD_MODE] == mode:
                        item.Check()

                    def select_mode(event, cplabels=cplabels, mode=mode):
                        cplabels[CPLD_MODE] = mode
                        self.update_line_labels(
                                self.subplot(x, y), params)
                        self.figure.canvas.draw()

                    self.Bind(wx.EVT_MENU, select_mode, id=menu_id)
                if cplabels[CPLD_MODE] == CPLDM_ALPHA:
                    menu_id = get_menu_id(MENU_LABELS_ALPHA, (x, y, i))
                    item = submenu.Append(
                            menu_id, "Adjust transparency",
                            "Change the alpha-blend for the labels overlay to make it more or less transparent")
                    self.Bind(wx.EVT_MENU,
                              lambda event, cplabels=cplabels:
                              self.on_adjust_labels_alpha(cplabels),
                              id=menu_id)
                popup.AppendMenu(-1, name, submenu)

        self.Bind(wx.EVT_MENU, open_image_in_new_figure, open_in_new_figure_item)
        self.Bind(wx.EVT_MENU, show_hist, show_hist_item)
        self.Bind(wx.EVT_MENU, change_contrast, id=MENU_CONTRAST_RAW)
        self.Bind(wx.EVT_MENU, change_contrast, id=MENU_CONTRAST_NORMALIZED)
        self.Bind(wx.EVT_MENU, change_contrast, id=MENU_CONTRAST_LOG)
        self.Bind(wx.EVT_MENU, change_interpolation, id=MENU_INTERPOLATION_NEAREST)
        self.Bind(wx.EVT_MENU, change_interpolation, id=MENU_INTERPOLATION_BICUBIC)
        self.Bind(wx.EVT_MENU, change_interpolation, id=MENU_INTERPOLATION_BILINEAR)
        self.Bind(wx.EVT_MENU,
                  lambda event: self.on_file_save_subplot(event, x, y),
                  id=MENU_SAVE_SUBPLOT[(x, y)])
        return popup

    def on_adjust_labels_alpha(self, cplabels):
        with wx.Dialog(self, title="Adjust labels transparency") as dlg:
            name = cplabels.get(CPLD_NAME, "Objects")
            orig_alpha = int(cplabels[CPLD_ALPHA_VALUE] * 100 + .5)
            dlg.Sizer = wx.BoxSizer(wx.VERTICAL)
            sizer = wx.BoxSizer(wx.VERTICAL)
            dlg.Sizer.Add(sizer, 1, wx.EXPAND | wx.ALL, 8)
            sizer.Add(wx.StaticText(dlg, label="%s transparency"),
                      0, wx.ALIGN_CENTER_HORIZONTAL)
            sizer.AddSpacer(4)
            slider = wx.Slider(
                    dlg, value=orig_alpha, minValue=0, maxValue=100,
                    style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS)
            sizer.Add(slider, 1, wx.EXPAND)
            button_sizer = wx.StdDialogButtonSizer()
            button_sizer.AddButton(wx.Button(dlg, wx.ID_OK))
            button_sizer.AddButton(wx.Button(dlg, wx.ID_CANCEL))
            dlg.Sizer.Add(button_sizer)
            button_sizer.Realize()

            def on_slider(event, cplabels=cplabels,
                          draw_fn=self.figure.canvas.draw_idle):
                cplabels[CPLD_ALPHA_VALUE] = float(slider.Value) / 100.
                draw_fn()

            dlg.Layout()
            slider.Bind(wx.EVT_SLIDER, on_slider)
            if dlg.ShowModal() != wx.ID_OK:
                slider.Value = orig_alpha
                on_slider(None)

    def imshow(self, x, y, image, cmap):
        subplot = self.subplot(x, y)

        subplot.imshow(image, cmap=cmap)

    def set_grids(self, shape):
        self.__gridspec = matplotlib.gridspec.GridSpec(*shape)

    def __set_grid_config(self, image):
        # Center the grid at the slice with the greatest mean intensity.
        means = [numpy.mean(img) for _idx, img in enumerate(image)]
        max_mean = numpy.max(means)
        center = means.index(max_mean)

        # Determine the indices of the first and last slices to display.
        start = max(0, center - 4)
        stop = min(center + 5, image.shape[0])

        # From the number of slices that are displayable, determine the
        # shape of the grid these can be displayed in.
        n_slices = stop - start
        n_rows = n_slices / 3 + (1 if n_slices % 3 != 0 else 0)
        n_cols = 3 if n_rows != 1 else n_slices

        self.__grid_config = ((n_rows, n_cols), start, stop)

    def gridshow(self, x, y, image, cmap='gray'):
        if self.__grid_config is None:
            self.__set_grid_config(image)

        dimensions, start, stop = self.__grid_config

        gx, gy = self.__gridspec.get_geometry()
        idx = gx * x + y

        gridspec = matplotlib.gridspec.GridSpecFromSubplotSpec(*dimensions, subplot_spec=self.__gridspec[idx])

        for idx, img in enumerate(image[start:stop]):
            ax = matplotlib.pyplot.Subplot(self.figure, gridspec[idx])

            if idx / 3 != (dimensions[0] - 1):
                ax.set_xticklabels([])

            if idx % 3 != 0:
                ax.set_yticklabels([])

            ax.imshow(img, cmap=cmap)

            self.figure.add_subplot(ax)

        matplotlib.pyplot.show()

    def plot(self, x, y, image, markers, markercolor="+r", markersize=15):
        subplot = self.subplot(x, y)

        subplot.imshow(image, cmap='gray')
        subplot.plot(markers[:, 1], markers[:, 0], markercolor, markersize=markersize)

    @allow_sharexy
    def subplot_imshow(self, x, y, image, title=None, clear=True, colormap=None,
                       colorbar=False, normalize=None, vmin=0, vmax=1,
                       rgb_mask=(1, 1, 1), sharex=None, sharey=None,
                       use_imshow=False, interpolation=None, cplabels=None):
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
        orig_vmin = vmin
        orig_vmax = vmax
        if interpolation is None:
            interpolation = get_matplotlib_interpolation_preference()
        if normalize is None:
            normalize = cellprofiler.preferences.get_intensity_mode()
            if normalize == cellprofiler.preferences.INTENSITY_MODE_RAW:
                normalize = False
            elif normalize == cellprofiler.preferences.INTENSITY_MODE_LOG:
                normalize = "log"
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
                        d[CPLD_OUTLINE_COLOR] = cellprofiler.preferences.get_primary_outline_color()
                    elif i == 1:
                        d[CPLD_OUTLINE_COLOR] = cellprofiler.preferences.get_secondary_outline_color()
                    elif i == 2:
                        d[CPLD_OUTLINE_COLOR] = cellprofiler.preferences.get_tertiary_outline_color()
                if CPLD_MODE not in d:
                    d[CPLD_MODE] = CPLDM_OUTLINES
                if CPLD_LINE_WIDTH not in d:
                    d[CPLD_LINE_WIDTH] = 1
                if CPLD_ALPHA_COLORMAP not in d:
                    d[CPLD_ALPHA_COLORMAP] = cellprofiler.preferences.get_default_colormap()
                if CPLD_ALPHA_VALUE not in d:
                    d[CPLD_ALPHA_VALUE] = .25
                new_cplabels.append(d)
            cplabels = new_cplabels

        # NOTE: self.subplot_user_params is used to store changes that are made
        #    to the display through GUI interactions (eg: hiding a channel).
        #    Once a subplot that uses this mechanism has been drawn, it will
        #    continually load defaults from self.subplot_user_params instead of
        #    the default values specified in the function definition.
        kwargs = {'title': title,
                  'clear': False,
                  'colormap': colormap,
                  'colorbar': colorbar,
                  'normalize': normalize,
                  'vmin': vmin,
                  'vmax': vmax,
                  'rgb_mask': rgb_mask,
                  'use_imshow': use_imshow,
                  'interpolation': interpolation,
                  'cplabels': cplabels}
        if (x, y) not in self.subplot_user_params:
            self.subplot_user_params[(x, y)] = {}
        if (x, y) not in self.subplot_params:
            self.subplot_params[(x, y)] = {}
        # overwrite keyword arguments with user-set values
        kwargs.update(self.subplot_user_params[(x, y)])
        self.subplot_params[(x, y)].update(kwargs)
        if kwargs["colormap"] is None:
            kwargs["colormap"] = cellprofiler.preferences.get_default_colormap()

        # and fetch back out
        title = kwargs['title']
        colormap = kwargs['colormap']
        colorbar = kwargs['colorbar']
        normalize = kwargs['normalize']
        vmin = kwargs['vmin']
        vmax = kwargs['vmax']
        rgb_mask = kwargs['rgb_mask']
        interpolation = kwargs['interpolation']

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
        subplot = self.subplot(x, y, sharex=sharex, sharey=sharey)
        subplot._adjustable = 'box-forced'
        subplot.plot([0, 0], list(image.shape[:2]), 'k')
        subplot.set_xlim([-0.5, image.shape[1] - 0.5])
        subplot.set_ylim([image.shape[0] - 0.5, -0.5])
        subplot.set_aspect('equal')

        # Set title
        if title is not None:
            self.set_subplot_title(title, x, y)

        # Update colorbar
        if orig_vmin is not None:
            tick_vmin = orig_vmin
        elif normalize == 'log':
            tick_vmin = image[image > 0].min()
        else:
            tick_vmin = image.min()
        if orig_vmax is not None:
            tick_vmax = orig_vmax
        else:
            tick_vmax = image.max()

        if isinstance(colormap, basestring):
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
                    self.show_imshow_popup_menu((evt.x, self.figure.canvas.GetSize()[1] - evt.y), (x, y))

        self.event_bindings[(x, y)] = [
            self.figure.canvas.mpl_connect('button_release_event', on_release)]

        if colorbar and not is_color_image(image):
            colormap.set_array(self.images[(x, y)])
            colormap.autoscale()
        if use_imshow or g_use_imshow:
            image = self.images[(x, y)]
            subplot.imshow(self.normalize_image(image, **kwargs))
        else:
            subplot.add_artist(CPImageArtist(self.images[(x, y)], self, kwargs))

        self.update_line_labels(subplot, kwargs)
        #
        # Colorbar support
        #
        if colorbar and not is_color_image(image):

            if not subplot in self.colorbar:
                cax = matplotlib.colorbar.make_axes(subplot)[0]
                bar = subplot.figure.colorbar(colormap, cax, subplot, use_gridspec=False)
                self.colorbar[subplot] = (cax, bar)
            else:
                cax, bar = self.colorbar[subplot]
                bar.set_array(self.images[(x, y)])
                bar.update_normal(colormap)
                bar.update_ticks()

        # Also add this menu to the main menu
        if (x, y) in self.subplot_menus:
            # First trash the existing menu if there is one
            self.menu_subplots.RemoveItem(self.subplot_menus[(x, y)])
        menu_pos = 0
        for yy in range(y + 1):
            if yy == y:
                cols = x
            else:
                cols = self.subplots.shape[0]
            for xx in range(cols):
                if (xx, yy) in self.images:
                    menu_pos += 1
        self.subplot_menus[(x, y)] = self.menu_subplots.InsertMenu(menu_pos,
                                                                   -1, (title or 'Subplot (%s,%s)' % (x, y)),
                                                                   self.get_imshow_menu((x, y)))

        # Attempt to update histogram plot if one was created
        hist_fig = find_fig(self, name='%s %s image histogram' % (self.Name,
                                                                  (x, y)))
        if hist_fig:
            hist_fig.subplot_histogram(0, 0, self.images[(x, y)].flatten(),
                                       bins=200, xlabel='pixel intensity')
            hist_fig.figure.canvas.draw()
        return subplot

    @staticmethod
    def update_line_labels(subplot, kwargs):
        outlines = [x for x in subplot.collections
                    if isinstance(x, CPOutlineArtist)]
        for outline in outlines:
            outline.remove()
        for cplabels in kwargs['cplabels']:
            if not cplabels.get(CPLD_SHOW, True):
                continue
            if cplabels[CPLD_MODE] == CPLDM_LINES:
                subplot.add_collection(CPOutlineArtist(
                        cplabels[CPLD_NAME],
                        cplabels[CPLD_LABELS],
                        linewidth=cplabels[CPLD_LINE_WIDTH],
                        colors=numpy.array(cplabels[CPLD_OUTLINE_COLOR], float) / 255.))

    @allow_sharexy
    def subplot_imshow_color(self, x, y, image, title=None,
                             normalize=False, rgb_mask=None, **kwargs):
        if rgb_mask is None:
            rgb_mask = [1, 1, 1]
        return self.subplot_imshow(
                x, y, image, title, normalize=normalize, rgb_mask=rgb_mask, **kwargs)

    @allow_sharexy
    def subplot_imshow_labels(self, x, y, labels, title=None, clear=True,
                              renumber=True, sharex=None, sharey=None,
                              use_imshow=False):
        """Show a labels matrix using the default color map

        x,y - the subplot's coordinates
        image - the binary image to show
        title - the caption for the image
        clear - clear the axis before showing
        sharex, sharey - the coordinates of the subplot that dictates
                panning and zooming, if any
        use_imshow - Use matplotlib's imshow to display instead of creating
                     our own artist.
        """
        if renumber:
            labels = tools.renumber_labels_for_display(labels)

        cm = matplotlib.cm.get_cmap(cellprofiler.preferences.get_default_colormap())
        cm.set_bad((0, 0, 0))
        labels = numpy.ma.array(labels, mask=labels == 0)
        mappable = matplotlib.cm.ScalarMappable(cmap=cm)

        if all([c0x == 0 for c0x in cm(0)[:3]]):
            # Set the lower limit to 0 if the color for index 0 is already black.
            mappable.set_clim(0, labels.max())
            cm = None
        elif numpy.any(labels != 0):
            mappable.set_clim(1, labels.max())
            cm = None
        image = mappable.to_rgba(labels)[:, :, :3]
        return self.subplot_imshow(x, y, image, title, clear, colormap=cm,
                                   normalize=False, vmin=None, vmax=None,
                                   sharex=sharex, sharey=sharey,
                                   use_imshow=use_imshow)

    @allow_sharexy
    def subplot_imshow_ijv(self, x, y, ijv, shape=None, title=None,
                           clear=True, renumber=True, sharex=None, sharey=None,
                           use_imshow=False):
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
        image = numpy.zeros(list(shape) + [3], numpy.float)
        if len(ijv) > 0:
            cm = matplotlib.cm.get_cmap(cellprofiler.preferences.get_default_colormap())
            max_label = numpy.max(ijv[:, 2])
            if renumber:
                numpy.random.seed(0)
                order = numpy.random.permutation(max_label)
            else:
                order = numpy.arange(max_label)
            order = numpy.hstack(([0], order + 1))
            colors = matplotlib.cm.ScalarMappable(cmap=cm).to_rgba(order)
            r, g, b, a = [scipy.sparse.coo_matrix((colors[ijv[:, 2], i], (ijv[:, 0], ijv[:, 1])),
                                                  shape=shape).toarray()
                          for i in range(4)]
            for i, plane in enumerate((r, g, b)):
                image[a != 0, i] = plane[a != 0] / a[a != 0]
        return self.subplot_imshow(x, y, image, title, clear,
                                   normalize=False, vmin=None, vmax=None,
                                   sharex=sharex, sharey=sharey,
                                   use_imshow=use_imshow)

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
        kwargs['colormap'] = matplotlib.cm.Greys_r
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
        kwargs['colormap'] = matplotlib.cm.binary_r
        return self.subplot_imshow(x, y, image, title=title, **kwargs)

    @staticmethod
    def normalize_image(image, **kwargs):
        """Produce a color image normalized according to user spec"""
        colormap = kwargs['colormap']
        normalize = kwargs['normalize']
        vmin = kwargs['vmin']
        vmax = kwargs['vmax']
        rgb_mask = kwargs['rgb_mask']
        image = image.astype(numpy.float32)
        if isinstance(colormap, matplotlib.cm.ScalarMappable):
            colormap = colormap.cmap

        # Perform normalization
        if normalize == 'log':
            if is_color_image(image):
                image = numpy.dstack([log_transform(image[:, :, ch]) for ch in range(image.shape[2])])
            else:
                image = log_transform(image)
        elif normalize:
            if is_color_image(image):
                image = numpy.dstack([auto_contrast(image[:, :, ch]) for ch in range(image.shape[2])])
            else:
                image = auto_contrast(image)

        # Apply rgb mask to hide/show channels
        if is_color_image(image):
            rgb_mask = match_rgbmask_to_image(rgb_mask, image)
            image *= rgb_mask
            if image.shape[2] == 2:
                image = numpy.dstack([image[:, :, 0],
                                   image[:, :, 1],
                                      numpy.zeros(image.shape[:2], image.dtype)])
        if not is_color_image(image):
            mappable = matplotlib.cm.ScalarMappable(cmap=colormap)
            mappable.set_clim(vmin, vmax)
            image = mappable.to_rgba(image)[:, :, :3]
        #
        # add the segmentations
        #
        for cplabel in kwargs['cplabels']:
            if cplabel[CPLD_MODE] == CPLDM_NONE or \
                    not cplabel.get(CPLD_SHOW, True):
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
                        dti, dtj = numpy.where((d < hw + .5) & ~lm)
                        lo[dti, dtj] = numpy.minimum(1, hw + .5 - d[dti, dtj])
                    image = image * (1 - lo[:, :, numpy.newaxis]) + \
                            lo[:, :, numpy.newaxis] * oc[numpy.newaxis, numpy.newaxis, :]
                elif cplabel[CPLD_MODE] == CPLDM_ALPHA:
                    #
                    # For alpha overlays, renumber
                    lnumbers = tools.renumber_labels_for_display(labels) + loffset
                    mappable = matplotlib.cm.ScalarMappable(
                            cmap=cplabel[CPLD_ALPHA_COLORMAP])
                    mappable.set_clim(1, ltotal)
                    limage = mappable.to_rgba(lnumbers[labels != 0])[:, :3]
                    alpha = cplabel[CPLD_ALPHA_VALUE]
                    image[labels != 0, :] *= 1 - alpha
                    image[labels != 0, :] += limage * alpha
                loffset += numpy.max(labels)

        return image

    def subplot_table(self, x, y, statistics,
                      col_labels=None,
                      row_labels=None,
                      n_cols=1,
                      n_rows=1, **kwargs):
        """Put a table into a subplot

        x,y - subplot's column and row
        statistics - a sequence of sequences that form the values to
                     go into the table
        col_labels - labels for the column header

        row_labels - labels for the row header

        **kwargs - for backwards compatibility, old argument values
        """

        nx, ny = self.subplots.shape
        xstart = float(x) / float(nx)
        ystart = float(y) / float(ny)
        width = float(n_cols) / float(nx)
        height = float(n_rows) / float(ny)
        cw, ch = self.figure.canvas.GetSizeTuple()
        ctrl = wx.grid.Grid(self.figure.canvas)
        self.widgets.append(
                (xstart, ystart, width, height,
                 wx.ALIGN_CENTER, wx.ALIGN_CENTER, ctrl))
        nrows = len(statistics)
        ncols = 0 if nrows == 0 else len(statistics[0])
        ctrl.CreateGrid(nrows, ncols)
        if col_labels is not None:
            for i, value in enumerate(col_labels):
                ctrl.SetColLabelValue(i, unicode(value))
        else:
            ctrl.SetColLabelSize(0)
        if row_labels is not None:
            ctrl.GridRowLabelWindow.Font = ctrl.GetLabelFont()
            ctrl.SetRowLabelAlignment(wx.ALIGN_LEFT, wx.ALIGN_CENTER)
            max_width = 0
            for i, value in enumerate(row_labels):
                value = unicode(value)
                ctrl.SetRowLabelValue(i, value)
                max_width = max(
                        max_width,
                        ctrl.GridRowLabelWindow.GetTextExtent(value + "M")[0])
            ctrl.SetRowLabelSize(max_width)
        else:
            ctrl.SetRowLabelSize(0)

        for i, row in enumerate(statistics):
            for j, value in enumerate(row):
                ctrl.SetCellValue(i, j, unicode(value))
                ctrl.SetReadOnly(i, j, True)
        ctrl.AutoSize()
        ctrl.Show()
        self.align_widget(ctrl, xstart, ystart, width, height,
                          wx.ALIGN_CENTER, wx.ALIGN_CENTER, cw, ch)
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

    def subplot_scatter(self, x, y,
                        xvals, yvals,
                        xlabel='', ylabel='',
                        xscale='linear', yscale='linear',
                        title='',
                        clear=True):
        """Put a scatterplot into a subplot

        x, y - subplot's column and row
        xvals, yvals - values to scatter
        xlabel - string label for x axis
        ylabel - string label for y axis
        xscale - scaling of the x axis (e.g. 'log' or 'linear')
        yscale - scaling of the y axis (e.g. 'log' or 'linear')
        title  - string title for the plot
        """
        xvals = numpy.array(xvals).flatten()
        yvals = numpy.array(yvals).flatten()
        if clear:
            self.clear_subplot(x, y)

        self.figure.set_facecolor((1, 1, 1))
        self.figure.set_edgecolor((1, 1, 1))

        axes = self.subplot(x, y)
        plot = axes.scatter(xvals, yvals,
                            facecolor=(0.0, 0.62, 1.0),
                            edgecolor='none',
                            alpha=0.75)
        axes.set_title(title)
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)

        return plot

    def subplot_histogram(self, x, y, values,
                          bins=20,
                          xlabel='',
                          xscale=None,
                          yscale='linear',
                          title='',
                          clear=True):
        """Put a histogram into a subplot

        x,y - subplot's column and row
        values - values to plot
        bins - number of bins to aggregate data in
        xlabel - string label for x axis
        xscale - 'log' to log-transform the data
        yscale - scaling of the y axis (e.g. 'log')
        title  - string title for the plot
        """
        if clear:
            self.clear_subplot(x, y)
        axes = self.subplot(x, y)
        self.figure.set_facecolor((1, 1, 1))
        self.figure.set_edgecolor((1, 1, 1))
        values = numpy.array(values).flatten()
        if xscale == 'log':
            values = numpy.log(values[values > 0])
            xlabel = 'Log(%s)' % (xlabel or '?')
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
        plot = axes.hist(values, bins,
                         facecolor=(0.0, 0.62, 1.0),
                         edgecolor='none',
                         log=(yscale == 'log'),
                         alpha=0.75)
        axes.set_xlabel(xlabel)
        axes.set_title(title)

        return plot

    def subplot_density(self, x, y, points,
                        gridsize=100,
                        xlabel='',
                        ylabel='',
                        xscale='linear',
                        yscale='linear',
                        bins=None,
                        cmap='jet',
                        title='',
                        clear=True):
        """Put a histogram into a subplot

        x,y - subplot's column and row
        points - values to plot
        gridsize - x & y bin size for data aggregation
        xlabel - string label for x axis
        ylabel - string label for y axis
        xscale - scaling of the x axis (e.g. 'log' or 'linear')
        yscale - scaling of the y axis (e.g. 'log' or 'linear')
        bins - scaling of the color map (e.g. None or 'log', see mpl.hexbin)
        title  - string title for the plot
        """
        if clear:
            self.clear_subplot(x, y)
        axes = self.subplot(x, y)
        self.figure.set_facecolor((1, 1, 1))
        self.figure.set_edgecolor((1, 1, 1))

        points = numpy.array(points)

        # Clip to positives if in log space
        if xscale == 'log':
            points = points[(points[:, 0] > 0)]
        if yscale == 'log':
            points = points[(points[:, 1] > 0)]

        # nothing to plot?
        if len(points) == 0 or points == [[]]: return

        plot = axes.hexbin(points[:, 0], points[:, 1],
                           gridsize=gridsize,
                           xscale=xscale,
                           yscale=yscale,
                           bins=bins,
                           cmap=matplotlib.cm.get_cmap(cmap))
        cb = self.figure.colorbar(plot)
        if bins == 'log':
            cb.set_label('log10(N)')

        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_title(title)

        xmin = numpy.nanmin(points[:, 0])
        xmax = numpy.nanmax(points[:, 0])
        ymin = numpy.nanmin(points[:, 1])
        ymax = numpy.nanmax(points[:, 1])

        # Pad all sides
        if xscale == 'log':
            xmin /= 1.5
            xmax *= 1.5
        else:
            xmin = xmin - (xmax - xmin) / 20.
            xmax = xmax + (xmax - xmin) / 20.

        if yscale == 'log':
            ymin /= 1.5
            ymax *= 1.5
        else:
            ymin = ymin - (ymax - ymin) / 20.
            ymax = ymax + (ymax - ymin) / 20.

        axes.axis([xmin, xmax, ymin, ymax])

        return plot

    def subplot_platemap(self, x, y, plates_dict, plate_type,
                         cmap=matplotlib.cm.jet, colorbar=True, title=''):
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

        if 'plate_choice' not in self.__dict__:
            platemap_plate = plate_names[0]
            # Add plate selection choice
            #
            # Make the text transparent so the gradient shows.
            # Intercept paint to paint the foreground only
            # Intercept erase background to do nothing
            # Intercept size to make sure we redraw
            #
            plate_static_text = wx.StaticText(
                    self.navtoolbar, -1, 'Plate: ',
                    style=wx.TRANSPARENT_WINDOW)

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

            self.plate_choice = wx.Choice(
                    self.navtoolbar, -1, choices=plate_names)

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
            self.plate_choice.cmap = cmap
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
        alphabet = 'ABCDEFGHIJKLMNOP'  # enough letters for a 384 well plate
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
        cmap.set_bad('gray', 1.)
        clean_data = numpy.ma.array(data, mask=numpy.isnan(data))
        plot = axes.imshow(clean_data, cmap=cmap, interpolation='nearest',
                           shape=data.shape)
        axes.set_title(title)
        axes.set_xticks(range(ncols))
        axes.set_yticks(range(nrows))
        axes.set_xticklabels(range(1, ncols + 1), minor=False)
        axes.set_yticklabels(alphabet[:nrows], minor=False)
        axes.axis('image')

        if colorbar and not numpy.all(numpy.isnan(data)):
            if self.colorbar.has_key(axes):
                cb = self.colorbar[axes]
                cb.set_clim(numpy.min(clean_data), numpy.max(clean_data))
                cb.update_normal(clean_data)
            else:
                self.colorbar[axes] = self.figure.colorbar(
                        plot, ax=axes)

        def format_coord(x, y):
            col = int(x + 0.5)
            row = int(y + 0.5)
            if (0 <= col < ncols) and (0 <= row < nrows):
                val = data[row, col]
                res = '%s%02d - %1.4f' % (alphabet[row], int(col + 1), val)
            else:
                res = '%s%02d' % (alphabet[row], int(col + 1))
                # TODO:
            ##            hint = wx.TipWindow(self, res)
            ##            wx.FutureCall(500, hint.Close)
            return res

        axes.format_coord = format_coord
        axes.figure.canvas.draw()
        return plot


def format_plate_data_as_array(plate_dict, plate_type):
    """ Returns an array shaped like the given plate type with the values from
    plate_dict stored in it.  Wells without data will be set to np.NaN
    plate_dict  -  dict mapping well names to data. eg: d["A01"] --> data
                   data values must be of numerical or string types
    plate_type  - '96' (return 8x12 array) or '384' (return 16x24 array)
    """
    if plate_type == '96':
        plate_shape = (8, 12)
    elif plate_type == '384':
        plate_shape = (16, 24)
    alphabet = 'ABCDEFGHIJKLMNOP'
    data = numpy.zeros(plate_shape)
    data[:] = numpy.nan
    display_error = True
    for well, val in plate_dict.items():
        r = alphabet.index(well[0].upper())
        c = int(well[1:]) - 1
        if r >= data.shape[0] or c >= data.shape[1]:
            if display_error:
                logging.getLogger("cellprofiler.gui.cpfigure").warning(
                        'A well value (%s) does not fit in the given plate type.\n' % well)
                display_error = False
            continue
        data[r, c] = val
    return data


def show_image(url, parent=None, needs_raise_after=True):
    """Show an image in a figure frame

    url - url of the image
    parent - parent frame to this one.
    """
    filename = url[(url.rfind("/") + 1):]
    try:
        if url.lower().endswith(".mat"):
            from scipy.io.matlab.mio import loadmat
            from cellprofiler.modules.loadimages import url2pathname
            image = loadmat(url2pathname(url), struct_as_record=True)["Image"]
        else:
            from bioformats import load_image_url
            image = load_image_url(url)
    except IOError:
        wx.MessageBox('Failed to open file, "%s"' % filename,
                      caption="File open error")
        return
    except javabridge.JavaException, je:
        wx.MessageBox(
                'Could not open "%s" as an image.' % filename,
                caption="File format error")
        return

    except Exception, e:
        from cellprofiler.gui.errordialog import display_error_dialog
        display_error_dialog(None, e, None,
                             "Failed to load %s" % url,
                             continue_only=True)
        return
    frame = Figure(parent=parent,
                   title=filename,
                   subplots=(1, 1))
    if image.ndim == 2:
        frame.subplot_imshow_grayscale(0, 0, image, title=filename)
    else:
        frame.subplot_imshow_color(0, 0, image[:, :, :3], title=filename)
    frame.panel.draw()
    if needs_raise_after:
        # %$@ hack hack hack
        wx.CallAfter(lambda: frame.Raise())
    return True


roundoff = True


class CPOutlineArtist(matplotlib.collections.LineCollection):
    """An artist that is a plot of the outline around an object

    This class is here so that we can add and remove artists for certain
    outlines.
    """

    def __init__(self, name, labels, *args, **kwargs):
        """Draw outlines for objects

        name - the name of the outline

        labels - a sequence of labels matrices

        kwargs - further arguments for Line2D
        """
        # get_outline_pts has its faults:
        # * it doesn't do holes
        # * it only does one of two disconnected objects
        #
        # We get around the second failing by resegmenting with
        # connected components and combining the original and new segmentation
        #
        lines = []
        for l in labels:
            new_labels, counts = scipy.ndimage.label(l != 0, numpy.ones((3, 3), bool))
            if counts == 0:
                continue
            l = l.astype(numpy.uint64) * counts + new_labels
            unique, idx = numpy.unique(l.flatten(), return_inverse=True)
            if unique[0] == 0:
                my_range = numpy.arange(len(unique))
            else:
                my_range = numpy.arange(1, len(unique))
            idx.shape = l.shape
            pts, offs, counts = centrosome.cpmorphology.get_outline_pts(idx, my_range)
            pts = pts[:, ::-1]  # matplotlib x, y reversed from i,j
            for off, count in zip(offs, counts):
                lines.append(numpy.vstack((pts[off:off + count], pts[off:off + 1])))
        matplotlib.collections.LineCollection.__init__(
                self, lines, *args, **kwargs)

    def get_outline_name(self):
        return self.__outline_name


class CPImageArtist(matplotlib.artist.Artist):
    def __init__(self, image, frame, kwargs):
        super(CPImageArtist, self).__init__()
        self.image = image
        self.frame = frame
        self.kwargs = kwargs
        #
        # The radius for the gaussian blur of 1 pixel sd
        #
        self.filterrad = 4.0
        self.interpolation = kwargs["interpolation"]

    def draw(self, renderer):
        global roundoff
        image = self.frame.normalize_image(self.image,
                                           **self.kwargs)
        magnification = renderer.get_image_magnification()
        numrows, numcols = self.image.shape[:2]
        if numrows == 0 or numcols == 0:
            return
        #
        # Limit the viewports to the image extents
        #
        view_x0 = int(min(numcols - 1, max(0, self.axes.viewLim.x0 - self.filterrad)))
        view_x1 = int(min(numcols, max(0, self.axes.viewLim.x1 + self.filterrad)))
        view_y0 = int(min(numrows - 1,
                          max(0, min(self.axes.viewLim.y0,
                                     self.axes.viewLim.y1) - self.filterrad)))
        view_y1 = int(min(numrows,
                          max(0, max(self.axes.viewLim.y0,
                                     self.axes.viewLim.y1) + self.filterrad)))
        xslice = slice(view_x0, view_x1)
        yslice = slice(view_y0, view_y1)
        image = image[yslice, xslice, :]

        #
        # Flip image upside-down if height is negative
        #
        flip_ud = self.axes.viewLim.height < 0
        if flip_ud:
            image = numpy.flipud(image)

        im = matplotlib.image.fromarray(image, 0)
        im.is_grayscale = False
        im.set_interpolation(self.interpolation)
        fc = self.axes.patch.get_facecolor()
        bg = matplotlib.colors.colorConverter.to_rgba(fc, 0)
        im.set_bg(*bg)

        # image input dimensions
        im.reset_matrix()

        # the viewport translation in the X direction
        tx = view_x0 - self.axes.viewLim.x0 - .5
        #
        # the viewport translation in the Y direction
        # which is from the bottom of the screen
        #
        if self.axes.viewLim.height < 0:
            ty = (self.axes.viewLim.y0 - view_y1) + .5
        else:
            ty = view_y0 - self.axes.viewLim.y0 - .5
        im.apply_translation(tx, ty)

        l, b, r, t = self.axes.bbox.extents
        widthDisplay = (r - l + 1) * magnification
        heightDisplay = (t - b + 1) * magnification

        # resize viewport to display
        sx = widthDisplay / self.axes.viewLim.width
        sy = abs(heightDisplay / self.axes.viewLim.height)
        im.apply_scaling(sx, sy)
        im.resize(widthDisplay, heightDisplay,
                  norm=1, radius=self.filterrad)
        bbox = self.axes.bbox.frozen()
        im._url = self.frame.Title

        # Two ways to do this, try by version
        mplib_version = matplotlib.__version__.split(".")
        if mplib_version[0] == '0':
            renderer.draw_image(l, b, im, bbox)
        else:
            gc = renderer.new_gc()
            gc.set_clip_rectangle(bbox)
            renderer.draw_image(gc, l, b, im)


def get_matplotlib_interpolation_preference():
    interpolation = cellprofiler.preferences.get_interpolation_mode()
    if interpolation == cellprofiler.preferences.IM_NEAREST:
        return matplotlib.image.NEAREST
    elif interpolation == cellprofiler.preferences.IM_BILINEAR:
        return matplotlib.image.BILINEAR
    elif interpolation == cellprofiler.preferences.IM_BICUBIC:
        return matplotlib.image.BICUBIC
    return matplotlib.image.NEAREST


__crosshair_cursor = None


def get_crosshair_cursor():
    global __crosshair_cursor
    if __crosshair_cursor is None:
        if sys.platform.lower().startswith('win'):
            #
            # Build the crosshair cursor image as a numpy array.
            #
            buf = numpy.ones((16, 16, 3), dtype='uint8') * 255
            buf[7, 1:-1, :] = buf[1:-1, 7, :] = 0
            abuf = numpy.ones((16, 16), dtype='uint8') * 255
            abuf[:6, :6] = abuf[9:, :6] = abuf[9:, 9:] = abuf[:6, 9:] = 0
            im = wx.ImageFromBuffer(16, 16, buf.tostring(), abuf.tostring())
            im.SetOptionInt(wx.IMAGE_OPTION_CUR_HOTSPOT_X, 7)
            im.SetOptionInt(wx.IMAGE_OPTION_CUR_HOTSPOT_Y, 7)
            __crosshair_cursor = wx.CursorFromImage(im)
        else:
            __crosshair_cursor = wx.CROSS_CURSOR
    return __crosshair_cursor


EVT_NAV_MODE_CHANGE = wx.PyEventBinder(wx.NewEventType())
NAV_MODE_ZOOM = 'zoom rect'
NAV_MODE_PAN = 'pan/zoom'
NAV_MODE_NONE = ''


class CPNavigationToolbar(matplotlib.backends.backend_wxagg.NavigationToolbar2WxAgg):
    """Navigation toolbar for EditObjectsDialog"""

    def set_cursor(self, cursor):
        """Set the cursor based on the mode"""
        if cursor == matplotlib.backend_bases.cursors.SELECT_REGION:
            self.canvas.SetCursor(get_crosshair_cursor())
        else:
            matplotlib.backends.backend_wxagg.NavigationToolbar2WxAgg.set_cursor(self, cursor)

    def cancel_mode(self):
        """Toggle the current mode to off"""
        if self.mode == NAV_MODE_ZOOM:
            self.zoom()
            if 'Zoom' in self.wx_ids:
                self.ToggleTool(self.wx_ids['Zoom'], False)
        elif self.mode == NAV_MODE_PAN:
            self.pan()
            if 'Pan' in self.wx_ids:
                self.ToggleTool(self.wx_ids['Pan'], False)

    def zoom(self, *args):
        matplotlib.backends.backend_wxagg.NavigationToolbar2WxAgg.zoom(self, *args)
        self.__send_mode_change_event()

    def pan(self, *args):
        matplotlib.backends.backend_wxagg.NavigationToolbar2WxAgg.pan(self, *args)
        self.__send_mode_change_event()

    def is_home(self):
        """Return True if zoom/pan is at the home position"""
        if self._views._pos <= 0:
            return True
        if self._views[0] == self._views[-1]:
            return True
        return False

    def reset(self):
        """Clear out the position stack"""
        # We differ from the reference implementation because we clear
        # the view stacks.
        self._views.clear()
        self._positions.clear()
        self.home()

    def save(self, event):
        #
        # Capture any file save event and redirect it to CPFigureFrame
        # Fixes issue #829 - Mac & PC display invalid save options when
        #                    you save using the icon.
        #
        parent = self.GetTopLevelParent()
        if isinstance(parent, Figure):
            parent.on_file_save(event)
        else:
            super(CPNavigationToolbar, self).save(event)

    def __send_mode_change_event(self):
        event = wx.NotifyEvent(EVT_NAV_MODE_CHANGE.evtType[0])
        event.EventObject = self
        self.GetEventHandler().ProcessEvent(event)
