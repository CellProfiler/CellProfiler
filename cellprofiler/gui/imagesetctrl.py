# coding=utf-8
"""ImageSetCtrl.py - A control to display an imageset
"""


import functools
import re
import urllib.parse

import numpy
import wx
import wx.adv
import wx.grid
import wx.lib.mixins.gridlabelrenderer as wxglr
from cellprofiler_core.constants.image import C_FRAME, CT_GRAYSCALE, CT_COLOR, CT_MASK, CT_OBJECTS, CT_FUNCTION
from cellprofiler_core.constants.image import C_SERIES
from cellprofiler_core.constants.measurement import C_FILE_NAME, C_C, C_Z, C_T, C_SERIES_NAME
from cellprofiler_core.constants.measurement import C_METADATA
from cellprofiler_core.constants.measurement import C_OBJECTS_FILE_NAME
from cellprofiler_core.constants.measurement import C_OBJECTS_PATH_NAME
from cellprofiler_core.constants.measurement import C_OBJECTS_URL
from cellprofiler_core.constants.measurement import C_PATH_NAME
from cellprofiler_core.constants.measurement import C_URL
from cellprofiler_core.constants.measurement import IMAGE
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.preferences import report_progress
from cellprofiler_core.setting import FileCollectionDisplay
from cellprofiler_core.setting.filter import DirectoryPredicate
from cellprofiler_core.setting.filter import ExtensionPredicate
from cellprofiler_core.setting.filter import FilePredicate
from cellprofiler_core.setting.filter import Filter
from cellprofiler_core.utilities.image import url_to_modpath
from cellprofiler_core.utilities.legacy import cmp

import cellprofiler.gui
from cellprofiler.gui.help.content import CREATING_A_PROJECT_CAPTION
import cellprofiler.gui.gridrenderers as cpglr
from cellprofiler.icons import get_builtin_image

"""Table column displays metadata"""
COL_METADATA = "Metadata"
"""Table column displays a URL"""
COL_URL = "URL"
"""Table column displays a file name"""
COL_FILENAME = "Filename"
"""Table column displays a path name"""
COL_PATHNAME = "Pathname"
"""Table column displays a series number"""
COL_SERIES = "Series"
"""Table column displays a frame number"""
COL_FRAME = "Frame"

COL_ORDER = [COL_PATHNAME, COL_FILENAME, COL_URL, COL_SERIES, COL_FRAME]

"""Display mode appropriate for novice users. Hide everything but file names"""
DISPLAY_MODE_SIMPLE = "Simple"

"""Display mode that shows more stuff"""
DISPLAY_MODE_COMPLEX = "Complex"

"""Revised display mode for CP5"""
DISPLAY_MODE_ALTERNATE = "Alternate"

ERROR_COLOR = wx.Colour(255, 0, 0)


class ImageSetGridTable(wx.grid.GridTableBase):
    DEFAULT_ATTR = wx.grid.GridCellAttr()
    ERROR_ATTR = wx.grid.GridCellAttr()
    ERROR_ATTR.SetTextColour(ERROR_COLOR)

    class ImageSetColumn(object):
        def __init__(
            self, name, channel, feature, column_type, channel_type, is_key=False
        ):
            """Initialize ImageSetColumn

            name - display name of the column

            feature - the measurement feature name

            column_type - one of COL_* indicating what it means

            channel_type - the channel type from the channel descriptor

            is_key - if metadata, it's part of the unique key for the
                     image set
            """
            self.name = name
            self.feature = feature
            self.channel = channel
            self.column_type = column_type
            self.channel_type = channel_type
            self.is_key = is_key

    def __init__(self, workspace, display_mode=DISPLAY_MODE_SIMPLE):
        super(self.__class__, self).__init__()
        self.workspace = workspace
        self.columns = []
        self.n_rows = 0
        self.display_mode = display_mode
        self.controller = None
        self.cache = ImageSetCache(workspace.measurements)

    def recompute(self):
        """Recompute the layout

        returns the number of rows and columns added or removed
        """
        self.cache = ImageSetCache(self.measurements)
        old_row_count = self.n_rows
        old_column_count = len(self.columns)
        self.columns = self.get_columns()
        self.n_rows = len(self.cache)
        if self.n_rows > 0:
            self.image_numbers = self.measurements.get_image_numbers().copy()
            self.metadata_tags = self.measurements.get_metadata_tags()
        return self.n_rows - old_row_count, len(self.columns) - old_column_count

    @property
    def measurements(self):
        m = self.workspace.measurements
        return m

    def get_columns(self):
        columns = []
        m = self.measurements
        if m is None or len(self.cache) == 0:
            return columns
        assert isinstance(m, Measurements)
        metadata_tags = m.get_metadata_tags()
        for feature in m.get_feature_names(IMAGE):
            is_key = False
            channel = None
            if self.display_mode == DISPLAY_MODE_COMPLEX:
                if feature.startswith(C_METADATA):
                    column_type = COL_METADATA
                    name = feature.split("_", 1)[1]
                    if feature in metadata_tags:
                        is_key = True
                elif feature.startswith(C_FILE_NAME) or feature.startswith(
                    C_OBJECTS_FILE_NAME
                ):
                    column_type = COL_FILENAME
                    channel = feature.split("_", 1)[1]
                    name = "%s File Name" % channel
                elif feature.startswith(C_PATH_NAME) or feature.startswith(
                    C_OBJECTS_PATH_NAME
                ):
                    column_type = COL_PATHNAME
                    channel = feature.split("_", 1)[1]
                    name = "%s Path Name" % channel
                elif feature.startswith(C_URL) or feature.startswith(C_OBJECTS_URL):
                    column_type = COL_URL
                    channel = feature.split("_", 1)[1]
                    name = "%s URL" % channel
                elif feature.startswith(C_SERIES):
                    column_type = COL_SERIES
                    channel = feature.split("_", 1)[1]
                    name = "%s Series" % channel
                elif feature.startswith(C_FRAME):
                    column_type = COL_FRAME
                    channel = feature.split("_", 1)[1]
                    name = "%s Frame" % channel
                else:
                    continue
            elif self.display_mode == DISPLAY_MODE_COMPLEX and feature.startswith(C_URL) or feature.startswith(C_OBJECTS_URL):
                column_type = COL_URL
                channel = feature.split("_", 1)[1]
                name = channel
            elif self.display_mode == DISPLAY_MODE_ALTERNATE and feature.startswith(C_FILE_NAME) or feature.startswith(C_OBJECTS_FILE_NAME):
                column_type = COL_FILENAME
                channel = feature.split("_", 1)[1]
                name = channel
            else:
                continue
            channel_type = m.get_channel_descriptor(channel)

            columns.append(
                self.ImageSetColumn(
                    name,
                    channel,
                    feature,
                    column_type,
                    channel_type,
                    is_key,
                )
            )

        def ordering_fn(a, b):
            """Put keys first, then sort by channel name"""
            #
            # If either is a key, put the one that is a key first
            #
            if a.is_key:
                if b.is_key:
                    return cmp(a.name, b.name)
                return -1
            elif b.is_key:
                return 1
            #
            # If either is metadata, put the metadata last
            #
            if a.column_type == COL_METADATA:
                if b.column_type == COL_METADATA:
                    return cmp(a.name, b.name)
                return 1
            elif b.column_type == COL_METADATA:
                return -1
            #
            # If different channels, order by channel
            #
            if a.channel != b.channel:
                return cmp(a.channel, b.channel)
            #
            # Otherwise, the order is given by COL_ORDER
            #
            return cmp(COL_ORDER.index(a.column_type), COL_ORDER.index(b.column_type))

        columns = sorted(columns, key=functools.cmp_to_key(ordering_fn))

        return columns

    def GetAttr(self, row, col, kind):
        attr = self.DEFAULT_ATTR
        attr.IncRef()  # OH so bogus, don't refcount = bus error
        return attr

    def CanHaveAttributes(self):
        return True

    def GetNumberRows(self):
        return self.n_rows

    def GetNumberCols(self):
        return len(self.columns)

    def IsEmptyCell(self, row, col):
        return row >= self.GetNumberRows() or col >= self.GetNumberCols()

    def GetValue(self, row, col):
        if (
            row >= self.n_rows
            or col >= len(self.columns)
            or row >= len(self.image_numbers)
        ):
            return ""
        image_set = self.image_numbers[row]
        column = self.columns[col]
        value = self.cache[column.feature, image_set]
        if isinstance(value, bytes):
            value = value.decode("unicode_escape")
        if (
            column.column_type == COL_URL
            and self.display_mode == DISPLAY_MODE_SIMPLE
            and value is not None
        ):
            last_slash = value.rfind("/")
            return urllib.parse.unquote(value[(last_slash + 1) :])
        elif (
                column.column_type == COL_FILENAME
                and self.display_mode == DISPLAY_MODE_ALTERNATE
                and value is not None
        ):
            # Check for and add a series name to the plane label
            meas = f"{C_SERIES_NAME}_{column.channel}"
            if self.measurements.has_measurements(
                    IMAGE, meas, image_set_number=image_set):
                res = self.measurements.get_measurement(
                    IMAGE, meas, image_set_number=image_set)
                if res:
                    value += f" ({res})"
            # Now check for and add CZT indexes
            keys = [C_SERIES, C_C, C_Z, C_T]
            for key in keys:
                meas = f"{key}_{column.channel}"
                if self.measurements.has_measurements(
                        IMAGE, meas, image_set_number=image_set):
                    res = self.measurements.get_measurement(
                        IMAGE, meas, image_set_number=image_set)
                    if res is not None:
                        value += f", {key[0]} {res}"
        return value

    def get_url(self, row, col):
        """Get the URL for a cell"""
        image_set = self.image_numbers[row]
        column = self.columns[col]
        if column.channel_type == CT_OBJECTS:
            feature = C_OBJECTS_URL + "_" + column.channel
        else:
            feature = C_URL + "_" + column.channel
        value = self.cache[feature, image_set]
        if value is not None:
            return value.encode()

    def GetRowLabelValue(self, row):
        if row >= len(self.image_numbers):
            return ""
        image_number = self.image_numbers[row]
        metadata_tags = self.metadata_tags
        if len(metadata_tags) > 0:
            key = [str(self.cache[tag, image_number]) for tag in metadata_tags]
            return " : ".join(key)

        return str(image_number)

    def GetColLabelValue(self, col):
        if col >= len(self.columns):
            return ""
        return self.columns[col].name

    def AppendCols(self, numCols=1):
        if self.controller is None:
            return False
        for i in range(numCols):
            self.controller.append_channel()
        return True

    def AppendRows(self, numRows=1, updateLabels=True):
        return True

    def InsertCols(self, pos=0, numCols=1, updateLabels=True):
        return True

    def InsertRows(self, pos=0, numRows=1, updateLabels=True):
        return True

    def DeleteCols(self, pos=0, numCols=1, updateLabels=True):
        if self.controller is None:
            return False
        channels = [x.channel for x in self.columns[pos : (pos + numCols)]]
        for channel in channels:
            self.controller.remove_channel(channel)
        return True

    def DeleteRows(self, pos=0, numRows=1, updateLabels=True):
        return True

    def SetColLabelValue(self, index, value):
        if self.controller is not None:
            self.controller.change_channel_name(self.columns[index].channel, value)


class ImageSetCache:
    """A cache for image set features

    This cache is optimized for display of pages of image sets.
    """

    def __init__(self, m, page_size=100, max_size=100):
        """Initialize the image set cache with a measurements structure

        m - measurements structure

        page_size - # of image sets per cached page

        max_size - maximum # of cached chunks (pages * features)
        """
        self.m = m
        if m is None:
            return
        self.page_size = page_size
        self.max_size = max_size
        self.cache = {}
        self.access_time = 0
        #
        # Break the image set into pages
        #
        image_set_numbers = m.get_image_numbers()
        if len(image_set_numbers) == 0:
            self.pages = []
            self.image_set_page = numpy.zeros(0, int)
            self.image_set_index = numpy.zeros(0, int)
        else:
            self.pages = [
                image_set_numbers[i : (i + page_size)]
                for i in range(0, len(image_set_numbers), page_size)
            ]
            self.image_set_page = numpy.zeros(numpy.max(image_set_numbers) + 1, int)
            self.image_set_index = numpy.zeros(numpy.max(image_set_numbers) + 1, int)
            for i, page in enumerate(self.pages):
                self.image_set_page[page] = i
                self.image_set_index[page] = numpy.arange(len(page))

    def __len__(self):
        if self.m is None:
            return 0
        else:
            return len(self.image_set_index)

    def __getitem__(self, idx):
        """Get a feature for an image number"""
        feature, image_number = idx
        if image_number > len(self.image_set_index):
            return ""
        page = self.image_set_page[image_number]
        index = self.image_set_index[image_number]
        key = (feature, page)
        if key not in self.cache:
            if len(self.cache) >= self.max_size:
                self.decimate()
            entry = [
                self.m[IMAGE, feature, self.pages[page]],
                self.access_time,
            ]
            self.cache[key] = entry
        else:
            entry = self.cache[key]
            entry[1] = self.access_time
        self.access_time += 1
        result = entry[0][index]
        if isinstance(result, bytes):
            result = result.decode("utf-8")
        return result

    def decimate(self):
        """Reduce the cache size by 1/2"""
        # Get the cache, sorted from low values to high
        cache_kv = sorted(list(self.cache.items()), key=lambda x: x[1][1])
        # Take 1/2 of the max size
        self.cache = dict(cache_kv[-int(self.max_size / 2) :])


class ImageSetCtrl(wx.grid.Grid, wxglr.GridWithLabelRenderersMixin):
    def __init__(self, workspace, *args, **kwargs):
        """Initialize the ImageSetCtrl

        workspace - display the image set using the measurements in
                    this workspace.

        *args, **kwargs - see the documentation for wx.grid.Grid
        """
        if "read_only" in kwargs:
            self.read_only = kwargs["read_only"]
            kwargs = dict(kwargs)
            del kwargs["read_only"]
        else:
            self.read_only = False
        if "display_mode" in kwargs:
            display_mode = kwargs["display_mode"]
            kwargs = dict(kwargs)
            del kwargs["display_mode"]
        else:
            display_mode = DISPLAY_MODE_ALTERNATE

        wx.grid.Grid.__init__(self, *args, **kwargs)
        wxglr.GridWithLabelRenderersMixin.__init__(self)
        self.SetCornerLabelRenderer(cpglr.CornerLabelRenderer(
            self, self.on_update, tooltip="Update and display the image set", label="Update"))
        self.SetDefaultRowLabelRenderer(cpglr.RowLabelRenderer())

        gclw = self.GetGridColLabelWindow()
        self.table = ImageSetGridTable(workspace, display_mode)
        self.SetTable(self.table)
        self.AutoSize()
        self.EnableEditing(False)
        self.SetDefaultCellOverflow(False)
        self.cell_renderer = EllipsisGridCellRenderer()
        self.SetDefaultRenderer(self.cell_renderer)
        self.col_label_renderer = ColLabelRenderer(self.read_only)
        self.SetDoubleBuffered(True)
        self.GetGridWindow().SetDoubleBuffered(True)
        self.GetGridColLabelWindow().SetDoubleBuffered(True)
        self.GetGridRowLabelWindow().SetDoubleBuffered(True)
        self.column_label_editor = wx.TextCtrl(
            gclw, style=wx.TE_PROCESS_ENTER, validator=ColumnNameValidator(self.table)
        )
        self.column_label_editor.Hide()
        self.column_label_editor.Bind(
            wx.EVT_KILL_FOCUS, self.on_col_label_editor_kill_focus
        )
        self.column_label_editor.Bind(
            wx.EVT_TEXT_ENTER, self.on_col_label_editor_text_enter
        )
        self.column_label_editor.Bind(wx.EVT_CHAR, self.on_col_label_editor_char)
        self.editor_column = None
        self.recompute()
        self.pressed_button = (-1, None, False)

        def get_builtin_bitmap(name):
            return wx.Bitmap(get_builtin_image(name))

        self.color_channel_image = get_builtin_bitmap("color")
        self.monochrome_channel_image = get_builtin_bitmap("monochrome")
        self.mask_image = get_builtin_bitmap("mask")
        self.objects_image = get_builtin_bitmap("objects")
        self.illumination_function_image = get_builtin_bitmap("illumination_function")
        self.filter_image = get_builtin_bitmap("filter")

        gclw.Bind(wx.EVT_PAINT, self.on_paint_gclw)
        gclw.Bind(wx.EVT_LEFT_DOWN, self.on_gclw_left_mouse_down)
        gclw.Bind(wx.EVT_LEFT_UP, self.on_gclw_left_mouse_up)
        gclw.Bind(wx.EVT_MOTION, self.on_gclw_motion)
        gclw.Bind(wx.EVT_MOUSE_CAPTURE_LOST, self.on_gclw_mouse_capture_lost)

        self.drop_location = None
        self.drop_target = ImageSetCtrlDropTarget(self)
        self.GetGridWindow().SetDropTarget(self.drop_target)

        self.EnableDragCell(True)
        self.Bind(wx.grid.EVT_GRID_CELL_BEGIN_DRAG, self.on_grid_begin_drag)

    def on_update(self, event):
        self.table.workspace.refresh_image_set()
        n_imagesets = self.table.workspace.measurements.image_set_count
        if n_imagesets == 0:

            wx.MessageBox(
                "Sorry, your pipeline doesn't produce any valid image sets "
                "as currently configured. Check your Input module settings, "
                "or see Help > {caption} for additional assistance "
                "on using the Input modules.".format(
                    caption=CREATING_A_PROJECT_CAPTION
                ),
                caption="No Image Sets Available",
                style=wx.OK | wx.ICON_INFORMATION,
                parent=self,
            )
        else:
            report_progress("ImageSetCount", None, "Found %d image sets" % n_imagesets)
        self.recompute()

    def set_controller(self, controller):
        """Set the image set controller

        controller - class derived from ImageSetController which is used
                     to update the image set.
        """
        self.table.controller = controller

    #######
    #
    # Grid column label window handling
    #
    #######

    def get_column_rect(self, col):
        _, height = self.GetGridColLabelWindow().GetClientSize()
        widths = [self.GetColSize(i) for i in range(col + 1)]
        x = 0 if col == 0 else sum(widths[:col])
        width = widths[-1]
        return wx.Rect(x, 0, width, height)

    def get_add_button_rect(self):
        last_column = self.table.GetNumberCols() - 1
        only = self.table.GetNumberCols() == 1
        label_size = self.GetGridColLabelWindow().GetTextExtent(
            self.table.GetColLabelValue(last_column)
        )
        return self.col_label_renderer.add_button_rect(
            self.get_column_rect(last_column), label_size, only
        )

    def on_paint_gclw(self, event):
        dc = wx.BufferedPaintDC(self.GetGridColLabelWindow())
        bkgnd_brush = wx.Brush(self.GetGridColLabelWindow().BackgroundColour)
        dc.SetBackground(bkgnd_brush)
        dc.Clear()
        dc.SetBackground(wx.NullBrush)
        bkgnd_brush.Destroy()
        if self.table.GetNumberCols() == 0:
            return
        selected_col, hit_code, pressed = self.pressed_button
        cols = self.CalcColLabelsExposed(self.GetGridColLabelWindow().GetUpdateRegion())
        x, y = self.CalcUnscrolledPosition((0, 0))
        pt = dc.GetDeviceOrigin()
        dc.SetDeviceOrigin(pt.x - x, pt.y)
        for col in cols:
            rect = self.get_column_rect(col)
            self.col_label_renderer.Draw(self, dc, rect, col)
        rect = self.get_add_button_rect()

    HIT_NOTHING = 0
    HIT_LABEL = 1
    HIT_CHANNEL_TYPE_BUTTON = 2
    HIT_FILTER_BUTTON = 3
    HIT_PLUS = 4
    HIT_MINUS = 5

    def gclw_hit_test(self, event):
        """Return a tuple of column # and hit test code for mouse event"""
        assert isinstance(event, wx.MouseEvent)

        x, y = self.CalcUnscrolledPosition(event.GetX(), event.GetY())
        if not self.read_only:
            r = self.get_add_button_rect()
            assert isinstance(r, wx.Rect)
            if r.Contains(event.GetPosition()):
                return self.table.GetNumberCols() - 1, self.HIT_PLUS
        if self.table.GetNumberCols() == 0:
            return None, None
        only = self.table.GetNumberCols() == 1
        for i in range(self.table.GetNumberCols()):
            last = i == self.table.GetNumberCols() - 1
            r = self.get_column_rect(i)
            if r.Contains(x, y):
                label = self.table.GetColLabelValue(i)
                label_size = self.GetGridColLabelWindow().GetTextExtent(label)
                rl = self.col_label_renderer.label_rect(r, label_size, last, only)
                if rl.Contains(x, y):
                    return i, self.HIT_LABEL
                rct = self.col_label_renderer.channel_type_icon_rect(
                    r, label_size, last, only
                )
                if self.read_only:
                    return i, self.HIT_NOTHING
                if rct.Contains(x, y):
                    return i, self.HIT_CHANNEL_TYPE_BUTTON
                rf = self.col_label_renderer.filter_icon_rect(r, label_size, last, only)
                if rf.Contains(x, y):
                    return i, self.HIT_FILTER_BUTTON
                if self.table.GetNumberCols() > 1:
                    rr = self.col_label_renderer.remove_icon_rect(r, label_size, last)
                    if rr.Contains(x, y):
                        return i, self.HIT_MINUS
                return i, self.HIT_NOTHING
        return None, None

    def on_gclw_left_mouse_down(self, event):
        assert isinstance(event, wx.MouseEvent)
        col, hit_code = self.gclw_hit_test(event)
        if col is None:
            event.Skip(True)
            return
        if hit_code in (
            self.HIT_CHANNEL_TYPE_BUTTON,
            self.HIT_FILTER_BUTTON,
            self.HIT_PLUS,
            self.HIT_MINUS,
        ):
            self.pressed_button = (col, hit_code, True)
            self.GetGridColLabelWindow().RefreshRect(
                self.get_column_rect(col), eraseBackground=False
            )
            self.GetGridColLabelWindow().CaptureMouse()
        elif hit_code == self.HIT_LABEL:
            self.activate_col_label_editor(col)
        else:
            event.Skip(True)

    def on_gclw_left_mouse_up(self, event):
        pb_col, pb_hit_code, pb_show = self.pressed_button

        if pb_hit_code is None:
            event.Skip(True)
            return
        col, hit_code = self.gclw_hit_test(event)
        self.pressed_button = (-1, None, False)
        if pb_hit_code is not None:
            self.GetGridColLabelWindow().ReleaseMouse()
        else:
            event.Skip(True)
            return
        if col == pb_col and hit_code == pb_hit_code:
            self.GetGridColLabelWindow().RefreshRect(
                self.get_column_rect(col), eraseBackground=False
            )
            if hit_code == self.HIT_CHANNEL_TYPE_BUTTON:
                self.on_channel_type_pressed(col)
            elif hit_code == self.HIT_FILTER_BUTTON:
                self.on_filter_pressed(col)
            elif hit_code == self.HIT_PLUS:
                self.on_add_column()
            elif hit_code == self.HIT_MINUS:
                self.on_remove_column(col)

    def on_gclw_motion(self, event):
        col, hit_code = self.gclw_hit_test(event)
        pb_col, pb_hit_code, pb_show = self.pressed_button
        if pb_hit_code is None:
            if hit_code == self.HIT_CHANNEL_TYPE_BUTTON:
                self.GetGridColLabelWindow().SetToolTip(
                    "Change the channel's type (monochrome, color, objects, etc)"
                )
            elif hit_code == self.HIT_FILTER_BUTTON:
                self.GetGridColLabelWindow().SetToolTip(
                    "Select items in this channel using a filter"
                )
            elif hit_code == self.HIT_PLUS:
                self.GetGridColLabelWindow().SetToolTip(
                    "Add an image column to the image set"
                )
            elif hit_code == self.HIT_MINUS:
                self.GetGridColLabelWindow().SetToolTip(
                    "Remove this image column from the image set."
                )
            elif col is not None:
                self.GetGridColLabelWindow().SetToolTip(
                    'Image column, "%s"' % self.table.columns[col].channel
                )
            event.Skip(True)
            return
        if (col != pb_col or hit_code != pb_hit_code) and pb_show:
            self.pressed_button = (pb_col, pb_hit_code, False)
            self.GetGridColLabelWindow().RefreshRect(
                self.get_column_rect(col), eraseBackground=False
            )
        elif col == pb_col and hit_code == pb_hit_code and not pb_show:
            self.pressed_button = (pb_col, pb_hit_code, True)
            self.GetGridColLabelWindow().RefreshRect(
                self.get_column_rect(col), eraseBackground=False
            )

    def on_gclw_mouse_capture_lost(self, event):
        pb_col, pb_hit_code, pb_show = self.pressed_button
        self.pressed_button = (-1, None, False)
        if pb_hit_code in (
            self.HIT_CHANNEL_TYPE_BUTTON,
            self.HIT_PLUS,
            self.HIT_FILTER_BUTTON,
            self.HIT_MINUS,
        ):
            self.GetGridColLabelWindow().RefreshRect(
                self.get_column_rect(pb_col), eraseBackground=False
            )

    def on_channel_type_pressed(self, col):
        with wx.Dialog(self) as dlg:
            assert isinstance(dlg, wx.Dialog)
            channel_name = self.table.columns[col].channel
            dlg.SetTitle(
                "Change image type for %s" % (self.table.GetColLabelValue(col))
            )
            dlg.Sizer = wx.BoxSizer(wx.VERTICAL)
            choices = [
                (
                    CT_GRAYSCALE,
                    self.monochrome_channel_image,
                    "Treat the image as monochrome, averaging colors if needed",
                ),
                (
                    CT_COLOR,
                    self.color_channel_image,
                    "Treat the image as color. Use ColorToGray to get individual colors",
                ),
                (
                    CT_MASK,
                    self.mask_image,
                    "Treat the image as a binary mask",
                ),
                (
                    CT_OBJECTS,
                    self.objects_image,
                    "Treat the image as objects",
                ),
                (
                    CT_FUNCTION,
                    self.illumination_function_image,
                    "Use the image for illumination correction",
                ),
            ]
            sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
            dlg.Sizer.AddSpacer(10)
            dlg.Sizer.Add(sub_sizer, 0, wx.EXPAND | wx.ALL, 10)
            sub_sizer.Add(wx.StaticText(dlg, label="Image type:"))
            channel_type = self.table.columns[col].channel_type
            choice = wx.adv.BitmapComboBox(
                dlg, value=channel_type, style=wx.CB_DROPDOWN
            )
            sub_sizer.Add(choice)
            selection = None
            current_help = ""
            for i, (text, bitmap, help_text) in enumerate(choices):
                choice.Append(text, bitmap, text)
                if text == channel_type:
                    selection = i
                    current_help = help_text

            help_text = wx.StaticText(dlg, label=current_help)
            help_text.Wrap(dlg.GetSize()[0] - 20)
            dlg.Sizer.Add(help_text, 1, wx.ALIGN_LEFT | wx.ALIGN_TOP | wx.ALL, 10)
            if selection is not None:
                choice.SetSelection(selection, -1)

            def on_combo(event):
                idx = choice.GetSelection()
                # noinspection PyChainedComparisons
                if 0 <= idx < len(choices):
                    help_text.Label = choices[idx][2]

            choice.Bind(wx.EVT_COMBOBOX, on_combo)

            button_sizer = wx.StdDialogButtonSizer()
            button_sizer.AddButton(wx.Button(dlg, wx.ID_OK))
            button_sizer.AddButton(wx.Button(dlg, wx.ID_CANCEL))
            dlg.Sizer.Add(button_sizer, 0, wx.ALIGN_BOTTOM)
            button_sizer.Realize()
            dlg.Layout()
            if dlg.ShowModal() == wx.ID_OK:
                channel_type = choice.GetStringSelection()
                if self.table.controller is not None:
                    self.table.controller.change_channel_type(
                        channel_name, channel_type
                    )

    def get_selected_cells(self):
        """Find the selected cells in the grid

        There are four selection mechanisms:

        * Selection by row

        * Selection by column

        * Selection by block

        * Selection by cell

        Scan through all of these, returning a list of row, column tuples.
        """
        selected_cells = set()
        for row in self.GetSelectedRows():
            selected_cells.update([(row, col) for col in range(self.GetNumberCols())])

        for col in self.GetSelectedCols():
            selected_cells.update([(row, col) for row in range(self.GetNumberRows())])

        for (row_min, col_min), (row_max, col_max) in zip(
            self.GetSelectionBlockTopLeft(), self.GetSelectionBlockBottomRight()
        ):
            for col in range(col_min, col_max + 1):
                selected_cells.update(
                    [(row, col) for row in range(row_min, row_max + 1)]
                )
        selected_cells.update(self.GetSelectedCells())
        return sorted(selected_cells)

    def on_filter_pressed(self, col):
        channel_name = self.table.columns[col].channel
        channel_name_dict = dict(
            [(x.channel, (i, x)) for i, x in enumerate(self.table.columns)]
        )

        def on_select_all(channel_name, fn_filter):
            idx, column = channel_name_dict[channel_name]
            self.SelectCol(idx, True)

        def on_select_none(channel_name, fn_filter):
            idx, column = channel_name_dict[channel_name]
            self.DeselectCol(idx)

        def on_select(channel_name, fn_filter):
            col_idx, column = channel_name_dict[channel_name]
            for row_idx in range(self.GetNumberRows()):
                url = self.table.get_url(row_idx, col_idx)
                if url is not None:
                    url = url
                    if fn_filter(url):
                        self.SelectBlock(row_idx, col_idx, row_idx, col_idx, True)

        def on_deselect(channel_name, fn_filter):
            col_idx, column = channel_name_dict[channel_name]
            for row_idx in range(self.GetNumberRows()):
                url = self.table.get_url(row_idx, col_idx)
                if url is not None:
                    url = url
                    if fn_filter(url):
                        self.DeselectCell(row_idx, col_idx)

        function_list = (
            (
                "Select channel",
                on_select_all,
                "Select all images in the current channel",
            ),
            (
                "Deselect channel",
                on_select_none,
                "Deselect all selected images in the current channel from the selection",
            ),
            (
                "Add to selection",
                on_select,
                "Add any image in the current channel that matches the filter to the current selection",
            ),
            (
                "Remove from selection",
                on_deselect,
                "Deselect any image in the current channel that is selected and matches the filter",
            ),
        )
        with FilterPanelDlg(
            self, channel_name, sorted(channel_name_dict.keys()), function_list
        ) as dlg:
            dlg.ShowModal()

    def on_add_column(self):
        print("Add column pressed")
        self.table.AppendCols(1)

    def on_remove_column(self, col):
        print("Remove column pressed")
        self.table.DeleteCols(col, 1)

    ####
    #
    # Grid column label editor handling
    #
    ####

    def on_col_label_editor_char(self, event):
        assert isinstance(event, wx.KeyEvent)
        key = event.GetKeyCode()
        if key in (wx.WXK_ESCAPE, wx.WXK_TAB):
            self.on_col_label_editor_done(False)
            return
        else:
            event.Skip()

    def on_col_label_editor_kill_focus(self, event):
        if not self.column_label_editor.IsShown():
            return
        self.on_col_label_editor_done()

    def on_col_label_editor_text_enter(self, event):
        self.on_col_label_editor_done()

    def on_col_label_editor_done(self, rename=True):
        if rename and self.column_label_editor.Validate():
            self.table.SetColLabelValue(
                self.editor_column, self.column_label_editor.GetValue()
            )
        rect = self.column_label_editor.GetRect()
        self.column_label_editor.Hide()
        self.editor_column = None
        self.GetGridColLabelWindow().RefreshRect(rect, eraseBackground=False)

    def activate_col_label_editor(self, col):
        last = col == self.table.GetNumberCols() - 1
        only = self.table.GetNumberCols() == 1
        self.editor_column = col
        rect = self.get_column_rect(col)
        self.GetGridColLabelWindow().RefreshRect(rect)
        rect = self.col_label_renderer.get_edit_rect(rect, last, only)
        self.column_label_editor.SetRect(rect)
        self.column_label_editor.Show()
        self.column_label_editor.SetValue(self.table.GetColLabelValue(col))
        self.column_label_editor.SetFocus()

    ##################
    #
    # Drop target control
    #
    ##################

    """Height of the drop graphic"""
    DROP_HEIGHT = 7

    def get_drop_location(self, x, y):
        """Return the row and column of the insert point of the drop

        x - x coordinate relative to the grid window
        y - y coordinate relative to the grid window

        returns the row and column of where the drop should be inserted. Returns
        a column of None if x is not within a column.
        """
        x, y = self.CalcUnscrolledPosition(x, y)
        if x < 0:
            col = None
        else:
            col = self.XToCol(x)
            if col == wx.NOT_FOUND:
                col = None
        bottom = self.GetGridWindow().GetVirtualSize()[1]
        if not self.GetNumberRows() or y <= self.GetRowSize(0) / 2:
            row = 0
        elif y >= bottom - self.GetRowSize(self.GetNumberRows() - 1):
            row = self.GetNumberRows()
        else:
            row = self.YToEdgeOfRow(y)
            if row == wx.NOT_FOUND:
                col = None
                row = None
            else:
                row += 1
        return row, col

    def get_drag_code(self, default_result):
        if self.drop_location is None:
            return wx.DragNone

        return (
            wx.DragNone
            if any([x is None for x in self.drop_location])
            else default_result
        )

    def on_drop_enter(self, x, y, result):
        self.drop_location = self.get_drop_location(x, y)
        self.refresh_drop_location()
        return self.get_drag_code(result)

    def on_drop_leave(self):
        self.refresh_drop_location()
        self.drop_location = None

    def on_drag_over(self, x, y, result):
        self.refresh_drop_location()
        self.drop_location = self.get_drop_location(x, y)
        self.refresh_drop_location()
        return self.get_drag_code(result)

    def on_drop_files(self, x, y, filenames):
        self.refresh_drop_location()
        row, col = self.drop_location
        self.drop_location = None
        pass

    def refresh_drop_location(self):
        if self.drop_location is None:
            return
        row, col = self.drop_location
        if col is not None:
            if row == self.GetNumberRows():
                rect = self.CellToRect(row - 1, col)
                rect.Y += rect.Height - int(self.DROP_HEIGHT / 2) - 1
                rect.Height = int(self.DROP_HEIGHT / 2) + 1
            else:
                rect = self.CellToRect(row, col)
                rect.Y -= int(self.DROP_HEIGHT / 2) + 1
                rect.Height = self.DROP_HEIGHT + 1
            rect.X, rect.Y = self.CalcScrolledPosition(rect.X, rect.Y)
            self.GetGridWindow().RefreshRect(rect, eraseBackground=False)

    ###############################
    #
    # Drag control
    #
    ###############################

    def on_grid_begin_drag(self, event):
        from cellprofiler_core.utilities.pathname import url2pathname

        selections = self.get_selected_cells()
        if len(selections) > 0:
            filenames = [
                url2pathname(self.table.get_url(row, col)) for row, col in selections
            ]
            data_object = wx.FileDataObject()
            for filename in filenames:
                data_object.AddFile(filename)
            drop_source = wx.DropSource(self)
            drop_source.SetData(data_object)
            result = drop_source.DoDragDrop(wx.Drag_AllowMove)
            if result == wx.DragMove:
                self.remove_selection()

    def remove_selection(self):
        if self.table.controller is None:
            return
        to_remove = [[] for col in range(self.GetNumberCols())]
        for row, col in self.get_selected_cells():
            to_remove[col].append(row)
        for rows in to_remove:
            if len(rows) > 0:
                image_sets = self.table

    def recompute(self):
        """Recompute the layout after a change to the image set"""

        n_rows_added, n_columns_added = self.table.recompute()

        need_column_layout = False
        if n_columns_added < 0:
            tm = wx.grid.GridTableMessage(
                self.table, wx.grid.GRIDTABLE_NOTIFY_COLS_DELETED, 0, -n_columns_added
            )
            self.ProcessTableMessage(tm)
            need_column_layout = True
        elif n_columns_added > 0:
            tm = wx.grid.GridTableMessage(
                self.table, wx.grid.GRIDTABLE_NOTIFY_COLS_INSERTED, 0, n_columns_added
            )
            self.ProcessTableMessage(tm)
            need_column_layout = True
        if n_rows_added < 0:
            tm = wx.grid.GridTableMessage(
                self.table, wx.grid.GRIDTABLE_NOTIFY_ROWS_DELETED, 0, -n_rows_added
            )
            self.ProcessTableMessage(tm)
        elif n_rows_added > 0:
            tm = wx.grid.GridTableMessage(
                self.table, wx.grid.GRIDTABLE_NOTIFY_ROWS_INSERTED, 0, n_rows_added
            )
            self.ProcessTableMessage(tm)

        only = self.table.GetNumberCols() == 1
        for i in range(self.table.GetNumberCols()):
            last = i == self.table.GetNumberCols() - 1
            label_size = self.GetGridColLabelWindow().GetTextExtent(
                self.table.GetColLabelValue(i)
            )
            min_width = self.col_label_renderer.minimum_width(label_size, last, only)
            self.SetColMinimalWidth(i, min_width)
            if need_column_layout:
                if self.table.GetNumberRows() > 0:
                    first_width, _ = self.GetGridWindow().GetTextExtent(
                        str(self.table.GetValue(0, i))
                    )
                    first_width += self.cell_renderer.padding * 4
                    width = max(first_width, min_width)
                else:
                    width = min_width
                self.SetColSize(i, width)


class EllipsisGridCellRenderer(wx.grid.GridCellRenderer):
    """Renders a grid cell with ellipsis in the middle if can't fit

    """

    def __init__(self, padding=2):
        super(self.__class__, self).__init__()
        self.padding = 2
        self.renderer = wx.RendererNative.Get()

    def Draw(self, grid, attr, dc, rect, row, col, is_selected):
        assert isinstance(dc, wx.DC)
        assert isinstance(attr, wx.grid.GridCellAttr)
        assert isinstance(rect, wx.Rect)
        assert isinstance(grid, ImageSetCtrl)
        s = str(grid.GetTable().GetValue(row, col))
        old_font = dc.GetFont()
        old_brush = dc.GetBrush()
        old_pen = dc.GetPen()
        old_foreground_color = dc.GetTextForeground()
        old_mode = dc.GetBackgroundMode()
        try:
            if attr.HasFont():
                dc.SetFont(attr.GetFont())
            else:
                dc.SetFont(grid.GetGridWindow().GetFont())
            if attr.HasBackgroundColour():
                brush = wx.Brush(attr.GetBackgroundColour())
                dc.SetBrush(brush)
            else:
                brush = wx.Brush(grid.GetGridWindow().BackgroundColour)
                dc.SetBrush(brush)
            dc.SetPen(wx.TRANSPARENT_PEN)
            dc.DrawRectangle(
                rect.GetX(), rect.GetY(), rect.GetWidth(), rect.GetHeight()
            )
            flags = 0
            if grid.IsInSelection(row, col):
                flags += wx.CONTROL_SELECTED
            if wx.Window.FindFocus() in (
                grid,
                grid.GetGridWindow(),
                grid.GetGridColLabelWindow(),
                grid.GetGridRowLabelWindow(),
                grid.GetGridCornerLabelWindow(),
            ):
                flags += wx.CONTROL_FOCUSED
            cellprofiler.gui.draw_item_selection_rect(
                grid.GetGridWindow(), dc, rect, flags
            )
            dc.SetBackgroundMode(wx.BRUSHSTYLE_TRANSPARENT)
            if attr.HasTextColour():
                dc.SetTextForeground(attr.GetTextColour())
            else:
                dc.SetTextForeground(grid.GetGridWindow().ForegroundColour)

            text_width = rect.GetWidth() - self.padding * 2
            if dc.GetTextExtent(s)[0] <= text_width:
                sdisplay = s
            else:
                increment = len(s)
                field_size = 0
                sdisplay = "..."
                while True:
                    increment = int((increment + 1) / 2)
                    test_size = field_size + increment
                    if len(s) > test_size:
                        half = int(test_size / 2)
                        stest = s[:half] + "..." + s[-half:]
                    else:
                        stest = s
                    width, _ = dc.GetTextExtent(stest)
                    if width <= text_width:
                        sdisplay = stest
                        field_size += increment
                    if increment == 1:
                        break

            dc.DrawText(sdisplay, rect.GetX() + self.padding, rect.GetY())
            if grid.drop_location is not None:
                drop_row, drop_col = grid.drop_location
                for do_once in range(1):
                    if drop_row == row and drop_col == col:
                        y = rect.GetY()
                    elif drop_row == row + 1 and drop_col == col:
                        y = rect.GetY() + rect.GetHeight() - 1
                    else:
                        break

                    dc.SetPen(wx.TRANSPARENT_PEN)
                    dc.SetBrush(wx.BLACK_BRUSH)
                    half_height = int(grid.DROP_HEIGHT) / 2
                    dc.DrawPolygon(
                        [
                            (rect.GetX(), y - half_height - 1),
                            (rect.GetX() + half_height + 1, y),
                            (rect.GetX() + rect.GetWidth() - half_height - 1, y),
                            (rect.GetX() + rect.GetWidth(), y - half_height - 1),
                            (rect.GetX() + rect.GetWidth(), y + half_height + 1),
                            (rect.GetX() + rect.GetWidth() - half_height, y + 1),
                            (rect.GetX() + half_height, y + 1),
                            (rect.GetX(), y + half_height + 1),
                            (rect.GetX(), y - half_height),
                        ]
                    )
        finally:
            dc.SetFont(old_font)
            dc.SetBrush(old_brush)
            dc.SetPen(old_pen)
            dc.SetTextForeground(old_foreground_color)
            dc.SetBackgroundMode(old_mode)

    def GetBestSize(self, grid, attr, dc, row, col):
        assert isinstance(dc, wx.DC)
        assert isinstance(grid, wx.grid.Grid)
        s = str(grid.GetTable().GetValue(row, col))
        width, height = grid.GetGridWindow().GetTextExtent(s)
        return wx.Size(width + 2 * self.padding, height)


class ColLabelRenderer(cpglr.ColLabelRenderer):
    """Renders the appearance of a column label

    A column label has the label text, an icon button for setting the
    column type and an icon button for activating the filter.
    """

    def __init__(self, read_only):
        super(self.__class__, self).__init__()
        #
        # Room reserved for button graphics
        self.icon_padding = 4
        self.icon_size = 16
        self.gap_size = 2
        self.read_only = read_only
        self.renderer = wx.RendererNative.Get()
        assert isinstance(self.renderer, wx.RendererNative)

    def Draw(self, grid, dc, rect, col):
        assert isinstance(grid, ImageSetCtrl)
        assert isinstance(dc, wx.DC)
        window = grid.GetGridColLabelWindow()
        bitmap = wx.Bitmap(width=rect.Width, height=rect.Height)
        last = col == grid.GetTable().GetNumberCols() - 1
        only = grid.GetTable().GetNumberCols() == 1
        try:
            mdc = wx.MemoryDC(bitmap)
            mdc.SetFont(window.Font)
            mdc.SetTextForeground(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT))
            mdc.SetBackground(wx.Brush(window.BackgroundColour))
            mdc.Clear()
            b_rect = wx.Rect(0, 0, rect.width, rect.height)
            self.DrawBorder(grid, mdc, b_rect)
            label = grid.GetTable().GetColLabelValue(col)
            label_size = mdc.GetTextExtent(label)
            draw_simple = False
            if (
                grid.GetTable().display_mode == DISPLAY_MODE_SIMPLE
                and not self.read_only
            ):
                column = grid.GetTable().columns[col]
                m = grid.GetTable().measurements
                channel_descriptors = m.get_channel_descriptors()
                if column.channel in channel_descriptors:
                    draw_simple = True
                    channel_type = channel_descriptors[column.channel]
            if draw_simple:
                x, y, width, height = self.label_rect(b_rect, label_size, last, only)
                if grid.editor_column is None:
                    for line_number, line in enumerate(label.split("\n")):
                        mdc.DrawText(line, x, y + label_size[1] * line_number)

                pb_col, pb_hit_code, pb_pressed = grid.pressed_button
                if channel_type == CT_GRAYSCALE:
                    image = grid.monochrome_channel_image
                elif channel_type == CT_COLOR:
                    image = grid.color_channel_image
                elif channel_type == CT_MASK:
                    image = grid.mask_image
                elif channel_type == CT_OBJECTS:
                    image = grid.objects_image
                else:
                    image = grid.illumination_function_image
                icon_info = [
                    (grid.HIT_CHANNEL_TYPE_BUTTON, image),
                    (grid.HIT_FILTER_BUTTON, grid.filter_image),
                ]
                if last:
                    icon_info.append((grid.HIT_PLUS, "+"))
                if not only:
                    icon_info.append((grid.HIT_MINUS, "-"))
                for idx, (hit_code, image) in enumerate(icon_info):
                    flags = 0
                    if (
                        (pb_col is None or pb_col == col)
                        and hit_code == pb_hit_code
                        and pb_pressed
                    ):
                        flags = wx.CONTROL_PRESSED
                    icon_rect = self.get_icon_rect(b_rect, label_size, idx, last, only)
                    self.draw_button(window, mdc, icon_rect, image, flags)
            else:
                x = (rect.width - label_size[0] + 1) / 2
                y = self.icon_padding + self.gap_size
                for line_number, line in enumerate(label.split("\n")):
                    mdc.DrawText(line, x, y + label_size[1] * line_number)
            dc.Blit(rect.X, rect.Y, rect.width, rect.height, mdc, 0, 0)
        finally:
            mdc.SelectObject(wx.NullBitmap)
            bitmap.Destroy()

    def channel_type_icon_rect(self, rect, label_size, last, only):
        """The position of the channel type icon

        rect - the drawing rectangle for the column header

        label_size - the width & height needed to draw the label

        last - column is the last column in the grid

        only - column is the only column in the grid
        """
        return self.get_icon_rect(rect, label_size, 0, last, only)

    def filter_icon_rect(self, rect, label_size, last, only):
        """The position of the filter icon

        rect - the drawing rectangle for the column header

        label_size - the width & height needed to draw the label
        """
        return self.get_icon_rect(rect, label_size, 1, last, only)

    def add_button_rect(self, rect, label_size, only):
        """The position of the add button in the last column"""
        return self.get_icon_rect(rect, label_size, 2, True, only)

    def remove_icon_rect(self, rect, label_size, last):
        """The position of the remove button"""
        return self.get_icon_rect(rect, label_size, 3 if last else 2, last, False)

    def get_icon_rect(self, rect, label_size, index, last, only):
        """Return the icon position of the index'th icon

        rect - rectangle for the column
        label_size - size of the label text
        index - icon index
        last - True if the last column (needs +)
        only - True if this is the only column (no -)
        """
        label_width, label_height = label_size
        n_icons = 3 if last else 2
        if not only:
            n_icons += 1
        n_to_right = n_icons - index - 1
        x = (
            rect.x
            + rect.width
            - self.icon_padding * (2 * n_to_right + 1)
            - self.icon_size * (n_to_right + 1)
            - self.gap_size * (n_to_right + 1)
        )
        y = self.icon_padding + self.gap_size
        return wx.Rect(x, y, self.icon_size, max(self.icon_size, label_height))

    def get_edit_rect(self, rect, last, only):
        """The rectangle for the column label editor"""
        x = rect.X + self.gap_size
        y = rect.Y + self.gap_size
        height = rect.Height - self.gap_size * 2
        first_icon_rect = self.get_icon_rect(rect, wx.Size(0, height), 0, last, only)
        width = first_icon_rect.GetX() - self.gap_size - x
        return wx.Rect(x, y, width, height)

    def draw_button(self, window, dc, rect, bitmap, flags):
        """Draw a button with the given flags"""
        assert isinstance(dc, wx.DC)
        x, y, width, height = rect
        rect = wx.Rect(
            x - self.icon_padding,
            y - self.icon_padding,
            width + self.icon_padding * 2,
            height + self.icon_padding * 2,
        )
        self.renderer.DrawPushButton(window, dc, rect, flags)
        if isinstance(bitmap, wx.Bitmap):
            dc.DrawBitmap(bitmap, x, y, useMask=True)
        elif isinstance(bitmap, str):
            dc.SetFont(window.Font)
            dc.SetBackgroundMode(wx.PENSTYLE_TRANSPARENT)
            width, height = dc.GetTextExtent(bitmap)
            x = rect.GetX() + (rect.GetWidth() - width) / 2
            y = rect.GetY() + (rect.GetHeight() - height) / 2
            dc.DrawText(bitmap, x, y)
            dc.SetFont(wx.NullFont)

    def label_rect(self, rect, label_size, last, only):
        """The position of the label

        rect - the drawing rectangle for the column header

        label_size - the width & height needed to draw the label
        """
        label_width, label_height = tuple(label_size)
        if self.read_only:
            available_width = rect.width
        else:
            first_button_rect = self.get_icon_rect(rect, label_size, 0, last, only)
            available_width = first_button_rect.GetX() - self.gap_size - rect.X
        x = rect.x + self.gap_size + (available_width - label_width) / 2
        y = self.icon_padding + self.gap_size
        return wx.Rect(x, y, label_width, max(self.icon_size, label_height))

    def minimum_width(self, label_size, last, only):
        if self.read_only:
            n_icons = 0
        else:
            n_icons = 2
            if last:
                n_icons += 1
            if not only:
                n_icons += 1
        return (
            label_size[0]
            + self.gap_size * (n_icons + 4)
            + self.icon_padding * n_icons * 2
            + self.icon_size * n_icons
        )


class ColumnNameValidator(wx.Validator):
    def __init__(self, table):
        super(self.__class__, self).__init__()
        self.table = table
        self.Bind(wx.EVT_CHAR, self.on_char)

    def Clone(self):
        return ColumnNameValidator(self.table)

    def Validate(self, win):
        value = self.GetWindow().GetValue()
        if any([value.lower() == column.channel for column in self.table.columns]):
            return False
        pattern = "^[A-Za-z_]\\w*$"
        return re.match(pattern, value) is not None

    def on_char(self, event):
        ctrl = self.GetWindow()
        assert isinstance(ctrl, wx.TextCtrl)
        key = event.GetKeyCode()
        if key < wx.WXK_SPACE or key == wx.WXK_DELETE or key > 255:
            event.Skip()
            return

        assert isinstance(ctrl, wx.TextCtrl)
        c = chr(key)
        if c.isalpha() or c == "_":
            event.Skip()
            return
        if ctrl.GetInsertionPoint() > 0 and c.isdigit():
            event.Skip()
            return
        if not wx.Validator.IsSilent():
            wx.Bell()


class ImageSetCtrlDropTarget(wx.FileDropTarget):
    def __init__(self, grid):
        wx.FileDropTarget.__init__(self)
        self.grid = grid

    def OnEnter(self, x, y, result):
        return self.grid.on_drop_enter(x, y, result)

    def OnLeave(self):
        return self.grid.on_drop_leave()

    def OnDragOver(self, x, y, result):
        return self.grid.on_drag_over(x, y, result)

    def OnDropFiles(self, x, y, filenames):
        return self.grid.on_drop_files(x, y, filenames)


class ImageSetController:
    """Modifies the image set according to GUI notifications"""

    def __init__(self):
        pass

    @staticmethod
    def can_edit():
        """Return True if the image set is editable"""
        return False

    @staticmethod
    def can_undo():
        """Return True if the image set can undo something"""
        return False

    @staticmethod
    def can_redo():
        """Return True if the image set can redo something"""
        return False

    def change_channel_name(self, old_name, new_name):
        """Change the name of a channel

        old_name - current name of the channel

        new_name - new name for the channel
        """
        raise NotImplementedError("Changing channel name is unsupported")

    def change_channel_type(self, name, channel_type):
        """Change the channel type

        name - channel name

        channel_type - channel descriptor type
        """
        raise NotImplementedError("Changing channel type is unsupported")

    def append_channel(self):
        """Append a new channel to the list of channels"""
        raise NotImplementedError("Appending channels is not supported")

    def remove_channel(self, channel_name):
        """Remove a channel from the list of channels

        channel_name - name of the channel
        """
        raise NotImplementedError("Removing channels is not supported")

    def add_files(self, file_names, channel_name, image_set):
        """Add files to a channel

        file_names - paths to the files

        channel_name - the name of the channel

        image_set - insert beginning at this image set
        """
        raise NotImplementedError("Adding files is not supported")

    def remove_image_sets(self, image_set_numbers):
        """Remove image sets from the image set list

        image_set_numbers - remove these image sets
        """
        raise NotImplementedError("Removing image sets is not supported")

    def remove_files(self, channel_name, image_set_numbers):
        """Remove files from a channel, shifting the remaining ones to fill"""
        raise NotImplementedError("Removing files from a channel is not supported")

    def undo(self):
        """Undo the last operation"""
        raise NotImplementedError("Undo not supported")

    def redo(self):
        """Redo the last operation"""
        raise NotImplementedError("Redo not supported")


class FilterPanelDlg(wx.Dialog):
    """A dialog box containing a filter panel"""

    def __init__(self, parent, channel_name, channel_names, function_list):
        """Initialize the dialog

        parent - parent window to this one

        channel_name - current channel name for the filter

        channel_names - list of all channel names

        function_dict - a list of triplets, each of which is converted
                        into a button. The triplet is (<function-name>,
                        <function>, <function-help>)

                        The dialog creates one button per entry and calls
                        the function when the button is pressed.
                        The function's signature is:
                        fn(channel_name, fn_filter) where
                        channel_name is the currently selected channel
                        fn_filter is a function that takes a URL
                        and returns True if the file passes the filter and
                        False if it doesn't.
                        """
        super(self.__class__, self).__init__(parent, size=(640, 480))
        self.SetTitle("Select images using a filter")

        self.filter_setting = Filter(
            "Filter",
            predicates=[FilePredicate(), DirectoryPredicate(), ExtensionPredicate()],
            value='and (file does contain "")',
        )

        self.make_ui(channel_name, channel_names, function_list)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.Destroy()

    def fn_filter(self, url):
        """A filter function that applies the current filter to a URL"""
        modpath = url_to_modpath(url)
        return self.filter_setting.evaluate(
            (FileCollectionDisplay.NODE_IMAGE_PLANE, modpath, None,)
        )

    def on_value_change(self, setting, panel, new_text, event, timeout):
        self.filter_setting.set_value_text(new_text)
        self.filter_panel_controller.update()
        self.Fit()

    def make_ui(self, channel_name, channel_names, function_list):
        """Construct the user interface

        channel_name - the channel that will be filtered

        channel_names - the list of channels that the user can switch to

        function_dict - dictionary of button name to function to run.
        """
        from cellprofiler.gui.module_view._filter_panel_controller import (
            FilterPanelController,
        )

        self.SetSizer(wx.BoxSizer(wx.VERTICAL))
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.Sizer.Add(sizer, 1, wx.EXPAND | wx.ALL, 20)
        sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(sub_sizer, 0, wx.EXPAND)
        sub_sizer.Add(
            wx.StaticText(self, label="Channel:"), 0, wx.ALIGN_LEFT | wx.ALIGN_TOP
        )
        self.channel_choice = wx.Choice(self, choices=channel_names)
        self.channel_choice.SetStringSelection(channel_name)
        sub_sizer.Add(self.channel_choice, 0, wx.EXPAND)

        self.filter_panel_controller = FilterPanelController(
            self, self.filter_setting, self.on_value_change
        )
        sizer.AddSpacer(5)
        sizer.Add(self.filter_panel_controller.panel, 1, wx.EXPAND)
        sizer.AddSpacer(5)
        sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.Sizer.Add(sub_sizer, 0, wx.EXPAND)

        for button_text, fn, help_text in function_list:
            button = wx.Button(self, label=button_text)
            button.SetToolTip(help_text)

            def on_button(event, fn=fn):
                current_channel = self.channel_choice.GetStringSelection()
                fn(current_channel, self.fn_filter)

            button.Bind(wx.EVT_BUTTON, on_button)
            sub_sizer.Add(button, 0, wx.EXPAND | wx.ALL, 2)
        sub_sizer.Add(wx.Button(self, wx.ID_OK, label="Exit"), 0, wx.ALIGN_RIGHT)
        self.Fit()
