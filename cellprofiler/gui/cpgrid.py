"""cpgrid.py - wx.grid helpers for cellprofiler
"""

import cellprofiler.gui
import wx
import wx.grid

BU_NORMAL = "normal"
BU_PRESSED = "pressed"


class GridButtonRenderer(wx.grid.PyGridCellRenderer):
    """Render a cell as a button

    The value of a cell should be organized like this: "key:state"
    where "key" is the key for the image to paint and "state" is the
    state of the button: "normal" or "pressed"
    GridButtonRenderer takes a dictionary organized as key: bitmap. This
    dictionary holds the images to render per key.
    """

    def __init__(self, bitmap_dictionary, bevel_width=2):
        super(GridButtonRenderer, self).__init__()
        self.__bitmap_dictionary = bitmap_dictionary
        self.__bevel_width = bevel_width

    def Draw(self, grid, attr, dc, rect, row, col, isSelected):
        dc.SetClippingRect(rect)
        dc.Clear()
        dc.DestroyClippingRegion()
        bitmap = self.get_bitmap(grid, attr, dc, row, col)
        state = self.get_state(grid, row, col)
        if state is not None:
            bv = ((state == BU_NORMAL and cellprofiler.gui.BV_UP) or
                  cellprofiler.gui.BV_DOWN)
            rect = cellprofiler.gui.draw_bevel(dc, rect, self.__bevel_width, bv)
        else:
            bw = self.__bevel_width
            rect = wx.Rect(rect.Left + bw, rect.Top + bw, rect.width - 2 * bw, rect.height - 2 * bw)
        dc.SetClippingRect(rect)
        if bitmap:
            dc.DrawBitmap(bitmap, rect.Left, rect.Top, True)
        dc.DestroyClippingRegion()

    def GetBestSize(self, grid, attr, dc, row, col):
        """Return the size of the cell's button"""
        bitmap = self.get_bitmap(grid, attr, dc, row, col)
        if bitmap:
            size = bitmap.Size
        else:
            size = wx.Size(0, 0)
        return wx.Size(size[0] + 2 * self.__bevel_width,
                       size[1] + 2 * self.__bevel_width)

    def Clone(self):
        return GridButtonRenderer(self.__bitmap_dictionary, self.__bevel_width)

    def get_bitmap(self, grid, attr, dc, row, col):
        """Get a cell's bitmap

        grid - the parent wx.grid
        attr - an instance of wx.grid.GriddCellAttr which provides rendering info
        dc   - the device context to be used for printing
        row,col - the coordinates of the cell to be rendered
        """
        value = grid.GetCellValue(row, col)
        key = value.split(':')[0]
        if self.__bitmap_dictionary.has_key(key):
            bitmap = self.__bitmap_dictionary[key]
            return bitmap
        return None

    @staticmethod
    def get_state(grid, row, col):
        """Get a cell's press-state

        grid - the grid control
        row,col - the row and column of the cell
        """
        value = grid.GetCellValue(row, col)
        values = value.split(':')
        if len(values) < 2:
            return None
        return values[1]

    @staticmethod
    def set_cell_value(grid, row, col, key, state):
        """Set a cell's value in a grid

        grid    - the grid control
        row,col - the cell coordinates
        key     - the keyword for the bitmap
        state   - either BU_NORMAL or BU_PRESSED
        """
        value = "%s:%s" % (key, state)
        grid.SetCellValue(row, col, value)
        grid.ForceRefresh()

    def set_cell_state(self, grid, row, col, state):
        key = grid.GetCellValue(row, col).split(':')[0]
        self.set_cell_value(grid, row, col, key, state)


EVT_GRID_BUTTON_TYPE = wx.NewEventType()
EVT_GRID_BUTTON = wx.PyEventBinder(EVT_GRID_BUTTON_TYPE)


class GridButtonClickedEvent(wx.PyCommandEvent):
    """Indicates that a grid button has been clicked"""

    def __init__(self, grid, row, col):
        super(GridButtonClickedEvent, self).__init__(EVT_GRID_BUTTON_TYPE,
                                                     grid.Id)
        self.SetEventObject(grid)
        self.SetEventType(EVT_GRID_BUTTON_TYPE)
        self.__row = row
        self.__col = col

    def get_col(self):
        """Column of clicked cell"""
        return self.__col

    def get_row(self):
        """Row of clicked cell"""
        return self.__row

    col = property(get_col)
    row = property(get_row)


def hook_grid_button_column(grid, col, bitmap_dictionary, bevel_width=2,
                            hook_events=True):
    """Attach hooks to a grid to make a column display grid buttons

    grid - the grid in question
    col  - the index of the column to modify
    bitmap_dictionary - a dictionary of bitmaps suitable for GridButtonRenderer
    """
    renderer = GridButtonRenderer(bitmap_dictionary, bevel_width)
    ui_dictionary = {"selected_row": None}
    event_handler = wx.EvtHandler()
    width = 0
    for bitmap in bitmap_dictionary.values():
        width = max(bitmap.Width, width)
    width += bevel_width * 2
    grid.SetColSize(col, width)

    def on_left_down(event):
        x = event.GetX()
        y = event.GetY()
        coords = grid.XYToCell(x, y)
        if coords and coords.Col == col:
            row = coords.Row
            if renderer.get_state(grid, row, col) == BU_NORMAL:
                ui_dictionary["selected_row"] = row
                renderer.set_cell_state(grid, row, col, BU_PRESSED)
                grid.GridWindow.CaptureMouse()
                event.Skip()
        else:
            if event_handler.NextHandler:
                event_handler.NextHandler.ProcessEvent(event)

    def on_mouse_move(event):
        if (ui_dictionary["selected_row"] is not None and
                grid.GridWindow.HasCapture()):
            x = event.GetX()
            y = event.GetY()
            coords = grid.XYToCell(x, y)
            row = ui_dictionary["selected_row"]
            selection_state = BU_NORMAL
            if coords and coords.Col == col and coords.Row == row:
                selection_state = BU_PRESSED
            if renderer.get_state(grid, row, col) != selection_state:
                renderer.set_cell_state(grid, row, col, selection_state)
        if event_handler.NextHandler:
            event_handler.NextHandler.ProcessEvent(event)

    def on_capture_lost(event):
        if ui_dictionary["selected_row"] is not None:
            renderer.set_cell_state(grid, ui_dictionary["selected_row"], col,
                                    BU_NORMAL)
            ui_dictionary["selected_row"] = None
        else:
            if event_handler.NextHandler:
                event_handler.NextHandler.ProcessEvent(event)

    def on_left_up(event):
        if (ui_dictionary["selected_row"] is not None and
                grid.GridWindow.HasCapture()):
            row = ui_dictionary["selected_row"]
            if renderer.get_state(grid, row, col) == BU_PRESSED:
                renderer.set_cell_state(grid, row, col, BU_NORMAL)
                grid.AddPendingEvent(GridButtonClickedEvent(grid, row, col))
            ui_dictionary["selected_row"] = None
            grid.GridWindow.ReleaseMouse()
            event.Skip()
        else:
            if event_handler.NextHandler:
                event_handler.NextHandler.ProcessEvent(event)

    col_attr = wx.grid.GridCellAttr()
    col_attr.SetReadOnly(True)
    col_attr.SetRenderer(renderer)
    grid.SetColAttr(col, col_attr)
    if hook_events:
        grid.GridWindow.PushEventHandler(event_handler)
        event_handler.Bind(wx.EVT_LEFT_DOWN, on_left_down, grid.GridWindow)
        event_handler.Bind(wx.EVT_LEFT_UP, on_left_up, grid.GridWindow)
        event_handler.Bind(wx.EVT_MOTION, on_mouse_move, grid.GridWindow)
        event_handler.Bind(wx.EVT_MOUSE_CAPTURE_LOST, on_capture_lost, grid.GridWindow)
    return renderer, width
