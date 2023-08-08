import wx
import wx.grid
from cellprofiler_core.preferences import get_error_color
from cellprofiler_core.setting import Table


class TableController(wx.grid.GridTableBase):
    DEFAULT_ATTR = wx.grid.GridCellAttr()
    ERROR_ATTR = wx.grid.GridCellAttr()
    ERROR_ATTR.SetTextColour(get_error_color())

    def __init__(self, v):
        super(self.__class__, self).__init__()
        assert isinstance(v, Table)
        self.v = v
        self.column_size = [v.max_field_size] * len(v.column_names)

    def bind_to_grid(self, grid):
        """Bind to intercept events on the grid

        Binds on_mouse_motion and on_column_resize in order to do tooltips.
        Sets up editing / auto size and other to customize for table type.
        """
        self.grid = grid
        grid.AutoSize()
        grid.EnableEditing(False)
        grid.SetDefaultCellOverflow(False)
        if self.v.corner_button is None:
            grid.fn_clicked = None
        else:
            fn_clicked = self.v.corner_button["fn_clicked"]

            def on_corner_button_clicked(e):
                fn_clicked(e)
                self.update_grid()
                grid.ForceRefresh()
                grid.Parent.Layout()

            grid.fn_clicked = on_corner_button_clicked
            grid.label = self.v.corner_button.get("label", "Update")
            grid.tooltip = self.v.corner_button.get("tooltip", "")
        #
        # Below largely taken from
        # http://wiki.wxpython.org/wxGrid%20ToolTips
        #
        self.last_pos = (None, None)
        grid.GetGridWindow().Bind(wx.EVT_MOTION, self.on_mouse_motion)
        grid.Bind(wx.grid.EVT_GRID_COL_SIZE, self.on_column_resize)

    def update_grid(self):
        """Update the grid after the table data has changed"""
        need_column_layout = False
        grid = self.grid
        v = self.v
        if len(v.column_names) < grid.GetNumberCols():
            tm = wx.grid.GridTableMessage(
                grid.Table,
                wx.grid.GRIDTABLE_NOTIFY_COLS_DELETED,
                0,
                grid.GetNumberCols() - len(v.column_names),
            )
            grid.ProcessTableMessage(tm)
            need_column_layout = True
        elif grid.GetNumberCols() < len(v.column_names):
            tm = wx.grid.GridTableMessage(
                grid.Table,
                wx.grid.GRIDTABLE_NOTIFY_COLS_INSERTED,
                0,
                len(v.column_names) - grid.GetNumberCols(),
            )
            grid.ProcessTableMessage(tm)
            need_column_layout = True
        if len(v.data) < grid.GetNumberRows():
            tm = wx.grid.GridTableMessage(
                grid.Table,
                wx.grid.GRIDTABLE_NOTIFY_ROWS_DELETED,
                0,
                grid.GetNumberRows() - len(v.data),
            )
            grid.ProcessTableMessage(tm)
        elif grid.GetNumberRows() < len(v.data):
            tm = wx.grid.GridTableMessage(
                grid.Table,
                wx.grid.GRIDTABLE_NOTIFY_ROWS_INSERTED,
                0,
                len(v.data) - grid.GetNumberRows(),
            )
            grid.ProcessTableMessage(tm)
        if need_column_layout:
            grid.AutoSizeColumns()

    def on_mouse_motion(self, event):
        x, y = self.grid.CalcUnscrolledPosition(event.GetPosition())
        row = self.grid.YToRow(y)
        col = self.grid.XToCol(x)
        this_pos = (row, col)
        if (
            this_pos != self.last_pos
            and row >= 0
            and col >= 0
            and row < len(self.v.data)
            and col < len(self.v.data[row])
        ):
            self.last_pos = this_pos
            s = self.v.data[row][col]
            if s is None:
                s = ""
            elif not isinstance(s, str):
                s = str(s)
            self.grid.GetGridWindow().SetToolTip(s)
        event.Skip()

    def on_column_resize(self, event):
        grid = self.grid
        col = event.GetRowOrCol()
        width = grid.GetColSize(col)
        table = grid.GetTable()
        self.column_size[col] = int(width * 1.1) / grid.CharWidth
        tm = wx.grid.GridTableMessage(self, wx.grid.GRIDTABLE_REQUEST_VIEW_GET_VALUES)
        grid.ProcessTableMessage(tm)
        grid.ForceRefresh()

    def GetAttr(self, row, col, kind):
        attrs = self.v.get_cell_attributes(row, self.v.column_names[col])
        attr = self.DEFAULT_ATTR
        if attrs is not None and self.v.ATTR_ERROR in attrs:
            attr = self.ERROR_ATTR
        attr.IncRef()  # OH so bogus, don't refcount = bus error
        return attr

    def CanHaveAttributes(self):
        return True

    def GetNumberRows(self):
        return len(self.v.data)

    def GetNumberCols(self):
        return len(self.v.column_names)

    def IsEmptyCell(self, row, col):
        return (
            len(self.v.data) <= row
            or len(self.v.data[row]) <= col
            or self.v.data[row][col] is None
        )

    def GetValue(self, row, col):
        if self.IsEmptyCell(row, col):
            return None
        s = str(self.v.data[row][col])
        if len(self.column_size) <= col:
            self.column_size += [self.v.max_field_size] * (
                col - len(self.column_size) + 1
            )
        field_size = self.column_size[col]
        if len(s) > field_size:
            half = int(field_size - 3) // 2
            s = s[:half] + "..." + s[-half:]
        return s

    def GetRowLabelValue(self, row):
        attrs = self.v.get_row_attributes(row)
        if attrs is not None and self.v.ATTR_ERROR in attrs:
            return "%d: Error" % (row + 1)
        return str(row + 1)

    def GetColLabelValue(self, col):
        return self.v.column_names[col]

    def AppendCols(self, numCols=1):
        return True

    def AppendRows(self, numRows=1, updateLabels=True):
        return True

    def InsertCols(self, pos=0, numCols=1, updateLabels=True):
        return True

    def InsertRows(self, pos=0, numRows=1, updateLabels=True):
        return True

    def DeleteCols(self, pos=0, numCols=1, updateLabels=True):
        return True

    def DeleteRows(self, pos=0, numRows=1, updateLabels=True):
        return True
