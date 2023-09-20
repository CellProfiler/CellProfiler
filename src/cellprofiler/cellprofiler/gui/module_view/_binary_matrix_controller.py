import wx
from cellprofiler_core.preferences import get_background_color
from cellprofiler_core.setting import BinaryMatrix

from ._module_view import ModuleView
from ..utilities.module_view import edit_control_name


class BinaryMatrixController:
    """A controller for the BinaryMatrix setting
    """

    def __init__(self, module_view, v):
        """Initialize the controller

        module_view - the module_view that contains the controller's panel

        v - the setting
        """
        self.module_view = module_view
        self.setting = v
        self.panel = wx.Panel(module_view.module_panel, name=edit_control_name(v))
        self.panel.controller = self
        self.panel.Sizer = wx.BoxSizer(wx.VERTICAL)
        self.matrix_ctrl = wx.Panel(self.panel)
        self.panel.Sizer.Add(self.matrix_ctrl, 0, wx.EXPAND | wx.ALL, 10)
        self.matrix_ctrl.SetMinSize(wx.Size(50, 50))
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.panel.Sizer.Add(sizer, 0, wx.EXPAND | wx.ALL, 2)
        sizer.Add(
            wx.StaticText(self.panel, label="Width:"),
            0,
            wx.ALIGN_RIGHT | wx.ALIGN_CENTER,
        )
        sizer.AddSpacer(1)
        self.width_ctrl = wx.SpinCtrl(self.panel)
        self.width_ctrl.SetRange(1, 100)
        sizer.Add(self.width_ctrl, 1, wx.EXPAND)
        sizer.AddSpacer(4)
        sizer.Add(
            wx.StaticText(self.panel, label="Height:"),
            0,
            wx.ALIGN_RIGHT | wx.ALIGN_CENTER,
        )
        sizer.AddSpacer(1)
        self.height_ctrl = wx.SpinCtrl(self.panel)
        self.height_ctrl.SetRange(1, 100)
        sizer.Add(self.height_ctrl, 1, wx.EXPAND)

        self.matrix_ctrl.Bind(wx.EVT_PAINT, self.on_matrix_ctrl_paint)
        self.matrix_ctrl.Bind(wx.EVT_LEFT_UP, self.on_matrix_ctrl_clicked)
        self.width_ctrl.Bind(wx.EVT_SPINCTRL, self.on_width_changed)
        self.height_ctrl.Bind(wx.EVT_SPINCTRL, self.on_height_changed)
        self.update()

    def update(self):
        h, w = self.setting.get_size()
        hh, ww = [(x - 1) / 2 for x in (h, w)]
        if self.height_ctrl.GetValue() != hh:
            self.height_ctrl.SetValue(hh)
        if self.width_ctrl.GetValue() != ww:
            self.width_ctrl.SetValue(ww)
        r = self.get_matrix_element_rect(h - 1, w - 1)
        b = wx.SystemSettings.GetMetric(wx.SYS_EDGE_X)
        self.matrix_ctrl.SetMinSize(wx.Size(r.GetRight() + b, r.GetBottom() + b))
        self.matrix_ctrl.Refresh(eraseBackground=False)

    def on_matrix_ctrl_clicked(self, event):
        assert isinstance(event, wx.MouseEvent)
        i, j = self.hit_test(event.GetX(), event.GetY())
        if i is not None:
            matrix = self.setting.get_matrix()
            matrix[i][j] = not matrix[i][j]
            value = BinaryMatrix.to_value(matrix)
            self.module_view.on_value_change(self.setting, self.panel, value, event)

    def on_matrix_ctrl_paint(self, event):
        paint_dc = wx.BufferedPaintDC(self.matrix_ctrl)
        matrix = self.setting.get_matrix()
        h = len(matrix)
        w = len(matrix[0])
        bx, ex, dx, by, ey, dy = [
            wx.SystemSettings.GetMetric(m)
            for m in (
                wx.SYS_BORDER_X,
                wx.SYS_EDGE_X,
                wx.SYS_SMALLICON_X,
                wx.SYS_BORDER_Y,
                wx.SYS_EDGE_Y,
                wx.SYS_SMALLICON_Y,
            )
        ]
        paint_dc.SetBackground(wx.Brush(get_background_color()))
        paint_dc.Clear()
        pShadow = wx.Pen(
            wx.SystemSettings.GetColour(wx.SYS_COLOUR_BTNSHADOW), 1, wx.PENSTYLE_SOLID
        )
        pHighlight = wx.Pen(
            wx.SystemSettings.GetColour(wx.SYS_COLOUR_BTNHIGHLIGHT),
            1,
            wx.PENSTYLE_SOLID,
        )
        bBackground, bForeground = [
            wx.Brush(color) for color in (wx.Colour(80, 80, 80, 255), wx.WHITE)
        ]
        rw = 2 * ex + dx
        rh = 2 * ey + dy
        for x in range(w):
            for y in range(h):
                rx = x * rw + x * bx
                ry = y * rh + y * by
                value = matrix[y][x]
                paint_dc.SetPen(pHighlight if value else pShadow)
                for k in range(ex):
                    paint_dc.DrawLine(rx + k, ry + k, rx + rw - k - 1, ry + k)
                    paint_dc.DrawLine(rx + k, ry + k, rx + k, ry + rh - k - 1)
                paint_dc.SetPen(pShadow if value else pHighlight)
                for k in range(ex):
                    paint_dc.DrawLine(
                        rx + k, ry + rh - k - 1, rx + rw - k - 1, ry + rh - k - 1
                    )
                    paint_dc.DrawLine(
                        rx + rw - k - 1, ry + k, rx + rw - k - 1, ry + rh - k - 1
                    )
                paint_dc.SetPen(wx.TRANSPARENT_PEN)
                paint_dc.SetBrush(bForeground if value else bBackground)
                paint_dc.DrawRectangle(rx + ex, ry + ey, dx, dy)
        event.Skip()

    @staticmethod
    def get_matrix_element_rect(i, j):
        bx, ex, dx, by, ey, dy = [
            wx.SystemSettings.GetMetric(m)
            for m in (
                wx.SYS_BORDER_X,
                wx.SYS_EDGE_X,
                wx.SYS_SMALLICON_X,
                wx.SYS_BORDER_Y,
                wx.SYS_EDGE_Y,
                wx.SYS_SMALLICON_Y,
            )
        ]
        return wx.Rect(
            ex * (2 * j + 1) + dx * j + bx * j,
            ey * (2 * i + 1) + dy * i + by * i,
            dx,
            dy,
        )

    def hit_test(self, x, y):
        """Return the i, j coordinates at the mouse

        returns i, j or None, None if misses the hit test
        """
        bx, ex, dx, by, ey, dy = [
            wx.SystemSettings.GetMetric(m)
            for m in (
                wx.SYS_BORDER_X,
                wx.SYS_EDGE_X,
                wx.SYS_SMALLICON_X,
                wx.SYS_BORDER_Y,
                wx.SYS_EDGE_Y,
                wx.SYS_SMALLICON_Y,
            )
        ]
        i = int((y - ey) / (2 * ey + dy + by))
        j = int((x - ex) / (2 * ex + dx + bx))
        h, w = self.setting.get_size()
        if (
            i < 0
            or j < 0
            or i >= h
            or j >= w
            or not self.get_matrix_element_rect(i, j).Contains(x, y)
        ):
            return None, None
        return i, j

    def on_width_changed(self, event):
        matrix = self.setting.get_matrix()
        h, w = self.setting.get_size()
        d = self.width_ctrl.GetValue() * 2 + 1 - w
        n = abs(int(d / 2))
        if d < 0:
            matrix = [row[n:-n] for row in matrix]
        elif d > 0:
            matrix = [[False] * n + row + [False] * n for row in matrix]
        else:
            return
        value = BinaryMatrix.to_value(matrix)
        self.module_view.on_value_change(self.setting, self.panel, value, event)

    def on_height_changed(self, event):
        matrix = self.setting.get_matrix()
        h, w = self.setting.get_size()
        d = self.height_ctrl.GetValue() * 2 + 1 - h
        n = abs(int(d / 2))
        if d < 0:
            matrix = matrix[n:-n]
        elif d > 0:
            matrix = (
                [[False] * w for _ in range(n)]
                + matrix
                + [[False] * w for _ in range(n)]
            )
        else:
            return
        value = BinaryMatrix.to_value(matrix)
        self.module_view.on_value_change(self.setting, self.panel, value, event)

    @classmethod
    def update_control(cls, module_view, v):
        """Update the Joiner setting's control

        returns the control
        """
        assert isinstance(module_view, ModuleView)
        control = module_view.module_panel.FindWindowByName(edit_control_name(v))
        if control is None:
            controller = BinaryMatrixController(module_view, v)
            return controller.panel
        else:
            control.controller.update()
            return control
