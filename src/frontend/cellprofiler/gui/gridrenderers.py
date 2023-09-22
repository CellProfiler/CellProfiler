import wx
import wx.lib.mixins.gridlabelrenderer as wxglr
from cellprofiler.icons import get_builtin_image

class GridLabelRenderer(wxglr.GridLabelRenderer):
    def DrawBorder(self, grid, dc, rect):
        """
        Draw a standard border around the label, to give a simple 3D
        effect like the stock wx.grid.Grid labels do.
        """
        top = rect.top
        bottom = rect.bottom
        left = rect.left
        right = rect.right
        old_pen = dc.GetPen()
        dc.SetPen(wx.Pen(wx.SystemSettings.GetColour(wx.SYS_COLOUR_3DSHADOW)))
        dc.DrawLine(right, top, right, bottom)
        dc.DrawLine(left, top, left, bottom)
        dc.DrawLine(left, bottom, right, bottom)
        dc.SetPen(wx.Pen(wx.SystemSettings.GetColour(wx.SYS_COLOUR_3DHIGHLIGHT)))
        if top == 0:
            dc.DrawLine(left + 1, top, left + 1, bottom)
            dc.DrawLine(left + 1, top, right, top)
        else:
            dc.DrawLine(left + 1, top + 1, left + 1, bottom)
            dc.DrawLine(left + 1, top + 1, right - 1, top + 1)
        dc.SetPen(old_pen)

class RowLabelRenderer(wxglr.GridDefaultRowLabelRenderer, GridLabelRenderer):
    def Draw(self, grid, dc, rect, row):
        super().Draw(grid, dc, rect, row)
class ColLabelRenderer(wxglr.GridDefaultColLabelRenderer, GridLabelRenderer):
    def Draw(self, grid, dc, rect, row):
        super().Draw(grid, dc, rect, row)

class CornerLabelRenderer(wxglr.GridDefaultCornerLabelRenderer):
    def __init__(self, grid, fn_clicked, tooltip, label):
        self._corner = grid.GetGridCornerLabelWindow()

        bmp = wx.Bitmap(get_builtin_image("IMG_UPDATE"))

        self._bmp_btn = wx.lib.buttons.GenBitmapTextButton(
            self._corner, bitmap=bmp, size=self._corner.GetSize()
        )
        self.fn_clicked = fn_clicked
        self.tooltip = tooltip
        self.label = label

    @property
    def fn_clicked(self):
        return self._fn_clicked

    @fn_clicked.setter
    def fn_clicked(self, func):
        self._fn_clicked = func
        self._corner.Bind(wx.EVT_BUTTON, self._fn_clicked)

    @property
    def tooltip(self):
        return self._bmp_btn.GetToolTip()

    @tooltip.setter
    def tooltip(self, val):
        self._bmp_btn.SetToolTip(val)

    @property
    def label(self):
        self._bmp_btn.GetLabel()

    @label.setter
    def label(self, val):
        self._bmp_btn.SetLabel(val)

    def Draw(self, grid, dc, rect, rc):
        pass
