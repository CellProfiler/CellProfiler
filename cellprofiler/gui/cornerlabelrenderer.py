import wx
import wx.lib.mixins.gridlabelrenderer as wxglr
from cellprofiler.icons import get_builtin_image

class CornerLabelRenderer(wxglr.GridDefaultCornerLabelRenderer):
    def __init__(self, grid, on_click, tooltip, label):
        self._label = label
        self._corner = grid.GetGridCornerLabelWindow()

        bmp = wx.Bitmap(get_builtin_image("IMG_UPDATE"))
        # mask = wx.Mask(bmp, wx.BLUE)
        # bmp.SetMask(mask)

        self._bmp_btn = wx.lib.buttons.GenBitmapTextButton(
            self._corner, bitmap=bmp, label=label, size=self._corner.GetSize()
        )
        self._bmp_btn.SetToolTip(tooltip)
        self._corner.Bind(wx.EVT_BUTTON, on_click)

    def Draw(self, grid, dc, rect, rc):
        top = rect.top
        bottom = rect.bottom
        left = rect.left
        right = rect.right
        dc.SetPen(wx.Pen(wx.SystemSettings.GetColour(wx.SYS_COLOUR_3DSHADOW)))
        dc.SetPen(wx.RED_PEN)
        # dc.DrawLine(right, top, right, bottom)
        # dc.DrawLine(left, top, left, bottom)
        # dc.DrawLine(left, bottom, right, bottom)
        # dc.DrawLine(left, top, right, top)
        # dc.SetPen(wx.RED_PEN)
        # if top == 0:
        #     dc.DrawLine(left + 1, top, left + 1, bottom)
        #     dc.DrawLine(left + 1, top, right+10, top)
        # else:
        #     dc.DrawLine(left + 1, top + 1, left + 1, bottom)
        #     dc.DrawLine(left + 1, top + 1, right - 1, top + 1)