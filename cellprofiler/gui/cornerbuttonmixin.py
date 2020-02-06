# coding=utf-8
"""CornerButtonMixin.py - a mixin for wx.grid.Grid that manages a button in the corner
"""

import wx
import wx.grid
import wx.lib.mixins.gridlabelrenderer


class CornerButtonMixin(object):
    """A mixin class for wx.grid.Grid that adds a button in the corner

    This should be added as a mixin to a class derived from wx.grid.Grid.
    It takes control of the grid's GridCornerLabelWindow, managing mouseclicks
    and painting to make it appear as if there is a button there
    """

    def __init__(self, fn_clicked, label="Update", tooltip="Update this table"):
        """Initialize the mixin - call after wx.grid.Grid.__init__

        fn_clicked - function to call upon button press

        label - the button's label
        """
        self.fn_clicked = fn_clicked
        self.label = label
        self.tooltip = tooltip
        corner = self.GetGridCornerLabelWindow()
        corner.SetDoubleBuffered(True)
        corner.Bind(wx.EVT_PAINT, self.on_paint_corner)
        corner.Bind(wx.EVT_LEFT_DOWN, self.on_corner_left_mouse_down)
        corner.Bind(wx.EVT_LEFT_UP, self.on_corner_left_mouse_up)
        corner.Bind(wx.EVT_MOTION, self.on_corner_motion)
        corner.Bind(wx.EVT_MOUSE_CAPTURE_LOST, self.on_corner_capture_lost)
        self.corner_hitcode = self.CORNER_HIT_NONE
        self.corner_button_pressed = False

    #######
    #
    # Grid corner handling
    #
    #######

    CORNER_HIT_NONE = None
    CORNER_HIT_UPDATE = 0
    BUTTON_PADDING = 4

    CORNER_ICON_PADDING = 2
    CORNER_ICON_SIZE = 16

    def get_corner_update_button_rect(self):
        crect = self.GetGridCornerLabelWindow().GetRect()
        w, h = self.GetGridCornerLabelWindow().GetTextExtent(self.label)
        w += 2 * self.BUTTON_PADDING
        h += 2 * self.BUTTON_PADDING
        x = crect.X + (crect.width - w) / 2
        y = crect.Y + (crect.height - h) / 2
        return wx.Rect(x, y, w, h)

    def corner_hit_test(self, x, y):
        if self.fn_clicked is None:
            return self.CORNER_HIT_NONE
        r = self.get_corner_update_button_rect()
        if r.Contains(x, y):
            return self.CORNER_HIT_UPDATE
        return self.CORNER_HIT_NONE

    def on_paint_corner(self, event):
        corner = self.GetGridCornerLabelWindow()
        dc = wx.BufferedPaintDC(corner)
        dc.SetFont(self.GetGridCornerLabelWindow().GetFont())
        old_brush = dc.GetBackground()
        new_brush = wx.Brush(self.GetGridCornerLabelWindow().GetBackgroundColour())
        dc.SetBackground(new_brush)
        try:
            dc.Clear()
            dc.SetBackgroundMode(wx.PENSTYLE_TRANSPARENT)
            rn = wx.RendererNative.Get()
            assert isinstance(rn, wx.RendererNative)
            cr = wx.lib.mixins.gridlabelrenderer.GridDefaultCornerLabelRenderer()
            cr.DrawBorder(self, dc, corner.GetRect())
            if self.fn_clicked is not None:
                r = self.get_corner_update_button_rect()
                if self.corner_hitcode == self.CORNER_HIT_UPDATE:
                    if self.corner_button_pressed:
                        flags = (
                            wx.CONTROL_PRESSED
                            | wx.CONTROL_CURRENT
                            | wx.CONTROL_FOCUSED
                            | wx.CONTROL_SELECTED
                        )
                    else:
                        flags = 0
                else:
                    flags = 0
                rn.DrawPushButton(corner, dc, r, flags)
                w, h = self.GetGridCornerLabelWindow().GetTextExtent(self.label)
                x = r.GetX() + (r.GetWidth() - w) / 2
                y = r.GetY() + (r.GetHeight() - h) / 2
                dc.DrawText(self.label, x, y)

        finally:
            dc.SetBackground(old_brush)
            new_brush.Destroy()

    def on_corner_left_mouse_down(self, event):
        corner = self.GetGridCornerLabelWindow()
        hit_code = self.corner_hit_test(event.X, event.Y)
        if hit_code != self.CORNER_HIT_NONE:
            self.corner_hitcode = hit_code
            self.corner_button_pressed = True
            corner.CaptureMouse()
            corner.Refresh(eraseBackground=False)

    def on_corner_left_mouse_up(self, event):
        corner = self.GetGridCornerLabelWindow()
        if self.corner_hitcode != self.CORNER_HIT_NONE:
            hit_code = self.corner_hit_test(event.X, event.Y)
            if hit_code == self.corner_hitcode:
                if hit_code == self.CORNER_HIT_UPDATE:
                    self.fn_clicked()
            self.corner_hitcode = self.CORNER_HIT_NONE
            if corner.HasCapture():
                corner.ReleaseMouse()
            corner.Refresh(eraseBackground=False)

    def on_corner_motion(self, event):
        corner = self.GetGridCornerLabelWindow()
        hit_code = self.corner_hit_test(event.X, event.Y)
        if self.corner_hitcode == self.CORNER_HIT_NONE:
            if hit_code == self.CORNER_HIT_NONE:
                corner.SetToolTip("")
            elif hit_code == self.CORNER_HIT_UPDATE:
                corner.SetToolTip(self.tooltip)
        else:
            was_pressed = self.corner_button_pressed
            self.corner_button_pressed = (
                self.corner_hitcode != self.CORNER_HIT_NONE
                and self.corner_hitcode == hit_code
            )
            if was_pressed != self.corner_button_pressed:
                corner.RefreshRect(
                    self.get_corner_update_button_rect(), eraseBackground=False
                )

    def on_corner_capture_lost(self, event):
        self.corner_hitcode = self.CORNER_HIT_NONE
