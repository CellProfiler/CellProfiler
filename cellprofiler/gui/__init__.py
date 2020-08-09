# coding=utf-8
"""CellProfilerGUI package

The CellProfilerGUI package holds the viewer and controller portions
of the cell profiler program
"""

import sys

BV_DOWN = "down"
BV_UP = "up"


def draw_item_selection_rect(window, dc, rect, flags):
    """Replacement for RendererNative.DrawItemSelectionRect

    window - draw in this window

    dc - device context to use for drawing

    rect - draw selection UI inside this rectangle

    flags - a combination of wx.CONTROL_SELECTED, wx.CONTROL_CURRENT and
            wx.CONTROL_FOCUSED

    This function fixes a bug in the Carbon implementation for drawing
    with wx.CONTROL_CURRENT and not wx.CONTROL_SELECTED.
    """
    # Bug in carbon DrawItemSelectionRect uses
    # uninitialized color for the rectangle
    # if it's not selected.
    #
    # Optimistically, I've coded it so that it
    # might work in Cocoa
    #
    import wx

    if sys.platform != "darwin":
        wx.RendererNative.Get().DrawItemSelectionRect(window, dc, rect, flags)
    elif flags & wx.CONTROL_SELECTED:
        if flags & wx.CONTROL_FOCUSED:
            color = wx.SystemSettings.GetColour(wx.SYS_COLOUR_HIGHLIGHT)
        else:
            color = wx.SystemSettings.GetColour(wx.SYS_COLOUR_INACTIVECAPTION)
        old_brush = dc.Brush
        new_brush = wx.Brush(color)
        dc.Brush = new_brush
        dc.Pen = wx.TRANSPARENT_PEN
        dc.DrawRectangle(rect)
        dc.Brush = old_brush
        new_brush.Destroy()
    elif flags & wx.CONTROL_CURRENT:
        #
        # On the Mac, draw a rectangle with the highlight pen and a null
        # brush.
        #
        if flags & wx.CONTROL_FOCUSED:
            pen_color = wx.SystemSettings.GetColour(wx.SYS_COLOUR_HIGHLIGHT)
        else:
            pen_color = wx.SystemSettings.GetColour(wx.SYS_COLOUR_GRAYTEXT)
        old_brush = dc.Brush
        dc.Brush = wx.TRANSPARENT_BRUSH
        old_pen = dc.Pen
        dc.Pen = wx.Pen(pen_color, width=2)
        dc.DrawRectangle(rect.Left, rect.Top, rect.Width, rect.Height)
