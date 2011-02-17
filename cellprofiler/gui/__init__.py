"""CellProfilerGUI package

The CellProfilerGUI package holds the viewer and controller portions
of the cell profiler program

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2010 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

__version__="$Revision$"

import os
import sys
import cellprofiler.preferences
from cellprofiler.icons import get_builtin_image

cp_image = None

def get_cp_image():
    """The CellProfiler icon as a wx.Image"""
    global cp_image
    if cp_image is None:
        cp_image = get_builtin_image('CellProfilerIcon')
    return cp_image

def get_cp_bitmap(size=None):
    """The CellProfiler icon as a wx.Bitmap"""
    import wx
    img = get_cp_image()
    if size is not None:
        img.Rescale(size, size, wx.IMAGE_QUALITY_HIGH)
    return wx.BitmapFromImage(img)
    
def get_cp_icon(size=None):
    """The CellProfiler icon as a wx.Icon"""
    import wx
    icon = wx.EmptyIcon()
    if size == None and sys.platform.startswith('win'):
        size = 32
    icon.CopyFromBitmap(get_cp_bitmap(size))
    return icon

BV_DOWN = "down"
BV_UP   = "up"
def draw_bevel(dc, rect, width, state, shadow_pen = None, highlight_pen = None):
    """Draw a bevel within the rectangle so the inside looks raised or lowered
    
    dc - device context for drawing
    rect - draw the bevel within this rectangle
    width - the width of the bevel in pixels
    state - either BV_DOWN for a bevel with a lowered appearance or BV_UP
            for raised appearance
    shadow_pen - pen to use for drawing the shadow portion of the bevel
    highlight_pen - pen to use for drawing the light portion of the bevel
    
    returns the coordinates of the inside rectangle
    """
    import wx
    if shadow_pen == None:
        shadow_pen = wx.Pen(wx.SystemSettings.GetColour(wx.SYS_COLOUR_3DSHADOW))
    if highlight_pen == None:
        highlight_pen = wx.Pen(wx.SystemSettings.GetColour(wx.SYS_COLOUR_3DHIGHLIGHT))
    top_left_pen = (state == BV_UP and highlight_pen) or shadow_pen
    bottom_right_pen = (state == BV_UP and shadow_pen) or highlight_pen
    for i in range(width):
        dc.Pen = top_left_pen
        dc.DrawLine(rect.Left,rect.Top,rect.Left,rect.Bottom)
        dc.DrawLine(rect.Left,rect.Top,rect.Right,rect.Top)
        dc.Pen = bottom_right_pen
        dc.DrawLine(rect.Right,rect.Bottom, rect.Left, rect.Bottom)
        dc.DrawLine(rect.Right,rect.Bottom, rect.Right, rect.Top)
        rect = wx.Rect(rect.Left+1, rect.Top+1, rect.width-2, rect.height-2)
    return rect
