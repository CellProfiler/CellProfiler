'''sashwindow_tools.py - custom painting of sashwindows

This module takes over painting the sash window to make it a little more obvious

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

import wx
from wx.aui import PyAuiDockArt
from cellprofiler.preferences import get_background_color

'''The size of the gripper the long way. This is about 5 dots worth.'''
GRIPPER_SIZE = 32
'''The size of the gripper the short way.'''
GRIPPER_HEIGHT = 8

def sw_bind_to_evt_paint(window):
    '''Bind to wx.EVT_PAINT to take over the painting
    
    window - a wx.SashWindow
    '''
    window.Bind(wx.EVT_PAINT, on_sashwindow_paint)

__art = None
__pane_info = None

def get_art_and_pane_info():
    global __art
    global __pane_info
    if __art is None:
        __art = PyAuiDockArt()
        __pane_info = wx.aui.AuiPaneInfo()
        __pane_info.Gripper(True)
    return __art, __pane_info


def on_sashwindow_paint(event):
    assert isinstance(event, wx.PaintEvent)
    window = event.EventObject
    assert isinstance(window, wx.SashWindow)
    dc = wx.PaintDC(window)
    dc.BeginDrawing()
    dc.Background = wx.Brush(get_background_color())
    dc.Clear()
    art, pane_info = get_art_and_pane_info()
    w, h = window.GetClientSizeTuple()
    for edge, orientation in (
        (wx.SASH_LEFT, wx.VERTICAL),
        (wx.SASH_TOP, wx.HORIZONTAL),
        (wx.SASH_RIGHT, wx.VERTICAL),
        (wx.SASH_BOTTOM, wx.HORIZONTAL)):
        if window.GetSashVisible(edge):
            margin = window.GetEdgeMargin(edge)
            if orientation == wx.VERTICAL:
                sy = 0
                sh = h
                sw = margin
                gw = GRIPPER_HEIGHT
                gh = GRIPPER_SIZE
                gy = (h - GRIPPER_SIZE) / 2
                pane_info.GripperTop(False)
                if edge == wx.SASH_LEFT:
                    gx = sx = 0
                else:
                    sx = w - margin
                    gx = w - gw
            else:
                sx = 0
                sw = w
                sh = margin
                gw = GRIPPER_SIZE
                gh = GRIPPER_HEIGHT
                gx = (w - GRIPPER_SIZE) / 2
                pane_info.GripperTop(True)
                if edge == wx.SASH_TOP:
                    gy = sy = 0
                else:
                    sy = h - margin
                    gy = h - gh
            art.DrawSash(dc, window, orientation, wx.Rect(sx, sy, sw, sh))
            art.DrawGripper(dc, window, wx.Rect(gx, gy, gw, gh), pane_info)
    dc.EndDrawing()

def sp_bind_to_evt_paint(window):
    '''Take over painting the splitter of a splitter window'''
    window.Bind(wx.EVT_PAINT, on_splitter_paint)
    
def on_splitter_paint(event):
    assert isinstance(event, wx.PaintEvent)
    window = event.EventObject
    assert isinstance(window, wx.SplitterWindow)
    dc = wx.PaintDC(window)
    dc.BeginDrawing()
    dc.Background = wx.Brush(get_background_color())
    dc.Clear()
    art, pane_info = get_art_and_pane_info()
    w, h = window.GetClientSizeTuple()
    margin = window.GetSashSize()
    pos = window.GetSashPosition()
    if window.GetSplitMode() == wx.SPLIT_VERTICAL:
        pane_info.GripperTop(False)
        sy = 0
        sh = h
        sw = margin
        gw = GRIPPER_HEIGHT
        sx = pos - margin/2
        gx = pos - gw / 2
        gy = (h - GRIPPER_SIZE) / 2
        gh = GRIPPER_SIZE
        orientation = wx.VERTICAL
    else:
        pane_info.GripperTop(True)
        sx = 0
        sw = h
        sh = margin
        gh = GRIPPER_HEIGHT
        sy = pos - margin / 2
        gx = (w - GRIPPER_SIZE) / 2
        gy = pos - GRIPPER_SIZE / 2
        gw = GRIPPER_SIZE
        orientation = wx.HORIZONTAL
    art.DrawSash(dc, window, orientation, wx.Rect(sx, sy, sw, sh))
    art.DrawGripper(dc, window, wx.Rect(gx, gy, gw, gh), pane_info)
    dc.EndDrawing()