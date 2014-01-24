'''bitmaplabelbutton.py - a button that displays a bitmap to the left of a label

'''
#CellProfiler is distributed under the GNU General Public License.
#See the accompanying file LICENSE for details.

#Copyright (c) 2003-2009 Massachusetts Institute of Technology
#Copyright (c) 2009-2014 Broad Institute
#All rights reserved.
#
# Some code below adapted from buttons.py
# Copyright:   (c) 1999 by Total Control Software
# Licence:     wxWindows license
#
#Please see the AUTHORS file for credits.
#
#Website: http://www.cellprofiler.org

import wx
from wx.lib.buttons import GenBitmapTextButton

class BitmapLabelButton(GenBitmapTextButton):
    LEFT_MARGIN = 5
    TEXT_PADDING = 3
    def OnPaint(self, event):
        width, height = self.GetClientSizeTuple()
        dc = wx.PaintDC(self)
        dc.SetBackground(wx.Brush(self.BackgroundColour))
        dc.Clear()
        flags = 0
        if not self.up:
            flags += wx.CONTROL_PRESSED
        if self.hasFocus:
            flags += wx.CONTROL_FOCUSED
        
        wx.RendererNative.Get().DrawPushButton(
            self, dc, wx.Rect(0, 0, width, height), flags)
        
        bmp = self.bmpLabel
        if self.bmpDisabled and not self.IsEnabled():
            bmp = self.bmpDisabled
        if self.bmpFocus and self.hasFocus:
            bmp = self.bmpFocus
        if self.bmpSelected and not self.up:
            bmp = self.bmpSelected
        bw,bh = bmp.GetWidth(), bmp.GetHeight()
        if not self.up:
            dx = dy = self.labelDelta
        else:
            dx = dy = 0
        hasMask = bmp.GetMask() != None
        bitmap_y = (height-bh)/2+dy
        dc.DrawBitmap(bmp, self.LEFT_MARGIN+dx, bitmap_y, hasMask)
        
        dc.SetFont(self.GetFont())
        if self.IsEnabled():
            dc.SetTextForeground(self.GetForegroundColour())
        else:
            dc.SetTextForeground(wx.SystemSettings.GetColour(wx.SYS_COLOUR_GRAYTEXT))

        label = self.GetLabel()
        if not self.up:
            dx = dy = self.labelDelta

        # Left justify the text after a little padding
        # Center.
        pos_x = self.LEFT_MARGIN + bw + self.TEXT_PADDING + dx
        text_width, text_height = dc.GetTextExtent(label)

        dc.DrawText(label, pos_x, (height - text_height)/2)
        
