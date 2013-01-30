'''bitmaplabelbutton.py - a button that displays a bitmap to the left of a label

'''
#CellProfiler is distributed under the GNU General Public License.
#See the accompanying file LICENSE for details.

#Copyright (c) 2003-2009 Massachusetts Institute of Technology
#Copyright (c) 2009-2013 Broad Institute
#All rights reserved.
#
#Please see the AUTHORS file for credits.
#
#Website: http://www.cellprofiler.org

import wx
from wx.lib.buttons import GenBitmapTextButton

class BitmapLabelButton(GenBitmapTextButton):
    def DrawBezel(self, dc, x1, y1, x2, y2):
        '''Use the native look and feel for the button outlines'''
        renderer = wx.RendererNative.Get()
        flags = 0
        if not self.up:
            flags += wx.CONTROL_PRESSED
        if self.hasFocus and self.useFocusInd:
            flags += wx.CONTROL_CURRENT
        renderer.DrawPushButton(self, dc, wx.Rect(x1, y1, x2-x1, y2 - y1), flags)
        
    def DrawFocusIndicator(self, dc, w, h):
        '''Focus indicator handled above'''
        pass