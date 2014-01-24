"""cellprofiler.gui.html.__init__

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
from htmlwindow import HtmlClickableWindow

__all__ = ['HtmlClickableWindow']

# Rewrite the help for the case where we have to use a differently named menu for wx 2.8.10.1 on Mac
import wx
import content
if wx.VERSION <= (2, 8, 10, 1, '') and wx.Platform == '__WXMAC__':
    content.startup_main = content.startup_main.replace('<i>Help</i> menu', '<i>CellProfiler Help</i> menu')
