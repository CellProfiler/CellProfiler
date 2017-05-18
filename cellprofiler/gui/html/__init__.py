# coding=utf-8
"""cellprofiler.gui.html.__init__
"""

from . import content
from . import htmlwindow
import wx

# Rewrite the help for the case where we have to use a differently named menu for wx 2.8.10.1 on Mac

__all__ = ['HtmlClickableWindow']

if wx.VERSION <= (2, 8, 10, 1, '') and wx.Platform == '__WXMAC__':
    content.startup_main = content.startup_main.replace('<i>Help</i> menu', '<i>CellProfiler Help</i> menu')
