""" an htmldialog with an ok button

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""


long_text = """\
This is a very important parameter which tells the module what you are
looking for. Most options within this module use this estimate of the
size range of the objects in order to distinguish them from noise in the
image. For example, for some of the identification methods, the smoothing
applied to the image is based on the minimum size of the objects. A comma
should be placed between the minimum and the maximum diameters. The units
here are pixels so that it is easy to zoom in on objects and determine
typical diameters.

<p>To measure distances easily, use the CellProfiler Image Tool,
'ShowOrHidePixelData', in any open window. Once this tool is
activated, you can draw a line across objects in your image and the
length of the line will be shown in pixel units. Note that for
non-round objects, the diameter here is actually the 'equivalent
diameter', meaning the diameter of a circle with the same area as the
object."""

import wx
from cellprofiler.gui.html.htmlwindow import HtmlClickableWindow
import webbrowser

class HTMLDialog(wx.Dialog):
    def __init__(self, parent, title, contents):
        super(HTMLDialog, self).__init__(parent, -1, title, 
                                         style=(wx.DEFAULT_DIALOG_STYLE | 
                                                wx.RESIZE_BORDER))
        html = HtmlClickableWindow(parent=self)
        html.SetPage(contents)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(html, 1, wx.EXPAND | wx.ALL, 5)
        sizer.Add(self.CreateStdDialogButtonSizer(wx.OK), flag=wx.CENTER)
        self.SetSizer(sizer)
        html.SetFocus()
        #self.Layout()

if __name__ == "__main__":
    app = wx.PySimpleApp()
    dialog = HTMLDialog(None, "Title", long_text)
    choice =  dialog.Show()
    app.MainLoop()

    
