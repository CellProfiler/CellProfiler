#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

import wx
from cellprofiler.icons import CellProfilerSplash
import cStringIO

class CellProfilerApp(wx.App):
    def OnInit(self):
        splashimage = wx.BitmapFromImage(wx.ImageFromStream(cStringIO.StringIO(CellProfilerSplash)))
        wx.SplashScreen(splashimage, wx.SPLASH_CENTRE_ON_SCREEN | wx.SPLASH_TIMEOUT, 2000, None, -1)
        wx.InitAllImageHandlers()
        from cellprofiler.gui.cpframe import CPFrame
        self.frame = CPFrame(None, -1, "Cell Profiler")
        self.SetTopWindow(self.frame)
        self.frame.Show()
        return 1

# end of class CellProfilerApp

if __name__ == "__main__":
    CellProfilerApp = CellProfilerApp(0)
    CellProfilerApp.MainLoop()
