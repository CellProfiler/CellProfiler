#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

import wx
from cellprofiler.icons import CellProfilerSplash
import cStringIO
import cellprofiler.preferences as cpp

class CellProfilerApp(wx.App):
    def __init__(self, *args, **kwargs):
        # allow suppression of version checking (primarily for nosetests). 
        self.check_for_new_version = kwargs.pop('check_for_new_version', False)
        super(CellProfilerApp, self).__init__(*args, **kwargs)

    def OnInit(self):
        # The wx.StandardPaths aren't available until this is set.
        # SVN version checking imports cellprofiler.modules, which
        # needs preferences that depend on StandardPaths.
        self.SetAppName('CellProfiler2.0')
        import cellprofiler.utilities.get_revision
        self.version = cellprofiler.utilities.get_revision.version

        wx.InitAllImageHandlers()

        # If the splash image has alpha, it shows up transparently on
        # windows, so we blend it into a white background.
        splashbitmap = wx.EmptyBitmapRGBA(CellProfilerSplash.GetWidth(), CellProfilerSplash.GetHeight(), 255, 255, 255, 255)
        dc = wx.MemoryDC()
        dc.SelectObject(splashbitmap)
        dc.DrawBitmap(wx.BitmapFromImage(CellProfilerSplash), 0, 0)
        dc.Destroy() # necessary to avoid a crash in splashscreen
        self.splash = wx.SplashScreen(splashbitmap, wx.SPLASH_CENTRE_ON_SCREEN | wx.SPLASH_TIMEOUT, 2000, None, -1)

        if self.check_for_new_version:
            self.new_version_check()

        from cellprofiler.gui.cpframe import CPFrame
        self.frame = CPFrame(None, -1, "Cell Profiler")

        self.SetTopWindow(self.frame)
        self.frame.Show()
        return 1

    def new_version_check(self, force=False):
        if cpp.get_check_new_versions() or force:
            import cellprofiler.utilities.check_for_updates as cfu
            cfu.check_for_updates('http://cellprofiler.org/CPupdate.html', 
                                  0 if force else max(self.version, cpp.get_skip_version()), 
                                  self.new_version_cb)

    def new_version_cb(self, new_version, new_version_info):
        # called from a child thread, so use CallAfter to bump it to the gui thread
        def cb2():
            def set_check_pref(val):
                cpp.set_check_new_versions(val)

            def skip_this_version():
                cpp.set_skip_version(new_version)

            # showing a modal dialog while the splashscreen is up causes a hang
            try: self.splash.Destroy()
            except: pass

            if new_version <= self.version:
                # special case: force must have been set in new_version_check, so give feedback to the user.
                wx.MessageBox('Your copy of CellProfiler is up to date.', '', wx.ICON_INFORMATION)
                return

            import cellprofiler.gui.newversiondialog as nvd
            dlg = nvd.NewVersionDialog(None, "CellProfiler update available (version %d)"%(new_version),
                                       new_version_info, 'http://cellprofiler.org/download.htm',
                                       cpp.get_check_new_versions(), set_check_pref, skip_this_version)
            dlg.ShowModal()
            dlg.Destroy()

        wx.CallAfter(cb2)

# end of class CellProfilerApp

if __name__ == "__main__":
    CellProfilerApp = CellProfilerApp(0)
    CellProfilerApp.MainLoop()
