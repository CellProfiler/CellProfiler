#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

import wx
from cellprofiler.icons import CellProfilerSplash
import cStringIO
import cellprofiler.preferences as cpp
import cellprofiler.utilities.get_revision as get_revision

class CellProfilerApp(wx.App):
    def OnInit(self):
        wx.InitAllImageHandlers()
        
        # splash
        splashimage = wx.BitmapFromImage(wx.ImageFromStream(cStringIO.StringIO(CellProfilerSplash)))
        self.splash = wx.SplashScreen(splashimage, wx.SPLASH_CENTRE_ON_SCREEN | wx.SPLASH_TIMEOUT, 2000, None, -1)

        self.new_version_check()

        from cellprofiler.gui.cpframe import CPFrame
        self.frame = CPFrame(None, -1, "Cell Profiler")

        self.SetTopWindow(self.frame)
        self.frame.Show()
        return 1

    def new_version_check(self, force=False):
        if cpp.get_check_new_versions() or force:
            import cellprofiler.utilities.check_for_updates as cfu
            cfu.check_for_updates('http://broad.mit.edu/~thouis/CPversion.html', 
                                  0 if force else max(get_revision.version, cpp.get_skip_version()), 
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

            if new_version <= get_revision.version:
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
