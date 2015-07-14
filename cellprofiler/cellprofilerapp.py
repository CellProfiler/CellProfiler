#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2015 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

import wx
from cellprofiler.icons import get_builtin_image
import cStringIO
import cellprofiler.preferences as cpp
from cellprofiler.gui.errordialog import display_error_dialog
import sys

# Make sure sys.excepthook is called for any uncaught exceptions, even in threads.
import cellprofiler.utilities.thread_excepthook
cellprofiler.utilities.thread_excepthook.install_thread_sys_excepthook()

CellProfilerSplash = get_builtin_image('CellProfilerSplash')

class CellProfilerApp(wx.App):
    def __init__(self, *args, **kwargs):
        # allow suppression of version checking (primarily for nosetests). 
        self.check_for_new_version = kwargs.pop('check_for_new_version', False)
        self.show_splashbox = kwargs.pop('show_splashbox', False)
        self.workspace_path = kwargs.pop('workspace_path', None)
        self.pipeline_path = kwargs.pop('pipeline_path', None)
        self.abort_initialization = False
        super(CellProfilerApp, self).__init__(*args, **kwargs)

    def OnInit(self):
        # The wx.StandardPaths aren't available until this is set.
        from cellprofiler.utilities.version import dotted_version
        self.SetAppName('CellProfiler%s' % dotted_version)

        if self.show_splashbox:
            # If the splash image has alpha, it shows up transparently on
            # windows, so we blend it into a white background.
            splashbitmap = wx.EmptyBitmapRGBA(
                CellProfilerSplash.GetWidth(), 
                CellProfilerSplash.GetHeight(), 255, 255, 255, 255)
            dc = wx.MemoryDC()
            dc.SelectObject(splashbitmap)
            dc.DrawBitmap(wx.BitmapFromImage(CellProfilerSplash), 0, 0)
            dc.SelectObject(wx.NullBitmap)
            dc.Destroy() # necessary to avoid a crash in splashscreen
            self.splash = wx.SplashScreen(
                splashbitmap, 
                wx.SPLASH_CENTRE_ON_SCREEN | wx.SPLASH_NO_TIMEOUT, 
                2000, None, -1)
            self.splash_timer = wx.Timer()
            self.splash_timer.Bind(wx.EVT_TIMER, self.destroy_splash_screen)
            self.splash_timer.Start(milliseconds = 2000, oneShot=True)
        else:
            self.splash = None

        if self.check_for_new_version:
            self.new_version_check()

        from javabridge import activate_awt
        activate_awt()
        from cellprofiler.gui.cpframe import CPFrame
        self.frame = CPFrame(None, -1, "Cell Profiler")
        self.destroy_splash_screen()
        try:
            self.frame.start(self.workspace_path, self.pipeline_path)
        except:
            return 0
        if self.abort_initialization:
            return 0

        # set up error dialog for uncaught exceptions
        def show_errordialog(type, exc, tb):
            def doit():
                cpp.cancel_progress()
                display_error_dialog(self.frame, exc, None, tb=tb, continue_only=True,
                                     message="Exception in CellProfiler core processing")
                # continue is really the only choice
            wx.CallAfter(doit)
        # replace default hook with error dialog
        self.orig_excepthook = sys.excepthook
        sys.excepthook = show_errordialog
        self.SetTopWindow(self.frame)
        self.frame.Show()
        if self.frame.startup_blurb_frame.IsShownOnScreen():
            self.frame.startup_blurb_frame.Raise()
        return 1

    def destroy_splash_screen(self, event = None):
        if self.splash is not None:
            self.splash.Destroy()
            self.splash = None
        
    def OnExit(self):
        from imagej.imagej2 import allow_quit
        allow_quit()
        from javabridge import deactivate_awt
        deactivate_awt()
        # restore previous exception hook
        sys.excepthook = self.orig_excepthook

    def new_version_check(self, force=False):
        if cpp.get_check_new_versions() or force:
            import cellprofiler.utilities.check_for_updates as cfu
            import platform
            import cellprofiler.utilities.version

            version_string = cellprofiler.utilities.version.version_string
            dotted_version = cellprofiler.utilities.version.dotted_version
            version_number = cellprofiler.utilities.version.version_number
            self.version = version_number
            cfu.check_for_updates('http://cellprofiler.org/CPupdate.html',
                                  0 if force else max(version_number, cpp.get_skip_version()),
                                  self.new_version_cb,
                                  user_agent='CellProfiler/%s %s' % (dotted_version, version_string))

    def new_version_cb(self, new_version, new_version_info):
        # called from a child thread, so use CallAfter to bump it to the gui thread
        def cb2():
            def set_check_pref(val):
                cpp.set_check_new_versions(val)

            def skip_this_version():
                cpp.set_skip_version(new_version)

            self.destroy_splash_screen()

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
