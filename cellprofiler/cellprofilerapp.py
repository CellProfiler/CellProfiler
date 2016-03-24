#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

import cStringIO
import sys

import wx

import cellprofiler.preferences as cpp
from cellprofiler.gui.errordialog import display_error_dialog
from cellprofiler.icons import get_builtin_image
# Make sure sys.excepthook is called for any uncaught exceptions, even in threads.
import cellprofiler.utilities.thread_excepthook

cellprofiler.utilities.thread_excepthook.install_thread_sys_excepthook()


class CellProfilerApp(wx.App):
    def __init__(self, *args, **kwargs):
        # allow suppression of version checking (primarily for nosetests).
        self.check_for_new_version = kwargs.pop('check_for_new_version', False)
        self.workspace_path = kwargs.pop('workspace_path', None)
        self.pipeline_path = kwargs.pop('pipeline_path', None)
        self.abort_initialization = False
        super(CellProfilerApp, self).__init__(*args, **kwargs)

    def OnInit(self):
        # The wx.StandardPaths aren't available until this is set.
        from cellprofiler.utilities.version import dotted_version
        self.SetAppName('CellProfiler%s' % dotted_version)

        from cellprofiler.gui.cpframe import CPFrame
        self.frame = CPFrame(None, -1, "Cell Profiler")
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

    def OnExit(self):
        from imagej.imagej2 import allow_quit
        allow_quit()
        # restore previous exception hook
        sys.excepthook = self.orig_excepthook

# end of class CellProfilerApp

if __name__ == "__main__":
    CellProfilerApp = CellProfilerApp(0)
    CellProfilerApp.MainLoop()
