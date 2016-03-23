#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

import cellprofiler.gui.errordialog
import preferences
import utilities.thread_excepthook
import sys
import wx

utilities.thread_excepthook.install_thread_sys_excepthook()


class Application(wx.App):
    def __init__(self, *args, **kwargs):
        # allow suppression of version checking (primarily for nosetests).
        self.check_for_new_version = kwargs.pop('check_for_new_version', False)
        self.workspace_path = kwargs.pop('workspace_path', None)
        self.pipeline_path = kwargs.pop('pipeline_path', None)
        self.abort_initialization = False
        super(Application, self).__init__(*args, **kwargs)

    def OnInit(self):
        # The wx.StandardPaths aren't available until this is set.
        from cellprofiler.utilities.version import dotted_version
        self.SetAppName('CellProfiler%s' % dotted_version)

        if self.check_for_new_version:
            self.new_version_check()

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
                cellprofiler.preferences.cancel_progress()

                cellprofiler.gui.errordialog.display_error_dialog(self.frame, exc, None, tb=tb, continue_only=True, message="Exception in CellProfiler core processing")
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

    def new_version_check(self, force=False):
        if preferences.get_check_new_versions() or force:
            import cellprofiler.utilities.check_for_updates as cfu
            import cellprofiler.utilities.version

            version_string = cellprofiler.utilities.version.version_string
            dotted_version = cellprofiler.utilities.version.dotted_version
            version_number = cellprofiler.utilities.version.version_number
            self.version = version_number
            cfu.check_for_updates('http://cellprofiler.org/CPupdate.html',
                                  0 if force else max(version_number, cellprofiler.preferences.get_skip_version()),
                                  self.new_version_cb,
                                  user_agent='CellProfiler/%s %s' % (dotted_version, version_string))

    def new_version_cb(self, new_version, new_version_info):
        # called from a child thread, so use CallAfter to bump it to the gui thread
        def cb2():
            def set_check_pref(val):
                cellprofiler.preferences.set_check_new_versions(val)

            def skip_this_version():
                cellprofiler.preferences.set_skip_version(new_version)

            if new_version <= self.version:
                # special case: force must have been set in new_version_check, so give feedback to the user.
                wx.MessageBox('Your copy of CellProfiler is up to date.', '', wx.ICON_INFORMATION)
                return

            import cellprofiler.gui.newversiondialog as nvd
            dlg = nvd.NewVersionDialog(None, "CellProfiler update available (version %d)" % (new_version),
                                       new_version_info, 'http://cellprofiler.org/download.htm',
                                       cellprofiler.preferences.get_check_new_versions(), set_check_pref, skip_this_version)
            dlg.ShowModal()
            dlg.Destroy()

        wx.CallAfter(cb2)


if __name__ == "__main__":
    CellProfilerApp = Application(0)
    CellProfilerApp.MainLoop()
