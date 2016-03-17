import sys
import wx
import cellprofiler.preferences
import cellprofiler.gui.errordialog
import cellprofiler.utilities.thread_excepthook

cellprofiler.utilities.thread_excepthook.install_thread_sys_excepthook()


class App(wx.App):
    def __init__(self, *args, **kwargs):
        self.original_excepthook = sys.excepthook
        self.check_for_new_version = kwargs.pop('check_for_new_version', False)

        self.workspace_path = kwargs.pop('workspace_path', None)

        self.pipeline_path = kwargs.pop('pipeline_path', None)

        self.abort_initialization = False

        super(App, self).__init__(*args, **kwargs)

    def OnInit(self):
        # The wx.StandardPaths aren't available until this is set.
        from cellprofiler.utilities.version import dotted_version
        self.SetAppName('CellProfiler%s' % dotted_version)

        if self.check_for_new_version:
            self.new_version_check()

        import cellprofiler.gui.cpframe

        self.frame = cellprofiler.gui.cpframe.CPFrame(None, -1, "Cell Profiler")

        try:
            self.frame.start(self.workspace_path, self.pipeline_path)
        except:
            return 0

        if self.abort_initialization:
            return 0

        def show_errordialog(exception, message, traceback):
            def doit():
                cellprofiler.preferences.cancel_progress()

                cellprofiler.gui.errordialog.display_error_dialog(self.frame, message, None, tb=traceback, continue_only=True, message="Exception in CellProfiler core processing")

            wx.CallAfter(doit)

        sys.excepthook = show_errordialog

        self.SetTopWindow(self.frame)

        self.frame.Show()

        if self.frame.startup_blurb_frame.IsShownOnScreen():
            self.frame.startup_blurb_frame.Raise()

        return 1

    def OnExit(self):
        import imagej.imagej2

        imagej.imagej2.allow_quit()

        sys.excepthook = self.original_excepthook

    def new_version_check(self, force=False):
        pass

    def new_version_cb(self, new_version, new_version_info):
        pass

if __name__ == "__main__":
    CellProfilerApp = App(0)

    CellProfilerApp.MainLoop()
