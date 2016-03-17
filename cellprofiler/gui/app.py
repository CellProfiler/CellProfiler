import sys
import wx
import cellprofiler.preferences
import cellprofiler.gui.errordialog
import cellprofiler.utilities.thread_excepthook


cellprofiler.utilities.thread_excepthook.install_thread_sys_excepthook()


class App(wx.App):
    def __init__(self, *args, **kwargs):
        self.abort_initialization = False

        self.frame = None

        self.original_excepthook = sys.excepthook

        self.pipeline_path = kwargs.pop('pipeline_path', None)

        self.workspace_path = kwargs.pop('workspace_path', None)

        super(App, self).__init__(*args, **kwargs)

    def OnInit(self):
        import cellprofiler.gui.cpframe
        import cellprofiler.utilities.version

        self.SetAppName("CellProfiler{0:s}".format(cellprofiler.utilities.version.dotted_version))

        self.frame = cellprofiler.gui.cpframe.CPFrame(None, -1, "Cell Profiler")

        self.frame.start(self.workspace_path, self.pipeline_path)

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


if __name__ == "__main__":
    CellProfilerApp = App(0)

    CellProfilerApp.MainLoop()
