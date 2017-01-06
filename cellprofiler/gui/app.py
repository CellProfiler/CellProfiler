# coding=utf-8
import sys
import wx
import cellprofiler
import cellprofiler.preferences
import cellprofiler.gui.errordialog


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

        self.SetAppName("CellProfiler{0:s}".format(cellprofiler.__version__))

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
        sys.excepthook = self.original_excepthook


if __name__ == "__main__":
    CellProfilerApp = App(0)

    CellProfilerApp.MainLoop()
