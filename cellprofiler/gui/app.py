# coding=utf-8

import logging
import os.path
import platform
import sys

import raven
import raven.conf
import raven.handlers.logging
import raven.transport.threaded_requests
import wx

import cellprofiler
import cellprofiler.preferences
import cellprofiler.utilities.cpjvm
import cellprofiler.gui.dialog
import cellprofiler.utilities.cpjvm

logger = logging.getLogger(__name__)

transport = raven.transport.threaded_requests.ThreadedRequestsHTTPTransport

dsn = "https://c0b47db2a1b34f12b33ca8e78067617e:3cee11601374464dadd4b44da8a22dbd@sentry.io/152399"

sentry = raven.Client(
    dsn=dsn,
    transport=transport,
    install_sys_hook=False
)

sentry.user_context({
    "machine": platform.machine(),
    "processor": platform.processor(),
    "python_implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
    "release": platform.release(),
    "system": platform.system(),
    "version": platform.version()
})


class App(wx.App):
    def __init__(self, *args, **kwargs):
        self.abort_initialization = False

        self.frame = None

        self.pipeline_path = kwargs.pop("pipeline_path", None)

        self.workspace_path = kwargs.pop("workspace_path", None)

        cellprofiler.utilities.cpjvm.cp_start_vm()

        super(App, self).__init__(*args, **kwargs)

    def OnInit(self):
        import cellprofiler.gui.cpframe

        self.SetAppName("CellProfiler{0:s}".format(cellprofiler.__version__))

        self.frame = cellprofiler.gui.cpframe.CPFrame(None, -1, "CellProfiler")

        self.frame.start(self.workspace_path, self.pipeline_path)

        if self.abort_initialization:
            return False

        self.SetTopWindow(self.frame)

        self.frame.Show()

        if cellprofiler.preferences.get_telemetry_prompt():
            telemetry = cellprofiler.gui.dialog.Telemetry()

            if telemetry.status == wx.ID_YES:
                cellprofiler.preferences.set_telemetry(True)
            else:
                cellprofiler.preferences.set_telemetry(False)

            cellprofiler.preferences.set_telemetry_prompt(False)

        if cellprofiler.preferences.get_telemetry():
            sentry_handler = raven.handlers.logging.SentryHandler(sentry)

            sentry_handler.setLevel(logging.ERROR)

            raven.conf.setup_logging(sentry_handler)

        if self.frame.startup_blurb_frame.IsShownOnScreen():
            self.frame.startup_blurb_frame.Raise()

        return True

    def OnExit(self):
        cellprofiler.utilities.cpjvm.cp_stop_vm()


if __name__ == "__main__":
    app = App(False)

    app.MainLoop()
