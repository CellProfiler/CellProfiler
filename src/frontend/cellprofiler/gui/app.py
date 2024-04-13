# coding=utf-8

import platform
import sys

import sentry_sdk
import wx
import wx.lib.inspection
from cellprofiler_core.preferences import get_telemetry_prompt
from cellprofiler_core.preferences import get_telemetry
from cellprofiler_core.preferences import set_telemetry
from cellprofiler_core.preferences import set_telemetry_prompt
from cellprofiler_core.utilities.java import stop_java

from cellprofiler import __version__ as cellprofiler_version
from .dialog import Telemetry

def init_telemetry():
    dsn = "https://c0b47db2a1b34f12b33ca8e78067617e:3cee11601374464dadd4b44da8a22dbd@sentry.io/152399"

    sentry = sentry_sdk.init(dsn=dsn, release=cellprofiler_version)

    sentry_sdk.set_user(
        {
            "architecture": platform.architecture(),
            "machine": platform.machine(),
            "node": platform.node(),
            "processor": platform.processor(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "release": platform.release(),
            "system": platform.system(),
            "version": platform.version(),
        }
    )

def stop_telemetry():
    sentry = sentry_sdk.init()

if get_telemetry():
    init_telemetry()

class App(wx.App):
    def __init__(self, *args, **kwargs):
        self.abort_initialization = False

        self.frame = None

        self.pipeline_path = kwargs.pop("pipeline_path", None)

        self.workspace_path = kwargs.pop("workspace_path", None)

        super(App, self).__init__(*args, **kwargs)

    def OnInit(self):
        if platform.system() == "Windows":
            import locale
            # Need to startup wx in English, otherwise C++ can't load images.
            self.locale = wx.Locale(wx.LANGUAGE_ENGLISH)
            # Ensure Python uses the same locale as wx
            try:
                locale.setlocale(locale.LC_ALL, self.locale.GetName())
            except:
                print(f"Python rejected the system locale detected by WX ('{self.locale.GetName()}').\n"
                      "This shouldn't cause problems, but please let us know if you encounter errors.")
        from .cpframe import CPFrame

        # This import is needed to populate the modules list
        import cellprofiler_core.modules

        # wx.lib.inspection.InspectionTool().Show()

        self.SetAppName("CellProfiler{0:s}".format(cellprofiler_version))

        self.frame = CPFrame(None, -1, "CellProfiler")

        self.frame.start(self.workspace_path, self.pipeline_path)

        if self.abort_initialization:
            return False

        self.SetTopWindow(self.frame)

        self.frame.Show()

        if hasattr(sys, "frozen"):
            from cellprofiler.gui.checkupdate import check_update
            try:
                check_update(self.frame)
            except Exception as e:
                print(f"Failed to check for updates - {e}")

        if get_telemetry_prompt():
            telemetry = Telemetry()

            if telemetry.status == wx.ID_YES:
                set_telemetry(True)
                init_telemetry()
            else:
                set_telemetry(False)
                stop_telemetry()

            set_telemetry_prompt(False)

        if self.frame.startup_blurb_frame is not None and self.frame.startup_blurb_frame.IsShownOnScreen():
            self.frame.startup_blurb_frame.Raise()

        return True

    def OnExit(self):
        stop_java()

        return 0


if __name__ == "__main__":
    app = App(False)

    app.MainLoop()
