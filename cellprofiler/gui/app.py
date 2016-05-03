# coding=utf-8

import cellprofiler.gui.dialog
import cellprofiler.preferences
import cellprofiler.utilities.thread_excepthook
import os
import os.path
import pip
import platform
import raven
import sys
import wx


cellprofiler.utilities.thread_excepthook.install_thread_sys_excepthook()


class App(wx.App):
    def __excepthook__(self, exception, message, tracback):
        def callback():
            modules = []

            for module in self.frame.pipeline.modules():
                description = module.__class__.__name__

                modules.append(description)

            pipeline = self.frame.pipeline

            pathnames = []

            for pathname in pipeline.file_list:
                description = os.path.basename(pathname)

                pathnames.append(description)

            self.client.captureException(
                exc_info=(exception, message, tracback),
                extra={
                    "modules": modules,
                    "pathnames": pathnames
                }
            )

            capitalized_description = message.message.capitalize()

            description = "{}.".format(capitalized_description)

            error = cellprofiler.gui.dialog.Error("Error", description)

            if error.status is wx.ID_CANCEL:
                cellprofiler.preferences.cancel_progress()

        wx.CallAfter(callback)

    def __init__(self, *args, **kwargs):
        self.abort_initialization = False

        path = os.path.join(os.path.dirname(__file__), os.pardir)

        try:
            self.release = raven.fetch_git_sha(path)
        except raven.versioning.InvalidGitRepository:
            self.release = raven.fetch_package_version("cellprofiler")

        dsn = "https://3d53494dbaaf4e858afd79f56506a749:8a7a767a1924423f89c1fdfd69717fd5@app.getsentry.com/70887"

        self.client = raven.Client(dsn=dsn, release=self.release)

        installed_distributions = []

        for distribution in pip.get_installed_distributions():
            description = "{}=={}".format(distribution.key, distribution.version)

            installed_distributions.append(description)

        installed_distributions = sorted(installed_distributions)

        self.client.user_context({
            "installed_distributions": installed_distributions,
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "release": platform.release(),
            "system": platform.system(),
            "version": platform.version()
        })

        self.frame = None

        self.original_excepthook = sys.excepthook

        self.pipeline_path = kwargs.pop('pipeline_path', None)

        self.workspace_path = kwargs.pop('workspace_path', None)

        super(App, self).__init__(*args, **kwargs)

    def OnInit(self):
        import cellprofiler.gui.cpframe

        name = "CellProfiler ({})".format(self.release)

        self.SetAppName(name)

        self.frame = cellprofiler.gui.cpframe.CPFrame(None, -1, "Cell Profiler")

        self.frame.start(self.workspace_path, self.pipeline_path)

        if self.abort_initialization:
            return 0

        sys.excepthook = self.__excepthook__

        self.SetTopWindow(self.frame)

        self.frame.Show()

        if self.frame.startup_blurb_frame.IsShownOnScreen():
            self.frame.startup_blurb_frame.Raise()

        return 1

    def OnExit(self):
        try:
            import imagej.imagej2

            imagej.imagej2.allow_quit()
        except ImportError:
            imagej = None

        sys.excepthook = self.original_excepthook
