import logging
import threading

import wx

from cellprofiler_core.utilities.appose import (
    environment_exists,
    get_environment,
    rebuild_environment,
)
from cellprofiler_core.utilities.core.plugins import (
    classify_plugin_directory,
    get_official_plugins_directory,
    official_plugins_repo_exists,
    download_official_plugins_repo,
    get_plugin_statuses,
    load_plugins,
)

from cellprofiler.icons import get_builtin_image
from ..errordialog import display_error_message

LOGGER = logging.getLogger(__name__)


class PluginsDialog(wx.Dialog):
    """
    Dialog for managing CellProfiler plugins.

    Lets the user fetch the officially-supported plugin set from the
    CellProfiler-plugins repository, and shows which plugins (from both
    the configured plugin directory and the official set) loaded
    successfully, with tracebacks available for the ones that didn't.

    An indicator is placed next to apposed plugins, with the ability
    to build their environments, rather than auto-building when they
    are run the first time.
    
    Environment builds have a two-phase progress bar,
    1) the dependency solve phase
    2) the download/install phase
    """

    def __init__(
        self,
        parent=None,
        ID=-1,
        title="CellProfiler Plugins",
        size=(700, 500),
        pos=wx.DefaultPosition,
        style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
        name=wx.DialogNameStr,
    ):
        wx.Dialog.__init__(self, parent, ID, title, pos, size, style, name)
        self.rows = []
        self.build_verb = "Building"

        self.image_list = wx.ImageList(16, 16)
        self.image_ok = self.image_list.Add(get_builtin_image("IMG_DISABLED").ConvertToBitmap())
        self.image_error = self.image_list.Add(get_builtin_image("IMG_ERROR").ConvertToBitmap())
        self.image_appose = self.image_list.Add(get_builtin_image("IMG_UPDATE").ConvertToBitmap())
        blank_image = wx.Image(16, 16)
        blank_image.InitAlpha()
        blank_image.SetAlpha(bytes(16 * 16))
        self.image_blank = self.image_list.Add(blank_image.ConvertToBitmap())

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(main_sizer)

        self.header_label = wx.StaticText(self)
        main_sizer.Add(self.header_label, 0, wx.EXPAND | wx.ALL, 10)

        download_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.download_button = wx.Button(self)
        self.Bind(wx.EVT_BUTTON, self.on_download, self.download_button)
        download_sizer.Add(self.download_button, 0, wx.RIGHT, 10)
        self.status_label = wx.StaticText(self)
        download_sizer.Add(self.status_label, 0, wx.ALIGN_CENTER_VERTICAL)
        main_sizer.Add(download_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        self.progress_gauge = wx.Gauge(self, range=100)
        self.progress_gauge.Hide()
        main_sizer.Add(
            self.progress_gauge, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10
        )
        # Appose reports nothing at all during pixi's solve phase, which is
        # often the longest part of a cold build, so the gauge pulses until
        # the first countable phase event arrives.
        self.pulse_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_pulse, self.pulse_timer)

        self.list_ctrl = wx.ListCtrl(self, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        self.list_ctrl.SetImageList(self.image_list, wx.IMAGE_LIST_SMALL)
        self.list_ctrl.InsertColumn(0, "")
        self.list_ctrl.InsertColumn(1, "Apposed")
        self.list_ctrl.InsertColumn(2, "Plugin")
        self.list_ctrl.InsertColumn(3, "Type")
        self.list_ctrl.InsertColumn(4, "Source")
        self.list_ctrl.SetColumnWidth(0, 28)
        self.list_ctrl.SetColumnWidth(1, 60)
        self.list_ctrl.SetColumnWidth(2, 300)
        self.list_ctrl.SetColumnWidth(3, 100)
        self.list_ctrl.SetColumnWidth(4, 100)
        main_sizer.Add(self.list_ctrl, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)
        self.Bind(wx.EVT_LIST_ITEM_SELECTED, self.on_select, self.list_ctrl)
        self.Bind(wx.EVT_LIST_ITEM_DESELECTED, self.on_select, self.list_ctrl)

        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.get_info_button = wx.Button(self, label="Get Info...")
        self.get_info_button.Disable()
        self.Bind(wx.EVT_BUTTON, self.on_get_info, self.get_info_button)
        button_sizer.Add(self.get_info_button, 0, wx.RIGHT, 10)
        self.build_env_button = wx.Button(self, label="Build Environment")
        self.build_env_button.Disable()
        self.Bind(wx.EVT_BUTTON, self.on_build_environment, self.build_env_button)
        button_sizer.Add(self.build_env_button, 0, wx.RIGHT, 10)
        self.refresh_button = wx.Button(self, label="Refresh")
        self.Bind(wx.EVT_BUTTON, self.on_refresh, self.refresh_button)
        button_sizer.Add(self.refresh_button, 0)
        button_sizer.AddStretchSpacer()
        close_button = wx.Button(self, wx.ID_CLOSE)
        self.Bind(wx.EVT_BUTTON, lambda event: self.Close(), close_button)
        button_sizer.Add(close_button, 0)
        main_sizer.Add(button_sizer, 0, wx.EXPAND | wx.ALL, 10)

        self.refresh_header()
        self.populate_list()

    def refresh_header(self):
        directory = get_official_plugins_directory()
        if official_plugins_repo_exists():
            self.header_label.SetLabel(
                f"Officially-supported plugins are installed at:\n{directory}"
            )
            self.download_button.SetLabel("Re-download / Update")
        else:
            self.header_label.SetLabel(
                "The officially-supported CellProfiler plugins are not installed.\n"
                f"They will be downloaded to:\n{directory}"
            )
            self.download_button.SetLabel("Download Official Plugins")
        self.Layout()

    def populate_list(self):
        self.list_ctrl.DeleteAllItems()
        self.rows = []
        for status in get_plugin_statuses():
            source_label = classify_plugin_directory(status["directory"])
            self.add_row(source_label, status)
        self.get_info_button.Disable()

    def add_row(self, source_label, status):
        row_index = self.list_ctrl.GetItemCount()
        image_index = self.image_ok if status["loaded"] else self.image_error
        self.list_ctrl.InsertItem(row_index, "", image_index)
        appose_image_index = self.image_appose if status.get("appose_env_spec") else self.image_blank
        self.list_ctrl.SetItemColumnImage(row_index, 1, appose_image_index)
        self.list_ctrl.SetItem(row_index, 2, status["name"])
        self.list_ctrl.SetItem(row_index, 3, (status["kind"] or "unknown").capitalize())
        self.list_ctrl.SetItem(row_index, 4, source_label)
        self.rows.append(status)

    def on_select(self, event):
        selected = self.list_ctrl.GetFirstSelected()
        can_show_info = selected != -1 and not self.rows[selected]["loaded"]
        self.get_info_button.Enable(can_show_info)
        spec = self.rows[selected].get("appose_env_spec") if selected != -1 else None
        self.build_env_button.Enable(bool(spec))
        self.build_env_button.SetLabel(
            "Rebuild Environment"
            if spec and self.environment_already_built(spec)
            else "Build Environment"
        )
        # The two labels differ in width, so the button needs re-laying out.
        self.Layout()

    @staticmethod
    def environment_already_built(spec):
        """
        Whether building `spec`'s environment would be a rebuild - i.e.
        whether Appose has already created a directory for it on disk.

        Any failure to tell (e.g. an unreadable or malformed spec file) is
        reported as "not built yet"
        """
        try:
            return environment_exists(spec)
        except Exception:
            LOGGER.warning(
                "Failed to determine environment status for %s", spec, exc_info=True
            )
            return False

    def on_get_info(self, event):
        selected = self.list_ctrl.GetFirstSelected()
        if selected == -1:
            return
        status = self.rows[selected]
        message = status["error"] or "No additional information is available."
        display_error_message(self, message, title=f"Error loading {status['name']}")

    def on_build_environment(self, event):
        selected = self.list_ctrl.GetFirstSelected()
        if selected == -1:
            return
        status = self.rows[selected]
        spec = status.get("appose_env_spec")
        if not spec:
            return
        rebuild = self.environment_already_built(spec)
        if rebuild:
            response = wx.MessageBox(
                f"This will delete the existing environment for {status['name']} and "
                "build it again from scratch, re-downloading its dependencies.\n\n"
                "Continue?",
                "Rebuild environment",
                style=wx.YES_NO | wx.ICON_WARNING,
            )
            if response != wx.YES:
                return
        self.build_env_button.Disable()
        self.refresh_button.Disable()
        self.build_verb = "Rebuilding" if rebuild else "Building"
        self.status_label.SetLabel(f"{self.build_verb} environment for {status['name']}...")
        self.progress_gauge.Show()
        self.pulse_timer.Start(100)
        self.Layout()
        threading.Thread(
            target=self._build_env_worker,
            args=(status["name"], spec, rebuild),
            daemon=True,
        ).start()

    def on_pulse(self, event):
        self.progress_gauge.Pulse()

    def _build_env_worker(self, name, spec, rebuild):
        def progress(title, current, maximum):
            wx.CallAfter(self._on_build_progress, name, title, current, maximum)

        try:
            # A rebuild discards the existing environment directory first,
            # so a build that failed partway can't leave state behind that
            # an incremental build would keep working against.
            if rebuild:
                rebuild_environment(spec, progress=progress)
            else:
                get_environment(spec, progress=progress)
            error = None
        except Exception as e:
            LOGGER.warning("Failed to build environment for %s", name, exc_info=True)
            error = str(e)
        wx.CallAfter(self._on_build_environment_complete, name, error)

    def _on_build_progress(self, name, title, current, maximum):
        # Queued from Appose's own threads, so the dialog may already be
        # gone by the time this runs; the build itself carries on either way.
        if not self:
            return
        if maximum > 0:
            self.pulse_timer.Stop()
            self.progress_gauge.SetRange(maximum)
            self.progress_gauge.SetValue(min(current, maximum))
            self.status_label.SetLabel(
                f"{self.build_verb} {name}: {title} ({current}/{maximum})"
            )
        else:
            self.status_label.SetLabel(f"{self.build_verb} {name}: {title}")
        self.Layout()

    def _on_build_environment_complete(self, name, error):
        if not self:
            return
        self.pulse_timer.Stop()
        self.progress_gauge.Hide()
        self.refresh_button.Enable()
        self.Layout()
        self.on_select(None)
        if error:
            self.status_label.SetLabel("Build failed.")
            wx.MessageBox(
                f"Failed to build environment for {name}:\n{error}",
                "Build failed",
                style=wx.ICON_ERROR,
            )
            return
        self.status_label.SetLabel(f"Environment for {name} is up to date.")

    def on_refresh(self, event):
        load_plugins()
        self.refresh_header()
        self.populate_list()

    def on_download(self, event):
        directory = get_official_plugins_directory()
        if official_plugins_repo_exists():
            response = wx.MessageBox(
                "This will replace the existing contents of:\n"
                f"{directory}\n\nContinue?",
                "Update official plugins",
                style=wx.YES_NO | wx.ICON_WARNING,
            )
            if response != wx.YES:
                return
        self.download_button.Disable()
        self.refresh_button.Disable()
        self.status_label.SetLabel("Downloading official plugins...")
        threading.Thread(target=self._download_worker, daemon=True).start()

    def _download_worker(self):
        try:
            download_official_plugins_repo()
            error = None
        except Exception as e:
            LOGGER.warning("Failed to download official plugins", exc_info=True)
            error = str(e)
        wx.CallAfter(self._on_download_complete, error)

    def _on_download_complete(self, error):
        self.download_button.Enable()
        self.refresh_button.Enable()
        if error:
            self.status_label.SetLabel("Download failed.")
            wx.MessageBox(
                f"Failed to download official plugins:\n{error}",
                "Download failed",
                style=wx.ICON_ERROR,
            )
            return
        self.status_label.SetLabel("")
        load_plugins()
        self.refresh_header()
        self.populate_list()
