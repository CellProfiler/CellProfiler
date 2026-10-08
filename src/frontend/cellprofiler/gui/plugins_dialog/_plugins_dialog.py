import logging
import os
import threading

import wx

from cellprofiler_core.preferences import get_plugin_directory
from cellprofiler_core.utilities.core.plugins import (
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

        self.image_list = wx.ImageList(16, 16)
        self.image_ok = self.image_list.Add(get_builtin_image("IMG_DISABLED").ConvertToBitmap())
        self.image_error = self.image_list.Add(get_builtin_image("IMG_ERROR").ConvertToBitmap())

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

        self.list_ctrl = wx.ListCtrl(self, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        self.list_ctrl.SetImageList(self.image_list, wx.IMAGE_LIST_SMALL)
        self.list_ctrl.InsertColumn(0, "")
        self.list_ctrl.InsertColumn(1, "Plugin")
        self.list_ctrl.InsertColumn(2, "Type")
        self.list_ctrl.InsertColumn(3, "Source")
        self.list_ctrl.SetColumnWidth(0, 28)
        self.list_ctrl.SetColumnWidth(1, 300)
        self.list_ctrl.SetColumnWidth(2, 100)
        self.list_ctrl.SetColumnWidth(3, 100)
        main_sizer.Add(self.list_ctrl, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)
        self.Bind(wx.EVT_LIST_ITEM_SELECTED, self.on_select, self.list_ctrl)
        self.Bind(wx.EVT_LIST_ITEM_DESELECTED, self.on_select, self.list_ctrl)

        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.get_info_button = wx.Button(self, label="Get Info...")
        self.get_info_button.Disable()
        self.Bind(wx.EVT_BUTTON, self.on_get_info, self.get_info_button)
        button_sizer.Add(self.get_info_button, 0, wx.RIGHT, 10)
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
        user_directory = get_plugin_directory()
        official_directory = get_official_plugins_directory()
        for status in get_plugin_statuses():
            source_label = self._source_label(status["directory"], user_directory, official_directory)
            self.add_row(source_label, status)
        self.get_info_button.Disable()

    @staticmethod
    def _source_label(directory, user_directory, official_directory):
        if directory is None:
            return "Unknown"
        real = os.path.realpath(directory)
        if user_directory and real == os.path.realpath(user_directory):
            return "User"
        if real == os.path.realpath(official_directory):
            return "Official"
        return "Unknown"

    def add_row(self, source_label, status):
        row_index = self.list_ctrl.GetItemCount()
        image_index = self.image_ok if status["loaded"] else self.image_error
        self.list_ctrl.InsertItem(row_index, "", image_index)
        self.list_ctrl.SetItem(row_index, 1, status["name"])
        self.list_ctrl.SetItem(row_index, 2, (status["kind"] or "unknown").capitalize())
        self.list_ctrl.SetItem(row_index, 3, source_label)
        self.rows.append(status)

    def on_select(self, event):
        selected = self.list_ctrl.GetFirstSelected()
        can_show_info = selected != -1 and not self.rows[selected]["loaded"]
        self.get_info_button.Enable(can_show_info)

    def on_get_info(self, event):
        selected = self.list_ctrl.GetFirstSelected()
        if selected == -1:
            return
        status = self.rows[selected]
        message = status["error"] or "No additional information is available."
        display_error_message(self, message, title=f"Error loading {status['name']}")

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
