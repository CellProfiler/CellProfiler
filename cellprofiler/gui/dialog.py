# coding=utf-8

import raven
import wx


class AboutDialogInfo(wx.AboutDialogInfo):
    def __init__(self):
        super(AboutDialogInfo, self).__init__()

        self.Copyright = "Copyright © 2003 - 2016 Broad Institute, Inc.\nAll rights reserved."

        self.Name = "CellProfiler"

        self.Version = raven.fetch_package_version("cellprofiler")


class Error(wx.MessageDialog):
    def __init__(self, message, extended_message):
        super(Error, self).__init__(parent=None, message=message, style=wx.CANCEL | wx.ICON_EXCLAMATION)

        self.SetExtendedMessage(extended_message)

        self.SetOKLabel("Continue Processing")

        self.status = self.ShowModal()

        self.Destroy()
