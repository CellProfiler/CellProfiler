# coding=utf-8

import wx
import wx.adv


class AboutDialogInfo(wx.adv.AboutDialogInfo):
    def __init__(self):
        super(AboutDialogInfo, self).__init__()

        self.Copyright = u"Copyright © 2003 - 2017 Broad Institute, Inc.\nAll rights reserved."

        self.Name = "CellProfiler"

        self.Version = "Nightly"


class Error(wx.MessageDialog):
    def __init__(self, message, extended_message):
        super(Error, self).__init__(parent=None, message=message, style=wx.CANCEL | wx.ICON_EXCLAMATION)

        self.SetExtendedMessage(extended_message)

        self.SetOKLabel("Continue Processing")
