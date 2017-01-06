# coding=utf-8

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *
import wx


class AboutDialogInfo(wx.AboutDialogInfo):
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
