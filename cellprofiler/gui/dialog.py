# coding=utf-8

import raven
import wx


class AboutDialogInfo(wx.AboutDialogInfo):
    def __init__(self):
        super(AboutDialogInfo, self).__init__()

        self.Copyright = u"Copyright © 2003 - 2018 Broad Institute, Inc." \
                         u"\nAll rights reserved."

        self.Name = "CellProfiler"

        self.Version = raven.fetch_package_version("cellprofiler")


class Error(wx.MessageDialog):
    def __init__(self, message, extended_message=""):
        super(Error, self).__init__(
            parent=None,
            message=message,
            style=wx.CANCEL | wx.ICON_EXCLAMATION
        )

        self.SetExtendedMessage(extended_message)

        self.SetOKLabel("Continue Processing")

        self.status = self.ShowModal()

        self.Destroy()


class Telemetry(wx.MessageDialog):
    def __init__(self):
        message = "Send diagnostic information to the CellProfiler Team"

        super(Telemetry, self).__init__(
            message=message,
            parent=None,
            style=wx.YES_NO | wx.ICON_QUESTION
        )

        extended_message = "Allow limited and anonymous usage statistics and " \
                           "exception reports to be sent to the CellProfiler " \
                           "team to help improve CellProfiler.\n\n" \
                           "(You can always update this setting in your " \
                           "CellProfiler preferences.)"

        self.SetExtendedMessage(extended_message)

        self.SetYesNoLabels("Send diagnostic information", "Stop sending diagnostic information")

        self.status = self.ShowModal()

        self.Destroy()
