import wx


class TelemetryMessageDialog(wx.MessageDialog):
    def __init__(self):
        message = "Send diagnostic information to the CellProfiler Team"

        super(TelemetryMessageDialog, self).__init__(
            message=message, parent=None, style=wx.YES_NO | wx.ICON_QUESTION
        )

        extended_message = (
            "Allow limited and anonymous usage statistics and "
            "exception reports to be sent to the CellProfiler "
            "team to help improve CellProfiler.\n\n"
            "(You can always update this setting in your "
            "CellProfiler preferences.)"
        )

        self.SetExtendedMessage(extended_message)

        self.SetYesNoLabels(
            "Send diagnostic information", "Stop sending diagnostic information"
        )

        self.status = self.ShowModal()

        self.Destroy()
