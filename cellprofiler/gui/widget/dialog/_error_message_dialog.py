import wx


class ErrorMessageDialog(wx.MessageDialog):
    def __init__(self, message, extended_message=""):
        super(ErrorMessageDialog, self).__init__(
            parent=None, message=message, style=wx.CANCEL | wx.ICON_EXCLAMATION
        )

        self.SetExtendedMessage(extended_message)

        self.SetOKLabel("Continue Processing")

        self.status = self.ShowModal()

        self.Destroy()
