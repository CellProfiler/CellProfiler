import wx

import cellprofiler.gui.figure


class Figure(cellprofiler.gui.figure.Figure):
    def on_close(self, event):
        """Hide instead of close"""
        if isinstance(event, wx.CloseEvent):
            event.Veto()
        self.Hide()
