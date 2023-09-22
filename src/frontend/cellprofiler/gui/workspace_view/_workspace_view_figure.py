import wx

from ..figure import Figure


class WorkspaceViewFigure(Figure):
    def on_close(self, event):
        """Hide instead of close"""
        if isinstance(event, wx.CloseEvent):
            event.Veto()
        self.Hide()
