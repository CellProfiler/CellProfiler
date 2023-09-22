import wx

import cellprofiler.gui.html.utils
import cellprofiler.gui.htmldialog


class Menu(wx.Menu):
    def __init__(self, frame):
        self.frame = frame

        super(Menu, self).__init__()

    def append(self, title, contents=None, event_fn=None):
        id = wx.NewId()

        self.Append(id, title)

        if event_fn:
            self.Bind(wx.EVT_MENU, event_fn, id=id)
        elif contents:
            self.Bind(wx.EVT_MENU, lambda _: self.__show_dialog(title, contents), id=id)

    def __show_dialog(self, title, contents):
        help_dialog = cellprofiler.gui.htmldialog.HTMLDialog(
            self.frame,
            title,
            cellprofiler.gui.html.utils.rst_to_html_fragment(contents),
        )

        help_dialog.Show()
