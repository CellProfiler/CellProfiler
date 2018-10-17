import wx

import cellprofiler.gui


class Menu(wx.Menu):
    def __init__(self, frame):
        self.frame = frame

        super(Menu, self).__init__()

    def append(self, title, contents=None, event_fn=None):
        id = wx.NewId()

        self.Append(id, title)

        if event_fn:
            wx.EVT_MENU(self, id, event_fn)
        elif contents:
            wx.EVT_MENU(self, id, lambda _: self.__show_dialog(title, contents))

    def __show_dialog(self, title, contents):
        help_dialog = cellprofiler.gui.htmldialog.HTMLDialog(
            self.frame,
            title,
            cellprofiler.gui.html.utils.rst_to_html_fragment(contents)
        )

        help_dialog.Show()
