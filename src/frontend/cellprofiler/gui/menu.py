import wx

import cellprofiler.gui.html.utils
import cellprofiler.gui.htmldialog
from cellprofiler.gui.i18n import _


class Menu(wx.Menu):
    def __init__(self, frame):
        self.frame = frame
        self._translatable_items = []
        self._translatable_submenus = []

        super(Menu, self).__init__()

    def append(self, title, contents=None, event_fn=None):
        item_id = wx.NewId()
        translated_title = _(title)

        self.Append(item_id, translated_title)
        self._translatable_items.append((item_id, title))

        if event_fn:
            self.Bind(wx.EVT_MENU, event_fn, id=item_id)
        elif contents:
            self.Bind(
                wx.EVT_MENU,
                lambda _: self.__show_dialog(translated_title, contents),
                id=item_id,
            )

    def append_submenu(self, submenu, title):
        translated_title = _(title)
        self.AppendSubMenu(submenu, translated_title)
        item = self.FindItemByPosition(self.GetMenuItemCount() - 1)
        self._translatable_submenus.append((item.GetId(), title))

    def refresh_translations(self):
        for item_id, title in self._translatable_items:
            self.SetLabel(item_id, _(title))
        for item_id, title in self._translatable_submenus:
            self.SetLabel(item_id, _(title))

    def __show_dialog(self, title, contents):
        help_dialog = cellprofiler.gui.htmldialog.HTMLDialog(
            self.frame,
            title,
            cellprofiler.gui.html.utils.rst_to_html_fragment(contents),
        )

        help_dialog.Show()
