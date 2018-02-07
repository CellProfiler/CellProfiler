# coding=utf-8
"""namesubscriber.py - implements a combobox with extra information
"""

import wx


def align_twosided_items(parent, items, min_spacing=8, left_texts=None, right_texts=None):
    """Find spacing for a list of pairs of text such that the left texts are
    left justified and the right texts (roughly) right justified.
    """
    if right_texts is None:
        right_texts = []
    if left_texts is None:
        left_texts = []
    if items:
        if wx.Platform == '__WXMSW__':
            # ignore minspacing for windows
            for item, left, right in zip(items, left_texts, right_texts):
                item.SetItemLabel("%s\t%s" % (left, right))
        else:
            # Mac and linux use spaces to align.
            widths = [parent.GetTextExtent("%s%s%s" % (left, " " * min_spacing, right))[0]
                      for left, right in zip(left_texts, right_texts)]
            maxwidth = max(widths)
            spacewidth = parent.GetTextExtent("  ")[0] - parent.GetTextExtent(" ")[0]
            for item, left, right, initial_width in \
                    zip(items, left_texts, right_texts, widths):
                numspaces = int(min_spacing + (maxwidth - initial_width) / spacewidth)
                item.SetItemLabel("%s%s%s" % (left, ' ' * numspaces, right))


class NameSubscriberComboBox(wx.Panel):
    """A read-only combobox with extra annotation, and a context menu.

    Mostly the same interface as wx.ComboBox, but choices is a list of (Name,
    Parent, modulenum).
    """

    def __init__(self, annotation, choices=None, value='', name=''):
        wx.Panel.__init__(self, annotation, name=name)
        if choices is None:
            choices = []
        self.orig_choices = choices
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.combo_dlg = wx.ComboBox(
                self, choices=[choice[0] for choice in choices],
                value=value, style=wx.CB_READONLY)
        self.annotation_dlg = wx.StaticText(self, label='', style=wx.ST_NO_AUTORESIZE)
        self.annotation_dlg.MinSize = (
            max([self.annotation_dlg.GetTextExtent(choice[1] + " (from #00)")[0]
                 for choice in self.orig_choices]), -1)
        self.update_annotation()

        sizer.AddStretchSpacer()
        sizer.Add(self.combo_dlg, flag=wx.ALL | wx.EXPAND | wx.ALIGN_CENTER, border=3)
        sizer.Add((5, 5))
        sizer.Add(self.annotation_dlg, flag=wx.ALIGN_CENTER)
        sizer.AddStretchSpacer()
        self.SetSizer(sizer)

        self.combo_dlg.Bind(wx.EVT_COMBOBOX, self.choice_made)
        self.combo_dlg.Bind(wx.EVT_RIGHT_DOWN, self.right_menu)
        for child in self.combo_dlg.Children:
            # Mac implements read_only combobox as a choice in a child
            child.Bind(wx.EVT_RIGHT_DOWN, self.right_menu)
        self.callbacks = []

    def choice_made(self, evt):
        choice = self.orig_choices[self.combo_dlg.Selection]
        self.update_annotation()
        for cb in self.callbacks:
            cb(evt)
        self.Refresh()

    def add_callback(self, cb):
        self.callbacks.append(cb)

    @staticmethod
    def get_choice_label(choice):
        name, module_name, module_num, is_input_module = choice[:4]
        if module_name:
            if is_input_module:
                return "(from %s)" % module_name
            return "(from %s #%02d)" % (module_name, module_num)
        return ""

    def update_annotation(self):
        self.annotation_dlg.Label = ''
        if self.orig_choices:
            ch = self.orig_choices[self.combo_dlg.Selection]
            self.annotation_dlg.Label = self.get_choice_label(ch)

    def right_menu(self, evt):
        menu = wx.Menu()
        all_menu = wx.Menu()

        fn_key = lambda x: (x[2], x)
        choices_sorted_by_num = sorted(self.orig_choices, key=fn_key)
        for name, annotation, num, is_input_module, choiceid in \
                choices_sorted_by_num:
            all_menu.Append(choiceid, "filler")

        align_twosided_items(
                self.combo_dlg,
                all_menu.MenuItems,
                left_texts=[name for name, _, _, _, _ in choices_sorted_by_num],
                right_texts=[self.get_choice_label(choice)
                             for choice in choices_sorted_by_num])

        submenus = {}
        for name, annotation, num, is_input_module, choiceid \
                in choices_sorted_by_num:
            if not annotation:
                continue
            key = (num, annotation, is_input_module)
            if key not in submenus:
                submenus[key] = wx.Menu()
            submenus[key].Append(choiceid, name)
        menu.AppendMenu(wx.ID_ANY, "All", all_menu)
        sorted_submenus = sorted(submenus.items())
        for (num, annotation, is_input_module), submenu in sorted_submenus:
            menu.AppendMenu(wx.ID_ANY, "filler", submenu)
        align_twosided_items(
                self.combo_dlg,
                menu.MenuItems,
                left_texts=['All'] + [k[1] for k, v in sorted_submenus],
                right_texts=[''] + [
                    "  " if is_input_module else "#%02d" % num
                    for (num, annotation, is_input_module), v
                    in sorted_submenus])
        self.PopupMenu(menu)
        menu.Destroy()

    def GetItems(self):
        return self.orig_choices

    def SetItems(self, choices):
        self.orig_choices = choices
        current = self.Value
        self.combo_dlg.Items = [name for name, _, _, _ in choices]
        # on Mac, changing the items clears the current selection
        self.SetValue(current)
        self.update_annotation()
        self.Refresh()

    Items = property(GetItems, SetItems)

    def GetValue(self):
        return self.combo_dlg.Value

    def SetValue(self, value):
        self.combo_dlg.Value = value
        self.update_annotation()
        self.Refresh()

    Value = property(GetValue, SetValue)
