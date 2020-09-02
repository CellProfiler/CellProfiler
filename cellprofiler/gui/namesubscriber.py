# coding=utf-8
"""namesubscriber.py - implements a combobox with extra information
"""

import platform

import wx


def align_twosided_items(
    parent, items, min_spacing=8, left_texts=None, right_texts=None
):
    """Find spacing for a list of pairs of text such that the left texts are
    left justified and the right texts (roughly) right justified.
    """
    if right_texts is None:
        right_texts = []
    if left_texts is None:
        left_texts = []
    if items:
        if wx.Platform == "__WXMSW__":
            # ignore minspacing for windows
            for item, left, right in zip(items, left_texts, right_texts):
                item.SetItemLabel("%s\t%s" % (left, right))
        else:
            # Mac and linux use spaces to align.
            widths = [
                parent.GetTextExtent("%s%s%s" % (left, " " * min_spacing, right))[0]
                for left, right in zip(left_texts, right_texts)
            ]
            maxwidth = max(widths)
            spacewidth = parent.GetTextExtent("  ")[0] - parent.GetTextExtent(" ")[0]
            for item, left, right, initial_width in zip(
                items, left_texts, right_texts, widths
            ):
                numspaces = int(min_spacing + (maxwidth - initial_width) / spacewidth)
                item.SetItemLabel("%s%s%s" % (left, " " * numspaces, right))


class NameSubscriberComboBox(wx.Panel):
    """A read-only combobox with extra annotation, and a context menu.

    Mostly the same interface as wx.ComboBox, but choices is a list of (Name,
    Parent, modulenum).
    """

    def __init__(self, annotation, choices=None, value="", name=""):
        wx.Panel.__init__(self, annotation, name=name)
        if choices is None:
            choices = []
        self.orig_choices = choices
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.combo_dlg = wx.ComboBox(
            self,
            choices=[choice[0] for choice in choices],
            value=value,
            style=wx.CB_READONLY,
        )
        self.annotation_dlg = wx.StaticText(self, label="", style=wx.ST_NO_AUTORESIZE)
        self.annotation_dlg.MinSize = (
            max(
                [
                    self.annotation_dlg.GetFullTextExtent(choice[1] + " (from #00)")[0]
                    for choice in self.orig_choices
                ]
            ),
            -1,
        )
        self.update_annotation()

        sizer.AddStretchSpacer()
        sizer.Add(self.combo_dlg, flag=wx.ALL | wx.EXPAND, border=3)
        sizer.Add((5, 5))
        sizer.Add(self.annotation_dlg, flag=wx.ALIGN_CENTER)
        sizer.AddStretchSpacer()
        self.SetSizer(sizer)

        self.combo_dlg.Bind(wx.EVT_COMBOBOX, self.choice_made)
        self.combo_dlg.Bind(wx.EVT_RIGHT_DOWN, self.right_menu)
        self.combo_dlg.Bind(wx.EVT_MOUSEWHEEL, self.ignore_mousewheel)
        for child in self.combo_dlg.Children:
            # Mac implements read_only combobox as a choice in a child
            child.Bind(wx.EVT_RIGHT_DOWN, self.right_menu)
        self.callbacks = []

    def ignore_mousewheel(self, event):
        return

    def choice_made(self, evt):
        choice = self.orig_choices[self.combo_dlg.GetSelection()]
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
        self.annotation_dlg.Label = ""
        if self.orig_choices:
            ch = self.orig_choices[self.combo_dlg.GetSelection()]
            self.annotation_dlg.Label = self.get_choice_label(ch)

    def right_menu(self, evt):
        menu = wx.Menu()
        all_menu = wx.Menu()

        def fn_key(x):
            return x[2], x

        def on_selection(event):
            self.Value = index[event.Id]
            self.choice_made(event)

        choices_sorted_by_num = sorted(self.orig_choices, key=fn_key)
        index = {}
        for idx, choice in enumerate(choices_sorted_by_num):
            choices_sorted_by_num[idx] = choice + (idx,)
            all_menu.Append(idx, "filler")
            index[idx] = choice[0]

        align_twosided_items(
            self.combo_dlg,
            all_menu.MenuItems,
            left_texts=[name for name, _, _, _, _ in choices_sorted_by_num],
            right_texts=[
                self.get_choice_label(choice) for choice in choices_sorted_by_num
            ],
        )

        submenus = {}
        for name, annotation, num, is_input_module, choiceid in choices_sorted_by_num:
            if not annotation:
                continue
            key = (num, annotation, is_input_module)
            if key not in submenus:
                submenus[key] = wx.Menu()
            submenus[key].Append(choiceid, name)
        menu.Append(wx.ID_ANY, "All", all_menu)
        sorted_submenus = sorted(submenus.items())
        for (num, annotation, is_input_module), submenu in sorted_submenus:
            menu.Append(wx.ID_ANY, "filler", submenu)
        align_twosided_items(
            self.combo_dlg,
            menu.MenuItems,
            left_texts=["All"] + [k[1] for k, v in sorted_submenus],
            right_texts=[""]
            + [
                "  " if is_input_module else "#%02d" % num
                for (num, annotation, is_input_module), v in sorted_submenus
            ],
        )
        menu.Bind(wx.EVT_MENU, on_selection)
        self.PopupMenu(menu)
        menu.Destroy()

    def GetItems(self):
        return self.orig_choices

    def SetItems(self, choices):
        self.orig_choices = choices
        current = self.Value
        self.combo_dlg.SetItems([name for name, _, _, _ in choices])
        # on Mac, changing the items clears the current selection
        self.SetValue(current)
        self.update_annotation()
        self.Refresh()

    Items = property(GetItems, SetItems)

    def GetValue(self):
        return self.combo_dlg.GetValue()

    def SetValue(self, value):
        self.combo_dlg.SetValue(value)
        self.update_annotation()
        self.Refresh()

    Value = property(GetValue, SetValue)


class NameSubscriberListBox(wx.Panel):
    """A list of checkboxes with extra annotation, and a context menu.

    Designed as an alternative to NameSubscriberCombobox which simplifies selection of
    multiple items.
    """

    def __init__(self, annotation, choices=None, checked=[], name="", nametype="Image"):
        wx.Panel.__init__(self, annotation, name=name)
        self.choices = choices
        if self.choices is None:
            self.choices = []
        if platform.system() == "darwin":
            self.text_width = 50
        else:
            self.text_width = 90
        self.checked = checked
        self.choice_names = self.get_choice_names()
        self.nametype = nametype
        self.list_dlg = wx.CheckListBox(self, style=wx.LB_NEEDED_SB | wx.LB_HSCROLL)
        self.list_dlg.SetMinSize((350, 125))
        self.SetItems(self.choices)
        self.SetChecked(self.checked)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.list_dlg, 0, flag=wx.ALL | wx.EXPAND, border=3)
        self.SetSizer(sizer)
        self.callbacks = []
        self.list_dlg.Bind(wx.EVT_CHECKLISTBOX, self.choice_made)
        self.list_dlg.Bind(wx.EVT_CONTEXT_MENU, self.right_menu)
        self.list_dlg.Bind(wx.EVT_LISTBOX, self.item_selected)

    def add_callback(self, cb):
        self.callbacks.append(cb)

    def right_menu(self, evt):
        menu = wx.Menu()
        sel_all = wx.MenuItem(menu, wx.NewId(), "Select All")
        sel_none = wx.MenuItem(menu, wx.NewId(), "Select None")
        force_refresh = wx.MenuItem(menu, wx.NewId(), "Refresh List")
        menu.Append(sel_all)
        menu.Append(sel_none)
        menu.Append(force_refresh)
        menu.Bind(wx.EVT_MENU, self.select_all, sel_all)
        menu.Bind(wx.EVT_MENU, self.select_none, sel_none)
        menu.Bind(wx.EVT_MENU, self.force_refresh, force_refresh)
        self.PopupMenu(menu)
        menu.Destroy()

    def select_all(self, evt):
        self.SetChecked(self.choice_names)
        self.checked = self.GetChecked()
        for cb in self.callbacks:
            cb(evt)

    def select_none(self, evt):
        self.SetChecked([])
        self.checked = self.GetChecked()
        for cb in self.callbacks:
            cb(evt)

    def force_refresh(self, evt):
        evt.refresh_now = True
        for cb in self.callbacks:
            cb(evt)

    def item_selected(self, evt):
        selected = self.choice_names[evt.Selection]
        if self.list_dlg.IsChecked(evt.Selection):
            self.checked.remove(selected)
        else:
            self.checked.append(selected)
        self.SetChecked(self.checked)
        self.checked = self.GetChecked()
        for cb in self.callbacks:
            cb(evt)
        self.list_dlg.Deselect(evt.Selection)

    def choice_made(self, evt):
        # Prevent selection of duplicate entires, marked with "Duplicate ___ Name!"
        if self.list_dlg.GetItems()[evt.Selection].endswith("Name!)"):
            if self.list_dlg.IsChecked(evt.Selection):
                self.list_dlg.Check(evt.Selection, False)
            return
        for cb in self.callbacks:
            cb(evt)
        self.Refresh()
        self.checked = self.GetChecked()

    def get_choice_labels(self):
        choice_labels = []
        for choice in self.choices:
            name, module_name, module_num, is_input_module = choice[:4]
            if module_name:
                if is_input_module:
                    end = "(from %s)" % module_name
                else:
                    end = "(from %s #%02d)" % (module_name, module_num)
            else:
                end = "(%s Missing!)" % self.nametype
            whitespace = " " * max(10, (self.text_width - len(name) - len(end)))
            choice_label = "".join((name, whitespace, end))
            if choice_label in choice_labels:
                # Name is duplicated
                end = "(Duplicate %s Name!)" % self.nametype
                whitespace = " " * max(10, (self.text_width - len(name) - len(end)))
                choice_label = "".join((name, whitespace, end))
            choice_labels.append(choice_label)
        return choice_labels

    def get_choice_names(self):
        choice_names = [choice[0] for choice in self.choices]
        if self.checked != "None":
            for item in self.checked:
                if item not in choice_names:
                    choice_names.insert(0, item)
                    self.choices.insert(0, (item, None, 0, True))
        return choice_names

    def GetItems(self):
        return self.list_dlg.GetItems()

    def SetItems(self, choices):
        self.choices = choices
        self.choice_names = self.get_choice_names()
        self.list_dlg.SetItems(self.get_choice_labels())
        labels = self.get_choice_labels()
        for i in range(len(self.choices)):
            if self.choices[i][1] is None:
                # Tag missing items
                self.list_dlg.SetItemBackgroundColour(i, "pink")
            elif labels[i].endswith("Name!)"):
                # Tag duplicated items
                self.list_dlg.SetItemForegroundColour(i, "grey")
        # on Mac, changing the items clears the current selection
        self.SetChecked(self.checked)
        self.Refresh()

    Items = property(GetItems, SetItems)

    def GetChecked(self):
        checked_indexes = self.list_dlg.GetCheckedItems()
        named_items = [self.choice_names[i] for i in checked_indexes]
        return named_items

    def SetChecked(self, values):
        if values != "None":
            selections = [self.choice_names.index(i) for i in values]
            # Filter out invalid duplicate entries
            labels = self.list_dlg.GetItems()
            selections = [i for i in selections if not labels[i].endswith("Name!)")]
            self.list_dlg.SetCheckedItems(selections)
        self.Refresh()

    Checked = property(GetChecked, SetChecked)
