# coding=utf-8
"""preferencesdlg.py Edit global preferences
"""

import cellprofiler.gui.help
import cellprofiler.gui.htmldialog
import cellprofiler.preferences
import matplotlib.cm
import os
import sys
import wx

DIRBROWSE = "Browse"
FILEBROWSE = "FileBrowse"
FONT = "Font"
COLOR = "Color"
CHOICE = "Choice"


class IntegerPreference(object):
    """User interface info for an integer preference

    This signals that a preference should be displayed and edited as
    an integer, optionally limited by a range.
    """

    def __init__(self, minval=None, maxval=None):
        self.minval = minval
        self.maxval = maxval


class ClassPathValidator(wx.PyValidator):
    def __init__(self):
        wx.PyValidator.__init__(self)

    def Validate(self, win):
        ctrl = self.GetWindow()
        for c in ctrl.Value:
            if ord(c) > 254:
                wx.MessageBox(
                        "Due to technical limitations, the path to the ImageJ plugins "
                        "folder cannot contain the character, \"%s\"." % c,
                        caption="Unsupported character in path name",
                        style=wx.OK | wx.ICON_ERROR, parent=win)
                ctrl.SetFocus()
                return False
        return True

    def TransferToWindow(self):
        return True

    def TransferFromWindow(self):
        return True

    def Clone(self):
        return ClassPathValidator()


class PreferencesDlg(wx.Dialog):
    """Display a dialog for setting preferences

    The dialog handles fetching current defaults and setting the
    defaults when the user hits OK.
    """

    def __init__(self, parent=None, ID=-1, title="CellProfiler preferences",
                 size=wx.DefaultSize, pos=wx.DefaultPosition,
                 style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER |
                       wx.THICK_FRAME,
                 name=wx.DialogNameStr):
        wx.Dialog.__init__(self, parent, ID, title, pos, size, style, name)
        self.SetExtraStyle(wx.WS_EX_VALIDATE_RECURSIVELY)
        p = self.get_preferences()
        top_sizer = wx.BoxSizer(wx.VERTICAL)
        scrollpanel_sizer = wx.BoxSizer(wx.VERTICAL)
        scrollpanel = wx.lib.scrolledpanel.ScrolledPanel(self)
        scrollpanel.SetMinSize((800, 600))
        scrollpanel_sizer.Add(scrollpanel, 1, wx.EXPAND)
        scrollpanel.SetSizer(top_sizer)
        self.SetSizer(scrollpanel_sizer)

        sizer = wx.GridBagSizer(len(p), 4)
        sizer.SetFlexibleDirection(wx.HORIZONTAL)
        top_sizer.Add(sizer, 1, wx.EXPAND | wx.ALL, 5)
        index = 0
        controls = []
        help_bitmap = wx.ArtProvider.GetBitmap(wx.ART_HELP,
                                               wx.ART_CMN_DIALOG,
                                               (16, 16))
        for text, getter, setter, ui_info, help_text in p:
            text_ctl = wx.StaticText(scrollpanel, label=text)
            sizer.Add(text_ctl, (index, 0))
            if (getattr(ui_info, "__getitem__", False) and not
            isinstance(ui_info, str)):
                ctl = wx.ComboBox(scrollpanel, -1,
                                  choices=ui_info, style=wx.CB_READONLY)
                ctl.SetStringSelection(getter())
            elif ui_info == COLOR:
                ctl = wx.Panel(scrollpanel, -1, style=wx.BORDER_SUNKEN)
                ctl.BackgroundColour = getter()
            elif ui_info == CHOICE:
                ctl = wx.CheckBox(scrollpanel, -1)
                ctl.Value = getter()
            elif isinstance(ui_info, IntegerPreference):
                minval = (-sys.maxint if ui_info.minval is None
                          else ui_info.minval)
                maxval = (sys.maxint if ui_info.maxval is None
                          else ui_info.maxval)
                ctl = wx.SpinCtrl(scrollpanel,
                                  min=minval,
                                  max=maxval,
                                  initial=getter())
            else:
                validator = wx.DefaultValidator
                current = getter()
                if current is None:
                    current = ""
                ctl = wx.TextCtrl(scrollpanel, -1, current,
                                  validator=validator)
                min_height = ctl.GetMinHeight()
                min_width = ctl.GetTextExtent("Make sure the window can display this")[0]
                ctl.SetMinSize((min_width, min_height))
            controls.append(ctl)
            sizer.Add(ctl, (index, 1), flag=wx.EXPAND)
            if ui_info == DIRBROWSE:
                def on_press(event, ctl=ctl, parent=self):
                    dlg = wx.DirDialog(parent)
                    dlg.Path = ctl.Value
                    if dlg.ShowModal() == wx.ID_OK:
                        ctl.Value = dlg.Path
                        dlg.Destroy()
            elif (isinstance(ui_info, basestring) and
                      ui_info.startswith(FILEBROWSE)):
                def on_press(event, ctl=ctl, parent=self, ui_info=ui_info):
                    dlg = wx.FileDialog(parent)
                    dlg.Path = ctl.Value
                    if len(ui_info) > len(FILEBROWSE) + 1:
                        dlg.Wildcard = ui_info[(len(FILEBROWSE) + 1):]
                    if dlg.ShowModal() == wx.ID_OK:
                        ctl.Value = dlg.Path
                    dlg.Destroy()

                ui_info = "Browse"
            elif ui_info == FONT:
                def on_press(event, ctl=ctl, parent=self):
                    name, size = ctl.Value.split(",")
                    fd = wx.FontData()
                    fd.SetInitialFont(wx.FFont(float(size),
                                               wx.FONTFAMILY_DEFAULT,
                                               face=name))
                    dlg = wx.FontDialog(parent, fd)
                    if dlg.ShowModal() == wx.ID_OK:
                        fd = dlg.GetFontData()
                        font = fd.GetChosenFont()
                        name = font.GetFaceName()
                        size = font.GetPointSize()
                        ctl.Value = "%s, %f" % (name, size)
                    dlg.Destroy()
            elif ui_info == COLOR:
                def on_press(event, ctl=ctl, parent=self):
                    color = wx.GetColourFromUser(self, ctl.BackgroundColour)
                    if any([x != -1 for x in color.Get()]):
                        ctl.BackgroundColour = color
                        ctl.Refresh()
            else:
                on_press = None
            if not on_press is None:
                identifier = wx.NewId()
                button = wx.Button(scrollpanel, identifier, ui_info)
                self.Bind(wx.EVT_BUTTON, on_press, button, identifier)
                sizer.Add(button, (index, 2))
            button = wx.Button(scrollpanel, -1, '?', (0, 0), (30, -1))

            def on_help(event, help_text=help_text):
                dlg = cellprofiler.gui.htmldialog.HTMLDialog(self, "Preferences help", help_text)
                dlg.Show()

            sizer.Add(button, (index, 3))
            self.Bind(wx.EVT_BUTTON, on_help, button)
            index += 1

        sizer.AddGrowableCol(1, 10)
        sizer.AddGrowableCol(3, 1)

        top_sizer.Add(wx.StaticLine(scrollpanel), 0, wx.EXPAND | wx.ALL, 2)
        btnsizer = self.CreateButtonSizer(wx.OK | wx.CANCEL)
        self.Bind(wx.EVT_BUTTON, self.save_preferences, id=wx.ID_OK)

        scrollpanel_sizer.Add(
                btnsizer, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALL, 5)
        scrollpanel.SetupScrolling(scrollToTop=False)
        self.Fit()
        self.controls = controls

    def save_preferences(self, event):
        event.Skip()
        p = self.get_preferences()
        for control, (text, getter, setter, ui_info, help_text) in \
                zip(self.controls, p):
            if ui_info == COLOR:
                value = control.BackgroundColour
            elif ui_info == FILEBROWSE:
                value = control.Value
                if not os.path.isfile(value):
                    continue
            elif ui_info == DIRBROWSE:
                value = control.Value
                if not os.path.isdir(value):
                    continue
            else:
                value = control.Value
            if value != getter():
                setter(value)

    def get_preferences(self):
        """Get the list of preferences.

        Each row in the list has the following form:
        Title - the text that appears to the right of the edit box
        get_function - retrieves the persistent preference value
        set_function - sets the preference value
        display - If this is a list, it represents the valid choices.
                  If it is "dirbrowse", put a directory browse button
                  to the right of the edit box.
        """
        return [
            [
                "Default Input Folder",
                cellprofiler.preferences.get_default_image_directory,
                cellprofiler.preferences.set_default_image_directory,
                DIRBROWSE,
                cellprofiler.gui.help.DEFAULT_IMAGE_FOLDER_HELP
            ],
            [
                "Default Output Folder",
                cellprofiler.preferences.get_default_output_directory,
                cellprofiler.preferences.set_default_output_directory,
                DIRBROWSE,
                cellprofiler.gui.help.DEFAULT_OUTPUT_FOLDER_HELP
            ],
            [
                "CellProfiler plugins directory",
                cellprofiler.preferences.get_plugin_directory,
                cellprofiler.preferences.set_plugin_directory,
                DIRBROWSE,
                cellprofiler.gui.help.PLUGINS_DIRECTORY_HELP
            ]
        ]
