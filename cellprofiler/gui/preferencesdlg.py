# coding=utf-8
"""preferencesdlg.py Edit global preferences
"""

import os
import sys

import matplotlib.cm
import six
import wx
import wx.lib.scrolledpanel

import cellprofiler.gui.help
import cellprofiler.gui.html.utils
import cellprofiler.gui.htmldialog
import cellprofiler_core.preferences

DIRBROWSE = "Browse"
FILEBROWSE = "FileBrowse"
FONT = "Font"
COLOR = "Color"
CHOICE = "Choice"
TEXT = "Text"


class IntegerPreference(object):
    """User interface info for an integer preference

    This signals that a preference should be displayed and edited as
    an integer, optionally limited by a range.
    """

    def __init__(self, minval=None, maxval=None):
        self.minval = minval
        self.maxval = maxval


class PreferencesDlg(wx.Dialog):
    """Display a dialog for setting preferences

    The dialog handles fetching current defaults and setting the
    defaults when the user hits OK.
    """

    def __init__(
        self,
        parent=None,
        ID=-1,
        title="CellProfiler preferences",
        size=wx.DefaultSize,
        pos=wx.DefaultPosition,
        style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
        name=wx.DialogNameStr,
    ):
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
        help_bitmap = wx.ArtProvider.GetBitmap(wx.ART_HELP, wx.ART_CMN_DIALOG, (16, 16))
        for text, getter, setter, ui_info, help_text in p:
            text_ctl = wx.StaticText(scrollpanel, label=text)
            sizer.Add(text_ctl, (index, 0))
            if getattr(ui_info, "__getitem__", False) and not isinstance(ui_info, str):
                ctl = wx.ComboBox(
                    scrollpanel, -1, choices=ui_info, style=wx.CB_READONLY
                )
                ctl.SetStringSelection(getter())
            elif ui_info == COLOR:
                ctl = wx.Panel(scrollpanel, -1, style=wx.BORDER_SUNKEN)
                ctl.BackgroundColour = getter()
            elif ui_info == CHOICE:
                ctl = wx.CheckBox(scrollpanel, -1)
                ctl.SetValue(getter())
            elif isinstance(ui_info, IntegerPreference):
                minval = -sys.maxsize if ui_info.minval is None else ui_info.minval
                maxval = sys.maxsize if ui_info.maxval is None else ui_info.maxval
                ctl = wx.SpinCtrl(scrollpanel, min=minval, max=maxval, initial=getter())
            else:
                validator = wx.DefaultValidator
                current = getter()
                if current is None:
                    current = ""
                ctl = wx.TextCtrl(
                    parent=scrollpanel, id=-1, value=current, validator=validator
                )
                min_height = ctl.GetMinHeight()
                min_width = ctl.GetFullTextExtent(
                    "Make sure the window can display this"
                )[0]
                ctl.SetMinSize((min_width, min_height))
            controls.append(ctl)
            sizer.Add(ctl, (index, 1), flag=wx.EXPAND)

            if ui_info == DIRBROWSE:

                def on_press(event, ctl=ctl, parent=self):
                    dlg = wx.DirDialog(parent)

                    dlg.SetPath(ctl.Value)

                    if dlg.ShowModal() == wx.ID_OK:
                        ctl.SetValue(dlg.GetPath())

                        dlg.Destroy()

            elif isinstance(ui_info, six.string_types) and ui_info.startswith(
                FILEBROWSE
            ):

                def on_press(event, ctl=ctl, parent=self, ui_info=ui_info):
                    dlg = wx.FileDialog(parent)

                    dlg.SetPath(ctl.Value)

                    if len(ui_info) > len(FILEBROWSE) + 1:
                        dlg.SetWildcard(ui_info[(len(FILEBROWSE) + 1) :])

                    if dlg.ShowModal() == wx.ID_OK:
                        ctl.SetValue(dlg.GetPath())

                    dlg.Destroy()

                ui_info = "Browse"
            elif ui_info == FONT:

                def on_press(event, ctl=ctl, parent=self):
                    name, size = ctl.Value.split(",")
                    fd = wx.FontData()
                    fd.SetInitialFont(
                        wx.FFont(
                            pointSize=float(size),
                            family=wx.FONTFAMILY_DEFAULT,
                            faceName=name,
                        )
                    )
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
            if on_press is not None:
                identifier = wx.NewId()
                button = wx.Button(scrollpanel, identifier, ui_info)
                self.Bind(wx.EVT_BUTTON, on_press, button, identifier)
                sizer.Add(button, (index, 2))
            button = wx.Button(scrollpanel, -1, "?", (0, 0), (30, -1))

            def on_help(event, help_text=help_text):
                dlg = cellprofiler.gui.htmldialog.HTMLDialog(
                    self,
                    "Preferences help",
                    cellprofiler.gui.html.utils.rst_to_html_fragment(help_text),
                )
                dlg.Show()

            sizer.Add(button, (index, 3))
            self.Bind(wx.EVT_BUTTON, on_help, button)
            index += 1

        sizer.AddGrowableCol(1, 10)
        sizer.AddGrowableCol(3, 1)

        top_sizer.Add(wx.StaticLine(scrollpanel), 0, wx.EXPAND | wx.ALL, 2)
        btnsizer = wx.StdDialogButtonSizer()
        btnSave = wx.Button(self, wx.ID_SAVE)
        btnCancel = wx.Button(self, wx.ID_CANCEL)
        btnsizer.SetAffirmativeButton(btnSave)
        btnsizer.SetCancelButton(btnCancel)
        self.Bind(wx.EVT_BUTTON, self.save_preferences, id=wx.ID_SAVE)
        btnsizer.Realize()
        scrollpanel_sizer.Add(btnsizer, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALL, 5)
        scrollpanel.SetupScrolling(scrollToTop=False)
        self.Fit()
        self.controls = controls

    def save_preferences(self, event):
        event.Skip()
        p = self.get_preferences()
        for control, (text, getter, setter, ui_info, help_text) in zip(
            self.controls, p
        ):
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
        self.Close()

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
        cmaps = list(matplotlib.cm.datad.keys())
        cmaps.sort()

        return [
            [
                "Send Telemetry to the CellProfiler Team",
                cellprofiler_core.preferences.get_telemetry,
                cellprofiler_core.preferences.set_telemetry,
                CHOICE,
                cellprofiler_core.preferences.SHOW_TELEMETRY_HELP,
            ],
            [
                "Default Input Folder",
                cellprofiler_core.preferences.get_default_image_directory,
                cellprofiler_core.preferences.set_default_image_directory,
                DIRBROWSE,
                cellprofiler_core.preferences.DEFAULT_IMAGE_FOLDER_HELP,
            ],
            [
                "Default Output Folder",
                cellprofiler_core.preferences.get_default_output_directory,
                cellprofiler_core.preferences.set_default_output_directory,
                DIRBROWSE,
                cellprofiler_core.preferences.DEFAULT_OUTPUT_FOLDER_HELP,
            ],
            [
                "Table font",
                self.get_table_font,
                self.set_table_font,
                FONT,
                cellprofiler_core.preferences.TABLE_FONT_HELP,
            ],
            [
                "Default colormap",
                cellprofiler_core.preferences.get_default_colormap,
                cellprofiler_core.preferences.set_default_colormap,
                cmaps,
                cellprofiler_core.preferences.DEFAULT_COLORMAP_HELP,
            ],
            [
                "Error color",
                cellprofiler_core.preferences.get_error_color,
                cellprofiler_core.preferences.set_error_color,
                COLOR,
                cellprofiler_core.preferences.ERROR_COLOR_HELP,
            ],
            [
                "Primary outline color",
                cellprofiler_core.preferences.get_primary_outline_color,
                cellprofiler_core.preferences.set_primary_outline_color,
                COLOR,
                cellprofiler_core.preferences.PRIMARY_OUTLINE_COLOR_HELP,
            ],
            [
                "Secondary outline color",
                cellprofiler_core.preferences.get_secondary_outline_color,
                cellprofiler_core.preferences.set_secondary_outline_color,
                COLOR,
                cellprofiler_core.preferences.SECONDARY_OUTLINE_COLOR_HELP,
            ],
            [
                "Tertiary outline color",
                cellprofiler_core.preferences.get_tertiary_outline_color,
                cellprofiler_core.preferences.set_tertiary_outline_color,
                COLOR,
                cellprofiler_core.preferences.TERTIARY_OUTLINE_COLOR_HELP,
            ],
            [
                "Interpolation mode",
                cellprofiler_core.preferences.get_interpolation_mode,
                cellprofiler_core.preferences.set_interpolation_mode,
                [
                    cellprofiler_core.preferences.IM_NEAREST,
                    cellprofiler_core.preferences.IM_BILINEAR,
                    cellprofiler_core.preferences.IM_BICUBIC,
                ],
                cellprofiler_core.preferences.INTERPOLATION_MODE_HELP,
            ],
            [
                "Intensity normalization",
                cellprofiler_core.preferences.get_intensity_mode,
                cellprofiler_core.preferences.set_intensity_mode,
                [
                    cellprofiler_core.preferences.INTENSITY_MODE_RAW,
                    cellprofiler_core.preferences.INTENSITY_MODE_NORMAL,
                    cellprofiler_core.preferences.INTENSITY_MODE_LOG,
                    cellprofiler_core.preferences.INTENSITY_MODE_GAMMA,
                ],
                cellprofiler_core.preferences.INTENSITY_MODE_HELP,
            ],
            [
                "Intensity normalization factor",
                cellprofiler_core.preferences.get_normalization_factor,
                cellprofiler_core.preferences.set_normalization_factor,
                TEXT,
                cellprofiler_core.preferences.NORMALIZATION_FACTOR_HELP,
            ],
            [
                "CellProfiler plugins directory",
                cellprofiler_core.preferences.get_plugin_directory,
                cellprofiler_core.preferences.set_plugin_directory,
                DIRBROWSE,
                cellprofiler_core.preferences.PLUGINS_DIRECTORY_HELP,
            ],
            [
                "Display welcome text on startup",
                cellprofiler_core.preferences.get_startup_blurb,
                cellprofiler_core.preferences.set_startup_blurb,
                CHOICE,
                cellprofiler_core.preferences.SHOW_STARTUP_BLURB_HELP,
            ],
            [
                "Warn if Java runtime environment not present",
                cellprofiler_core.preferences.get_report_jvm_error,
                cellprofiler_core.preferences.set_report_jvm_error,
                CHOICE,
                cellprofiler_core.preferences.REPORT_JVM_ERROR_HELP,
            ],
            [
                'Show the "Analysis complete" message at the end of a run',
                cellprofiler_core.preferences.get_show_analysis_complete_dlg,
                cellprofiler_core.preferences.set_show_analysis_complete_dlg,
                CHOICE,
                cellprofiler_core.preferences.SHOW_ANALYSIS_COMPLETE_HELP,
            ],
            [
                'Show the "Exiting test mode" message',
                cellprofiler_core.preferences.get_show_exiting_test_mode_dlg,
                cellprofiler_core.preferences.set_show_exiting_test_mode_dlg,
                CHOICE,
                cellprofiler_core.preferences.SHOW_EXITING_TEST_MODE_HELP,
            ],
            [
                "Warn if images are different sizes",
                cellprofiler_core.preferences.get_show_report_bad_sizes_dlg,
                cellprofiler_core.preferences.set_show_report_bad_sizes_dlg,
                CHOICE,
                cellprofiler_core.preferences.SHOW_REPORT_BAD_SIZES_DLG_HELP,
            ],
            [
                "Show the sampling menu",
                cellprofiler_core.preferences.get_show_sampling,
                cellprofiler_core.preferences.set_show_sampling,
                CHOICE,
                cellprofiler_core.preferences.SHOW_SAMPLING_MENU_HELP,
            ],
            [
                "Maximum number of workers",
                cellprofiler_core.preferences.get_max_workers,
                cellprofiler_core.preferences.set_max_workers,
                IntegerPreference(
                    1, cellprofiler_core.preferences.default_max_workers() * 4
                ),
                cellprofiler_core.preferences.MAX_WORKERS_HELP,
            ],
            [
                "Temporary folder",
                cellprofiler_core.preferences.get_temporary_directory,
                (
                    lambda x: cellprofiler_core.preferences.set_temporary_directory(
                        x, globally=True
                    )
                ),
                DIRBROWSE,
                cellprofiler_core.preferences.TEMP_DIR_HELP,
            ],
            [
                "Maximum memory for Java (MB)",
                cellprofiler_core.preferences.get_jvm_heap_mb,
                cellprofiler_core.preferences.set_jvm_heap_mb,
                IntegerPreference(128, 64000),
                cellprofiler_core.preferences.JVM_HEAP_HELP,
            ],
            [
                "Save pipeline and/or file list in addition to project",
                cellprofiler_core.preferences.get_save_pipeline_with_project,
                cellprofiler_core.preferences.set_save_pipeline_with_project,
                cellprofiler_core.preferences.SPP_ALL,
                cellprofiler_core.preferences.SAVE_PIPELINE_WITH_PROJECT_HELP,
            ],
            [
                "Pony",
                cellprofiler_core.preferences.get_wants_pony,
                cellprofiler_core.preferences.set_wants_pony,
                CHOICE,
                "The end is neigh.",
            ],
        ]

    @staticmethod
    def get_title_font():
        return "%s,%f" % (
            cellprofiler_core.preferences.get_title_font_name(),
            cellprofiler_core.preferences.get_title_font_size(),
        )

    @staticmethod
    def set_title_font(font):
        name, size = font.split(",")
        cellprofiler_core.preferences.set_title_font_name(name)
        cellprofiler_core.preferences.set_title_font_size(float(size))

    @staticmethod
    def get_table_font():
        return "%s,%f" % (
            cellprofiler_core.preferences.get_table_font_name(),
            cellprofiler_core.preferences.get_table_font_size(),
        )

    @staticmethod
    def set_table_font(font):
        name, size = font.split(",")
        cellprofiler_core.preferences.set_table_font_name(name)
        cellprofiler_core.preferences.set_table_font_size(float(size))
