import os
import sys

import matplotlib.cm
import wx
import wx.lib.scrolledpanel
from cellprofiler_core.preferences import ALWAYS_CONTINUE_HELP
from cellprofiler_core.preferences import export_to_json
from cellprofiler_core.preferences import CONSERVE_MEMORY_HELP
from cellprofiler_core.preferences import DEFAULT_COLORMAP_HELP
from cellprofiler_core.preferences import DEFAULT_IMAGE_FOLDER_HELP
from cellprofiler_core.preferences import DEFAULT_OUTPUT_FOLDER_HELP
from cellprofiler_core.preferences import ERROR_COLOR_HELP
from cellprofiler_core.preferences import FORCE_BIOFORMATS_HELP
from cellprofiler_core.preferences import IM_BICUBIC
from cellprofiler_core.preferences import IM_BILINEAR
from cellprofiler_core.preferences import IM_NEAREST
from cellprofiler_core.preferences import INTENSITY_MODE_GAMMA
from cellprofiler_core.preferences import INTENSITY_MODE_HELP
from cellprofiler_core.preferences import INTENSITY_MODE_LOG
from cellprofiler_core.preferences import INTENSITY_MODE_NORMAL
from cellprofiler_core.preferences import INTENSITY_MODE_RAW
from cellprofiler_core.preferences import INTERPOLATION_MODE_HELP
from cellprofiler_core.preferences import MAX_WORKERS_HELP
from cellprofiler_core.preferences import NORMALIZATION_FACTOR_HELP
from cellprofiler_core.preferences import PLUGINS_DIRECTORY_HELP
from cellprofiler_core.preferences import PRIMARY_OUTLINE_COLOR_HELP
from cellprofiler_core.preferences import REPORT_JVM_ERROR_HELP
from cellprofiler_core.preferences import SAVE_PIPELINE_WITH_PROJECT_HELP
from cellprofiler_core.preferences import SECONDARY_OUTLINE_COLOR_HELP
from cellprofiler_core.preferences import SHOW_ANALYSIS_COMPLETE_HELP
from cellprofiler_core.preferences import SHOW_EXITING_TEST_MODE_HELP
from cellprofiler_core.preferences import SHOW_REPORT_BAD_SIZES_DLG_HELP
from cellprofiler_core.preferences import SHOW_SAMPLING_MENU_HELP
from cellprofiler_core.preferences import SHOW_STARTUP_BLURB_HELP
from cellprofiler_core.preferences import SHOW_TELEMETRY_HELP
from cellprofiler_core.preferences import SPP_ALL
from cellprofiler_core.preferences import TABLE_FONT_HELP
from cellprofiler_core.preferences import TEMP_DIR_HELP
from cellprofiler_core.preferences import TERTIARY_OUTLINE_COLOR_HELP
from cellprofiler_core.preferences import UPDATER_HELP
from cellprofiler_core.preferences import WIDGET_INSPECTOR_HELP
from cellprofiler_core.preferences import default_max_workers
from cellprofiler_core.preferences import get_always_continue
from cellprofiler_core.preferences import get_check_update_bool
from cellprofiler_core.preferences import get_conserve_memory
from cellprofiler_core.preferences import get_default_colormap
from cellprofiler_core.preferences import get_default_image_directory
from cellprofiler_core.preferences import get_default_output_directory
from cellprofiler_core.preferences import get_error_color
from cellprofiler_core.preferences import get_force_bioformats
from cellprofiler_core.preferences import get_intensity_mode
from cellprofiler_core.preferences import get_interpolation_mode
from cellprofiler_core.preferences import get_max_workers
from cellprofiler_core.preferences import get_normalization_factor
from cellprofiler_core.preferences import get_plugin_directory
from cellprofiler_core.preferences import get_primary_outline_color
from cellprofiler_core.preferences import get_report_jvm_error
from cellprofiler_core.preferences import get_save_pipeline_with_project
from cellprofiler_core.preferences import get_secondary_outline_color
from cellprofiler_core.preferences import get_show_analysis_complete_dlg
from cellprofiler_core.preferences import get_show_exiting_test_mode_dlg
from cellprofiler_core.preferences import get_show_report_bad_sizes_dlg
from cellprofiler_core.preferences import get_show_sampling
from cellprofiler_core.preferences import get_startup_blurb
from cellprofiler_core.preferences import get_table_font_name
from cellprofiler_core.preferences import get_table_font_size
from cellprofiler_core.preferences import get_telemetry
from cellprofiler_core.preferences import get_temporary_directory
from cellprofiler_core.preferences import get_tertiary_outline_color
from cellprofiler_core.preferences import get_title_font_name
from cellprofiler_core.preferences import get_title_font_size
from cellprofiler_core.preferences import get_widget_inspector
from cellprofiler_core.preferences import get_wants_pony
from cellprofiler_core.preferences import set_always_continue
from cellprofiler_core.preferences import set_check_update
from cellprofiler_core.preferences import set_conserve_memory
from cellprofiler_core.preferences import set_default_colormap
from cellprofiler_core.preferences import set_default_image_directory
from cellprofiler_core.preferences import set_default_output_directory
from cellprofiler_core.preferences import set_error_color
from cellprofiler_core.preferences import set_force_bioformats
from cellprofiler_core.preferences import set_intensity_mode
from cellprofiler_core.preferences import set_interpolation_mode
from cellprofiler_core.preferences import set_max_workers
from cellprofiler_core.preferences import set_normalization_factor
from cellprofiler_core.preferences import set_plugin_directory
from cellprofiler_core.preferences import set_primary_outline_color
from cellprofiler_core.preferences import set_report_jvm_error
from cellprofiler_core.preferences import set_save_pipeline_with_project
from cellprofiler_core.preferences import set_secondary_outline_color
from cellprofiler_core.preferences import set_show_analysis_complete_dlg
from cellprofiler_core.preferences import set_show_exiting_test_mode_dlg
from cellprofiler_core.preferences import set_show_report_bad_sizes_dlg
from cellprofiler_core.preferences import set_show_sampling
from cellprofiler_core.preferences import set_startup_blurb
from cellprofiler_core.preferences import set_table_font_name
from cellprofiler_core.preferences import set_table_font_size
from cellprofiler_core.preferences import set_telemetry
from cellprofiler_core.preferences import set_temporary_directory
from cellprofiler_core.preferences import set_tertiary_outline_color
from cellprofiler_core.preferences import set_title_font_name
from cellprofiler_core.preferences import set_title_font_size
from cellprofiler_core.preferences import set_widget_inspector
from cellprofiler_core.preferences import set_wants_pony

from cellprofiler.gui.app import init_telemetry, stop_telemetry

from ._integer_preference import IntegerPreference
from ..constants.preferences_dialog import CHOICE
from ..constants.preferences_dialog import COLOR
from ..constants.preferences_dialog import DIRBROWSE
from ..constants.preferences_dialog import FILEBROWSE
from ..constants.preferences_dialog import FONT
from ..constants.preferences_dialog import TEXT
from ..html.utils import rst_to_html_fragment
from ..htmldialog import HTMLDialog


class PreferencesDialog(wx.Dialog):
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

            elif isinstance(ui_info, str) and ui_info.startswith(FILEBROWSE):

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
                dlg = HTMLDialog(
                    self, "Preferences help", rst_to_html_fragment(help_text),
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
        # We use ID_HELP here because we just need any preset button ID
        btnExport = wx.Button(self, wx.ID_HELP, "Export")
        self.Bind(wx.EVT_BUTTON, self.export_preferences, btnExport)
        btnsizer.AddButton(btnExport)
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
                if 'Send Telemetry' in text:
                    if value:
                        init_telemetry()
                    else:
                        stop_telemetry()
                setter(value)
        self.Close()

    def export_preferences(self, event):
        with wx.FileDialog(
            self,
            message="Save CellProfiler config to file",
            defaultFile="cp_settings.json",
            wildcard="*.json",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        ) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                path = dlg.GetPath()
                export_to_json(path)

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
                get_telemetry,
                set_telemetry,
                CHOICE,
                SHOW_TELEMETRY_HELP,
            ],
            [
                "Automatically check for updates",
                get_check_update_bool,
                set_check_update,
                CHOICE,
                UPDATER_HELP,
            ],
            [
                "Default Input Folder",
                get_default_image_directory,
                set_default_image_directory,
                DIRBROWSE,
                DEFAULT_IMAGE_FOLDER_HELP,
            ],
            [
                "Default Output Folder",
                get_default_output_directory,
                set_default_output_directory,
                DIRBROWSE,
                DEFAULT_OUTPUT_FOLDER_HELP,
            ],
            [
                "Table font",
                self.get_table_font,
                self.set_table_font,
                FONT,
                TABLE_FONT_HELP,
            ],
            [
                "Default colormap",
                get_default_colormap,
                set_default_colormap,
                cmaps,
                DEFAULT_COLORMAP_HELP,
            ],
            ["Error color", get_error_color, set_error_color, COLOR, ERROR_COLOR_HELP,],
            [
                "Primary outline color",
                get_primary_outline_color,
                set_primary_outline_color,
                COLOR,
                PRIMARY_OUTLINE_COLOR_HELP,
            ],
            [
                "Secondary outline color",
                get_secondary_outline_color,
                set_secondary_outline_color,
                COLOR,
                SECONDARY_OUTLINE_COLOR_HELP,
            ],
            [
                "Tertiary outline color",
                get_tertiary_outline_color,
                set_tertiary_outline_color,
                COLOR,
                TERTIARY_OUTLINE_COLOR_HELP,
            ],
            [
                "Interpolation mode",
                get_interpolation_mode,
                set_interpolation_mode,
                [IM_NEAREST, IM_BILINEAR, IM_BICUBIC,],
                INTERPOLATION_MODE_HELP,
            ],
            [
                "Intensity normalization",
                get_intensity_mode,
                set_intensity_mode,
                [
                    INTENSITY_MODE_RAW,
                    INTENSITY_MODE_NORMAL,
                    INTENSITY_MODE_LOG,
                    INTENSITY_MODE_GAMMA,
                ],
                INTENSITY_MODE_HELP,
            ],
            [
                "Intensity normalization factor",
                get_normalization_factor,
                set_normalization_factor,
                TEXT,
                NORMALIZATION_FACTOR_HELP,
            ],
            [
                "CellProfiler plugins directory",
                get_plugin_directory,
                set_plugin_directory,
                DIRBROWSE,
                PLUGINS_DIRECTORY_HELP,
            ],
            [
                "Display welcome text on startup",
                get_startup_blurb,
                set_startup_blurb,
                CHOICE,
                SHOW_STARTUP_BLURB_HELP,
            ],
            [
                "Warn if Java runtime environment not present",
                get_report_jvm_error,
                set_report_jvm_error,
                CHOICE,
                REPORT_JVM_ERROR_HELP,
            ],
            [
                'Show the "Analysis complete" message at the end of a run',
                get_show_analysis_complete_dlg,
                set_show_analysis_complete_dlg,
                CHOICE,
                SHOW_ANALYSIS_COMPLETE_HELP,
            ],
            [
                'Show the "Exiting test mode" message',
                get_show_exiting_test_mode_dlg,
                set_show_exiting_test_mode_dlg,
                CHOICE,
                SHOW_EXITING_TEST_MODE_HELP,
            ],
            [
                "Warn if images are different sizes",
                get_show_report_bad_sizes_dlg,
                set_show_report_bad_sizes_dlg,
                CHOICE,
                SHOW_REPORT_BAD_SIZES_DLG_HELP,
            ],
            [
                "Show the sampling menu",
                get_show_sampling,
                set_show_sampling,
                CHOICE,
                SHOW_SAMPLING_MENU_HELP,
            ],
            [
                "Maximum number of workers",
                get_max_workers,
                set_max_workers,
                IntegerPreference(1, default_max_workers() * 4),
                MAX_WORKERS_HELP,
            ],
            [
                "Temporary folder",
                get_temporary_directory,
                (lambda x: set_temporary_directory(x, globally=True)),
                DIRBROWSE,
                TEMP_DIR_HELP,
            ],
            [
                "Save pipeline and/or file list in addition to project",
                get_save_pipeline_with_project,
                set_save_pipeline_with_project,
                SPP_ALL,
                SAVE_PIPELINE_WITH_PROJECT_HELP,
            ],
            [
                'Conserve system memory',
                get_conserve_memory,
                set_conserve_memory,
                CHOICE,
                CONSERVE_MEMORY_HELP,
            ],
            [
                'Always use BioFormats to read images',
                get_force_bioformats,
                set_force_bioformats,
                CHOICE,
                FORCE_BIOFORMATS_HELP,
            ],
            [
                'Always skip failing images',
                get_always_continue,
                set_always_continue,
                CHOICE,
                ALWAYS_CONTINUE_HELP,
            ],
            [
                "Enable widget inspector",
                (lambda: get_widget_inspector(global_only=True)),
                self._set_widget_inspector,
                CHOICE,
                WIDGET_INSPECTOR_HELP,
            ],

            ["Pony", get_wants_pony, set_wants_pony, CHOICE, "The end is neigh.",],
        ]

    def _set_widget_inspector(self, val):
        from cellprofiler.gui.cpframe import ID_FILE_WIDGET_INSPECTOR, ID_DEBUG_HELP

        frame = wx.GetApp().frame
        menu_item_exists = frame.menu_item_exists(ID_FILE_WIDGET_INSPECTOR)

        # when setting true, inject menu item if not already present
        if val and not menu_item_exists:
            frame.inject_menu_item_by_title("&Test", ID_FILE_WIDGET_INSPECTOR, "Widget Inspector", sibling_id=ID_DEBUG_HELP)
        # when setting false, remove menu item if present
        elif not val and menu_item_exists:
            frame.remove_menu_item(ID_FILE_WIDGET_INSPECTOR)

        set_widget_inspector(val, globally=True)

    @staticmethod
    def get_title_font():
        return "%s,%f" % (get_title_font_name(), get_title_font_size(),)

    @staticmethod
    def set_title_font(font):
        name, size = font.split(",")
        set_title_font_name(name)
        set_title_font_size(float(size))

    @staticmethod
    def get_table_font():
        return "%s,%f" % (get_table_font_name(), get_table_font_size(),)

    @staticmethod
    def set_table_font(font):
        name, size = font.split(",")
        set_table_font_name(name)
        set_table_font_size(float(size))
