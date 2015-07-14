'''preferencesdlg.py Edit global preferences

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2015 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

import wx
import matplotlib.cm
import os
import sys

import cellprofiler.preferences as cpprefs
import cellprofiler.gui.help as cphelp
from cellprofiler.gui.htmldialog import HTMLDialog

DIRBROWSE = "Browse"
FILEBROWSE = "FileBrowse"
FONT = "Font"
COLOR = "Color"
CHOICE = "Choice"

class IntegerPreference(object):
    '''User interface info for an integer preference
    
    This signals that a preference should be displayed and edited as
    an integer, optionally limited by a range.
    '''
    def __init__(self, minval = None, maxval = None):
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
                    caption = "Unsupported character in path name",
                    style = wx.OK | wx.ICON_ERROR, parent = win)
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
    '''Display a dialog for setting preferences
    
    The dialog handles fetching current defaults and setting the
    defaults when the user hits OK.
    '''
    def __init__(self, parent=None, ID=-1, title="CellProfiler preferences",
                 size=wx.DefaultSize, pos=wx.DefaultPosition, 
                 style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | 
                 wx.THICK_FRAME,
                 name=wx.DialogNameStr):
        wx.Dialog.__init__(self, parent, ID, title, pos, size, style,name)
        self.SetExtraStyle(wx.WS_EX_VALIDATE_RECURSIVELY)
        p = self.get_preferences()
        top_sizer = wx.BoxSizer(wx.VERTICAL)
        scrollpanel_sizer = wx.BoxSizer(wx.VERTICAL)
        scrollpanel = wx.lib.scrolledpanel.ScrolledPanel(self)
        scrollpanel.SetMinSize((800, 600))
        scrollpanel_sizer.Add(scrollpanel, 1, wx.EXPAND)
        scrollpanel.SetSizer(top_sizer)
        self.SetSizer(scrollpanel_sizer)

        sizer = wx.GridBagSizer(len(p),4)
        sizer.SetFlexibleDirection(wx.HORIZONTAL)
        top_sizer.Add(sizer,1, wx.EXPAND|wx.ALL, 5)
        index = 0
        controls = []
        help_bitmap = wx.ArtProvider.GetBitmap(wx.ART_HELP,
                                               wx.ART_CMN_DIALOG,
                                               (16,16))
        for text, getter, setter, ui_info, help_text in p:
            text_ctl = wx.StaticText(scrollpanel, label=text)
            sizer.Add(text_ctl,(index,0))
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
                                  min = minval,
                                  max = maxval,
                                  initial = getter())
            else:
                if getter == cpprefs.get_ij_plugin_directory:
                    validator = ClassPathValidator()
                else:
                    validator = wx.DefaultValidator
                current = getter()
                if current is None:
                    current = ""
                ctl = wx.TextCtrl(scrollpanel, -1, current, 
                                  validator = validator)
                min_height = ctl.GetMinHeight()
                min_width  = ctl.GetTextExtent("Make sure the window can display this")[0]
                ctl.SetMinSize((min_width, min_height))
            controls.append(ctl)
            sizer.Add(ctl,(index,1),flag=wx.EXPAND)
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
                        ctl.Value = "%s, %f"%(name,size) 
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
                id = wx.NewId()
                button = wx.Button(scrollpanel, id, ui_info)
                self.Bind(wx.EVT_BUTTON, on_press, button,id)
                sizer.Add(button, (index, 2))
            button = wx.Button(scrollpanel, -1, '?', (0, 0), (30, -1))
            def on_help(event, help_text = help_text):
                dlg = HTMLDialog(self, "Preferences help", help_text)
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
                setter(control.BackgroundColour)
            elif ui_info == FILEBROWSE and os.path.isfile(control.Value):
                setter(control.Value)
            else:
                setter(control.Value)

    def get_preferences(self):
        '''Get the list of preferences.
        
        Each row in the list has the following form:
        Title - the text that appears to the right of the edit box
        get_function - retrieves the persistent preference value
        set_function - sets the preference value
        display - If this is a list, it represents the valid choices.
                  If it is "dirbrowse", put a directory browse button
                  to the right of the edit box.
        '''
        cmaps = list(matplotlib.cm.datad.keys())
        cmaps.sort()
        return [["Default Input Folder",
                 cpprefs.get_default_image_directory,
                 cpprefs.set_default_image_directory,
                 DIRBROWSE, cphelp.DEFAULT_IMAGE_FOLDER_HELP],
                ["Default Output Folder",
                 cpprefs.get_default_output_directory,
                 cpprefs.set_default_output_directory,
                 DIRBROWSE, cphelp.DEFAULT_OUTPUT_FOLDER_HELP],
                [ "Title font", 
                  self.get_title_font, 
                  self.set_title_font, 
                  FONT, cphelp.TITLE_FONT_HELP],
                ["Table font", 
                 self.get_table_font, 
                 self.set_table_font, 
                 FONT, cphelp.TABLE_FONT_HELP],
                ["Default colormap", 
                 cpprefs.get_default_colormap, 
                 cpprefs.set_default_colormap, 
                 cmaps, cphelp.DEFAULT_COLORMAP_HELP],
                ["Window background", 
                 cpprefs.get_background_color, 
                 cpprefs.set_background_color, 
                 COLOR, cphelp.WINDOW_BACKGROUND_HELP],
                ["Error color",
                 cpprefs.get_error_color,
                 cpprefs.set_error_color,
                 COLOR, cphelp.ERROR_COLOR_HELP],
                ["Primary outline color",
                 cpprefs.get_primary_outline_color,
                 cpprefs.set_primary_outline_color,
                 COLOR, cphelp.PRIMARY_OUTLINE_COLOR_HELP],
                ["Secondary outline color",
                 cpprefs.get_secondary_outline_color,
                 cpprefs.set_secondary_outline_color,
                 COLOR, cphelp.SECONDARY_OUTLINE_COLOR_HELP],
                ["Tertiary outline color",
                 cpprefs.get_tertiary_outline_color,
                 cpprefs.set_tertiary_outline_color,
                 COLOR, cphelp.TERTIARY_OUTLINE_COLOR_HELP],
                ["Interpolation mode",
                 cpprefs.get_interpolation_mode,
                 cpprefs.set_interpolation_mode,
                 [cpprefs.IM_NEAREST, cpprefs.IM_BILINEAR, cpprefs.IM_BICUBIC],
                 cphelp.INTERPOLATION_MODE_HELP],
                ["Intensity normalization",
                 cpprefs.get_intensity_mode,
                 cpprefs.set_intensity_mode,
                 [cpprefs.INTENSITY_MODE_RAW, cpprefs.INTENSITY_MODE_NORMAL,
                  cpprefs.INTENSITY_MODE_LOG],
                 cphelp.INTENSITY_MODE_HELP],
                ["CellProfiler plugins directory",
                 cpprefs.get_plugin_directory,
                 cpprefs.set_plugin_directory,
                 DIRBROWSE, cphelp.PLUGINS_DIRECTORY_HELP],
                ["ImageJ plugins directory",
                 cpprefs.get_ij_plugin_directory,
                 cpprefs.set_ij_plugin_directory,
                 DIRBROWSE, cphelp.IJ_PLUGINS_DIRECTORY_HELP],
                ["Check for updates", 
                 cpprefs.get_check_new_versions, 
                 cpprefs.set_check_new_versions, 
                 CHOICE, cphelp.CHECK_FOR_UPDATES_HELP],
                ["Display welcome text on startup", 
                 cpprefs.get_startup_blurb, 
                 cpprefs.set_startup_blurb, 
                 CHOICE, cphelp.SHOW_STARTUP_BLURB_HELP],
                ["Warn if Java runtime environment not present",
                 cpprefs.get_report_jvm_error,
                 cpprefs.set_report_jvm_error,
                 CHOICE, cphelp.REPORT_JVM_ERROR_HELP],
                ['Show the "Analysis complete" message at the end of a run',
                 cpprefs.get_show_analysis_complete_dlg,
                 cpprefs.set_show_analysis_complete_dlg,
                 CHOICE, cphelp.SHOW_ANALYSIS_COMPLETE_HELP],
                ['Show the "Exiting test mode" message',
                 cpprefs.get_show_exiting_test_mode_dlg,
                 cpprefs.set_show_exiting_test_mode_dlg,
                 CHOICE, cphelp.SHOW_EXITING_TEST_MODE_HELP],
                ['Warn if images are different sizes',
                 cpprefs.get_show_report_bad_sizes_dlg,
                 cpprefs.set_show_report_bad_sizes_dlg,
                 CHOICE, cphelp.SHOW_REPORT_BAD_SIZES_DLG_HELP],
                ['Show the sampling menu',
                 cpprefs.get_show_sampling,
                 cpprefs.set_show_sampling,
                 CHOICE, """<p>Show the sampling menu </p>
                 <p><i>Note that CellProfiler must be restarted after setting.</i></p>
                 <p>The sampling menu is an interplace for Paramorama, a plugin for an interactive visualization 
                 program for exploring the parameter space of image analysis algorithms.
                 will generate a text file, which specifies: (1) all unique combinations of 
                 the sampled parameter values; (2) the mapping from each combination of parameter values to 
                 one or more output images; and (3) the actual output images.</p>
                 <p>More information on how to use the plugin can be found 
                 <a href="http://www.comp.leeds.ac.uk/scsajp/applications/paramorama2/">here</a>.</p>
                 <p><b>References</b>
                 <ul>
                 <li>Visualization of parameter space for image analysis. Pretorius AJ, Bray MA, Carpenter AE 
                 and Ruddle RA. (2011) IEEE Transactions on Visualization and Computer Graphics, 17(12), 2402-2411.</li>
                 </ul>"""],
                ['Warn if a pipeline was saved in an old version of CellProfiler',
                 cpprefs.get_warn_about_old_pipeline,
                 cpprefs.set_warn_about_old_pipeline,
                 CHOICE,
                 cphelp.WARN_ABOUT_OLD_PIPELINES_HELP],
                ['Use more figure space',
                 cpprefs.get_use_more_figure_space,
                 cpprefs.set_use_more_figure_space,
                 CHOICE,
                 cphelp.USE_MORE_FIGURE_SPACE_HELP
                ],
                ['Maximum number of workers',
                 cpprefs.get_max_workers,
                 cpprefs.set_max_workers,
                 IntegerPreference(1, cpprefs.default_max_workers() * 4),
                 cphelp.MAX_WORKERS_HELP],
                ['Temporary folder',
                 cpprefs.get_temporary_directory,
                 cpprefs.set_temporary_directory,
                 DIRBROWSE,
                 cphelp.TEMP_DIR_HELP],
                ['Maximum memory for Java (MB)',
                 cpprefs.get_jvm_heap_mb,
                 cpprefs.set_jvm_heap_mb,
                 IntegerPreference(128, 64000),
                 cphelp.JVM_HEAP_HELP],
                ['Save pipeline and/or file list in addition to project',
                 cpprefs.get_save_pipeline_with_project,
                 cpprefs.set_save_pipeline_with_project,
                 cpprefs.SPP_ALL,
                 cphelp.SAVE_PIPELINE_WITH_PROJECT_HELP],
                ['Folder name regular expression guesses',
                 cpprefs.get_pathname_re_guess_file,
                 cpprefs.set_pathname_re_guess_file,
                 FILEBROWSE,
                 cphelp.FOLDER_RE_GUESS_HELP],
                ['File name regular expression guesses',
                 cpprefs.get_filename_re_guess_file,
                 cpprefs.set_filename_re_guess_file,
                 FILEBROWSE,
                 cphelp.FILE_RE_GUESS_HELP]
                ]
    
    def get_title_font(self):
        return "%s,%f"%(cpprefs.get_title_font_name(),
                         cpprefs.get_title_font_size())
    
    def set_title_font(self, font):
        name, size = font.split(",")
        cpprefs.set_title_font_name(name)
        cpprefs.set_title_font_size(float(size))
    
    def get_table_font(self):
        return "%s,%f"%(cpprefs.get_table_font_name(),
                         cpprefs.get_table_font_size())
    
    def set_table_font(self, font):
        name, size = font.split(",")
        cpprefs.set_table_font_name(name)
        cpprefs.set_table_font_size(float(size))

if __name__=='__main__':
    class MyApp(wx.App):
        def OnInit(self):
            dlg = PreferencesDlg()
            dlg.show_modal()
            return 1
    app = MyApp(0)
