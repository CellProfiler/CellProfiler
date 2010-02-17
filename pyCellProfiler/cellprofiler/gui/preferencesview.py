"""PreferencesView.py - displays the default preferences in the lower right corner

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2010

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import os
import string
import wx
import cellprofiler.preferences
from cellprofiler.gui.htmldialog import HTMLDialog
from cellprofiler.gui.help import \
     DEFAULT_IMAGE_FOLDER_HELP, DEFAULT_OUTPUT_FOLDER_HELP, OUTPUT_FILENAME_HELP

WELCOME_MESSAGE = 'Welcome to CellProfiler'

class PreferencesView:
    """View / controller for the preferences that get displayed in the main window
    
    """
    def __init__(self,panel):
        self.__panel = panel
        self.__sizer = wx.BoxSizer(wx.VERTICAL)
        self.__image_folder_panel = wx.Panel(panel,-1)
        self.__image_edit_box = self.__make_folder_panel(self.__image_folder_panel,
                                                         cellprofiler.preferences.get_default_image_directory(),
                                                         'default image folder',
                                                         DEFAULT_IMAGE_FOLDER_HELP,
                                                         [cellprofiler.preferences.set_default_image_directory,
                                                          self.__notify_pipeline_list_view_directory_change])
        self.__output_folder_panel = wx.Panel(panel,-1)
        self.__output_edit_box = self.__make_folder_panel(self.__output_folder_panel,
                                                          cellprofiler.preferences.get_default_output_directory(),
                                                          'default output folder',
                                                          DEFAULT_OUTPUT_FOLDER_HELP,
                                                          [cellprofiler.preferences.set_default_output_directory,
                                                           self.__notify_pipeline_list_view_directory_change])
        self.__odds_and_ends_panel = wx.Panel(panel,-1)
        self.__make_odds_and_ends_panel()
        self.__status_text = wx.StaticText(panel,-1,style=wx.SUNKEN_BORDER,label=WELCOME_MESSAGE)
        self.__sizer.AddMany([(self.__image_folder_panel,0,wx.EXPAND|wx.ALL,1),
                              (self.__output_folder_panel,0,wx.EXPAND|wx.ALL,1),
                              (self.__odds_and_ends_panel,0,wx.EXPAND|wx.ALL,1),
                              (self.__status_text,0,wx.EXPAND|wx.ALL,2)])
        panel.SetSizer(self.__sizer)
        self.__errors = set()
        self.__pipeline_list_view = None
        
    def __make_folder_panel(self, panel, value, text, help_text, actions):
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        help_button = wx.Button(panel,-1,'?',(0,0),(25,25))
        text_static = wx.StaticText(panel,-1,string.capitalize(text)+':')
        edit_box = wx.TextCtrl(panel,-1)
        edit_box.SetValue(value)
        browse_bmp = wx.ArtProvider.GetBitmap(wx.ART_FOLDER_OPEN,
                                              wx.ART_CMN_DIALOG,
                                              (16,16))
        browse_button = wx.BitmapButton(panel,-1,bitmap = browse_bmp)
        browse_button.SetToolTipString("Browse for %s folder" % text)
        
        new_bmp = wx.ArtProvider.GetBitmap(wx.ART_NEW_DIR,
                                           wx.ART_CMN_DIALOG,
                                           (16,16))
        new_button = wx.BitmapButton(panel,-1,bitmap=new_bmp)
        new_button.SetToolTipString("Make a new sub-folder")
        if os.path.isdir(value):
            new_button.Disable()
        sizer.AddMany([(help_button,0,wx.ALL,1),
                       (text_static,0,wx.EXPAND,1),
                       (edit_box,3,wx.EXPAND|wx.ALL,1),
                       (browse_button,0,0|wx.ALL,1),
                       (new_button,0,0|wx.ALL,1)])
        panel.SetSizer(sizer)
        def on_new_folder(event):
            if os.path.exists(edit_box.Value):
                return
            if wx.MessageBox("Do you really want to create the %s folder?" %
                             edit_box.Value,style=wx.YES_NO) == wx.YES:
                os.makedirs(edit_box.Value)
                self.__on_edit_box_change(event, edit_box, text, actions)
            
        def on_edit_box_change(event):
            if os.path.isdir(edit_box.Value):
                new_button.Disable()
                new_button.SetToolTipString("%s is a directory" % 
                                            edit_box.Value)
            else:
                new_button.Enable()
                new_button.SetToolTipString("Press button to create the %s directory" %
                                            edit_box.Value)
            self.__on_edit_box_change(event, edit_box, text, actions)
            
        panel.Bind(wx.EVT_BUTTON, lambda event: self.__on_help(event, help_text),
                   help_button)
        panel.Bind(wx.EVT_BUTTON, lambda event: self.__on_browse(event, edit_box, text), browse_button)
        panel.Bind(wx.EVT_TEXT, on_edit_box_change, edit_box)
        panel.Bind(wx.EVT_BUTTON, on_new_folder, new_button)
        return edit_box
    
    def __make_odds_and_ends_panel(self):
        panel = self.__odds_and_ends_panel
        output_filename_text = wx.StaticText(panel,-1,'Output filename:')
        self.__output_filename_edit_box = wx.TextCtrl(panel,-1,'DefaultOUT.mat')
        output_filename_help_button = wx.Button(panel,-1,'?',(0,0),(25,25))
        self.__analyze_images_button = wx.Button(panel,-1,'Analyze images')
        self.__stop_analysis_button = wx.Button(panel,-1,'Stop analysis')
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.AddMany([(output_filename_text,0,wx.ALL,1),
                       (self.__output_filename_edit_box,3,wx.EXPAND|wx.ALL,1),
                       (output_filename_help_button,0,wx.ALL,1),
                       (self.__analyze_images_button,0,wx.ALL,1),
                       (self.__stop_analysis_button, 0, wx.ALL,1)])
        sizer.Hide(self.__stop_analysis_button)
        panel.SetSizer(sizer)
        panel.Bind(wx.EVT_BUTTON,
                   lambda event: self.__on_help(event, OUTPUT_FILENAME_HELP),
                   output_filename_help_button)
        panel.Bind(wx.EVT_TEXT, self.__on_output_filename_changed, self.__output_filename_edit_box)
        cellprofiler.preferences.add_output_file_name_listener(self.__on_preferences_output_filename_event)
        cellprofiler.preferences.add_image_directory_listener(self.__on_preferences_image_directory_event)
        cellprofiler.preferences.add_output_directory_listener(self.__on_preferences_output_directory_event)
        panel.Bind(wx.EVT_WINDOW_DESTROY, self.__on_destroy, panel)
    
    def check_preferences(self):
        '''Return True if preferences are OK (e.g. directories exist)'''
        path = self.__image_edit_box.Value
        if not os.path.isdir(path):
            if wx.MessageBox(('The default image directory is "%s", but '
                              'the directory does not exist. Do you want to '
                              'create it?') % path, 
                             "Warning, cannot run pipeline",
                             style = wx.YES_NO) == wx.NO:
                return False, "Image directory does not exist"
            os.makedirs(path)
            cellprofiler.preferences.set_default_image_directory(path)
        path = self.__output_edit_box.Value
        if not os.path.isdir(path):
            if wx.MessageBox(('The default output directory is "%s", but '
                              'the directory does not exist. Do you want to '
                              'create it?') % path, 
                             "Warning, cannot run pipeline",
                             style = wx.YES_NO) == wx.NO:
                return False, "Output directory does not exist"
            os.makedirs(path)
            cellprofiler.preferences.set_default_output_directory(path)
        return True, "OK"
                          
    def __on_destroy(self, event):
        cellprofiler.preferences.remove_image_directory_listener(self.__on_preferences_image_directory_event)
        cellprofiler.preferences.remove_output_directory_listener(self.__on_preferences_output_directory_event)
        cellprofiler.preferences.remove_output_file_name_listener(self.__on_preferences_output_filename_event)

    def attach_to_pipeline_controller(self, pipeline_controller):
        self.__panel.Bind(wx.EVT_BUTTON,
                          pipeline_controller.on_analyze_images, 
                          self.__analyze_images_button)
        self.__panel.Bind(wx.EVT_BUTTON,
                          pipeline_controller.on_stop_running,
                          self.__stop_analysis_button)
    
    def attach_to_pipeline_list_view(self, pipeline_list_view):
        self.__pipeline_list_view = pipeline_list_view
    
    def start_debugging(self):
        self.__analyze_images_button.Disable()
        self.__stop_analysis_button.Disable()
        
    def stop_debugging(self):
        self.__analyze_images_button.Enable()
        self.__stop_analysis_button.Enable()
        
    def on_analyze_images(self):
        self.__odds_and_ends_panel.Sizer.Hide(self.__analyze_images_button)
        self.__odds_and_ends_panel.Sizer.Show(self.__stop_analysis_button)
        self.__odds_and_ends_panel.Layout()
    
    def on_stop_analysis(self):
        self.__odds_and_ends_panel.Sizer.Show(self.__analyze_images_button)
        self.__odds_and_ends_panel.Sizer.Hide(self.__stop_analysis_button)
        self.__odds_and_ends_panel.Layout()
        
    def set_message_text(self,text):
        saved_size = self.__status_text.GetSize()
        self.__status_text.SetLabel(text)
        self.__status_text.SetSize(saved_size)
    
    def pop_error_text(self,error_text):
        if error_text in self.__errors:
            self.__errors.remove(error_text)
            if len(self.__errors) == 0:
                self.set_message_color(wx.Colour(0,0,0))
                self.set_message_text(WELCOME_MESSAGE)
            else:
                self.set_message_text(self.__errors.__iter__().next())
        
    def set_message_color(self,color):
        self.__status_text.SetForegroundColour(color)
    
    def set_error_text(self,error_text):
        self.set_message_text(error_text)
        self.set_message_color(wx.Colour(255,0,0))
        self.__errors.add(error_text)
        
    def __on_browse(self, event, edit_box, text):
        dir_dialog = wx.DirDialog(self.__panel,string.capitalize(text), edit_box.GetValue())
        if dir_dialog.ShowModal() == wx.ID_OK:
            edit_box.SetValue(dir_dialog.GetPath())

    def __on_edit_box_change(self, event, edit_box, text, actions):
        path = edit_box.GetValue()
        error_text = 'The %s is not a directory'%(text)
        if os.path.isdir(path):
            for action in actions:
                action(path)
            self.pop_error_text(error_text)
        else:
            self.set_error_text(error_text)
    
    def __on_help(self,event, help_text):
        dlg = HTMLDialog(self.__panel, "Help", help_text)
        dlg.Show()
    
    def __on_pixel_size_changed(self,event):
        error_text = 'Pixel size must be a number'
        text = self.__pixel_size_edit_box.Value
        if text.isdigit():
            cellprofiler.preferences.set_pixel_size(int(text))
            self.pop_error_text(error_text)
        else:
            self.set_error_text(error_text)
    
    def __on_output_filename_changed(self,event):
        cellprofiler.preferences.set_output_file_name(self.__output_filename_edit_box.Value)
    
    def __on_preferences_output_filename_event(self,event):
        old_selection = self.__output_filename_edit_box.Selection
        if self.__output_filename_edit_box.Value != cellprofiler.preferences.get_output_file_name():
            self.__output_filename_edit_box.Value = cellprofiler.preferences.get_output_file_name()
            self.__output_filename_edit_box.SetSelection(*old_selection)
        
    def __on_preferences_output_directory_event(self,event):
        old_selection = self.__output_edit_box.Selection
        if self.__output_edit_box.Value != cellprofiler.preferences.get_default_output_directory():
            self.__output_edit_box.Value = cellprofiler.preferences.get_default_output_directory()
            self.__output_edit_box.SetSelection(*old_selection)
    
    def __on_preferences_image_directory_event(self, event):
        old_selection = self.__image_edit_box.Selection
        if self.__image_edit_box.Value != cellprofiler.preferences.get_default_image_directory():
            self.__image_edit_box.Value = cellprofiler.preferences.get_default_image_directory()
            self.__image_edit_box.SetSelection(*old_selection)

    def __notify_pipeline_list_view_directory_change(self, path):
        # modules may need revalidation
        if self.__pipeline_list_view is not None:
            self.__pipeline_list_view.notify_directory_change()
