"""PreferencesView.py - displays the default preferences in the lower right corner

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import os
import string
import wx
import cellprofiler.preferences

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
                                                         'default input folder',
                                                         'HelpDefaultImageFolder.m',
                                                         cellprofiler.preferences.set_default_image_directory)
        self.__output_folder_panel = wx.Panel(panel,-1)
        self.__output_edit_box = self.__make_folder_panel(self.__output_folder_panel,
                                                          cellprofiler.preferences.get_default_output_directory(),
                                                          'default output folder',
                                                          'HelpDefaultOutputFolder.m',
                                                          cellprofiler.preferences.set_default_output_directory)
        self.__odds_and_ends_panel = wx.Panel(panel,-1)
        self.__make_odds_and_ends_panel()
        self.__status_text = wx.StaticText(panel,-1,style=wx.SUNKEN_BORDER,label=WELCOME_MESSAGE)
        self.__sizer.AddMany([(self.__image_folder_panel,0,wx.EXPAND|wx.ALL,1),
                              (self.__output_folder_panel,0,wx.EXPAND|wx.ALL,1),
                              (self.__odds_and_ends_panel,0,wx.EXPAND|wx.ALL,1),
                              (self.__status_text,0,wx.EXPAND|wx.ALL,2)])
        panel.SetSizer(self.__sizer)
        self.__errors = set()
        
    def __make_folder_panel(self,panel,value, text,helpfile,action):
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        help_button = wx.Button(panel,-1,'?',(0,0),(25,25))
        text_static = wx.StaticText(panel,-1,string.capitalize(text)+':')
        edit_box = wx.TextCtrl(panel,-1)
        edit_box.SetValue(value)
        browse_button = wx.Button(panel,-1,'Browse...')
        sizer.AddMany([(help_button,0,wx.ALL,1),
                       (text_static,0,wx.EXPAND,1),
                       (edit_box,3,wx.EXPAND|wx.ALL,1),
                       (browse_button,0,0|wx.ALL,1)])
        panel.SetSizer(sizer)
        panel.Bind(wx.EVT_BUTTON,lambda event: self.__on_help(event, helpfile))
        panel.Bind(wx.EVT_BUTTON,lambda event: self.__on_browse(event,edit_box,text,action),browse_button)
        panel.Bind(wx.EVT_TEXT,lambda event: self.__on_edit_box_change(event, edit_box, text,action),edit_box)
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
        panel.Bind(wx.EVT_BUTTON,lambda event: self.__on_help(event,"HelpOutputFileName.m"),output_filename_help_button)
        panel.Bind(wx.EVT_TEXT, self.__on_output_filename_changed, self.__output_filename_edit_box)
        cellprofiler.preferences.add_output_file_name_listener(self.__on_preferences_output_filename_event)
        cellprofiler.preferences.add_image_directory_listener(self.__on_preferences_image_directory_event)
        cellprofiler.preferences.add_output_directory_listener(self.__on_preferences_output_directory_event)
        panel.Bind(wx.EVT_WINDOW_DESTROY, self.__on_destroy, panel)
    
    def __on_destroy(self, event):
        cellprofiler.preferences.remove_image_directory_listener(self.__on_preferences_image_directory_event)
        cellprofiler.preferences.remove_output_directory_listener(self.__on_preferences_output_directory_event)
        cellprofiler.preferences.remove_output_file_name_listener(self.__on_preferences_output_filename_event)

    def attach_to_pipeline_controller(self,pipeline_controller):
        self.__panel.Bind(wx.EVT_BUTTON,
                          pipeline_controller.on_analyze_images, 
                          self.__analyze_images_button)
        self.__panel.Bind(wx.EVT_BUTTON,
                          pipeline_controller.on_stop_running,
                          self.__stop_analysis_button)
    
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
                self.set_message_color(wx.Color(0,0,0))
                self.set_message_text(WELCOME_MESSAGE)
            else:
                self.set_message_text(self.__errors.__iter__().next())
        
    def set_message_color(self,color):
        self.__status_text.SetForegroundColour(color)
    
    def set_error_text(self,error_text):
        self.set_message_text(error_text)
        self.set_message_color(wx.Color(255,0,0))
        self.__errors.add(error_text)
        
    def __on_browse(self,event,edit_box,text,action):
        dir_dialog = wx.DirDialog(self.__panel,string.capitalize(text),edit_box.GetValue())
        if dir_dialog.ShowModal() == wx.ID_OK:
            edit_box.SetValue(dir_dialog.GetPath())

    def __on_edit_box_change(self,event,edit_box,text,action):
        path = edit_box.GetValue()
        error_text = 'The %s is not a directory'%(text)
        if os.path.isdir(path):
            action(path)
            self.pop_error_text(error_text)
        else:
            self.set_error_text(error_text)
    
    def __on_help(self,event,helpfile):
        pass
    
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
