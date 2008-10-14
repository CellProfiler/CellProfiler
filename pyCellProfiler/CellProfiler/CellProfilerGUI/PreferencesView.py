"""PreferencesView.py - displays the default preferences in the lower right corner

    $Revision$
"""

import os
import string
import wx
import CellProfiler.Preferences

WELCOME_MESSAGE = 'Welcome to CellProfiler'

class PreferencesView:
    """View / controller for the preferences that get displayed in the main window
    
    """
    def __init__(self,panel):
        self.__panel = panel
        self.__sizer = wx.BoxSizer(wx.VERTICAL)
        self.__image_folder_panel = wx.Panel(panel,-1)
        self.__image_edit_box = self.__make_folder_panel(self.__image_folder_panel,
                                                         CellProfiler.Preferences.GetDefaultImageDirectory(),
                                                         'default image folder',
                                                         'HelpDefaultImageFolder.m',
                                                         CellProfiler.Preferences.SetDefaultImageDirectory)
        self.__output_folder_panel = wx.Panel(panel,-1)
        self.__output_edit_box = self.__make_folder_panel(self.__output_folder_panel,
                                                          CellProfiler.Preferences.GetDefaultOutputDirectory(),
                                                          'default output folder',
                                                          'HelpDefaultOutputFolder.m',
                                                          CellProfiler.Preferences.SetDefaultOutputDirectory)
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
        panel.Bind(wx.EVT_BUTTON,lambda event: self.__OnHelp(event, helpfile))
        panel.Bind(wx.EVT_BUTTON,lambda event: self.__OnBrowse(event,edit_box,text,action),browse_button)
        panel.Bind(wx.EVT_TEXT,lambda event: self.__OnEditBoxChange(event, edit_box, text,action),edit_box)
        return edit_box
    
    def __make_odds_and_ends_panel(self):
        panel = self.__odds_and_ends_panel
        pixel_help_button = wx.Button(panel,-1,'?',(0,0),(25,25))
        pixel_size_text = wx.StaticText(panel,-1,'Pixel size:')
        self.__pixel_size_edit_box = wx.TextCtrl(panel,-1,'1',(0,0),(25,20))
        output_filename_text = wx.StaticText(panel,-1,'Output filename:')
        self.__output_filename_edit_box = wx.TextCtrl(panel,-1,'DefaultOUT.mat')
        output_filename_help_button = wx.Button(panel,-1,'?',(0,0),(25,25))
        self.__analyze_images_button = wx.Button(panel,-1,'Analyze images')
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.AddMany([(pixel_help_button,0,wx.ALL,1),
                       (pixel_size_text,0,wx.ALL,1),
                       (self.__pixel_size_edit_box,0,wx.ALL,1),
                       (output_filename_text,0,wx.ALL,1),
                       (self.__output_filename_edit_box,3,wx.EXPAND|wx.ALL,1),
                       (output_filename_help_button,0,wx.ALL,1),
                       (self.__analyze_images_button,0,wx.ALL,1)])
        panel.SetSizer(sizer)
        panel.Bind(wx.EVT_BUTTON,lambda event: self.__OnHelp(event,"HelpPixelSize.m"),pixel_help_button)
        panel.Bind(wx.EVT_BUTTON,lambda event: self.__OnHelp(event,"HelpOutputFileName.m"),output_filename_help_button)
        panel.Bind(wx.EVT_TEXT, self.__OnPixelSizeChanged, self.__pixel_size_edit_box)
        panel.Bind(wx.EVT_TEXT, self.__OnOutputFilenameChanged, self.__output_filename_edit_box)
        CellProfiler.Preferences.AddOutputFileNameListener(self.__OnPreferencesOutputFilenameEvent)

    def AttachToPipelineController(self,pipeline_controller):
        self.__panel.Bind(wx.EVT_BUTTON,pipeline_controller.OnAnalyzeImages, self.__analyze_images_button)
        
    def SetMessageText(self,text):
        saved_size = self.__status_text.GetSize()
        self.__status_text.SetLabel(text)
        self.__status_text.SetSize(saved_size)
    
    def PopErrorText(self,error_text):
        if error_text in self.__errors:
            self.__errors.remove(error_text)
            if len(self.__errors) == 0:
                self.SetMessageColor(wx.Color(0,0,0))
                self.SetMessageText(WELCOME_MESSAGE)
        
    def SetMessageColor(self,color):
        self.__status_text.SetForegroundColour(color)
    
    def SetErrorText(self,error_text):
        self.SetMessageText(error_text)
        self.SetMessageColor(wx.Color(255,0,0))
        self.__errors.add(error_text)
        
    def __OnBrowse(self,event,edit_box,text,action):
        dir_dialog = wx.DirDialog(self.__panel,string.capitalize(text),edit_box.GetValue())
        if dir_dialog.ShowModal() == wx.ID_OK:
            edit_box.SetValue(dir_dialog.GetPath())

    def __OnEditBoxChange(self,event,edit_box,text,action):
        path = edit_box.GetValue()
        error_text = 'The %s is not a directory'%(text)
        if os.path.isdir(path):
            action(path)
            self.PopErrorText(error_text)
        else:
            self.SetErrorText(error_text)
    
    def __OnHelp(self,event,helpfile):
        pass
    
    def __OnPixelSizeChanged(self,event):
        error_text = 'Pixel size must be a number'
        text = self.__pixel_size_edit_box.Value
        if text.isdigit():
            CellProfiler.Preferences.SetPixelSize(int(text))
            self.PopErrorText(error_text)
        else:
            self.SetErrorText(error_text)
    
    def __OnOutputFilenameChanged(self,event):
        CellProfiler.Preferences.SetOutputFileName(self.__output_filename_edit_box.Value)
    
    def __OnPreferencesOutputFilenameEvent(self,event):
        if self.__output_filename_edit_box.Value != CellProfiler.Preferences.GetOutputFileName():
            self.__output_filename_edit_box.Value = CellProfiler.Preferences.GetOutputFileName()
        
