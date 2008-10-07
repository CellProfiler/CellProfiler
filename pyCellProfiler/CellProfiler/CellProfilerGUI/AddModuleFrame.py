"""AddModuleFrame.py - this is the window frame and the subwindows
that give you the GUI to add a module to a pipeline

    $Revision$
"""

import os
import re
import wx
import CellProfiler.Preferences
class AddModuleFrame(wx.Frame):
    """The window frame that lets you add modules to a pipeline
    
    """
    def __init__(self, *args, **kwds):
        """Initialize the frame and its layout
        
        """
        kwds["style"] = wx.DEFAULT_FRAME_STYLE
        wx.Frame.__init__(self, *args, **kwds)
        self.SetSize(wx.Size(320,360))
        # Top level panels
        left_panel = wx.Panel(self,-1)
        right_panel = wx.Panel(self,-1)
        self.SetBackgroundColour(left_panel.GetBackgroundColour())
        # Module categories (in left panel)
        module_categories_text = wx.StaticText(left_panel,-1,'Module Categories',style=wx.TEXT_ALIGNMENT_CENTER)
        font = module_categories_text.GetFont()
        module_categories_text.SetFont(wx.Font(font.GetPointSize()*1.2,font.GetFamily(),font.GetStyle(),wx.FONTWEIGHT_BOLD))
        self.__module_categories_list_box = wx.ListBox(left_panel,-1)
        # Control panel for the selected module
        selected_module_panel = wx.Panel(left_panel,-1)
        selected_module_static_box = wx.StaticBox(selected_module_panel,-1,'For Selected Module')
        add_to_pipeline_button = wx.Button(selected_module_panel,-1,'+ Add to Pipeline')
        module_help_button = wx.Button(selected_module_panel,-1,'? Module Help')
        # Other buttons
        getting_started_button = wx.Button(left_panel,-1,'? Getting Started')
        browse_button = wx.Button(left_panel,-1,'Browse...')
        done_button = wx.Button(left_panel,-1,'Done')
        # Right-side panel
        self.__module_list_box = wx.ListBox(right_panel,-1)
        # Sizers
        top_sizer = wx.BoxSizer(wx.HORIZONTAL)
        top_sizer.AddMany([(left_panel,1,wx.EXPAND|wx.LEFT,5),
                           (right_panel,1,wx.EXPAND)])
        self.SetSizer(top_sizer)
        left_sizer = wx.BoxSizer(wx.VERTICAL)
        left_sizer.Add(module_categories_text,0,wx.CENTER|wx.ALL,5)
        left_sizer.AddSpacer(4)
        left_sizer.Add(self.__module_categories_list_box,0,wx.EXPAND|wx.LEFT|wx.RIGHT,10)
        left_sizer.AddStretchSpacer(1)
        left_sizer.Add(selected_module_panel,0,wx.EXPAND)
        left_sizer.AddStretchSpacer(1)
        left_sizer.Add(getting_started_button,0,wx.EXPAND)
        left_sizer.AddSpacer(2)
        left_sizer.Add(browse_button,0,wx.EXPAND)
        left_sizer.AddSpacer(2)
        left_sizer.Add(done_button,0,wx.EXPAND |wx.BOTTOM,5)
        left_panel.SetSizer(left_sizer)
        
        right_sizer = wx.BoxSizer(wx.VERTICAL)
        right_sizer.Add(self.__module_list_box,1,wx.EXPAND|wx.ALL,5)
        right_panel.SetSizer(right_sizer)
        
        selected_module_panel_sizer = wx.StaticBoxSizer(selected_module_static_box,wx.VERTICAL)
        selected_module_panel_sizer.Add(add_to_pipeline_button,0,wx.EXPAND)
        selected_module_panel_sizer.AddSpacer(2)
        selected_module_panel_sizer.Add(module_help_button,0,wx.EXPAND)
        selected_module_panel.SetSizer(selected_module_panel_sizer)
        
        self.__set_icon()
        self.Bind(wx.EVT_CLOSE,self.__onClose, self)
        self.Bind(wx.EVT_LISTBOX,self.__onCategorySelected,self.__module_categories_list_box)
        self.Bind(wx.EVT_BUTTON,self.__onAddToPipeline,add_to_pipeline_button)
        self.Bind(wx.EVT_BUTTON,self.__onClose,done_button)
        self.__get_module_files()
        self.__set_categories()
        self.__listeners = []
        self.__module_categories_list_box.Select(0)
        self.__onCategorySelected(None)
        self.Layout()
        
    def __onClose(self,event):
        self.Hide()
        
    def __set_icon(self):
        filename=os.path.join(CellProfiler.Preferences.PythonRootDirectory(),'CellProfilerIcon.png')
        icon = wx.Icon(filename,wx.BITMAP_TYPE_PNG)
        self.SetIcon(icon)
        
    def __get_module_files(self):
        self.__module_files = [ 'File Processing',
                                'Image Processing',
                                'Object Processing',
                                'Measurement',
                                'Other'
                               ]
        self.__module_dict = {}
        for key in self.__module_files:
            self.__module_dict[key] = {}
            
        files = [x for x in os.listdir(CellProfiler.Preferences.ModuleDirectory())
                 if os.path.splitext(x)[1] == CellProfiler.Preferences.ModuleExtension()]
        files.sort()
        for file in files:
            module_path = os.path.join(CellProfiler.Preferences.ModuleDirectory(),file)
            fid = open(module_path,'r')
            try:
                category = 'Other'
                for line in fid:
                    match = re.match('^% Category: (.+)$',line)
                    if match:
                        category = match.groups()[0]
                        break
                if not self.__module_dict.has_key(category):
                    self.__module_files.insert(-1,category)
                    self.__module_dict[category] = {}
                self.__module_dict[category][os.path.splitext(file)[0]] = module_path
            finally:
                fid.close()
    
    def __set_categories(self):
        self.__module_categories_list_box.AppendItems(self.__module_files)
        
    def __onCategorySelected(self,event):
        category=self.__GetSelectedCategory()
        self.__module_list_box.Clear()
        keys = self.__module_dict[category].keys()
        keys.sort()
        self.__module_list_box.AppendItems(keys)
        self.__module_list_box.Select(0)

    def __GetSelectedCategory(self):
        return self.__module_files[self.__module_categories_list_box.GetSelection()]

    def __onAddToPipeline(self,event):
        category = self.__GetSelectedCategory()
        idx = self.__module_list_box.GetSelection()
        if idx != wx.NOT_FOUND:
            file = self.__module_list_box.GetItems()[idx]
            self.Notify(AddToPipelineEvent(file,self.__module_dict[category][file]))
            
    def AddListener(self,listener):
        self.__listeners.append(listener)
        
    def RemoveListener(self,listener):
        self.__listeners.remove(listener)
    
    def Notify(self,event):
        for listener in self.__listeners:
            listener(self,event)

class AddToPipelineEvent:
    def __init__(self,ModuleName,ModulePath):
        self.ModuleName = ModuleName
        self.ModulePath = ModulePath
