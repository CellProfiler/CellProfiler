""" CellProfiler.CellProfilerGUI.CPFrame - Cell Profiler's main window

    $Revision$   
    """

import os
import wx
import wx.lib.inspection
import wx.lib.scrolledpanel
import cellprofiler.preferences
from cellprofiler.gui.pipelinelistview import PipelineListView
from cellprofiler.pipeline import Pipeline
from cellprofiler.gui.pipelinecontroller import PipelineController
from cellprofiler.gui.moduleview import ModuleView
from cellprofiler.gui.preferencesview import PreferencesView
from cellprofiler.gui.directoryview import DirectoryView
import traceback
import sys

ID_FILE_LOAD_PIPELINE=100
ID_FILE_EXIT=101
ID_FILE_WIDGET_INSPECTOR=102
ID_FILE_SAVE_PIPELINE=103
ID_FILE_CLEAR_PIPELINE=104
ID_FILE_ANALYZE_IMAGES=105

ID_HELP_MODULE=200

class CPFrame(wx.Frame):
    def __init__(self, *args, **kwds):
        """Initialize the frame and its layout
        
        """
        kwds["style"] = wx.DEFAULT_FRAME_STYLE
        wx.Frame.__init__(self, *args, **kwds)
        self.__top_left_panel = wx.Panel(self,-1)
        self.__logo_panel = wx.Panel(self,-1,style=wx.RAISED_BORDER)
        self.__module_list_panel = wx.Panel(self.__top_left_panel,-1)
        self.__module_controls_panel = wx.Panel(self.__top_left_panel,-1)
        self.__module_panel = wx.lib.scrolledpanel.ScrolledPanel(self,-1,style=wx.SUNKEN_BORDER)
        self.__file_list_panel = wx.Panel(self,-1)
        self.__preferences_panel = wx.Panel(self,-1)
        self.__pipeline = Pipeline()
        self.__add_menu()
        self.__attach_views()
        self.__set_properties()
        self.__set_icon()
        self.__layout_logo()
        self.__do_layout()
 
    def __set_properties(self):
        self.SetTitle("CellProfiler")
        self.SetSize((640, 480))
 
    def __add_menu(self):
        """Add the menu to the frame
        
        """
        self.__menu_bar = wx.MenuBar()
        self.__menu_file = wx.Menu()
        self.__menu_file.Append(ID_FILE_LOAD_PIPELINE,'Load Pipeline...\tctrl+P','Load a pipeline from a .MAT file')
        self.__menu_file.Append(ID_FILE_SAVE_PIPELINE,'Save Pipeline as...','Save a pipeline as a .MAT file')
        self.__menu_file.Append(ID_FILE_CLEAR_PIPELINE,'Clear pipeline','Remove all modules from the current pipeline')
        self.__menu_file.AppendSeparator()
        self.__menu_file.Append(ID_FILE_ANALYZE_IMAGES,'Analyze images\tctrl+L','Run the pipeline on the images in the image directory')
        self.__menu_file.AppendSeparator()
        self.__menu_file.Append(ID_FILE_WIDGET_INSPECTOR,'Widget inspector','Run the widget inspector for debugging the UI')
        self.__menu_file.Append(ID_FILE_EXIT,'E&xit\tctrl+Q','Quit the application')
        self.__menu_bar.Append(self.__menu_file,'&File')
        self.__menu_help = wx.Menu()
        self.__menu_help.Append(ID_HELP_MODULE,'Module help','Display help from the module''s .m file')
        self.__menu_bar.Append(self.__menu_help,'&Help')
        self.SetMenuBar(self.__menu_bar)
        wx.EVT_MENU(self,ID_FILE_EXIT,lambda event: self.Close())
        wx.EVT_MENU(self,ID_FILE_WIDGET_INSPECTOR,self.__on_widget_inspector)
        wx.EVT_MENU(self,ID_HELP_MODULE,self.__on_help_module)
        accelerator_table = wx.AcceleratorTable([(wx.ACCEL_CTRL,ord('L'),ID_FILE_ANALYZE_IMAGES),
                                                 (wx.ACCEL_CTRL,ord('P'),ID_FILE_LOAD_PIPELINE),
                                                 (wx.ACCEL_CTRL,ord('Q'),ID_FILE_EXIT)])
        self.SetAcceleratorTable(accelerator_table)
    
    def __on_widget_inspector(self, evt):
        wx.lib.inspection.InspectionTool().Show()

    def __on_help_module(self,event):
        modules = self.__pipeline_list_view.get_selected_modules()
        filename = self.__get_icon_filename()
        icon = wx.Icon(filename,wx.BITMAP_TYPE_PNG)
        font = wx.SystemSettings.GetFont(wx.SYS_SYSTEM_FIXED_FONT)
        bgcolor = wx.SystemSettings.GetColour(wx.SYS_COLOUR_LISTBOX)
        for module in modules:
            helpframe = wx.Frame(self,-1,'Help for module, "%s"'%(module.module_name),size=(640,480))
            sizer = wx.BoxSizer()
            panel = wx.lib.scrolledpanel.ScrolledPanel(helpframe,-1,style=wx.SUNKEN_BORDER)
            panel.SetBackgroundColour(bgcolor)
            sizer.Add(panel,1,wx.EXPAND)
            helpframe.SetSizer(sizer)
            statictext = wx.StaticText(panel,-1,module.GetHelp())
            statictext.SetFont(font)
            statictext.SetBackgroundColour(bgcolor)
            sizer = wx.BoxSizer()
            sizer.Add(statictext,1,wx.EXPAND|wx.ALL,5)
            panel.SetSizer(sizer)
            panel.SetupScrolling()
            helpframe.SetIcon(icon)
            helpframe.Layout()
            helpframe.Show()
        
    def __attach_views(self):
        self.__pipeline_list_view = PipelineListView(self.__module_list_panel)
        self.__pipeline_controller = PipelineController(self.__pipeline,self)
        self.__pipeline_list_view.attach_to_pipeline(self.__pipeline,self.__pipeline_controller)
        self.__pipeline_controller.attach_to_module_controls_panel(self.__module_controls_panel)
        self.__module_view = ModuleView(self.__module_panel,self.__pipeline)
        self.__pipeline_controller.attach_to_module_view(self.__module_view)
        self.__pipeline_list_view.attach_to_module_view((self.__module_view))
        self.__preferences_view = PreferencesView(self.__preferences_panel)
        self.__preferences_view.attach_to_pipeline_controller(self.__pipeline_controller)
        self.__directory_view = DirectoryView(self.__file_list_panel)
        self.__pipeline_controller.attach_to_directory_view(self.__directory_view)
        
    def __do_layout(self):
        self.__sizer = CPSizer(2,2,0,1)
        self.__top_left_sizer = wx.FlexGridSizer(3,1,1,1)
        self.__top_left_sizer.Add(self.__logo_panel,0,wx.EXPAND|wx.ALL,1)
        self.__top_left_sizer.Add(self.__module_list_panel,1,wx.EXPAND|wx.ALL,1)
        self.__top_left_sizer.Add(self.__module_controls_panel,0,wx.EXPAND|wx.ALL,2)
        self.__top_left_sizer.AddGrowableRow(1)
        self.__top_left_panel.SetSizer(self.__top_left_sizer)
        self.__sizer.AddMany([(self.__top_left_panel,0,wx.EXPAND),
                         (self.__module_panel,1,wx.EXPAND),
                         (self.__file_list_panel,0,wx.EXPAND),
                         (self.__preferences_panel,0,wx.EXPAND)])
        self.__sizer.set_ignore_height(0,1) # Ignore the best height for the file list panel
        self.__sizer.set_ignore_height(0,0) # Ignore the best height for the module list panel
        self.SetSizer(self.__sizer)
        self.Layout()
        self.__directory_view.set_height(self.__preferences_panel.GetBestSize()[1])

    def __layout_logo(self):
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        image = wx.Image(self.__get_icon_filename(),wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        logopic = wx.StaticBitmap(self.__logo_panel,-1,image)
        logotext = wx.StaticText(self.__logo_panel,-1,"Cell Profiler\nimage analysis\npipeline",style=wx.ALIGN_CENTER)
        sizer.AddMany([(logopic,0,wx.ALIGN_LEFT|wx.ALIGN_TOP|wx.ALL,5),
                       (logotext,1,wx.EXPAND)])
        self.__logo_panel.SetSizer(sizer)
    
    def __get_icon_filename(self):
        return os.path.join(cellprofiler.preferences.python_root_directory(),'CellProfilerIcon.png')
    
    def __set_icon(self):
        filename = self.__get_icon_filename()
        icon = wx.Icon(filename,wx.BITMAP_TYPE_PNG)
        self.SetIcon(icon)
 
    def display_error(self,message,error):
        """Displays an exception in a standardized way
        
        """
        tb = sys.exc_info()[2]
        traceback.print_tb(tb)
        text = '\n'.join(traceback.format_list(traceback.extract_tb(tb)))
        text = error.message + '\n'+text
        wx.MessageBox(text,"Caught exception during operation")
    
    def get_preferences_view(self):
        return self.__preferences_view
    
    preferences_view = property(get_preferences_view)

class CPSizer(wx.PySizer):
    """A grid sizer that deals out leftover sizes to the hungry row and column
    
    """
    # If this were for use outside of here, it would look at the positioning flags such
    # as wx.EXPAND and wx.ALIGN... in RecalcSizes, but we assume everything wants
    # to be expanded
    def __init__(self,rows,cols,hungry_row,hungry_col):
        wx.PySizer.__init__(self)
        self.__rows = rows
        self.__cols = cols
        self.__hungry_row = hungry_row
        self.__hungry_col = hungry_col
        self.__ignore_width = [[False for j in range(0,rows)] for i in range(0,cols)]
        self.__ignore_height = [[False for j in range(0,rows)] for i in range(0,cols)]
    
    def set_ignore_width(self,col,row,ignore=True):
        """Don't pay any attention to the minimum width of the item in grid cell col,row
        
        """
        self.__ignore_width[col][row]=ignore
    
    def get_ignore_width(self,col,row):
        """Return true if we should ignore the minimum width of the item at col,row
        
        """
        return self.__ignore_width[col][row]
    
    def set_ignore_height(self,col,row,ignore=True):
        """Don't pay any attention to the minimum height of the item in grid cell col,row
        
        """
        self.__ignore_height[col][row]=ignore
    
    def get_ignore_height(self,col,row):
        """Return true if we should ignore the minimum height of the item at col,row
        
        """
        return self.__ignore_height[col][row]
    
    def CalcMin(self):
        """Calculate the minimum row and column and add
        """
        (row_heights, col_widths) = self.__get_min_sizes()
        return wx.Size(sum(col_widths),sum(row_heights))
    
    def __get_min_sizes(self):
        row_heights=[0 for i in range(0,self.__rows)]
        col_widths=[0 for i in range(0,self.__cols)]
        idx = 0
        for item in self.GetChildren():
            row,col = divmod(idx,self.__rows)
            size = item.CalcMin()
            if not self.get_ignore_width(col,row):
                col_widths[col]=max(col_widths[col],size[0])
            if not self.get_ignore_height(col,row):
                row_heights[row]=max(row_heights[row],size[1])
            idx+=1
        return (row_heights,col_widths)
    
    def RecalcSizes(self):
        """Recalculate the sizes of our items, distributing leftovers among them  
        """
        (row_heights, col_widths) = self.__get_min_sizes()
        size = self.GetSize()
        leftover_width = size[0]- sum(col_widths)
        leftover_height = size[1] - sum(row_heights)
        col_widths[self.__hungry_col]+=leftover_width
        row_heights[self.__hungry_row]+=leftover_height
        idx = 0
        for item in self.GetChildren():
            row,col = divmod(idx,self.__rows)
            item_size = wx.Size(col_widths[col],row_heights[row])
            item_pos = wx.Point(sum(col_widths[:col]),sum(row_heights[:row]))
            item.SetDimension(item_pos,item_size)
            idx+=1
    
        
