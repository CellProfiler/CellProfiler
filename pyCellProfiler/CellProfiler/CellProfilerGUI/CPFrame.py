""" CellProfiler.CellProfilerGUI.CPFrame - Cell Profiler's main window

    $Revision$   
    """

import os
import wx
import wx.lib.inspection
import CellProfiler.Preferences
from CellProfiler.CellProfilerGUI.PipelineListView import PipelineListView
from CellProfiler.Pipeline import Pipeline
from CellProfiler.CellProfilerGUI.PipelineController import PipelineController
from CellProfiler.CellProfilerGUI.ModuleView import ModuleView
from CellProfiler.CellProfilerGUI.PreferencesView import PreferencesView
from CellProfiler.CellProfilerGUI.DirectoryView import DirectoryView
import traceback
import sys

ID_FILE_LOAD_PIPELINE=100
ID_FILE_EXIT=101
ID_FILE_WIDGET_INSPECTOR=102
ID_FILE_SAVE_PIPELINE=103
ID_FILE_CLEAR_PIPELINE=104

class CPFrame(wx.Frame):
    def __init__(self, *args, **kwds):
        """Initialize the frame and its layout
        
        """
        kwds["style"] = wx.DEFAULT_FRAME_STYLE
        wx.Frame.__init__(self, *args, **kwds)
        self.__TopLeftPanel = wx.Panel(self,-1)
        self.__LogoPanel = wx.Panel(self,-1)
        self.__ModuleListPanel = wx.Panel(self.__TopLeftPanel,-1)
        self.__ModuleControlsPanel = wx.Panel(self.__TopLeftPanel,-1)
        self.__ModulePanel = wx.Panel(self,-1)
        self.__FileListPanel = wx.Panel(self,-1)
        self.__PreferencesPanel = wx.Panel(self,-1)
        self.__Pipeline = Pipeline()
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
        self.__menu_file.Append(ID_FILE_LOAD_PIPELINE,'Load Pipeline...','Load a pipeline from a .MAT file')
        self.__menu_file.Append(ID_FILE_SAVE_PIPELINE,'Save Pipeline as...','Save a pipeline as a .MAT file')
        self.__menu_file.Append(ID_FILE_CLEAR_PIPELINE,'Clear pipeline','Remove all modules from the current pipeline')
        self.__menu_file.AppendSeparator()
        self.__menu_file.Append(ID_FILE_WIDGET_INSPECTOR,'Widget inspector','Run the widget inspector for debugging the UI')
        self.__menu_file.Append(ID_FILE_EXIT,'E&xit','Quit the application')
        wx.EVT_MENU(self,ID_FILE_EXIT,lambda event: self.Close())
        wx.EVT_MENU(self,ID_FILE_WIDGET_INSPECTOR,self.__OnWidgetInspector)
        self.__menu_bar.Append(self.__menu_file,'&File')
        self.SetMenuBar(self.__menu_bar)
    
    def __OnWidgetInspector(self, evt):
        wx.lib.inspection.InspectionTool().Show()

    def __attach_views(self):
        self.__PipelineListView = PipelineListView(self.__ModuleListPanel)
        self.__PipelineController = PipelineController(self.__Pipeline,self)
        self.__PipelineListView.AttachToPipeline(self.__Pipeline,self.__PipelineController)
        self.__PipelineController.AttachToModuleControlsPanel(self.__ModuleControlsPanel)
        self.__ModuleView = ModuleView(self.__ModulePanel,self.__Pipeline)
        self.__PipelineController.AttachToModuleView(self.__ModuleView)
        self.__PipelineListView.AttachToModuleView((self.__ModuleView))
        self.__PreferencesView = PreferencesView(self.__PreferencesPanel)
        self.__DirectoryView = DirectoryView(self.__FileListPanel)
        
    def __do_layout(self):
        #self.__sizer = wx.FlexGridSizer(2,2,1,1)
        self.__sizer = CPSizer(2,2,0,1)
        self.__TopLeftSizer = wx.FlexGridSizer(3,1,1,1)
        self.__TopLeftSizer.Add(self.__LogoPanel,0,wx.EXPAND|wx.ALL,1)
        self.__TopLeftSizer.Add(self.__ModuleListPanel,1,wx.EXPAND|wx.ALL,1)
        self.__TopLeftSizer.Add(self.__ModuleControlsPanel,0,wx.EXPAND|wx.ALL,1)
        self.__TopLeftSizer.AddGrowableRow(1)
        self.__TopLeftPanel.SetSizer(self.__TopLeftSizer)
        self.__sizer.AddMany([(self.__TopLeftPanel,0,wx.EXPAND),
                         (self.__ModulePanel,1,wx.EXPAND),
                         (self.__FileListPanel,0,wx.EXPAND),
                         (self.__PreferencesPanel,0,wx.EXPAND)])
        self.__sizer.SetIgnoreHeight(0,1) # Ignore the best height for the file list panel
        self.__sizer.SetIgnoreHeight(0,0) # Ignore the best height for the module list panel
        self.SetSizer(self.__sizer)
        self.Layout()
        self.__DirectoryView.SetHeight(self.__PreferencesPanel.GetBestSize()[1])

    def __layout_logo(self):
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        image = wx.Image(self.__get_icon_filename(),wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        logopic = wx.StaticBitmap(self.__LogoPanel,-1,image)
        logotext = wx.StaticText(self.__LogoPanel,-1,"Cell Profiler\nimage analysis\npipeline",style=wx.ALIGN_CENTER)
        sizer.AddMany([(logopic,0,wx.ALIGN_LEFT|wx.ALIGN_TOP|wx.ALL,5),
                       (logotext,1,wx.EXPAND)])
        self.__LogoPanel.SetSizer(sizer)
    
    def __get_icon_filename(self):
        return os.path.join(CellProfiler.Preferences.PythonRootDirectory(),'CellProfilerIcon.png')
    
    def __set_icon(self):
        filename = self.__get_icon_filename()
        icon = wx.Icon(filename,wx.BITMAP_TYPE_PNG)
        self.SetIcon(icon)
 
    def DisplayError(self,message,error):
        """Displays an exception in a standardized way
        
        """
        tb = sys.exc_info()[2]
        text = '\n'.join(traceback.format_list(traceback.extract_tb(tb)))
        wx.MessageBox(text,error.message)

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
    
    def SetIgnoreWidth(self,col,row,ignore=True):
        """Don't pay any attention to the minimum width of the item in grid cell col,row
        
        """
        self.__ignore_width[col][row]=ignore
    
    def GetIgnoreWidth(self,col,row):
        """Return true if we should ignore the minimum width of the item at col,row
        
        """
        return self.__ignore_width[col][row]
    
    def SetIgnoreHeight(self,col,row,ignore=True):
        """Don't pay any attention to the minimum height of the item in grid cell col,row
        
        """
        self.__ignore_height[col][row]=ignore
    
    def GetIgnoreHeight(self,col,row):
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
            if not self.GetIgnoreWidth(col,row):
                col_widths[col]=max(col_widths[col],size[0])
            if not self.GetIgnoreHeight(col,row):
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
    
        
