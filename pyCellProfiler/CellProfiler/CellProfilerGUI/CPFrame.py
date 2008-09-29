""" CellProfiler.CellProfilerGUI.CPFrame - Cell Profiler's main window

    $Revision$   
    """
    
import wx
import wx.lib.inspection
from CellProfiler.CellProfilerGUI.PipelineListView import PipelineListView
from CellProfiler.Pipeline import Pipeline
from CellProfiler.CellProfilerGUI.PipelineController import PipelineController
from CellProfiler.CellProfilerGUI.ModuleView import ModuleView
import traceback
import sys

ID_FILE_LOAD_PIPELINE=100
ID_FILE_EXIT=101
ID_FILE_WIDGET_INSPECTOR=102

class CPFrame(wx.Frame):
    def __init__(self, *args, **kwds):
        """Initialize the frame and its layout
        
        """
        kwds["style"] = wx.DEFAULT_FRAME_STYLE
        wx.Frame.__init__(self, *args, **kwds)
        self.__sizer = wx.FlexGridSizer(2,2,1,1)
        self.__TopLeftPanel = wx.Panel(self,-1)
        self.__TopLeftSizer = wx.FlexGridSizer(3,1,1,1)
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
        self.__do_layout()
 
    def __set_properties(self):
        self.SetTitle("CellProfiler")
        self.SetSize((400, 300))
 
    def __add_menu(self):
        """Add the menu to the frame
        
        """
        self.__menu_bar = wx.MenuBar()
        self.__menu_file = wx.Menu()
        self.__menu_file.Append(ID_FILE_LOAD_PIPELINE,'Load Pipeline...','Load a pipeline from a .MAT file')
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
        self.__PipelineListView.AttachToModuleView((self.__ModuleView))
        
    def __do_layout(self):
        self.__TopLeftSizer.Add(self.__LogoPanel,0,wx.EXPAND|wx.ALL,1)
        self.__TopLeftSizer.Add(self.__ModuleListPanel,1,wx.EXPAND|wx.ALL,1)
        self.__TopLeftSizer.Add(self.__ModuleControlsPanel,0,wx.EXPAND|wx.ALL,1)
        self.__TopLeftSizer.AddGrowableRow(1)
        self.__TopLeftPanel.SetSizer(self.__TopLeftSizer)
        self.__sizer.AddMany([(self.__TopLeftPanel,0,wx.EXPAND),
                         (self.__ModulePanel,1,wx.EXPAND | wx.ALL),
                         (self.__FileListPanel,0,wx.EXPAND),
                         (self.__PreferencesPanel,1,wx.EXPAND)])
        self.__sizer.AddGrowableCol(1)
        self.__sizer.AddGrowableRow(0)
        self.SetSizer(self.__sizer)
        self.Layout()
 
    def DisplayError(self,message,error):
        """Displays an exception in a standardized way
        
        """
        tb = sys.exc_info()[2]
        text = '\n'.join(traceback.format_list(traceback.extract_tb(tb)))
        wx.MessageBox(text,error.message)
