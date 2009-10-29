""" CellProfiler.CellProfilerGUI.CPFrame - Cell Profiler's main window

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import os
import wx
import wx.html
import wx.lib.scrolledpanel
import cellprofiler.preferences
from cellprofiler.gui import get_icon, get_cp_bitmap
from cellprofiler.gui.pipelinelistview import PipelineListView
from cellprofiler.gui.cpfigure import close_all
from cellprofiler.pipeline import Pipeline
from cellprofiler.gui.pipelinecontroller import PipelineController
from cellprofiler.gui.moduleview import ModuleView
from cellprofiler.gui.preferencesview import PreferencesView
from cellprofiler.gui.directoryview import DirectoryView
import cellprofiler.gui.preferencesdlg
import traceback
import sys

ID_FILE_LOAD_PIPELINE=wx.NewId()
ID_FILE_EXIT=wx.NewId()
ID_FILE_WIDGET_INSPECTOR=wx.NewId()
ID_FILE_SAVE_PIPELINE=wx.NewId()
ID_FILE_CLEAR_PIPELINE=wx.NewId()
ID_FILE_ANALYZE_IMAGES=wx.NewId()
ID_FILE_STOP_ANALYSIS=wx.NewId()
ID_FILE_PRINT=wx.NewId()

ID_EDIT_SELECT_ALL = wx.NewId()
ID_EDIT_COPY = wx.NewId()

ID_OPTIONS_PREFERENCES = wx.NewId()

ID_DEBUG_START = wx.NewId()
ID_DEBUG_STOP = wx.NewId()
ID_DEBUG_STEP = wx.NewId()
ID_DEBUG_NEXT_IMAGE_SET = wx.NewId()
ID_DEBUG_NEXT_GROUP = wx.NewId()
ID_DEBUG_CHOOSE_GROUP = wx.NewId()
ID_DEBUG_CHOOSE_IMAGE_SET = wx.NewId()
ID_DEBUG_RELOAD = wx.NewId()

ID_WINDOW = wx.NewId()
ID_WINDOW_CLOSE_ALL = wx.NewId()
ID_WINDOW_SHOW_ALL_FRAMES = wx.NewId()
ID_WINDOW_HIDE_ALL_FRAMES = wx.NewId()

ID_HELP_MODULE=wx.NewId()
ID_HELP_DEVELOPERS_GUIDE = wx.NewId()

class CPFrame(wx.Frame):
    def __init__(self, *args, **kwds):
        """Initialize the frame and its layout
        
        """
        kwds["style"] = wx.DEFAULT_FRAME_STYLE
        wx.Frame.__init__(self, *args, **kwds)
        self.BackgroundColour = cellprofiler.preferences.get_background_color()
        self.__top_left_panel = wx.Panel(self,-1)
        self.__top_left_panel.BackgroundColour = cellprofiler.preferences.get_background_color()
        self.__logo_panel = wx.Panel(self,-1,style=wx.RAISED_BORDER)
        self.__logo_panel.BackgroundColour = cellprofiler.preferences.get_background_color()
        self.__module_list_panel = wx.lib.scrolledpanel.ScrolledPanel(self.__top_left_panel, -1)
        self.__module_list_panel.SetBackgroundColour('white')
        self.__pipeline_test_panel = wx.Panel(self.__top_left_panel,-1)
        self.__pipeline_test_panel.Hide()
        self.__pipeline_test_panel.BackgroundColour = cellprofiler.preferences.get_background_color()
        self.__module_controls_panel = wx.Panel(self.__top_left_panel,-1)
        self.__module_controls_panel.BackgroundColour = cellprofiler.preferences.get_background_color()
        self.__module_panel = wx.lib.scrolledpanel.ScrolledPanel(self,-1,style=wx.SUNKEN_BORDER)
        self.__module_panel.BackgroundColour = cellprofiler.preferences.get_background_color()
        self.__file_list_panel = wx.Panel(self,-1)
        self.__file_list_panel.BackgroundColour = cellprofiler.preferences.get_background_color()
        self.__preferences_panel = wx.Panel(self,-1)
        self.__preferences_panel.BackgroundColour = cellprofiler.preferences.get_background_color()
        self.__pipeline = Pipeline()
        self.__add_menu()
        self.__attach_views()
        self.__set_properties()
        self.__set_icon()
        self.__layout_logo()
        self.__do_layout()
        self.__error_listeners = []
        self.Bind(wx.EVT_SIZE,self.__on_size,self)
        self.tbicon = wx.TaskBarIcon()
        self.tbicon.SetIcon(get_icon(), "CellProfiler2.0")

    def OnClose(self, event):
        self.tbicon.Destroy()
        self.Destroy()
 
    def __set_properties(self):
        self.SetTitle("CellProfiler")
        self.SetSize((640, 480))
 
    def __add_menu(self):
        """Add the menu to the frame
        
        """
        self.__menu_bar = wx.MenuBar()
        self.__menu_file = wx.Menu()
        self.__menu_file.Append(ID_FILE_LOAD_PIPELINE,'Load Pipeline...\tctrl+P','Load a pipeline from a .MAT file')
        self.__menu_file.Append(ID_FILE_SAVE_PIPELINE,'Save Pipeline as...\tctrl+shift+S','Save a pipeline as a .MAT file')
        self.__menu_file.Append(ID_FILE_CLEAR_PIPELINE,'Clear pipeline','Remove all modules from the current pipeline')
        self.__menu_file.AppendSeparator()
        self.__menu_file.Append(ID_FILE_ANALYZE_IMAGES,'Analyze images\tctrl+N','Run the pipeline on the images in the image directory')
        self.__menu_file.Append(ID_FILE_STOP_ANALYSIS,'Stop analysis','Stop running the pipeline')
        self.__menu_file.AppendSeparator()
        self.__menu_file.Append(ID_OPTIONS_PREFERENCES,"&Preferences...","Set global application preferences")
        # self.__menu_file.Append(ID_FILE_WIDGET_INSPECTOR,'Widget inspector','Run the widget inspector for debugging the UI')
        self.__menu_file.Append(ID_FILE_EXIT,'E&xit\tctrl+Q','Quit the application')
        self.__menu_bar.Append(self.__menu_file,'&File')
        self.__menu_debug = wx.Menu()
        self.__menu_debug.Append(ID_DEBUG_START,'&Start test run\tF5','Start the pipeline debugger')
        self.__menu_debug.Append(ID_DEBUG_STOP,'S&top test run\tctrl+F5','Stop the pipeline debugger')
        self.__menu_debug.Append(ID_DEBUG_STEP,'Ste&p to next module\tF6','Execute the currently selected module')
        self.__menu_debug.Append(ID_DEBUG_NEXT_IMAGE_SET,'&Next image set\tF7','Advance to the next image set')
        self.__menu_debug.Append(ID_DEBUG_NEXT_GROUP, 'Next &group\tF8','Advance to the next group in the image set')
        self.__menu_debug.Append(ID_DEBUG_CHOOSE_GROUP, 'Choose group', 'Choose which image set group to process in test-mode')
        self.__menu_debug.Append(ID_DEBUG_CHOOSE_IMAGE_SET, 'Choose image set','Choose any of the available image sets in the current image set list')
        if not hasattr(sys, 'frozen'):
            self.__menu_debug.Append(ID_DEBUG_RELOAD, "Reload modules' source")
        self.__menu_debug.Enable(ID_DEBUG_STOP,False)
        self.__menu_debug.Enable(ID_DEBUG_STEP,False)
        self.__menu_debug.Enable(ID_DEBUG_NEXT_IMAGE_SET,False)
        self.__menu_debug.Enable(ID_DEBUG_NEXT_GROUP, False)
        self.__menu_debug.Enable(ID_DEBUG_CHOOSE_GROUP, False)
        self.__menu_debug.Enable(ID_DEBUG_CHOOSE_IMAGE_SET, False)
        self.__menu_bar.Append(self.__menu_debug,'&Test')
        self.__menu_window = wx.Menu()
        self.__menu_window.Append(ID_WINDOW_CLOSE_ALL, "Close &all\tctrl+L", 
                                  "Close all figure windows")
        self.__menu_window.Append(ID_WINDOW_SHOW_ALL_FRAMES,"Show all frames",
                                  "Show all module display frames for all modules during analysis")
        self.__menu_window.Append(ID_WINDOW_HIDE_ALL_FRAMES,"Hide all frames",
                                  "Hide all module display frames for all modules during analysis")
        self.__menu_bar.Append(self.__menu_window,"&Window")
        self.__menu_help = wx.Menu()
        self.__menu_help.Append(ID_HELP_MODULE,'Module help','Display help from the module''s .m file')
        self.__menu_help.Append(ID_HELP_DEVELOPERS_GUIDE,"Developer's guide",
                                "Launch the developer's guide webpage")
        self.__menu_bar.Append(self.__menu_help,'&Help')
        self.SetMenuBar(self.__menu_bar)
        wx.EVT_MENU(self,ID_FILE_EXIT,lambda event: self.Close())
        wx.EVT_MENU(self,ID_FILE_WIDGET_INSPECTOR,self.__on_widget_inspector)
        wx.EVT_MENU(self,ID_HELP_MODULE,self.__on_help_module)
        wx.EVT_MENU(self,ID_HELP_DEVELOPERS_GUIDE, self.__on_help_developers_guide)
        wx.EVT_MENU(self,ID_OPTIONS_PREFERENCES, self.__on_preferences)
        wx.EVT_MENU(self,ID_WINDOW_CLOSE_ALL, self.__on_close_all)
        accelerator_table = wx.AcceleratorTable([(wx.ACCEL_CMD,ord('N'),ID_FILE_ANALYZE_IMAGES),
                                                 (wx.ACCEL_CMD,ord('P'),ID_FILE_LOAD_PIPELINE),
                                                 (wx.ACCEL_CMD|wx.ACCEL_SHIFT,ord('S'),ID_FILE_SAVE_PIPELINE),
                                                 (wx.ACCEL_CMD,ord('L'),ID_WINDOW_CLOSE_ALL),
                                                 (wx.ACCEL_CMD,ord('Q'),ID_FILE_EXIT),
                                                 (wx.ACCEL_NORMAL,wx.WXK_F5,ID_DEBUG_START),
                                                 (wx.ACCEL_CMD,wx.WXK_F5,ID_DEBUG_STOP),
                                                 (wx.ACCEL_NORMAL,wx.WXK_F6,ID_DEBUG_STEP),
                                                 (wx.ACCEL_NORMAL,wx.WXK_F7,ID_DEBUG_NEXT_IMAGE_SET),
                                                 (wx.ACCEL_NORMAL,wx.WXK_F8,ID_DEBUG_NEXT_GROUP)])
        self.SetAcceleratorTable(accelerator_table)
    
    def enable_debug_commands(self, enable=True):
        """Enable or disable the debug commands (like ID_DEBUG_STEP)"""
        self.__menu_debug.Enable(ID_DEBUG_START,not enable)
        self.__menu_debug.Enable(ID_DEBUG_STOP,enable)
        self.__menu_debug.Enable(ID_DEBUG_STEP,enable)
        self.__menu_debug.Enable(ID_DEBUG_NEXT_IMAGE_SET,enable)
        self.__menu_debug.Enable(ID_DEBUG_NEXT_GROUP, enable)
        self.__menu_debug.Enable(ID_DEBUG_CHOOSE_GROUP, enable)
        self.__menu_debug.Enable(ID_DEBUG_CHOOSE_IMAGE_SET, enable)
        
    def __on_widget_inspector(self, evt):
        try:
            import wx.lib.inspection
            wx.lib.inspection.InspectionTool().Show()
        except:
            wx.MessageBox("Inspection tool is not available on this platform")

    def __on_preferences(self, event):
        dlg = cellprofiler.gui.preferencesdlg.PreferencesDlg()
        dlg.show_modal()
    
    def __on_close_all(self, event):
        close_all(self)
    
    def __on_help_developers_guide(self, event):
        import webbrowser
        filename = os.path.abspath("developers-guide/developer.html")
        webbrowser.open(filename)
        
    def __on_help_module(self,event):
        modules = self.__pipeline_list_view.get_selected_modules()
        self.do_help_modules(modules)
        
    def do_help_modules(self, modules):
        for module in modules:
            self.do_help_module(module.module_name, module.get_help())
    
    def do_help_module(self, module_name, help_text):
        helpframe = wx.Frame(self,-1,'Help for module, "%s"' %
                             (module_name),size=(640,480))
        helpframe.MenuBar = wx.MenuBar()
        ####################################################
        #
        # Add the HTML window
        #
        ####################################################
        
        sizer = wx.BoxSizer()
        helpframe.SetSizer(sizer)
        window = wx.html.HtmlWindow(helpframe)
        sizer.Add(window,1,wx.EXPAND)
        window.AppendToPage(help_text)

        ################################################
        #
        # Add a file menu for the frame
        #
        ################################################
        menu = wx.Menu()
        menu.Append(ID_FILE_SAVE_PIPELINE, "&Save...")
        menu.Append(ID_FILE_PRINT, "&Print...")
        menu.Append(ID_FILE_EXIT, "E&xit")
        def on_save(event):
            self.save_help(event, module_name, help_text)
        
        def on_print(event):
            self.print_help(event, module_name, help_text)
            
        def on_exit(event):
            helpframe.Close()
        
        helpframe.MenuBar.Append(menu, '&File')
        helpframe.Bind(wx.EVT_MENU, on_save, id=ID_FILE_SAVE_PIPELINE)
        helpframe.Bind(wx.EVT_MENU, on_print, id=ID_FILE_PRINT)
        helpframe.Bind(wx.EVT_MENU, on_exit, id = ID_FILE_EXIT)
        
        ####################################################
        #
        # Add an edit menu
        #
        ####################################################
        menu = wx.Menu()
        copy_menu_item = menu.Append(ID_EDIT_COPY, "Copy")
        copy_menu_item.Enable(False)
        menu.Append(ID_EDIT_SELECT_ALL, "Select All")
        
        def on_idle(event):
            copy_menu_item.Enable(len(window.SelectionToText()) > 0)
        
        def on_edit_select_all(event):
            window.SelectAll()
        
        def on_copy(event):
            data_object = wx.TextDataObject(window.SelectionToText())
            if wx.TheClipboard.Open():
                try:
                    wx.TheClipboard.SetData(data_object)
                    wx.TheClipboard.Flush()
                finally:
                    wx.TheClipboard.Close()
            else:
                wx.MessageBox("Failed to copy to the clipboard","Error",
                              wx.OK | wx.ICON_ERROR)
        helpframe.MenuBar.Append(menu, '&Edit')
        helpframe.Bind(wx.EVT_MENU, on_copy, id=ID_EDIT_COPY)
        helpframe.Bind(wx.EVT_MENU, on_edit_select_all, id= ID_EDIT_SELECT_ALL)
        helpframe.Bind(wx.EVT_IDLE, on_idle)
        
        helpframe.SetIcon(get_icon())
        helpframe.Layout()
        helpframe.Show()
    
    def print_help(self, event, module_name, help_text):
        '''Print the help text for a module'''
        printer = wx.html.HtmlEasyPrinting("Printing %s"%module_name,
                                           event.GetEventObject())
        printer.GetPrintData().SetPaperId(wx.PAPER_LETTER)
        printer.PrintText(help_text)
    
    def save_help(self, event, module_name, help_text):
        '''Save the help text for a module'''
        save_dlg = wx.FileDialog(event.GetEventObject(),
                                 message = "Save help for %s to file"%module_name,
                                 defaultFile = "%s.html"%module_name,
                                 wildcard = "*.html",
                                 style = wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        result = save_dlg.ShowModal()
        if result == wx.ID_OK:
            pathname = os.path.join(save_dlg.GetDirectory(),
                                    save_dlg.GetFilename())
            fd = open(pathname, "wt")
            fd.write(help_text)
            fd.close()
        
    def __attach_views(self):
        self.__pipeline_list_view = PipelineListView(self.__module_list_panel)
        self.__pipeline_controller = PipelineController(self.__pipeline,self)
        self.__pipeline_list_view.attach_to_pipeline(self.__pipeline,self.__pipeline_controller)
        self.__pipeline_controller.attach_to_test_controls_panel(self.__pipeline_test_panel)
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
        if False:
            self.__top_left_sizer = wx.FlexGridSizer(3,1,1,1)
        else:
            self.__top_left_sizer = wx.BoxSizer(wx.VERTICAL)
        self.__top_left_sizer.Add(self.__logo_panel,0,wx.EXPAND|wx.ALL,1)
        self.__top_left_sizer.Add(self.__module_list_panel,1,wx.EXPAND|wx.ALL,1)
        self.__top_left_sizer.Add(self.__pipeline_test_panel, 0, wx.EXPAND|wx.ALL,2)
        self.__top_left_sizer.Add(self.__module_controls_panel,0,wx.EXPAND|wx.ALL,2)
        if False:
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
        import base64
        import cStringIO
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        # cp logo
        data64 = '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'
        data = base64.b64decode(data64)
        stream = cStringIO.StringIO(data)
        bitmap = wx.BitmapFromImage(wx.ImageFromStream(stream))
        logopic = wx.StaticBitmap(self.__logo_panel,-1,bitmap)
        sizer.Add(logopic)
        self.__logo_panel.SetSizer(sizer)
    
    def __set_icon(self):
        self.SetIcon(get_icon())
    
    def __on_size(self, event):
        self.Layout()
 
    def display_error(self,message,error):
        """Displays an exception in a standardized way
        
        """
        for listener in self.__error_listeners:
            listener(message, error)
        tb = sys.exc_info()[2]
        traceback.print_tb(tb)
        text = '\n'.join(traceback.format_list(traceback.extract_tb(tb)))
        text = error.message + '\n'+text
        wx.MessageBox(text,"Caught exception during operation")
    
    def add_error_listener(self,listener):
        """Add a listener for display errors"""
        self.__error_listeners.append(listener)
    
    def remove_error_listener(self,listener):
        """Remove a listener for display errors"""
        self.__error_listeners.remove(listener)
    
    def get_preferences_view(self):
        return self.__preferences_view
    
    preferences_view = property(get_preferences_view)
    
    def get_pipeline_controller(self):
        """Get the pipeline controller to drive testing"""
        return self.__pipeline_controller
    
    pipeline_controller = property(get_pipeline_controller)
    
    def get_pipeline(self):
        """Get the pipeline - mostly to drive testing"""
        return self.__pipeline
    
    pipeline = property(get_pipeline)
    
    def get_module_view(self):
        """Return the module view window"""
        return self.__module_view
    
    module_view = property(get_module_view)
    
    def get_pipeline_list_view(self):
        return self.__pipeline_list_view
    
    pipeline_list_view = property(get_pipeline_list_view)

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
    
        
