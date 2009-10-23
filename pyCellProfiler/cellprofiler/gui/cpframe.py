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
        self.__module_list_panel = wx.Panel(self.__top_left_panel,-1)
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
    
        
