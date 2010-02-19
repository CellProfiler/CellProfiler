""" CellProfiler.CellProfilerGUI.CPFrame - Cell Profiler's main window

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2010

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import os
import wx
import wx.html
import wx.lib.scrolledpanel
import cellprofiler.preferences
from cellprofiler.modules import get_data_tool_names
from cellprofiler.gui import get_icon, get_cp_bitmap
from cellprofiler.gui.pipelinelistview import PipelineListView
from cellprofiler.gui.cpfigure import close_all
from cellprofiler.gui.help import MAIN_HELP, make_help_menu
from cellprofiler.pipeline import Pipeline
from cellprofiler.gui.pipelinecontroller import PipelineController
from cellprofiler.gui.moduleview import ModuleView
from cellprofiler.gui.preferencesview import PreferencesView
from cellprofiler.gui.directoryview import DirectoryView
from cellprofiler.gui.datatoolframe import DataToolFrame
import cellprofiler.gui.preferencesdlg
import cellprofiler.utilities.get_revision as get_revision
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
ID_EDIT_UNDO = wx.NewId()
ID_EDIT_MOVE_UP = wx.NewId()
ID_EDIT_MOVE_DOWN = wx.NewId()
ID_EDIT_DELETE = wx.NewId()

ID_OPTIONS_PREFERENCES = wx.NewId()
ID_CHECK_NEW_VERSION = wx.NewId()

ID_DEBUG_TOGGLE = wx.NewId()
ID_DEBUG_STEP = wx.NewId()
ID_DEBUG_NEXT_IMAGE_SET = wx.NewId()
ID_DEBUG_NEXT_GROUP = wx.NewId()
ID_DEBUG_CHOOSE_GROUP = wx.NewId()
ID_DEBUG_CHOOSE_IMAGE_SET = wx.NewId()
ID_DEBUG_RELOAD = wx.NewId()

ID_WINDOW = wx.NewId()
ID_WINDOW_CLOSE_ALL = wx.NewId()
ID_WINDOW_SHOW_ALL_WINDOWS = wx.NewId()
ID_WINDOW_HIDE_ALL_WINDOWS = wx.NewId()
ID_WINDOW_ALL = (ID_WINDOW_CLOSE_ALL, ID_WINDOW_SHOW_ALL_WINDOWS,
                 ID_WINDOW_HIDE_ALL_WINDOWS)

window_ids = []

ID_HELP_MODULE=wx.NewId()
ID_HELP_DEVELOPERS_GUIDE = wx.NewId()

class CPFrame(wx.Frame):
    def __init__(self, *args, **kwds):
        """Initialize the frame and its layout

        """
        kwds["style"] = wx.DEFAULT_FRAME_STYLE
        wx.Frame.__init__(self, *args, **kwds)
        self.BackgroundColour = cellprofiler.preferences.get_background_color()
        self.__splitter = wx.SplitterWindow(self, -1)
        self.__left_win = wx.Panel(self.__splitter, style=wx.BORDER_NONE)
        self.__right_win = wx.Panel(self.__splitter, style=wx.BORDER_NONE)

        self.__logo_panel = wx.Panel(self.__left_win,-1,style=wx.RAISED_BORDER)
        self.__logo_panel.BackgroundColour = cellprofiler.preferences.get_background_color()
        self.__module_list_panel = wx.lib.scrolledpanel.ScrolledPanel(self.__left_win, -1)
        self.__module_list_panel.SetBackgroundColour('white')
        self.__pipeline_test_panel = wx.Panel(self.__left_win,-1)
        self.__pipeline_test_panel.Hide()
        self.__pipeline_test_panel.BackgroundColour = cellprofiler.preferences.get_background_color()
        self.__module_controls_panel = wx.Panel(self.__left_win,-1)
        self.__module_controls_panel.BackgroundColour = cellprofiler.preferences.get_background_color()
        self.__module_panel = wx.lib.scrolledpanel.ScrolledPanel(self.__right_win,-1,style=wx.SUNKEN_BORDER)
        self.__module_panel.BackgroundColour = cellprofiler.preferences.get_background_color()
        self.__file_list_panel = wx.Panel(self.__left_win,-1)
        self.__file_list_panel.BackgroundColour = cellprofiler.preferences.get_background_color()
        self.__preferences_panel = wx.Panel(self.__right_win,-1)
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
        self.SetTitle("CellProfiler (v.%d)"%(get_revision.version))
        self.SetSize((1024, 600))

    def __add_menu(self):
        """Add the menu to the frame

        """
        self.__menu_bar = wx.MenuBar()
        self.__menu_file = wx.Menu()
        self.__menu_file.Append(ID_FILE_LOAD_PIPELINE,'Load Pipeline...\tctrl+O','Load a pipeline from a .MAT file')
        self.__menu_file.Append(ID_FILE_SAVE_PIPELINE,'Save Pipeline as...\tctrl+shift+S','Save a pipeline as a .MAT file')
        self.__menu_file.Append(ID_FILE_CLEAR_PIPELINE,'Clear pipeline','Remove all modules from the current pipeline')
        self.__menu_file.AppendSeparator()
        self.__menu_file.Append(ID_FILE_ANALYZE_IMAGES,'Analyze images\tctrl+N','Run the pipeline on the images in the image directory')
        self.__menu_file.Append(ID_FILE_STOP_ANALYSIS,'Stop analysis','Stop running the pipeline')
        self.__menu_file.AppendSeparator()
        self.__menu_file.Append(ID_OPTIONS_PREFERENCES,"&Preferences...","Set global application preferences")
        self.__menu_file.Append(ID_CHECK_NEW_VERSION,"Check for updates...","Check for a new version of CellProfiler")
        self.recent_files = wx.Menu()
        self.__menu_file.AppendSubMenu(self.recent_files, "&Recent")
        # self.__menu_file.Append(ID_FILE_WIDGET_INSPECTOR,'Widget inspector','Run the widget inspector for debugging the UI')
        self.__menu_file.Append(ID_FILE_EXIT,'E&xit\tctrl+Q','Quit the application')
        self.__menu_bar.Append(self.__menu_file,'&File')
        self.menu_edit = wx.Menu()
        self.menu_edit.Append(ID_EDIT_UNDO, "&Undo\tctrl+Z", "Undo last action")
        self.menu_edit.Append(ID_EDIT_MOVE_UP, "Move &up", "Move module toward the start of the pipeline")
        self.menu_edit.Append(ID_EDIT_MOVE_DOWN, "Move &down", "Move module toward the end of the pipeline")
        self.menu_edit.Append(ID_EDIT_DELETE, "&Delete", "Delete selected modules")
        self.menu_edit_add_module = wx.Menu()
        self.menu_edit.AppendSubMenu(self.menu_edit_add_module, "&Add module")

        self.__menu_bar.Append(self.menu_edit, '&Edit')
        self.__menu_debug = wx.Menu()
        self.__menu_debug.Append(ID_DEBUG_TOGGLE,'&Start test run\tF5','Start the pipeline debugger')
        self.__menu_debug.Append(ID_DEBUG_STEP,'Ste&p to next module\tF6','Execute the currently selected module')
        self.__menu_debug.Append(ID_DEBUG_NEXT_IMAGE_SET,'&Next image set\tF7','Advance to the next image set')
        self.__menu_debug.Append(ID_DEBUG_NEXT_GROUP, 'Next &group\tF8','Advance to the next group in the image set')
        self.__menu_debug.Append(ID_DEBUG_CHOOSE_GROUP, 'Choose group', 'Choose which image set group to process in test-mode')
        self.__menu_debug.Append(ID_DEBUG_CHOOSE_IMAGE_SET, 'Choose image set','Choose any of the available image sets in the current image set list')
        if not hasattr(sys, 'frozen'):
            self.__menu_debug.Append(ID_DEBUG_RELOAD, "Reload modules' source")
        self.__menu_debug.Enable(ID_DEBUG_STEP,False)
        self.__menu_debug.Enable(ID_DEBUG_NEXT_IMAGE_SET,False)
        self.__menu_debug.Enable(ID_DEBUG_NEXT_GROUP, False)
        self.__menu_debug.Enable(ID_DEBUG_CHOOSE_GROUP, False)
        self.__menu_debug.Enable(ID_DEBUG_CHOOSE_IMAGE_SET, False)
        self.__menu_bar.Append(self.__menu_debug,'&Test')
        self.__menu_bar.Append(self.data_tools_menu(),'&Data tools')
        self.__menu_window = wx.Menu()
        self.__menu_window.Append(ID_WINDOW_CLOSE_ALL, "Close &all windows\tctrl+L", 
                                  "Close all figure windows")
        self.__menu_window.Append(ID_WINDOW_SHOW_ALL_WINDOWS,"Show all windows",
                                  "Show all module display windows for all modules during analysis")
        self.__menu_window.Append(ID_WINDOW_HIDE_ALL_WINDOWS,"Hide all windows",
                                  "Hide all module display windows for all modules during analysis")
        self.__menu_window.AppendSeparator()
        self.__menu_bar.Append(self.__menu_window,"&Window")
        self.__menu_help = make_help_menu(MAIN_HELP, self)
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
        wx.EVT_MENU(self,ID_CHECK_NEW_VERSION, self.__on_check_new_version)
        wx.EVT_MENU(self,ID_WINDOW_CLOSE_ALL, self.__on_close_all)
        accelerator_table = wx.AcceleratorTable(
            [(wx.ACCEL_CMD,ord('N'),ID_FILE_ANALYZE_IMAGES),
             (wx.ACCEL_CMD,ord('O'),ID_FILE_LOAD_PIPELINE),
             (wx.ACCEL_CMD|wx.ACCEL_SHIFT,ord('S'),ID_FILE_SAVE_PIPELINE),
             (wx.ACCEL_CMD,ord('L'),ID_WINDOW_CLOSE_ALL),
             (wx.ACCEL_CMD,ord('Q'),ID_FILE_EXIT),
             (wx.ACCEL_CMD,ord('W'),ID_FILE_EXIT),
             (wx.ACCEL_NORMAL,wx.WXK_F5,ID_DEBUG_TOGGLE),
             (wx.ACCEL_NORMAL,wx.WXK_F6,ID_DEBUG_STEP),
             (wx.ACCEL_NORMAL,wx.WXK_F7,ID_DEBUG_NEXT_IMAGE_SET),
             (wx.ACCEL_NORMAL,wx.WXK_F8,ID_DEBUG_NEXT_GROUP),
             (wx.ACCEL_CMD,ord('Z'),ID_EDIT_UNDO) ])
        self.SetAcceleratorTable(accelerator_table)
        
    def data_tools_menu(self):
        '''Create a menu of data tools'''
        if not hasattr(self, "__data_tools_menu"):
            self.__data_tools_menu = wx.Menu()
            for data_tool_name in get_data_tool_names():
                new_id = wx.NewId()
                self.__data_tools_menu.Append(new_id, data_tool_name)
                def on_data_tool(event, data_tool_name=data_tool_name):
                    self.__on_data_tool(event, data_tool_name)
                wx.EVT_MENU(self, new_id, on_data_tool)
        return self.__data_tools_menu

    def enable_debug_commands(self, enable=True):
        """Enable or disable the debug commands (like ID_DEBUG_STEP)"""
        startstop = self.__menu_debug.FindItemById(ID_DEBUG_TOGGLE)
        assert isinstance(startstop, wx.MenuItem)
        startstop.Text = '&Stop test run\tF5' if enable else '&Start test run\tF5'
        startstop.Help = ('Stop the pipeline debugger' if enable 
                          else 'Start the pipeline debugger')
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
        dlg.Destroy()

    def __on_check_new_version(self, event):
        wx.GetApp().new_version_check(force=True)

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

        ####################################################
        #
        # Build an accelerator table for some of the commands
        #
        ####################################################
        accelerator_table = wx.AcceleratorTable(
            [(wx.ACCEL_CMD,ord('Q'), ID_FILE_EXIT),
             (wx.ACCEL_CMD,ord('P'), ID_FILE_PRINT),
             (wx.ACCEL_CMD,ord('C'), ID_EDIT_COPY)])
        helpframe.SetAcceleratorTable(accelerator_table)
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
        self.__pipeline_list_view = PipelineListView(self.__module_list_panel, self)
        self.__pipeline_controller = PipelineController(self.__pipeline,self)
        self.__pipeline_list_view.attach_to_pipeline(self.__pipeline,self.__pipeline_controller)
        self.__pipeline_controller.attach_to_test_controls_panel(self.__pipeline_test_panel)
        self.__pipeline_controller.attach_to_module_controls_panel(self.__module_controls_panel)
        self.__module_view = ModuleView(self.__module_panel,self.__pipeline)
        self.__pipeline_controller.attach_to_module_view(self.__module_view)
        self.__pipeline_list_view.attach_to_module_view((self.__module_view))
        self.__preferences_view = PreferencesView(self.__preferences_panel)
        self.__preferences_view.attach_to_pipeline_controller(self.__pipeline_controller)
        self.__preferences_view.attach_to_pipeline_list_view(self.__pipeline_list_view)
        self.__directory_view = DirectoryView(self.__file_list_panel)
        self.__pipeline_controller.attach_to_directory_view(self.__directory_view)

    def __do_layout(self):
        width = wx.SystemSettings.GetMetric(wx.SYS_SCREEN_X)
        height = wx.SystemSettings.GetMetric(wx.SYS_SCREEN_Y)
        self.SetSize((int(width * 2 / 3), int(height * 2 / 3)))
        splitter = self.__splitter
        left_win = self.__left_win
        right_win = self.__right_win
        
        self.__splitter.SetMinimumPaneSize(self.__logopic.GetBestSize()[0] + 5)
        self.__splitter.SplitVertically(self.__left_win, self.__right_win, 300)

        left_sizer = wx.BoxSizer(wx.VERTICAL)
        left_sizer.Add(self.__logo_panel,0,wx.EXPAND|wx.ALL,1)
        left_sizer.Add(self.__module_list_panel,1,wx.EXPAND|wx.ALL,1)
        left_sizer.Add(self.__pipeline_test_panel, 0, wx.EXPAND|wx.ALL,2)
        left_sizer.Add(self.__module_controls_panel,0,wx.EXPAND|wx.ALL,2)
        left_sizer.Add(self.__file_list_panel,0,wx.EXPAND|wx.ALL|wx.CENTER,2)
        left_win.SetSizer(left_sizer)

        right_sizer = wx.BoxSizer(wx.VERTICAL)
        right_sizer.Add(self.__module_panel, 1, wx.EXPAND|wx.ALL, 1)
        right_sizer.Add(self.__preferences_panel, 0, wx.EXPAND|wx.ALL, 1)
        right_win.SetSizer(right_sizer)
        self.__directory_view.set_height(self.__preferences_panel.GetBestSize()[1])

        border = wx.BoxSizer()
        border.Add(splitter, 1, wx.EXPAND|wx.ALL,1)
        self.SetSizer(border)
        self.Layout()

    def __layout_logo(self):
        import cStringIO
        sizer = wx.BoxSizer(wx.HORIZONTAL)

        from cellprofiler.icons import CP_logo
        bitmap = wx.BitmapFromImage(CP_logo)
        self.__logopic = wx.StaticBitmap(self.__logo_panel,-1,bitmap)
        sizer.Add(self.__logopic)
        self.__logo_panel.SetSizer(sizer)

    def __set_icon(self):
        self.SetIcon(get_icon())

    def __on_size(self, event):
        self.Layout()
        
    def __on_data_tool(self, event, tool_name):
        dlg = wx.FileDialog(self, "Choose data output file for %s data tool" %
                            tool_name, wildcard="*.mat",
                            style=(wx.FD_OPEN | wx.FILE_MUST_EXIST))
        if dlg.ShowModal() == wx.ID_OK:
            DataToolFrame(self, 
                          module_name=tool_name,
                          measurements_file_name = dlg.Path)
    
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


