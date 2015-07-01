# -*- Encoding: utf-8 -*-
""" CellProfiler.CellProfilerGUI.CPFrame - Cell Profiler's main window

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2015 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
import logging
logger = logging.getLogger(__name__)
import inspect
import os
import pdb
import wx
import wx.html
import wx.lib.scrolledpanel
import cellprofiler.preferences as cpprefs
import cellprofiler.measurements as cpmeas
import cellprofiler.workspace as cpw

from cellprofiler.icons import get_builtin_image, get_icon_copyrights
from cellprofiler.modules import get_data_tool_names, instantiate_module
from cellprofiler.gui import get_cp_icon, get_cp_bitmap
from cellprofiler.gui.pipelinelistview import PipelineListView
from cellprofiler.gui.cpfigure import close_all
from cellprofiler.gui.help import MAIN_HELP, make_help_menu, \
     HELP_ON_MODULE_BUT_NONE_SELECTED, HELP_ON_FILE_LIST, PLATEVIEWER_HELP
from cellprofiler.pipeline import Pipeline
from cellprofiler.gui.pipelinecontroller import PipelineController
from cellprofiler.gui.moduleview import ModuleView, stop_validation_queue_thread
from cellprofiler.gui.preferencesview import PreferencesView
from cellprofiler.gui.directoryview import DirectoryView
from cellprofiler.gui.datatoolframe import DataToolFrame
from cellprofiler.gui.html.htmlwindow import HtmlClickableWindow, WELCOME_SCREEN_FRAME
from cellprofiler.gui.errordialog import display_error_message
from cellprofiler.gui.pathlist import PathListCtrl
from cellprofiler.gui.imagesetctrl import ImageSetCtrl
from cellprofiler.gui.sashwindow_tools import sw_bind_to_evt_paint, sp_bind_to_evt_paint
from cellprofiler.gui.bitmaplabelbutton import BitmapLabelButton
import cellprofiler.gui.html
import cellprofiler.gui.preferencesdlg
import cellprofiler.utilities.version as version
import traceback
import sys

ID_FILE_NEW_WORKSPACE = wx.ID_NEW
ID_FILE_LOAD=wx.ID_OPEN
ID_FILE_LOAD_PIPELINE=wx.NewId()
ID_FILE_URL_LOAD_PIPELINE = wx.NewId()
ID_FILE_OPEN_IMAGE = wx.NewId()
ID_FILE_EXIT=wx.NewId()
ID_FILE_WIDGET_INSPECTOR=wx.NewId()
ID_FILE_SAVE_PIPELINE=wx.NewId()
ID_FILE_SAVE=wx.ID_SAVE
ID_FILE_SAVE_AS = wx.ID_SAVEAS
ID_FILE_REVERT_TO_SAVED = wx.NewId()
ID_FILE_CLEAR_PIPELINE=wx.NewId()
ID_FILE_EXPORT_IMAGE_SETS = wx.NewId()
ID_FILE_IMPORT_FILE_LIST = wx.NewId()
ID_FILE_ANALYZE_IMAGES=wx.NewId()
ID_FILE_STOP_ANALYSIS=wx.NewId()
ID_FILE_RESTART = wx.NewId()
ID_FILE_PRINT=wx.NewId()
ID_FILE_PLATEVIEWER = wx.NewId()
ID_FILE_RUN_MULTIPLE_PIPELINES = wx.NewId()
ID_FILE_NEW_CP=wx.NewId()

ID_EDIT_SELECT_ALL = wx.NewId()
ID_EDIT_COPY = wx.NewId()
ID_EDIT_DUPLICATE = wx.NewId()
ID_EDIT_UNDO = wx.ID_UNDO
ID_EDIT_MOVE_UP = wx.NewId()
ID_EDIT_MOVE_DOWN = wx.NewId()
ID_EDIT_DELETE = wx.NewId()

ID_EDIT_EXPAND_ALL = wx.NewId()
ID_EDIT_COLLAPSE_ALL = wx.NewId()
ID_EDIT_BROWSE_FOR_FILES = wx.NewId()
ID_EDIT_CLEAR_FILE_LIST = wx.NewId()
ID_EDIT_REMOVE_FROM_FILE_LIST = wx.NewId()
ID_EDIT_SHOW_FILE_LIST_IMAGE = wx.NewId()
ID_EDIT_ENABLE_MODULE = wx.NewId()
ID_EDIT_GO_TO_MODULE = wx.NewId()

ID_OPTIONS_PREFERENCES = wx.ID_PREFERENCES
ID_CHECK_NEW_VERSION = wx.NewId()

ID_DEBUG_TOGGLE = wx.NewId()
ID_DEBUG_STEP = wx.NewId()
ID_DEBUG_NEXT_IMAGE_SET = wx.NewId()
ID_DEBUG_NEXT_GROUP = wx.NewId()
ID_DEBUG_CHOOSE_GROUP = wx.NewId()
ID_DEBUG_CHOOSE_IMAGE_SET = wx.NewId()
ID_DEBUG_CHOOSE_RANDOM_IMAGE_SET = wx.NewId()
ID_DEBUG_RELOAD = wx.NewId()
ID_DEBUG_NUMPY = wx.NewId()
ID_DEBUG_PDB = wx.NewId()
ID_DEBUG_VIEW_WORKSPACE = wx.NewId()

# ~*~
ID_SAMPLE_INIT = wx.NewId()
# ~^~

ID_WINDOW = wx.NewId()
ID_WINDOW_CLOSE_ALL = wx.NewId()
ID_WINDOW_SHOW_ALL_WINDOWS = wx.NewId()
ID_WINDOW_HIDE_ALL_WINDOWS = wx.NewId()
ID_WINDOW_ALL = (ID_WINDOW_CLOSE_ALL, ID_WINDOW_SHOW_ALL_WINDOWS,
                 ID_WINDOW_HIDE_ALL_WINDOWS)

window_ids = []

ID_HELP_WELCOME = wx.NewId()
ID_HELP_MODULE = wx.NewId()
ID_HELP_SEARCH = wx.NewId()
ID_HELP_DATATOOLS = wx.NewId()
ID_HELP_ONLINE_MANUAL = wx.NewId()
ID_HELP_RELEASE_NOTES = wx.NewId()
ID_HELP_DEVELOPERS_GUIDE = wx.NewId()
ID_HELP_SOURCE_CODE = wx.NewId()
ID_HELP_ABOUT = wx.ID_ABOUT

class CPFrame(wx.Frame):
    def __init__(self, *args, **kwds):
        """Initialize the frame and its layout

        """
        kwds["style"] = wx.DEFAULT_FRAME_STYLE
        wx.Frame.__init__(self, *args, **kwds)
        self.__pipeline = Pipeline()
        self.__workspace = cpw.Workspace(
            self.__pipeline, None, None, None, None, None)
        background_color = cpprefs.get_background_color()
        self.BackgroundColour = background_color
        self.__splitter = wx.SplitterWindow(self, -1, style=wx.SP_BORDER)
        sp_bind_to_evt_paint(self.__splitter)
        #
        # Screen size metrics might be used below
        #
        screen_width = wx.SystemSettings.GetMetric(wx.SYS_SCREEN_X)
        screen_height = wx.SystemSettings.GetMetric(wx.SYS_SCREEN_Y)

        # Crappy splitters leave crud on the screen because they want custom
        # background painting but fail to do it. Here, we have a fight with
        # them and beat them.
        self.__splitter.BackgroundStyle = 0

        self.__right_win = wx.Panel(self.__splitter, style=wx.BORDER_NONE)
        self.__right_win.BackgroundColour = background_color
        self.__right_win.AutoLayout = True

        self.__left_win = wx.Panel(self.__splitter, style=wx.BORDER_NONE)
        # bottom left will be the file browser

        self.__module_list_panel = wx.Panel(self.__left_win)
        self.__module_list_panel.SetBackgroundColour(background_color)
        self.__module_list_panel.SetToolTipString("The pipeline panel contains the modules in the pipeline. Click on the '+' button below or right-click in the panel to begin adding modules.")
        self.__pipeline_test_panel = wx.Panel(self.__left_win,-1)
        self.__pipeline_test_panel.SetToolTipString("The test mode panel is used for previewing the module settings prior to an analysis run. Click the buttons or use the 'Test' menu item to begin testing your module settings.")
        self.__pipeline_test_panel.BackgroundColour = cpprefs.get_background_color()
        self.__module_controls_panel = wx.Panel(self.__left_win,-1, style=wx.BORDER_NONE)
        self.__module_controls_panel.BackgroundColour = cpprefs.get_background_color()
        self.__module_controls_panel.SetToolTipString("The module controls add, remove, move and get help for modules. Click on the '+' button to begin adding modules.")
        #
        # The right window has the following structure:
        #
        #  right_win
        #    Notes window
        #    path_module_imageset_panel
        #        path_list_sash
        #            group_box
        #            path_list_ctrl
        #            path_list_filter_checkbox
        #            path_list_help_button
        #        
        #        module_panel
        #        image_set_list_sash
        #            image_set_list_ctrl
        #
        self.__right_win.Sizer = wx.BoxSizer(wx.VERTICAL)
        self.__notes_panel = wx.Panel(self.__right_win)
        self.__right_win.Sizer.Add(self.__notes_panel, 0, wx.EXPAND | wx.ALL)
        self.__right_win.Sizer.AddSpacer(4)
        self.__path_module_imageset_panel = wx.Panel(self.__right_win)
        self.__right_win.Sizer.Add(self.__path_module_imageset_panel, 1,
                                   wx.EXPAND | wx.ALL)
        self.__path_module_imageset_panel.Bind(
            wx.EVT_SIZE, self.__on_path_module_imageset_panel_size)
        
        ########################################################################
        #
        # The path list control that holds all of the files being dealt with
        # by the pipeline
        #
        ########################################################################
        
        #
        # Path list sash controls path list sizing
        #
        self.__path_list_sash = wx.SashLayoutWindow(
            self.__path_module_imageset_panel, style=wx.NO_BORDER)
        sw_bind_to_evt_paint(self.__path_list_sash)
        self.__path_list_sash.Bind(wx.EVT_SASH_DRAGGED,
                                   self.__on_sash_drag)
        self.__path_list_sash.SetOrientation(wx.LAYOUT_HORIZONTAL)
        self.__path_list_sash.SetAlignment(wx.LAYOUT_TOP)
        self.__path_list_sash.SetDefaultSize((screen_width,  screen_height/4))
        self.__path_list_sash.SetDefaultBorderSize(4)
        self.__path_list_sash.SetSashVisible(wx.SASH_BOTTOM, True)
        self.__path_list_sash.BackgroundColour = cpprefs.get_background_color()
        self.__path_list_sash.AutoLayout = True
        self.__path_list_sash.Hide()
        path_list_group_box = wx.StaticBox(
            self.__path_list_sash,
            label = "File list")
        sizer = wx.StaticBoxSizer(path_list_group_box, wx.VERTICAL)
        self.__path_list_sash.Sizer = wx.BoxSizer(wx.VERTICAL)
        self.__path_list_sash.Sizer.Add(sizer, 1, wx.EXPAND)
        # Add spacer so that group box doesn't cover sash's handle
        self.__path_list_sash.Sizer.AddSpacer(6)
        #
        # Path list control
        #
        self.__path_list_ctrl = PathListCtrl(self.__path_list_sash)
        sizer.Add(self.__path_list_ctrl, 1, wx.EXPAND | wx.ALL)
        #
        # Path list tools horizontal sizer
        #
        sizer.AddSpacer(2)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(hsizer, 0, wx.EXPAND)
        #
        # Path list show/hide filtered files checkbox
        #
        hsizer.AddSpacer(5)
        self.__path_list_filter_checkbox = wx.CheckBox(
            self.__path_list_sash,
            label = "Show files excluded by filters")
        hsizer.Add(self.__path_list_filter_checkbox, 0, wx.EXPAND)
        def show_disabled(event):
            self.__path_list_ctrl.set_show_disabled(
                self.__path_list_filter_checkbox.Value)
        self.__path_list_filter_checkbox.Bind(wx.EVT_CHECKBOX, show_disabled)
        hsizer.AddStretchSpacer()
        #
        # Help
        #
        hsizer.AddSpacer(5)
        self.__path_list_help_button = wx.Button(
            self.__path_list_sash, label="?", style=wx.BU_EXACTFIT)
        self.__path_list_help_button.Bind(wx.EVT_BUTTON, self.__on_help_path_list)
        hsizer.Add(self.__path_list_help_button, 0, wx.EXPAND)
        
        ######################################################################
        #
        # Module view panel
        #
        ######################################################################

        self.__module_panel = wx.Panel(self.__path_module_imageset_panel)
        self.__module_panel.BackgroundColour = cpprefs.get_background_color()
        
        ######################################################################
        #
        # The imageset panel
        #
        ######################################################################

        self.__imageset_sash = wx.SashLayoutWindow(
            self.__path_module_imageset_panel, style=wx.NO_BORDER)
        self.__imageset_sash.SetOrientation(wx.LAYOUT_HORIZONTAL)
        self.__imageset_sash.SetAlignment(wx.LAYOUT_BOTTOM)
        self.__imageset_sash.SetDefaultSize((screen_width, screen_height/4))
        self.__imageset_sash.SetDefaultBorderSize(4)
        self.__imageset_sash.SetExtraBorderSize(2)
        self.__imageset_sash.SetSashVisible(wx.SASH_TOP, True)
        self.__imageset_sash.Bind(wx.EVT_SASH_DRAGGED,
                                  self.__on_sash_drag)
        sw_bind_to_evt_paint(self.__imageset_sash)
        self.__imageset_sash.Hide()
        self.__imageset_panel = wx.Panel(self.__imageset_sash)
        self.__imageset_panel.Sizer = wx.BoxSizer()
        self.__imageset_panel.SetAutoLayout(True)
        self.__imageset_ctrl = ImageSetCtrl(
            self.__workspace, self.__imageset_panel, read_only=True)
        self.__imageset_panel.Sizer.Add(self.__imageset_ctrl, 1, wx.EXPAND)
        self.__grid_ctrl = ModuleView.CornerButtonGrid(
            self.__imageset_panel)
        self.__imageset_panel.Sizer.Add(self.__grid_ctrl, 1, wx.EXPAND)

        self.__right_win.Sizer.AddSpacer(4)
        #
        # Preferences panel
        #
        self.__preferences_panel = wx.Panel(self.__right_win,-1)
        self.__right_win.Sizer.Add(self.__preferences_panel, 1, wx.EXPAND)
        self.__preferences_panel.BackgroundColour = cpprefs.get_background_color()
        self.__preferences_panel.SetToolTipString(
            "The folder panel sets/creates the input and output folders and output filename. Once your pipeline is ready and your folders set, click 'Analyze Images' to begin the analysis run.")
        #
        # Progress and status panels
        #
        self.__progress_panel = wx.Panel(self.__right_win)
        self.__progress_panel.AutoLayout = True
        self.__right_win.Sizer.Add(self.__progress_panel, 0, wx.EXPAND)
        self.__status_panel = wx.Panel(self.__right_win)
        self.__status_panel.AutoLayout = True
        self.__right_win.Sizer.Add(self.__status_panel, 0, wx.EXPAND)
        self.__add_menu()
        self.__attach_views()
        self.__set_properties()
        self.__set_icon()
        self.__do_layout()
        self.__make_search_frame()
        self.__make_startup_blurb_frame()
        self.__error_listeners = []
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        self.SetAutoLayout(True)
        if cpprefs.get_startup_blurb():
            self.show_welcome_screen(True)
        self.show_module_ui(True)
        
    def start(self, workspace_path, pipeline_path):
        '''Handle resource loading after the GUI has been constructed
        
        workspace_path - one of the following: a pathname to the workspace
                         to load, False to ask the user for a new workspace
                         or None to leave the decision to the user's
                         preference.
                         
        pipeline_path - the pipeline to load after the workspace has been
                        loaded or None for the workspace's pipeline.
        '''
        self.__pipeline_controller.start(workspace_path, pipeline_path)
        self.__module_view.start()
        #
        # Do a little placement after the UI has been constructed
        #
        # Put the welcome screen over the module settings.
        #
        r = self.__right_win.GetScreenRect()
        self.startup_blurb_frame.SetRect(r)

    def show_path_list_ctrl(self, show):
        '''Show or hide the path list control

        show - true to show, false to hide
        '''
        if bool(show) == bool(self.__path_list_sash.IsShown()):
            return
        self.__path_list_sash.Show(show)
        self.layout_pmi_panel()
        self.__path_list_sash.Layout()
        
    def show_imageset_sash(self, show):
        '''Show or hide the imageset control

        show - true to show, false to hide
        '''
        if bool(show) == bool(self.__imageset_sash.IsShown()):
            return
        self.__imageset_sash.Show(show)
        self.layout_pmi_panel()
        self.__imageset_sash.Layout()
        
    def show_imageset_ctrl(self):
        sizer = self.__imageset_panel.Sizer
        assert isinstance(sizer, wx.Sizer)
        if (sizer.IsShown(self.__imageset_ctrl) == False or
            self.__imageset_sash.IsShown() == False):
            sizer.Show(self.__imageset_ctrl, True)
            sizer.Show(self.__grid_ctrl, False)
            self.show_imageset_sash(True)
            self.__imageset_panel.Layout()
        
    def show_grid_ctrl(self, table = None):
        if table is not None:
            self.__grid_ctrl.SetTable(table)
        sizer = self.__imageset_panel.Sizer
        if (sizer.IsShown(self.__imageset_ctrl) == True or
            self.__imageset_sash.IsShown() == False):
            sizer.Show(self.__imageset_ctrl, False)
            sizer.Show(self.__grid_ctrl, True)
            self.show_imageset_sash(True)
            self.__imageset_sash.Layout()
            self.__imageset_panel.Layout()
            self.__grid_ctrl.Layout()
        
    def get_grid_ctrl(self):
        return self.__grid_ctrl

    def reset_imageset_ctrl(self, refresh_image_set=True):
        if refresh_image_set:
            self.__workspace.refresh_image_set()
        self.__imageset_ctrl.recompute()

    def show_module_ui(self, show):
        '''Show or hide the module and notes panel'''
        if (show == self.__path_module_imageset_panel.IsShownOnScreen() and
            show == self.__notes_panel.IsShownOnScreen()):
            return
        right_sizer = self.__right_win.Sizer
        assert isinstance(right_sizer, wx.Sizer)
        right_sizer.Show(self.__notes_panel, show)
        right_sizer.Show(self.__path_module_imageset_panel, show)
        self.__right_win.Layout()
        if show:
            self.show_preferences(False)
            self.layout_pmi_panel()
            self.__path_list_sash.Layout()
            self.__module_panel.Layout()
            self.__module_view.module_panel.SetupScrolling(
                scroll_x=True,
                scroll_y=True,
                scrollToTop=False)
            self.__imageset_sash.Layout()
      
    def show_welcome_screen(self, show):
        '''Show or hide the welcome screen
        
        show - If True, show the welcome screen and hide the preferences 
               and module UI, otherwise hide the welcome screen.
        '''
        self.startup_blurb_frame.Show(show)
        if (show):
            self.startup_blurb_frame.Raise()
            
    def show_preferences(self, show):
        '''Show or hide the preferences panel
        
        show - if True, show the preferences panel and hide the welcome
               and module UI. If false, just hide the preferences.
        '''
        self.__preferences_panel.Show(show)
        if show:
            self.show_module_ui(False)
            self.show_welcome_screen(False)
            self.__preferences_panel.Layout()
            self.__preferences_panel.Parent.Layout()

    def __on_sash_drag(self, event):
        sash = event.GetEventObject()
        width, _ = sash.GetSize()
        sash.SetDefaultSize((width, event.GetDragRect().height))
        self.layout_pmi_panel()
        sash.Layout()

    def __on_path_module_imageset_panel_size(self, event):
        self.layout_pmi_panel()
        if self.__path_list_sash.IsShown():
            self.__path_list_sash.Layout()
        if self.__imageset_sash.IsShown():
            self.__imageset_sash.Layout()

    def layout_pmi_panel(self):
        '''Run the sash layout algorithm on the path/module/imageset panel'''
        wx.LayoutAlgorithm().LayoutWindow(self.__path_module_imageset_panel,
                                          self.__module_panel)
        self.__right_win.Layout()


    def OnClose(self, event):
        if event.CanVeto() and not self.pipeline_controller.check_close():
            event.Veto()
            return
        try:
            self.__workspace.measurements.flush()
        except:
            logger.warn("Failed to flush temporary measurements file during close", exc_info=True)
        try:
            self.__preferences_view.close()
        except:
            logger.warn("Failed during close", exc_info=True)
        try:
            self.pipeline_controller.on_close()
        except:
            logger.warn("Failed to close the pipeline controller", exc_info=True)
        try:
            stop_validation_queue_thread()
        except:
            logger.warn("Failed to stop pipeline validation thread", exc_info=True)
        wx.GetApp().ExitMainLoop()

    def __set_properties(self):
        self.SetTitle("CellProfiler %s"%(version.title_string))
        self.SetSize((1024, 600))

    def enable_edit_commands(self, ids):
        '''Enable the edit commands that are supported by the focused window
        
        ids - a list of the IDs supported by the window that has the focus.
        
        This should be called when a window receives an EVT_SET_FOCUS or
        when its state has changed to the point where it needs to enable
        different sets of commands.
        
        Commands that can be passed through here:
        wx.ID_COPY
        wx.ID_CUT
        wx.ID_PASTE
        wx.ID_DELETE
        wx.ID_SELECTALL
        '''
        d = dict([(x, False) for x in
                  (wx.ID_COPY, wx.ID_CUT, wx.ID_PASTE, wx.ID_SELECTALL)])
        for eyedee in ids:
            d[eyedee] = True
        for k, v in d.iteritems():
            self.menu_edit.Enable(k, v)
        
    def __add_menu(self):
        """Add the menu to the frame

        """
        self.__menu_file = wx.Menu()
        self.__menu_file.Append(
            wx.ID_NEW,
            "New Project",
            help="Create an empty project")
        self.__menu_file.Append(
            wx.ID_OPEN,
            "Open Project...\tctrl+O",
            help='Open a project from a .%s project file' % cpprefs.EXT_PROJECT)
        self.recent_workspace_files = wx.Menu()
        self.__menu_file.AppendSubMenu(
            self.recent_workspace_files, 
            "Open Recent")
        self.__menu_file.Append(
            wx.ID_SAVE,
            "Save Project\tctrl+S",
            help='Save the project to the current project file')
        self.__menu_file.Append(
            wx.ID_SAVEAS,
            "Save Project As...",
            help='Save the project to a file of your choice')
        self.__menu_file.Append(
            ID_FILE_REVERT_TO_SAVED,
            "Revert to Saved",
            help="Reload the project file, discarding changes")
        submenu = wx.Menu()
        submenu.Append(
            ID_FILE_LOAD_PIPELINE,
            'Pipeline from File...',
            'Import a pipeline into the project from a .%s file' % 
            cpprefs.EXT_PIPELINE)
        submenu.Append(
            ID_FILE_URL_LOAD_PIPELINE, 
            'Pipeline from URL...', 
            'Load a pipeline from the web')
        submenu.Append(
            ID_FILE_IMPORT_FILE_LIST,
            "File List...",
            "Add files or URLs to the Images module file list")
        self.__menu_file.AppendSubMenu(submenu, "Import")
        
        submenu = wx.Menu()
        submenu.Append(
            ID_FILE_SAVE_PIPELINE,
            'Pipeline...\tctrl+P',
            "Save the project's pipeline to a .%s file" % cpprefs.EXT_PIPELINE)
        submenu.Append(
            ID_FILE_EXPORT_IMAGE_SETS,
            "Image Set Listing...",
            "Export the project's image sets as a CSV file suitable for LoadData")
        self.__menu_file.AppendSubMenu(submenu, "Export")
        self.__menu_file.Append(
            ID_FILE_CLEAR_PIPELINE,
            'Clear Pipeline',
            'Remove all modules from the current pipeline')
        self.__menu_file.AppendSeparator()
        self.__menu_file.Append(
            ID_FILE_OPEN_IMAGE,
            'View Image',
            'Open an image file for viewing')
        self.__menu_file.AppendSeparator()
        self.__menu_file.Append(ID_FILE_ANALYZE_IMAGES,'Analyze Images\tctrl+N','Run the pipeline on the images in the image directory')
        self.__menu_file.Append(ID_FILE_STOP_ANALYSIS,'Stop Analysis','Stop running the pipeline')
        self.__menu_file.Append(ID_FILE_RUN_MULTIPLE_PIPELINES, 'Run Multiple Pipelines')
        if os.name=='posix':
            self.__menu_file.Append(ID_FILE_NEW_CP, 'Open a New CP Window')
        self.__menu_file.Append(ID_FILE_RESTART, 'Resume Pipeline', 'Resume a pipeline from a saved measurements file.')
        self.__menu_file.AppendSeparator()
        self.__menu_file.Append(ID_OPTIONS_PREFERENCES,"&Preferences...","Set global application preferences")

        self.recent_files = wx.Menu()
        self.recent_pipeline_files = wx.Menu()
        self.__menu_file.Append(ID_FILE_EXIT,'E&xit\tctrl+Q','Quit the application')

        self.menu_edit = wx.Menu()
        self.menu_edit.Append(wx.ID_UNDO, help="Undo last action")
        self.menu_edit.AppendSeparator()
        
        self.menu_edit.Append(wx.ID_CUT)
        self.menu_edit.Append(wx.ID_COPY)
        self.menu_edit.Append(wx.ID_PASTE)
        self.menu_edit.Append(wx.ID_SELECTALL)
        
        self.menu_edit.AppendSeparator()
        self.menu_edit.Append(ID_EDIT_MOVE_UP, "Move Module &Up", "Move module toward the start of the pipeline")
        self.menu_edit.Append(ID_EDIT_MOVE_DOWN, "Move Module &Down", "Move module toward the end of the pipeline")
        self.menu_edit.Append(ID_EDIT_DELETE, "&Delete Module", "Delete selected modules")
        self.menu_edit.Append(ID_EDIT_DUPLICATE, "Duplicate Module", "Duplicate selected modules")
        self.menu_edit.Append(
            ID_EDIT_ENABLE_MODULE, "Disable Module",
            "Disable a module to skip it when running the pipeline")
        self.menu_edit_add_module = wx.Menu()
        self.menu_edit.AppendSubMenu(self.menu_edit_add_module, "&Add Module")
        self.menu_edit_goto_module = wx.Menu()
        self.menu_edit.AppendSubMenu(
            self.menu_edit_goto_module, "&Go to Module")

        self.menu_edit.AppendSeparator()
        self.menu_edit.Append(ID_EDIT_SHOW_FILE_LIST_IMAGE,
                              "Show Selected Image",
                              "Display the first selected image in the file list")
        self.menu_edit.Append(ID_EDIT_REMOVE_FROM_FILE_LIST,
                              "Remove From File List",
                              "Remove the selected files from the file list")
        self.menu_edit.Append(ID_EDIT_BROWSE_FOR_FILES,
                              "Browse for Images",
                              "Select images to add to the file list using a file browser")
        self.menu_edit.Append(ID_EDIT_CLEAR_FILE_LIST, "Clear File List",
                              "Remove all files from the file list")
        self.menu_edit.Append(ID_EDIT_EXPAND_ALL, "Expand All Folders", 
                              "Expand all folders in the file list and show all file names")
        self.menu_edit.Append(ID_EDIT_COLLAPSE_ALL, "Collapse All Folders",
                              "Collapse all folders in the file list, hiding all file names")

        self.__menu_debug = wx.Menu()
        self.__menu_debug.Append(ID_DEBUG_TOGGLE,'&Start Test Mode\tF5','Start the pipeline debugger')
        self.__menu_debug.Append(ID_DEBUG_STEP,'Ste&p to Next Module\tF6','Execute the currently selected module')
        self.__menu_debug.Append(ID_DEBUG_NEXT_IMAGE_SET,'&Next Image Set\tF7','Advance to the next image set')
        self.__menu_debug.Append(ID_DEBUG_NEXT_GROUP, 'Next Image &Group\tF8','Advance to the next group in the image set list')
        self.__menu_debug.Append(ID_DEBUG_CHOOSE_RANDOM_IMAGE_SET, 'Random Image Set','Advance to a random image set')
        self.__menu_debug.Append(ID_DEBUG_CHOOSE_GROUP, 'Choose Image Group', 'Choose which image set group to process in test-mode')
        self.__menu_debug.Append(ID_DEBUG_CHOOSE_IMAGE_SET, 'Choose Image Set','Choose any of the available image sets')
        self.__menu_debug.Append(ID_DEBUG_VIEW_WORKSPACE, "View Workspace","Show the workspace viewer")
        if not hasattr(sys, 'frozen') or os.getenv('CELLPROFILER_DEBUG'):
            self.__menu_debug.Append(ID_DEBUG_RELOAD, "Reload Modules' Source")
            self.__menu_debug.Append(ID_DEBUG_NUMPY, "Numpy Memory Usage...")
            self.__menu_debug.Append(ID_DEBUG_PDB, "Break Into Debugger")
            #
            # Lee wants the wx debugger
            #
            if os.environ.get("USERNAME", "").lower() == "leek":
                self.__menu_debug.Append(ID_FILE_WIDGET_INSPECTOR, "Widget inspector")
        self.__menu_debug.Enable(ID_DEBUG_STEP,False)
        self.__menu_debug.Enable(ID_DEBUG_NEXT_IMAGE_SET,False)
        self.__menu_debug.Enable(ID_DEBUG_NEXT_GROUP, False)
        self.__menu_debug.Enable(ID_DEBUG_CHOOSE_GROUP, False)
        self.__menu_debug.Enable(ID_DEBUG_CHOOSE_IMAGE_SET, False)
        self.__menu_debug.Enable(ID_DEBUG_CHOOSE_RANDOM_IMAGE_SET, False)
        self.__menu_debug.Enable(ID_DEBUG_VIEW_WORKSPACE, False)

        self.__menu_window = wx.Menu()
        self.__menu_window.Append(ID_WINDOW_CLOSE_ALL, "Close &All Open Windows\tctrl+L", 
                                  "Close all open module display windows")
        self.__menu_window.Append(ID_WINDOW_SHOW_ALL_WINDOWS,"Show All Windows On Run",
                                  "Show all module display windows for all modules during analysis")
        self.__menu_window.Append(ID_WINDOW_HIDE_ALL_WINDOWS,"Hide All Windows On Run",
                                  "Hide all module display windows for all modules during analysis")
        self.__menu_window.AppendSeparator()
        
        self.__menu_help = wx.Menu()
        # We must add a non-submenu menu item before
        # make_help_menu adds submenus, otherwise the submenus
        # will disappear on the Mac.
        self.__menu_help.Append(ID_HELP_WELCOME, "Show Welcome Screen","Display the welcome screen shown at startup")
        self.__menu_help.Append(ID_HELP_RELEASE_NOTES, "Release Notes","Show the release notes in a browser")
        self.__menu_help.Append(ID_HELP_ONLINE_MANUAL, "Online Manual","Launch the HTML help in a browser")
        self.__menu_help.AppendSeparator()
        make_help_menu(MAIN_HELP, self, self.__menu_help)
        self.__menu_help.AppendSeparator()
        self.__menu_help.AppendSubMenu(self.data_tools_help(), 'Data Tool Help','Display documentation for available data tools')
        self.__menu_help.Append(ID_HELP_MODULE,'Module Help','Display Documentation for the Current Module')
        self.__menu_help.Append(ID_HELP_SEARCH, "Search Help...",
                                "Search for help pages that match a search term.")
        self.__menu_help.AppendSeparator()
        self.__menu_help.Append(ID_HELP_DEVELOPERS_GUIDE,"Developer's Guide",
                                "Launch the developer's guide webpage")
        self.__menu_help.Append(ID_HELP_SOURCE_CODE, "Source Code",
                                "Visit CellProfiler's Github repository")
        self.__menu_help.Append(
            ID_CHECK_NEW_VERSION,
            "Check for Updates...",
            "Check for a new version of CellProfiler")
        self.__menu_help.Append(ID_HELP_ABOUT, "&About...",
                                "About CellProfiler")

        self.__menu_bar = wx.MenuBar()
        self.__menu_bar.Append(self.__menu_file,'&File')
        self.__menu_bar.Append(self.menu_edit, '&Edit')
        self.__menu_bar.Append(self.__menu_debug,'&Test')
        if cpprefs.get_show_sampling():
            self.__menu_sample = wx.Menu()
            self.__menu_sample.Append(ID_SAMPLE_INIT, 'Initialize Sampling', 'Initialize sampling up to current module')
            self.__menu_bar.Append(self.__menu_sample, '&Sample')
        self.__menu_bar.Append(self.data_tools_menu(), '&Data Tools')
        self.__menu_bar.Append(self.__menu_window, "&Window")
        if wx.VERSION <= (2, 8, 10, 1, '') and wx.Platform == '__WXMAC__':
            self.__menu_bar.Append(self.__menu_help, 'CellProfiler Help')
        else:
            self.__menu_bar.Append(self.__menu_help, '&Help')
        self.SetMenuBar(self.__menu_bar)
        self.enable_edit_commands([])

        wx.EVT_MENU(self, ID_FILE_OPEN_IMAGE, self.on_open_image)
        wx.EVT_MENU(self,ID_FILE_EXIT,lambda event: self.Close())
        wx.EVT_MENU(self,ID_FILE_WIDGET_INSPECTOR,self.__on_widget_inspector)
        wx.EVT_MENU(self, ID_FILE_NEW_CP,self.__on_new_cp)
        
        wx.EVT_MENU(self, wx.ID_CUT, self.on_cut)
        self.Bind(wx.EVT_UPDATE_UI, self.on_update_cut_ui, id=wx.ID_CUT)
        wx.EVT_MENU(self, wx.ID_COPY, self.on_copy)
        self.Bind(wx.EVT_UPDATE_UI, self.on_update_copy_ui, id=wx.ID_COPY)
        wx.EVT_MENU(self, wx.ID_PASTE, self.on_paste)
        self.Bind(wx.EVT_UPDATE_UI, self.on_update_paste_ui, id=wx.ID_PASTE)
        wx.EVT_MENU(self, wx.ID_SELECTALL, self.on_select_all)
        self.Bind(wx.EVT_UPDATE_UI, self.on_update_select_all_ui, 
                  id=wx.ID_SELECTALL)
        
        wx.EVT_MENU(self, ID_HELP_WELCOME, self.__on_help_welcome)
        wx.EVT_MENU(self,ID_HELP_MODULE,self.__on_help_module)
        wx.EVT_BUTTON(self,ID_HELP_MODULE,self.__on_help_module)
        wx.EVT_MENU(self,ID_HELP_RELEASE_NOTES,self.__on_help_release_notes)
        wx.EVT_MENU(self,ID_HELP_ONLINE_MANUAL,self.__on_help_online_manual)
        wx.EVT_MENU(self,ID_HELP_DEVELOPERS_GUIDE, self.__on_help_developers_guide)
        wx.EVT_MENU(self, ID_HELP_SOURCE_CODE, self.__on_help_source_code)
        wx.EVT_MENU(self, ID_HELP_SEARCH, self.__on_search_help)
        wx.EVT_MENU(self, ID_HELP_ABOUT, self.__on_help_about)
        wx.EVT_MENU(self,ID_OPTIONS_PREFERENCES, self.__on_preferences)
        wx.EVT_MENU(self,ID_CHECK_NEW_VERSION, self.__on_check_new_version)
        wx.EVT_MENU(self,ID_WINDOW_CLOSE_ALL, self.__on_close_all)
        wx.EVT_MENU(self, ID_DEBUG_NUMPY, self.__debug_numpy_references)
        wx.EVT_MENU(self, ID_DEBUG_PDB, self.__debug_pdb)
        accelerator_table = wx.AcceleratorTable(
            [(wx.ACCEL_CMD, ord('N'), ID_FILE_ANALYZE_IMAGES),
             (wx.ACCEL_CMD, ord('O'), ID_FILE_LOAD),
             (wx.ACCEL_CMD, ord('P'), ID_FILE_SAVE_PIPELINE),
             (wx.ACCEL_CMD|wx.ACCEL_SHIFT,ord('S'),ID_FILE_SAVE),
             (wx.ACCEL_CMD,ord('L'),ID_WINDOW_CLOSE_ALL),
             (wx.ACCEL_CMD,ord('Q'),ID_FILE_EXIT),
             (wx.ACCEL_CMD,ord('W'),ID_FILE_EXIT),
             (wx.ACCEL_CMD, ord('A'), wx.ID_SELECTALL),
             (wx.ACCEL_CMD, ord('C'), wx.ID_CUT),
             (wx.ACCEL_CMD, ord('V'), wx.ID_PASTE),
             (wx.ACCEL_NORMAL,wx.WXK_F5,ID_DEBUG_TOGGLE),
             (wx.ACCEL_NORMAL,wx.WXK_F6,ID_DEBUG_STEP),
             (wx.ACCEL_NORMAL,wx.WXK_F7,ID_DEBUG_NEXT_IMAGE_SET),
             (wx.ACCEL_NORMAL,wx.WXK_F8,ID_DEBUG_NEXT_GROUP),
             (wx.ACCEL_CMD,ord('Z'),ID_EDIT_UNDO) ])
        self.SetAcceleratorTable(accelerator_table)
        self.enable_launch_commands()

    def data_tools_help(self):
        '''Create a help menu for the data tools'''
        if not hasattr(self, "__data_tools_help_menu"):
            self.__menu_data_tools_help_menu = wx.Menu()
            def on_plate_viewer_help(event):
                import htmldialog
                dlg = htmldialog.HTMLDialog(
                    self, "Help on plate viewer", PLATEVIEWER_HELP)
                dlg.Show()
            new_id = wx.NewId()
            self.__menu_data_tools_help_menu.Append(new_id, "Plate viewer")
            wx.EVT_MENU(self, new_id, on_plate_viewer_help)
                
            for data_tool_name in get_data_tool_names():
                new_id = wx.NewId()
                self.__menu_data_tools_help_menu.Append(new_id, data_tool_name)

                def on_data_tool_help(event, data_tool_name=data_tool_name):
                    self.__on_data_tool_help(event, data_tool_name)
                wx.EVT_MENU(self, new_id, on_data_tool_help)
        return self.__menu_data_tools_help_menu

    def data_tools_menu(self):
        '''Create a menu of data tools'''

        if not hasattr(self, "__data_tools_menu"):
            self.__data_tools_menu = wx.Menu()
            
            def on_data_tool_overview(event):
                import htmldialog
                from cellprofiler.gui.help import MENU_BAR_DATATOOLS_HELP
                dlg = htmldialog.HTMLDialog(self, 'Data Tool Overview', MENU_BAR_DATATOOLS_HELP)
                dlg.Show()            
            new_id = wx.NewId()
            self.__data_tools_menu.Append(
                new_id, 'Data Tool Overview', 'Overview of the Data Tools') 
            wx.EVT_MENU(self,new_id,on_data_tool_overview)
            
            self.__data_tools_menu.AppendSeparator()
            
            self.__data_tools_menu.Append(
                ID_FILE_PLATEVIEWER, 'Plate Viewer', 
                'Open the plate viewer to inspect the images in the current workspace')
            
            self.__data_tools_menu.AppendSeparator()
            
            for data_tool_name in get_data_tool_names():
                new_id = wx.NewId()
                self.__data_tools_menu.Append(new_id, data_tool_name)
                def on_data_tool(event, data_tool_name=data_tool_name):
                    self.__on_data_tool(event, data_tool_name)
                wx.EVT_MENU(self, new_id, on_data_tool)

            self.__data_tools_menu.AppendSeparator()
            
            self.__data_tools_menu.AppendSubMenu(self.data_tools_help(), '&Help')

        return self.__data_tools_menu

    #########################################################
    #
    # Handlers for ID_CUT / ID_COPY / ID_DELETE / ID_PASTE
    #
    # Adapted from a post reply by Robin Dunn:
    # http://wxpython-users.1045709.n5.nabble.com/how-to-implement-copy-paste-with-accelerators-td3337472.html
    #########################################################
    def on_cut(self, event):
        '''Handle ID_CUT'''
        focus = wx.Window.FindFocus()
        if (focus is not None and hasattr(focus, "Cut") 
            and hasattr(focus, 'CanCut') and focus.CanCut()):
            focus.Cut()
            
    def on_update_cut_ui(self, event):
        focus = wx.Window.FindFocus() 
        event.Enable(bool(
            focus and hasattr(focus, 'CanCut') and focus.CanCut()))
       
    def on_copy(self, event):
        '''Handle ID_COPY'''
        focus = wx.Window.FindFocus()
        if focus is not None and hasattr(focus, "Copy") and\
           hasattr(focus, 'CanCopy') and focus.CanCopy():
            focus.Copy()
            
    def on_update_copy_ui(self, event):
        focus = wx.Window.FindFocus() 
        event.Enable(bool(
            focus and hasattr(focus, 'CanCopy') and focus.CanCopy()))

    def on_paste(self, event):
        '''Handle ID_PASTE'''
        focus = wx.Window.FindFocus()
        if focus is not None and hasattr(focus, "Paste") and\
           hasattr(focus, "CanPaste") and focus.CanPaste():
            focus.Paste()
            
    def on_update_paste_ui(self, event):
        focus = wx.Window.FindFocus() 
        event.Enable(bool(
            focus and hasattr(focus, 'CanPaste') and focus.CanPaste()))
        
    def on_select_all(self, event):
        focus = wx.Window.FindFocus()
        if (focus and hasattr(focus, "SelectAll")):
            focus.SelectAll()

    def on_update_select_all_ui(self, event):
        focus = wx.Window.FindFocus()
        if hasattr(focus, "CanSelect") and not focus.CanSelect():
            event.Enable(False)
            return
        event.Enable(bool(focus and hasattr(focus, "SelectAll")))

    debug_commands = (ID_DEBUG_STEP, ID_DEBUG_NEXT_IMAGE_SET,
                      ID_DEBUG_NEXT_GROUP, ID_DEBUG_CHOOSE_GROUP,
                      ID_DEBUG_CHOOSE_IMAGE_SET, 
                      ID_DEBUG_CHOOSE_RANDOM_IMAGE_SET,
                      ID_DEBUG_VIEW_WORKSPACE)
    def enable_debug_commands(self):
        """Enable or disable the debug commands (like ID_DEBUG_STEP)"""
        startstop = self.__menu_debug.FindItemById(ID_DEBUG_TOGGLE)
        self.__menu_file.Enable(ID_FILE_ANALYZE_IMAGES, False)
        self.__menu_file.Enable(ID_FILE_RESTART, False)
        self.__menu_file.Enable(ID_FILE_RUN_MULTIPLE_PIPELINES, False)

        assert isinstance(startstop, wx.MenuItem)
        startstop.Text = '&Exit Test Mode\tF5'
        startstop.Help = 'Stop testing your pipeline'
        for cmd in self.debug_commands:
            self.__menu_debug.Enable(cmd, True)

    def enable_launch_commands(self):
        '''Enable commands to start analysis or test mode'''
        startstop = self.__menu_debug.FindItemById(ID_DEBUG_TOGGLE)
        startstop.Text =  '&Start Test Mode\tF5'
        startstop.Help =  'Start testing your pipeline'
        for cmd in self.debug_commands:
            self.__menu_debug.Enable(cmd, False)
        self.__menu_file.Enable(ID_FILE_ANALYZE_IMAGES, True)
        self.__menu_file.Enable(ID_FILE_RESTART, True)
        self.__menu_debug.Enable(ID_DEBUG_TOGGLE, True)
        self.__menu_file.Enable(ID_FILE_RUN_MULTIPLE_PIPELINES, True)

        self.__menu_file.Enable(ID_FILE_STOP_ANALYSIS, False)

    def enable_analysis_commands(self):
        '''Enable commands to pause or stop analysis'''
        self.__menu_file.Enable(ID_FILE_ANALYZE_IMAGES, False)
        self.__menu_file.Enable(ID_FILE_RESTART, False)
        self.__menu_debug.Enable(ID_DEBUG_TOGGLE, False)
        self.__menu_file.Enable(ID_FILE_RUN_MULTIPLE_PIPELINES, False)

        self.__menu_file.Enable(ID_FILE_STOP_ANALYSIS, True)

    def __on_widget_inspector(self, evt):
        try:
            import wx.lib.inspection
            wx.lib.inspection.InspectionTool().Show()
        except:
            wx.MessageBox("Inspection tool is not available on this platform")

    def __on_preferences(self, event):
        dlg = cellprofiler.gui.preferencesdlg.PreferencesDlg()
        dlg.Show()

    def __on_check_new_version(self, event):
        wx.GetApp().new_version_check(force=True)

    def __on_close_all(self, event):
        close_all(self)

    def __on_new_cp(self, event):
        import os
        if not hasattr(sys, 'frozen'):
            os.system('open CellProfiler_python.command')
        else:       
            os.system('open -na CellProfiler.app')

    def __on_help_release_notes(self, event):
            import webbrowser
            webbrowser.open("http://github.com/CellProfiler/CellProfiler/wiki/CellProfiler-release-notes")    
    
    def __on_help_online_manual(self, event):
        import webbrowser
        webbrowser.open("http://cellprofiler.org/CPmanual/")

    def __on_help_developers_guide(self, event):
        import webbrowser
        webbrowser.open("http://www.cellprofiler.org/wiki/index.php/Main_Page")
        
    def __on_help_source_code(self, event):
        import webbrowser
        webbrowser.open("https://github.com/CellProfiler/CellProfiler")
        
    def __on_help_path_list(self, event):
        import htmldialog
        dlg = htmldialog.HTMLDialog(self, "Help on file list", HELP_ON_FILE_LIST)
        dlg.Show()

    def __on_help_about(self, event):
        CellProfilerSplash = get_builtin_image('CellProfilerSplash')
        splashbitmap = wx.BitmapFromImage(CellProfilerSplash)

        dlg = wx.Dialog(self)
        dlg.Title = "About CellProfiler"
        sizer = wx.BoxSizer(wx.VERTICAL)
        dlg.SetSizer(sizer)
        sizer.Add(wx.StaticBitmap(dlg, -1, splashbitmap), 0, wx.EXPAND | wx.ALL, 5)
        sizer.Add(wx.StaticText(
            dlg, label="Version: %s" % version.dotted_version),
                  0, wx.EXPAND | wx.ALL, 5)
        sizer.Add(wx.StaticText(
            dlg, label="GIT hash: %s" % version.git_hash),
                  0, wx.EXPAND | wx.ALL, 5)
        sizer.Add(wx.StaticText(
            dlg, label="Revision author timestamp: %d" % version.version_number),
                  0, wx.EXPAND | wx.ALL, 5)

        cellprofiler_copyright = u"""
CellProfiler (BI-2013-070) is cell image analysis software that was
developed with funding from the U.S. National Institutes of Health and
National Science Foundation and is made available through the
generosity of the Broad Institute, Inc.
Copyright © 2009-2015 Broad Institute, Inc.
Copyright © 2008-2009 Massachusetts Institute of Technology
All rights reserved."""
        copyright_ctl = wx.StaticText(dlg, -1, cellprofiler_copyright)
        sizer.Add(copyright_ctl, 0, wx.ALIGN_LEFT | wx.ALL, 5)

        license_ctl = wx.StaticText(dlg, -1, cellprofiler_license)
        sizer.Add(license_ctl, 0, wx.ALIGN_LEFT | wx.ALL, 5)
        
        button_list = [("GPL License", gpl_license),
                       ("BSD License", bsd_license)]
        
        additional_copyrights = u""" (a) h5py © 2008 Andrew Collette, et al.; ilastik © 2012 C. Sommer,
et al.,; ImageJ2 © 2009-2013 Board of Regents of the University of
Wisconsin-Madison, Max Planck Institute of Molecular Cell Biology and
Genetics; libhdf5 © 2006-2013 HDF Group; libtiff © 1988-1997 Sam
Leffler and © 1991- 1997 Silicon Graphics, Inc.; NumPy © 2005-2013
NumPy Developers; openGL © 2013 Khronos Group; Pillow © 1990-2013
Python Software Foundation; progressbar © 2005 Nilton Volpato; SciPy ©
2013 SciPy Developers; sklearn © 2007-2013 Scikit-Learn Developers;
and zlib © 1995-2013 Jean-loup Gailly, Mark Adler under BSD licenses;
(b) altgraph © 2004 Istvan Albert, Bob Ippolito and © 2010-2013 Ronald
Oussoren; macholib © 2010-2013 Bob Ippolito, Ronald Oussoren;
modulegraph © 2010-2013 Bob Ippolito, Ronald Oussoren; py2app ©
2004-2006 Bob Ippolito, © 2010-2011 Ronald Oussoren; and vigra ©
1998-2013 by Ullrich Koethe under MIT licenses; (c) libjpeg ©
1991-1998 Thomas G. Lane under a license from the Independent JPEG
Group found at https://github.com/aseprite/aseprite/blob/master/docs/
licenses/libjpeg-LICENSE.txt; (d) matplotlib © 2003-2012 John Hunter,
© 2012-2013 Michael Droettboom et al. under licenses found at
http://matplotlib.org/users/license.html#copyright-policy; (e) Java
Runtime Environment © 2013 Oracle, Inc. under the Oracle Binary Code
License Agreement for the Java SE Platform Products and JavaFX found
at http://www.oracle.com/technetwork/java/javase/terms/license/
index.html; (f) Bioformats © 2000-2013 University of Dundee & Open
Microscopy Environment; ice © 2013 ZeroC, Inc.; mySQLdb © 2013 Andy
Dustman; omero © 2002-2013 Open Microscopy Environment; and ZeroMQ ©
2007-2012 iMatix Corporation and Contributors; (g) wxWidgets ©
1999-2005 Julian Smart, et al.; and wxPython © 1992-2006 Julian Smart,
et al. and © 1996 Artificial Intelligence Applications Institute under
the wxWindows license; (h) Intel Visual Fortran Compiler Run-time
Library, Intel Fortran Portability Library, Intel OMP Runtime Library,
Math Library for Intel Compilers, Short Vector Math Library for Intel
Compilers under the Single User license provision for
Resdistributables © 2013 Intel Corporation under the End User License
Agreement for the Intel Software Development Products; and
(i) FastEMD © Copyright (c) 2009-2012, Ofir Pele. under BSD licenses

"""

        icon_copyrights = get_icon_copyrights()
        if additional_copyrights is not None:
            additional_copyrights += icon_copyrights
        button_list.append(("Additional licenses and attributions", 
                            additional_copyrights))

        for button_text, license_text in button_list:
            sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(sub_sizer, 0, wx.EXPAND | wx.ALL, 5)
            license_button = wx.Button(dlg, -1, button_text + " >>")
            sub_sizer.Add(license_button, 0, wx.ALIGN_LEFT | wx.RIGHT, 3)
            sub_sizer.Add(wx.StaticLine(dlg), 0, wx.ALIGN_CENTER_VERTICAL | wx.EXPAND)

            panel = wx.lib.scrolledpanel.ScrolledPanel(dlg)
            panel.Sizer = wx.BoxSizer(wx.VERTICAL)
            license_ctl = wx.StaticText(panel, -1, license_text)
            panel.SetMinSize((license_ctl.Size[0] + 20, 320))
            panel.Sizer.Add(license_ctl, 1, 
                            wx.EXPAND | wx.ALL, 2)
            sizer.Add(panel, 0, wx.EXPAND | wx.ALL, 5)
            panel.SetupScrolling()
            panel.Hide()

            def on_license(event, panel=panel):
                if panel.IsShown():
                    panel.Hide()
                else:
                    panel.Show()
                dlg.Fit()
                dlg.Refresh()
            license_button.Bind(wx.EVT_BUTTON, on_license)

        sizer.Add(dlg.CreateStdDialogButtonSizer(wx.OK), 0, 
                  wx.ALIGN_RIGHT | wx.ALL, 5)
        dlg.Fit()
        dlg.ShowModal()

    def __on_help_welcome(self, event):
        self.show_welcome_screen(True)
        
    def __on_help_module(self, event):
        modules = self.__pipeline_list_view.get_selected_modules()
        active_module = self.__pipeline_list_view.get_active_module()
        if len(modules) > 0:
            self.do_help_modules(modules)
        elif active_module is not None:
            self.do_help_module(active_module.module_name, 
                                active_module.get_help())
        else:
            wx.MessageBox(HELP_ON_MODULE_BUT_NONE_SELECTED, 
                          "No module selected",
                          style=wx.OK|wx.ICON_INFORMATION)

    def __debug_numpy_references(self, event):
        try:
            import contrib.objgraph as objgraph
            numpyobj = [(o, objgraph.numpy_size(o)) for o in objgraph.by_instanceof(objgraph.numpyarray)]
            numpyobj = [o for o, sz in numpyobj if (sz is None) or (sz > 1024)]
            objgraph.show_backrefs(numpyobj, max_depth=4,
                                   filename=os.path.join(cpprefs.get_default_output_directory(),
                                                         'cellprofiler_numpy.dot'))
        except Exception, e:
            print "Couldn't generate objgraph: %s"%(e)
            import pdb
            pdb.post_mortem(sys.exc_traceback)

    def __debug_pdb(self, event):
        pdb.set_trace()

    def do_help_modules(self, modules):
        for module in modules:
            ## An attempt to place images inline with the help. However, the
            ## images will not scale properly in size (yet)
            #result = module.get_help()
            #root = os.path.split(__file__)[0]
            #if len(root) == 0:
                #root = os.curdir
            #root = os.path.split(os.path.abspath(root))[0] # Back up one level
            #absolute_image_path = os.path.join(root, 'icons','%s.png'%(module.module_name,))
            ## Check if the file that goes with this module exists on this computer
            #if os.path.exists(absolute_image_path) and os.path.isfile(absolute_image_path):
                ## If so, strip out end html tags so I can add more stuff
                #result = result.replace('</body>','').replace('</html>','')
                ## Include images specific to the module
                #result += '\n\n<div><p><img src="%s", width="50%%"></p></div>\n'%absolute_image_path
                ## Now end the help text
                #result += '</body></html>'
            #self.do_help_module(module.module_name, result)
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
        window = HtmlClickableWindow(helpframe)
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
        helpframe.SetIcon(get_cp_icon())
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
            pathname = save_dlg.GetPath()
            fd = open(pathname, "wt")
            fd.write(help_text)
            fd.close()

    def on_open_image(self, event):
        dlg = wx.FileDialog(self,
                            message = "Open an image file",
                            wildcard = "Image file (*.tif,*.tiff,*.jpg,*.jpeg,*.png,*.gif,*.bmp)|*.tif;*.tiff;*.jpg;*.jpeg;*.png;*.gif;*.bmp|*.* (all files)|*.*",
                            style = wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            from cellprofiler.modules.loadimages import LoadImagesImageProvider
            from cellprofiler.gui.cpfigure import CPFigureFrame
            lip = LoadImagesImageProvider("dummy", "", dlg.Path)
            image = lip.provide_image(None).pixel_data
            frame = CPFigureFrame(self, title=dlg.Path, subplots=(1,1))
            if image.ndim == 3:
                frame.subplot_imshow_color(0, 0, image, title = dlg.Path)
            else:
                frame.subplot_imshow_grayscale(0, 0, image, title=dlg.Path)
            frame.panel.draw()    

    def __attach_views(self):
        self.__pipeline_list_view = PipelineListView(self.__module_list_panel, self)
        self.__pipeline_controller = PipelineController(self.__workspace, self)
        self.__pipeline_list_view.attach_to_pipeline(self.__pipeline,self.__pipeline_controller)
        self.__pipeline_controller.attach_to_test_controls_panel(self.__pipeline_test_panel)
        self.__pipeline_controller.attach_to_module_controls_panel(self.__module_controls_panel)
        self.__pipeline_controller.attach_to_path_list_ctrl(
            self.__path_list_ctrl, 
            self.__path_list_filter_checkbox)
        self.__module_view = ModuleView(
            self.__module_panel,
            self.__workspace,
            frame = self,
            notes_panel = self.__notes_panel)
        self.__pipeline_controller.attach_to_module_view(self.__module_view)
        self.__pipeline_list_view.attach_to_module_view((self.__module_view))
        self.__preferences_view = PreferencesView(
            self.__right_win.Sizer,
            self.__preferences_panel, 
            self.__progress_panel, 
            self.__status_panel)
        self.__preferences_view.attach_to_pipeline_list_view(self.__pipeline_list_view)

    def __do_layout(self):
        width = wx.SystemSettings.GetMetric(wx.SYS_SCREEN_X)
        height = wx.SystemSettings.GetMetric(wx.SYS_SCREEN_Y)
        self.SetSize((int(width * 2 / 3), int(height * 2 / 3)))
        splitter = self.__splitter
        right_win = self.__right_win
        top_left_win = self.__left_win

        self.__splitter.SetMinimumPaneSize(120)
        self.__splitter.SplitVertically(self.__left_win, self.__right_win, 300)
        self.__splitter.BorderSize = 0
        self.__splitter.SashSize = 5
        self.__splitter.BackgroundColour = self.BackgroundColour

        top_left_sizer = wx.BoxSizer(wx.VERTICAL)
        top_left_sizer.Add(self.__module_list_panel,1,wx.EXPAND|wx.ALL,1)
        top_left_sizer.Add(self.__module_controls_panel,0,wx.EXPAND|wx.ALL,2)
        top_left_sizer.Add(self.__pipeline_test_panel, 0, wx.EXPAND|wx.ALL,2)
        top_left_win.SetSizer(top_left_sizer)

        border = wx.BoxSizer()
        border.Add(splitter, 1, wx.EXPAND|wx.ALL,1)
        self.SetSizer(border)
        self.Layout()
        right_win.Layout()
        top_left_win.Layout()

    def __set_icon(self):
        self.SetIcon(get_cp_icon())

    def __make_search_frame(self):
        '''Make and hide the "search the help" frame'''
        background_color = cpprefs.get_background_color()
        size = (wx.SystemSettings.GetMetric(wx.SYS_SCREEN_X) / 2,
                wx.SystemSettings.GetMetric(wx.SYS_SCREEN_Y) / 2)
        self.search_frame = wx.Frame(
            self, title = "Search CellProfiler help",
            size = size,
            style = wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL)
        self.search_frame.AutoLayout = True
        self.search_frame.SetIcon(get_cp_icon())
        self.search_frame.Sizer = wx.BoxSizer(wx.VERTICAL)
        self.search_frame.BackgroundColour = background_color
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.search_frame.Sizer.Add(sizer, 0, wx.EXPAND | wx.ALL, 4)
        sizer.Add(wx.StaticText(self.search_frame, label = "Search:"), 0,
                  wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        sizer.AddSpacer(2)
        search_text_ctrl = wx.TextCtrl(self.search_frame)
        sizer.Add(search_text_ctrl, 1, wx.EXPAND)
        search_button = wx.Button(self.search_frame, label = "Search")
        search_button.SetDefault()
        sizer.AddSpacer(2)
        sizer.Add(search_button, 0, wx.EXPAND)

        html_window = HtmlClickableWindow(self.search_frame)
        self.search_frame.Sizer.Add(html_window, 1, wx.EXPAND | wx.ALL, 4)
        def on_search(event):
            from cellprofiler.gui.html.manual import search_module_help
            search_text = search_text_ctrl.Value
            html = search_module_help(search_text)
            if html is None:
                so_sorry = """<html>
		<header><title>"%s" not found in help</title></header>
		<body>Could not find "%s" in CellProfiler's help documentation</body>
		</html>""" % (search_text, search_text)
                html_window.SetPage(so_sorry)
            else:
                html_window.SetPage(html)
        search_button.Bind(wx.EVT_BUTTON, on_search)
        def on_link_clicked(event):
            '''Handle anchor clicks manually

            The HTML window (on Windows at least) jams the anchor to the
            top of the window which obscures it.
            '''
            linkinfo = event.GetLinkInfo()
            if linkinfo.GetHref()[0] != "#":
                event.Skip()
                return
            html_window.ScrollToAnchor(linkinfo.GetHref()[1:])
            html_window.ScrollLines(-1)

        html_window.Bind(wx.html.EVT_HTML_LINK_CLICKED, on_link_clicked)
        def on_close(event):
            assert isinstance(event, wx.CloseEvent)
            self.search_frame.Hide()
            event.Veto()

        self.search_frame.Bind(wx.EVT_CLOSE, on_close)
        self.search_frame.Layout()
        self.search_frame.SetIcon(get_cp_icon())

    def __on_search_help(self, event):
        if self.search_frame is not None:
            self.search_frame.Show()
            self.search_frame.Raise()
            
    def __make_startup_blurb_frame(self):
        '''Make the frame surrounding the startup blurb panel'''
        background_color = cpprefs.get_background_color()
        frame = self.startup_blurb_frame = wx.Frame(
            self, title = "Welcome to CellProfiler",
            size = (640, 480),
            name=WELCOME_SCREEN_FRAME)
        frame.BackgroundColour = background_color
        frame.Sizer = wx.BoxSizer()
        content = HtmlClickableWindow(frame)
        content.load_startup_blurb()
        frame.Sizer.Add(content, 1, wx.EXPAND)
        frame.SetIcon(get_cp_icon())
        
        def on_close(event):
            assert isinstance(event, wx.CloseEvent)
            event.EventObject.Hide()
            event.Veto()
        
        frame.Bind(wx.EVT_CLOSE, on_close)
        frame.Layout()

    def __on_data_tool(self, event, tool_name):
        module = instantiate_module(tool_name)
        args, varargs, varkw, vardef = inspect.getargspec(module.run_as_data_tool)
        if len(args) + (0 if varargs is None else len(varargs)) == 1:
            # Data tool doesn't need the data tool frame because it doesn't
            # take the "workspace" argument
            #
            module.run_as_data_tool()
            return
        dlg = wx.FileDialog(
            self, "Choose data output file for %s data tool" %
            tool_name, wildcard = "Measurements file(*.mat,*.h5)|*.mat;*.h5",
            style=(wx.FD_OPEN | wx.FILE_MUST_EXIST))
        if dlg.ShowModal() == wx.ID_OK:
            DataToolFrame(self, 
                          module_name=tool_name,
                          measurements_file_name = dlg.Path)

    def __on_data_tool_help(self, event, tool_name):
        module = instantiate_module(tool_name)
        self.do_help_module(tool_name, module.get_help())

    def display_error(self,message,error):
        """Displays an exception in a standardized way

        """
        for listener in self.__error_listeners:
            listener(message, error)
        tb = sys.exc_info()[2]
        traceback.print_tb(tb)
        text = '\n'.join(traceback.format_list(traceback.extract_tb(tb)))
        text = error.message + '\n'+text
        display_error_message(self, text, "Caught exception during operation")

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

cellprofiler_license = """
CellProfiler is licensed under the terms of the GNU General Public
License (GPL) version 2. The files in the "CellProfiler/cpmath" and
"CellProfiler/utilities" subdirectories are licensed under the more
permissive BSD license. CellProfiler is distributed, in part, under
the provisions of the additional licenses and attributions below.

CELLPROFILER IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF CELLPROFILER, EVEN
IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  """

gpl_license = """
		    GNU GENERAL PUBLIC LICENSE
		       Version 2, June 1991

 Copyright (C) 1989, 1991 Free Software Foundation, Inc.,
 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

			    Preamble

  The licenses for most software are designed to take away your
freedom to share and change it.  By contrast, the GNU General Public
License is intended to guarantee your freedom to share and change free
software--to make sure the software is free for all its users.  This
General Public License applies to most of the Free Software
Foundation's software and to any other program whose authors commit to
using it.  (Some other Free Software Foundation software is covered by
the GNU Lesser General Public License instead.)  You can apply it to
your programs, too.

  When we speak of free software, we are referring to freedom, not
price.  Our General Public Licenses are designed to make sure that you
have the freedom to distribute copies of free software (and charge for
this service if you wish), that you receive source code or can get it
if you want it, that you can change the software or use pieces of it
in new free programs; and that you know you can do these things.

  To protect your rights, we need to make restrictions that forbid
anyone to deny you these rights or to ask you to surrender the rights.
These restrictions translate to certain responsibilities for you if you
distribute copies of the software, or if you modify it.

  For example, if you distribute copies of such a program, whether
gratis or for a fee, you must give the recipients all the rights that
you have.  You must make sure that they, too, receive or can get the
source code.  And you must show them these terms so they know their
rights.

  We protect your rights with two steps: (1) copyright the software, and
(2) offer you this license which gives you legal permission to copy,
distribute and/or modify the software.

  Also, for each author's protection and ours, we want to make certain
that everyone understands that there is no warranty for this free
software.  If the software is modified by someone else and passed on, we
want its recipients to know that what they have is not the original, so
that any problems introduced by others will not reflect on the original
authors' reputations.

  Finally, any free program is threatened constantly by software
patents.  We wish to avoid the danger that redistributors of a free
program will individually obtain patent licenses, in effect making the
program proprietary.  To prevent this, we have made it clear that any
patent must be licensed for everyone's free use or not licensed at all.

  The precise terms and conditions for copying, distribution and
modification follow.

		    GNU GENERAL PUBLIC LICENSE
   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

  0. This License applies to any program or other work which contains
a notice placed by the copyright holder saying it may be distributed
under the terms of this General Public License.  The "Program", below,
refers to any such program or work, and a "work based on the Program"
means either the Program or any derivative work under copyright law:
that is to say, a work containing the Program or a portion of it,
either verbatim or with modifications and/or translated into another
language.  (Hereinafter, translation is included without limitation in
the term "modification".)  Each licensee is addressed as "you".

Activities other than copying, distribution and modification are not
covered by this License; they are outside its scope.  The act of
running the Program is not restricted, and the output from the Program
is covered only if its contents constitute a work based on the
Program (independent of having been made by running the Program).
Whether that is true depends on what the Program does.

  1. You may copy and distribute verbatim copies of the Program's
source code as you receive it, in any medium, provided that you
conspicuously and appropriately publish on each copy an appropriate
copyright notice and disclaimer of warranty; keep intact all the
notices that refer to this License and to the absence of any warranty;
and give any other recipients of the Program a copy of this License
along with the Program.

You may charge a fee for the physical act of transferring a copy, and
you may at your option offer warranty protection in exchange for a fee.

  2. You may modify your copy or copies of the Program or any portion
of it, thus forming a work based on the Program, and copy and
distribute such modifications or work under the terms of Section 1
above, provided that you also meet all of these conditions:

    a) You must cause the modified files to carry prominent notices
    stating that you changed the files and the date of any change.

    b) You must cause any work that you distribute or publish, that in
    whole or in part contains or is derived from the Program or any
    part thereof, to be licensed as a whole at no charge to all third
    parties under the terms of this License.

    c) If the modified program normally reads commands interactively
    when run, you must cause it, when started running for such
    interactive use in the most ordinary way, to print or display an
    announcement including an appropriate copyright notice and a
    notice that there is no warranty (or else, saying that you provide
    a warranty) and that users may redistribute the program under
    these conditions, and telling the user how to view a copy of this
    License.  (Exception: if the Program itself is interactive but
    does not normally print such an announcement, your work based on
    the Program is not required to print an announcement.)

These requirements apply to the modified work as a whole.  If
identifiable sections of that work are not derived from the Program,
and can be reasonably considered independent and separate works in
themselves, then this License, and its terms, do not apply to those
sections when you distribute them as separate works.  But when you
distribute the same sections as part of a whole which is a work based
on the Program, the distribution of the whole must be on the terms of
this License, whose permissions for other licensees extend to the
entire whole, and thus to each and every part regardless of who wrote it.

Thus, it is not the intent of this section to claim rights or contest
your rights to work written entirely by you; rather, the intent is to
exercise the right to control the distribution of derivative or
collective works based on the Program.

In addition, mere aggregation of another work not based on the Program
with the Program (or with a work based on the Program) on a volume of
a storage or distribution medium does not bring the other work under
the scope of this License.

  3. You may copy and distribute the Program (or a work based on it,
under Section 2) in object code or executable form under the terms of
Sections 1 and 2 above provided that you also do one of the following:

    a) Accompany it with the complete corresponding machine-readable
    source code, which must be distributed under the terms of Sections
    1 and 2 above on a medium customarily used for software interchange; or,

    b) Accompany it with a written offer, valid for at least three
    years, to give any third party, for a charge no more than your
    cost of physically performing source distribution, a complete
    machine-readable copy of the corresponding source code, to be
    distributed under the terms of Sections 1 and 2 above on a medium
    customarily used for software interchange; or,

    c) Accompany it with the information you received as to the offer
    to distribute corresponding source code.  (This alternative is
    allowed only for noncommercial distribution and only if you
    received the program in object code or executable form with such
    an offer, in accord with Subsection b above.)

The source code for a work means the preferred form of the work for
making modifications to it.  For an executable work, complete source
code means all the source code for all modules it contains, plus any
associated interface definition files, plus the scripts used to
control compilation and installation of the executable.  However, as a
special exception, the source code distributed need not include
anything that is normally distributed (in either source or binary
form) with the major components (compiler, kernel, and so on) of the
operating system on which the executable runs, unless that component
itself accompanies the executable.

If distribution of executable or object code is made by offering
access to copy from a designated place, then offering equivalent
access to copy the source code from the same place counts as
distribution of the source code, even though third parties are not
compelled to copy the source along with the object code.

  4. You may not copy, modify, sublicense, or distribute the Program
except as expressly provided under this License.  Any attempt
otherwise to copy, modify, sublicense or distribute the Program is
void, and will automatically terminate your rights under this License.
However, parties who have received copies, or rights, from you under
this License will not have their licenses terminated so long as such
parties remain in full compliance.

  5. You are not required to accept this License, since you have not
signed it.  However, nothing else grants you permission to modify or
distribute the Program or its derivative works.  These actions are
prohibited by law if you do not accept this License.  Therefore, by
modifying or distributing the Program (or any work based on the
Program), you indicate your acceptance of this License to do so, and
all its terms and conditions for copying, distributing or modifying
the Program or works based on it.

  6. Each time you redistribute the Program (or any work based on the
Program), the recipient automatically receives a license from the
original licensor to copy, distribute or modify the Program subject to
these terms and conditions.  You may not impose any further
restrictions on the recipients' exercise of the rights granted herein.
You are not responsible for enforcing compliance by third parties to
this License.

  7. If, as a consequence of a court judgment or allegation of patent
infringement or for any other reason (not limited to patent issues),
conditions are imposed on you (whether by court order, agreement or
otherwise) that contradict the conditions of this License, they do not
excuse you from the conditions of this License.  If you cannot
distribute so as to satisfy simultaneously your obligations under this
License and any other pertinent obligations, then as a consequence you
may not distribute the Program at all.  For example, if a patent
license would not permit royalty-free redistribution of the Program by
all those who receive copies directly or indirectly through you, then
the only way you could satisfy both it and this License would be to
refrain entirely from distribution of the Program.

If any portion of this section is held invalid or unenforceable under
any particular circumstance, the balance of the section is intended to
apply and the section as a whole is intended to apply in other
circumstances.

It is not the purpose of this section to induce you to infringe any
patents or other property right claims or to contest validity of any
such claims; this section has the sole purpose of protecting the
integrity of the free software distribution system, which is
implemented by public license practices.  Many people have made
generous contributions to the wide range of software distributed
through that system in reliance on consistent application of that
system; it is up to the author/donor to decide if he or she is willing
to distribute software through any other system and a licensee cannot
impose that choice.

This section is intended to make thoroughly clear what is believed to
be a consequence of the rest of this License.

  8. If the distribution and/or use of the Program is restricted in
certain countries either by patents or by copyrighted interfaces, the
original copyright holder who places the Program under this License
may add an explicit geographical distribution limitation excluding
those countries, so that distribution is permitted only in or among
countries not thus excluded.  In such case, this License incorporates
the limitation as if written in the body of this License.

  9. The Free Software Foundation may publish revised and/or new versions
of the General Public License from time to time.  Such new versions will
be similar in spirit to the present version, but may differ in detail to
address new problems or concerns.

Each version is given a distinguishing version number.  If the Program
specifies a version number of this License which applies to it and "any
later version", you have the option of following the terms and conditions
either of that version or of any later version published by the Free
Software Foundation.  If the Program does not specify a version number of
this License, you may choose any version ever published by the Free Software
Foundation.

  10. If you wish to incorporate parts of the Program into other free
programs whose distribution conditions are different, write to the author
to ask for permission.  For software which is copyrighted by the Free
Software Foundation, write to the Free Software Foundation; we sometimes
make exceptions for this.  Our decision will be guided by the two goals
of preserving the free status of all derivatives of our free software and
of promoting the sharing and reuse of software generally.

			    NO WARRANTY

  11. BECAUSE THE PROGRAM IS LICENSED FREE OF CHARGE, THERE IS NO WARRANTY
FOR THE PROGRAM, TO THE EXTENT PERMITTED BY APPLICABLE LAW.  EXCEPT WHEN
OTHERWISE STATED IN WRITING THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES
PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED
OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  THE ENTIRE RISK AS
TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU.  SHOULD THE
PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING,
REPAIR OR CORRECTION.

  12. IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING
WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MAY MODIFY AND/OR
REDISTRIBUTE THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES,
INCLUDING ANY GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING
OUT OF THE USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED
TO LOSS OF DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY
YOU OR THIRD PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER
PROGRAMS), EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE
POSSIBILITY OF SUCH DAMAGES.

		     END OF TERMS AND CONDITIONS

	    How to Apply These Terms to Your New Programs

  If you develop a new program, and you want it to be of the greatest
possible use to the public, the best way to achieve this is to make it
free software which everyone can redistribute and change under these terms.

  To do so, attach the following notices to the program.  It is safest
to attach them to the start of each source file to most effectively
convey the exclusion of warranty; and each file should have at least
the "copyright" line and a pointer to where the full notice is found.

    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) <year>  <name of author>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

Also add information on how to contact you by electronic and paper mail.

If the program is interactive, make it output a short notice like this
when it starts in an interactive mode:

    Gnomovision version 69, Copyright (C) year name of author
    Gnomovision comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
    This is free software, and you are welcome to redistribute it
    under certain conditions; type `show c' for details.

The hypothetical commands `show w' and `show c' should show the appropriate
parts of the General Public License.  Of course, the commands you use may
be called something other than `show w' and `show c'; they could even be
mouse-clicks or menu items--whatever suits your program.

You should also get your employer (if you work as a programmer) or your
school, if any, to sign a "copyright disclaimer" for the program, if
necessary.  Here is a sample; alter the names:

  Yoyodyne, Inc., hereby disclaims all copyright interest in the program
  `Gnomovision' (which makes passes at compilers) written by James Hacker.

  <signature of Ty Coon>, 1 April 1989
  Ty Coon, President of Vice

This General Public License does not permit incorporating your program into
proprietary programs.  If your program is a subroutine library, you may
consider it more useful to permit linking proprietary applications with the
library.  If this is what you want to do, use the GNU Lesser General
Public License instead of this License.
"""

bsd_license = """
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.

    * Neither the name of the Massachusetts Institute of Technology
      nor the Broad Institute nor the names of its contributors may be
      used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL MASSACHUSETTS
INSTITUTE OF TECHNOLOGY OR THE BROAD INSTITUTE BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."""
