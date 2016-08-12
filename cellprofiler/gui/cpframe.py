# -*- Encoding: utf-8 -*-
""" CellProfiler.CellProfilerGUI.CPFrame - Cell Profiler's main window
"""

import cellprofiler.gui
import cellprofiler.gui.figure
import cellprofiler.gui.datatoolframe
import cellprofiler.gui.dialog
import cellprofiler.gui.errordialog
import cellprofiler.gui.help
import cellprofiler.gui.html
import cellprofiler.gui.html.htmlwindow
import cellprofiler.gui.imagesetctrl
import cellprofiler.gui.moduleview
import cellprofiler.gui.pathlist
import cellprofiler.gui.pipelinecontroller
import cellprofiler.gui.pipelinelistview
import cellprofiler.gui.preferencesdlg
import cellprofiler.gui.preferencesview
import cellprofiler.icons
import cellprofiler.modules
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.utilities.version
import cellprofiler.workspace
import inspect
import logging
import os
import pdb
import sys
import traceback
import wx
import wx.html
import wx.lib.scrolledpanel

logger = logging.getLogger(__name__)

ID_FILE_NEW_WORKSPACE = wx.ID_NEW
ID_FILE_LOAD = wx.ID_OPEN
ID_FILE_LOAD_PIPELINE = wx.NewId()
ID_FILE_URL_LOAD_PIPELINE = wx.NewId()
ID_FILE_OPEN_IMAGE = wx.NewId()
ID_FILE_EXIT = wx.NewId()
ID_FILE_WIDGET_INSPECTOR = wx.NewId()
ID_FILE_SAVE_PIPELINE = wx.NewId()
ID_FILE_SAVE = wx.ID_SAVE
ID_FILE_SAVE_AS = wx.ID_SAVEAS
ID_FILE_REVERT_TO_SAVED = wx.NewId()
ID_FILE_CLEAR_PIPELINE = wx.NewId()
ID_FILE_EXPORT_IMAGE_SETS = wx.NewId()
ID_FILE_EXPORT_PIPELINE_NOTES = wx.NewId()
ID_FILE_IMPORT_FILE_LIST = wx.NewId()
ID_FILE_ANALYZE_IMAGES = wx.NewId()
ID_FILE_STOP_ANALYSIS = wx.NewId()
ID_FILE_RESTART = wx.NewId()
ID_FILE_PRINT = wx.NewId()
ID_FILE_PLATEVIEWER = wx.NewId()
ID_FILE_RUN_MULTIPLE_PIPELINES = wx.NewId()
ID_FILE_NEW_CP = wx.NewId()

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
        self.__pipeline = cellprofiler.pipeline.Pipeline()
        self.__workspace = cellprofiler.workspace.Workspace(
                self.__pipeline, None, None, None, None, None)
        # background_color = cellprofiler.preferences.get_background_color()
        self.__splitter = wx.SplitterWindow(self, -1, style=wx.SP_BORDER)
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
        self.__right_win.AutoLayout = True

        self.__left_win = wx.Panel(self.__splitter, style=wx.BORDER_NONE)
        # bottom left will be the file browser

        self.__module_list_panel = wx.Panel(self.__left_win)
        self.__module_list_panel.SetToolTipString(
                "The pipeline panel contains the modules in the pipeline. Click on the '+' button below or right-click in the panel to begin adding modules.")
        self.__pipeline_test_panel = wx.Panel(self.__left_win, -1)
        self.__pipeline_test_panel.SetToolTipString(
                "The test mode panel is used for previewing the module settings prior to an analysis run. Click the buttons or use the 'Test' menu item to begin testing your module settings.")
        self.__module_controls_panel = wx.Panel(self.__left_win, -1, style=wx.BORDER_NONE)
        self.__module_controls_panel.SetToolTipString(
                "The module controls add, remove, move and get help for modules. Click on the '+' button to begin adding modules.")
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
        self.__pmi_layout_in_progress = False
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
        self.__path_list_sash.Bind(wx.EVT_SASH_DRAGGED,
                                   self.__on_sash_drag)
        self.__path_list_sash.SetOrientation(wx.LAYOUT_HORIZONTAL)
        self.__path_list_sash.SetAlignment(wx.LAYOUT_TOP)
        self.__path_list_sash.SetDefaultSize((screen_width, screen_height / 4))
        self.__path_list_sash.SetDefaultBorderSize(4)
        self.__path_list_sash.SetSashVisible(wx.SASH_BOTTOM, True)
        self.__path_list_sash.AutoLayout = True
        self.__path_list_sash.Hide()
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.__path_list_sash.Sizer = wx.BoxSizer(wx.VERTICAL)
        self.__path_list_sash.Sizer.Add(sizer, 1, wx.EXPAND)
        # Add spacer so that group box doesn't cover sash's handle
        self.__path_list_sash.Sizer.AddSpacer(6)
        #
        # Path list control
        #
        self.__path_list_ctrl = cellprofiler.gui.pathlist.PathListCtrl(self.__path_list_sash)
        self.__path_list_ctrl.SetBackgroundColour(wx.WHITE)
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

        ######################################################################
        #
        # The imageset panel
        #
        ######################################################################

        self.__imageset_sash = wx.SashLayoutWindow(
                self.__path_module_imageset_panel, style=wx.NO_BORDER)
        self.__imageset_sash.SetOrientation(wx.LAYOUT_HORIZONTAL)
        self.__imageset_sash.SetAlignment(wx.LAYOUT_BOTTOM)
        self.__imageset_sash.SetDefaultSize((screen_width, screen_height / 4))
        self.__imageset_sash.SetDefaultBorderSize(4)
        self.__imageset_sash.SetExtraBorderSize(2)
        self.__imageset_sash.SetSashVisible(wx.SASH_TOP, True)
        self.__imageset_sash.Bind(wx.EVT_SASH_DRAGGED,
                                  self.__on_sash_drag)
        self.__imageset_sash.Hide()
        self.__imageset_panel = wx.Panel(self.__imageset_sash)
        self.__imageset_panel.Sizer = wx.BoxSizer()
        self.__imageset_panel.SetAutoLayout(True)
        self.__imageset_ctrl = cellprofiler.gui.imagesetctrl.ImageSetCtrl(
                self.__workspace, self.__imageset_panel, read_only=True)
        self.__imageset_panel.Sizer.Add(self.__imageset_ctrl, 1, wx.EXPAND)
        self.__grid_ctrl = cellprofiler.gui.moduleview.ModuleView.CornerButtonGrid(
                self.__imageset_panel)
        self.__imageset_panel.Sizer.Add(self.__grid_ctrl, 1, wx.EXPAND)

        self.__right_win.Sizer.AddSpacer(4)
        #
        # Preferences panel
        #
        self.__preferences_panel = wx.Panel(self.__right_win, -1)
        self.__right_win.Sizer.Add(self.__preferences_panel, 1, wx.EXPAND)
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
        if cellprofiler.preferences.get_startup_blurb():
            self.show_welcome_screen(True)
        self.show_module_ui(True)

    def start(self, workspace_path, pipeline_path):
        """Handle resource loading after the GUI has been constructed

        workspace_path - one of the following: a pathname to the workspace
                         to load, False to ask the user for a new workspace
                         or None to leave the decision to the user's
                         preference.

        pipeline_path - the pipeline to load after the workspace has been
                        loaded or None for the workspace's pipeline.
        """
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
        """Show or hide the path list control

        show - true to show, false to hide
        """
        if bool(show) == bool(self.__path_list_sash.IsShown()):
            return
        self.__path_list_sash.Show(show)
        self.layout_pmi_panel()
        self.__path_list_sash.Layout()

    def show_imageset_sash(self, show):
        """Show or hide the imageset control

        show - true to show, false to hide
        """
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

    def show_grid_ctrl(self, table=None):
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
        """Show or hide the module and notes panel"""
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
        """Show or hide the welcome screen

        show - If True, show the welcome screen and hide the preferences
               and module UI, otherwise hide the welcome screen.
        """
        self.startup_blurb_frame.Show(show)
        if show:
            self.startup_blurb_frame.Raise()

    def show_preferences(self, show):
        """Show or hide the preferences panel

        show - if True, show the preferences panel and hide the welcome
               and module UI. If false, just hide the preferences.
        """
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
        if not self.__pmi_layout_in_progress:
            self.layout_pmi_panel()
        if self.__path_list_sash.IsShown():
            self.__path_list_sash.Layout()
        if self.__imageset_sash.IsShown():
            self.__imageset_sash.Layout()

    def layout_pmi_panel(self):
        """Run the sash layout algorithm on the path/module/imageset panel"""
        self.__pmi_layout_in_progress = True
        try:
            wx.LayoutAlgorithm().LayoutWindow(self.__path_module_imageset_panel,
                                              self.__module_panel)
            self.__right_win.Layout()
        finally:
            self.__pmi_layout_in_progress = False

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
            cellprofiler.gui.moduleview.stop_validation_queue_thread()
        except:
            logger.warn("Failed to stop pipeline validation thread", exc_info=True)
        wx.GetApp().ExitMainLoop()

    def __set_properties(self):
        self.SetTitle("CellProfiler %s" % cellprofiler.utilities.version.title_string)
        self.SetSize((1024, 600))

    def enable_edit_commands(self, ids):
        """Enable the edit commands that are supported by the focused window

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
        """
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
                help='Open a project from a .%s project file' % cellprofiler.preferences.EXT_PROJECT)
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
                cellprofiler.preferences.EXT_PIPELINE)
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
                "Save the project's pipeline to a .%s file" % cellprofiler.preferences.EXT_PIPELINE)
        submenu.Append(
                ID_FILE_EXPORT_IMAGE_SETS,
                "Image Set Listing...",
                "Export the project's image sets as a CSV file suitable for LoadData")
        submenu.Append(
                ID_FILE_EXPORT_PIPELINE_NOTES,
                "Pipeline notes...",
                "Save a text file outlining the pipeline's modules and module notes")
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
        self.__menu_file.Append(ID_FILE_ANALYZE_IMAGES, 'Analyze Images\tctrl+N',
                                'Run the pipeline on the images in the image directory')
        self.__menu_file.Append(ID_FILE_STOP_ANALYSIS, 'Stop Analysis', 'Stop running the pipeline')
        self.__menu_file.Append(ID_FILE_RUN_MULTIPLE_PIPELINES, 'Run Multiple Pipelines')
        if os.name == 'posix':
            self.__menu_file.Append(ID_FILE_NEW_CP, 'Open a New CP Window')
        self.__menu_file.Append(ID_FILE_RESTART, 'Resume Pipeline', 'Resume a pipeline from a saved measurements file.')
        self.__menu_file.AppendSeparator()
        self.__menu_file.Append(ID_OPTIONS_PREFERENCES, "&Preferences...", "Set global application preferences")

        self.recent_files = wx.Menu()
        self.recent_pipeline_files = wx.Menu()
        self.__menu_file.Append(ID_FILE_EXIT, 'E&xit\tctrl+Q', 'Quit the application')

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
        self.__menu_debug.Append(ID_DEBUG_TOGGLE, '&Start Test Mode\tF5', 'Start the pipeline debugger')
        self.__menu_debug.Append(ID_DEBUG_STEP, 'Ste&p to Next Module\tF6', 'Execute the currently selected module')
        self.__menu_debug.Append(ID_DEBUG_NEXT_IMAGE_SET, '&Next Image Set\tF7', 'Advance to the next image set')
        self.__menu_debug.Append(ID_DEBUG_NEXT_GROUP, 'Next Image &Group\tF8',
                                 'Advance to the next group in the image set list')
        self.__menu_debug.Append(ID_DEBUG_CHOOSE_RANDOM_IMAGE_SET, 'Random Image Set', 'Advance to a random image set')
        self.__menu_debug.Append(ID_DEBUG_CHOOSE_GROUP, 'Choose Image Group',
                                 'Choose which image set group to process in test-mode')
        self.__menu_debug.Append(ID_DEBUG_CHOOSE_IMAGE_SET, 'Choose Image Set',
                                 'Choose any of the available image sets')
        self.__menu_debug.Append(ID_DEBUG_VIEW_WORKSPACE, "View Workspace", "Show the workspace viewer")
        if not hasattr(sys, 'frozen') or os.getenv('CELLPROFILER_DEBUG'):
            self.__menu_debug.Append(ID_DEBUG_RELOAD, "Reload Modules' Source")
            self.__menu_debug.Append(ID_DEBUG_PDB, "Break Into Debugger")
            #
            # Lee wants the wx debugger
            #
            if os.environ.get("USERNAME", "").lower() == "leek":
                self.__menu_debug.Append(ID_FILE_WIDGET_INSPECTOR, "Widget inspector")
        self.__menu_debug.Enable(ID_DEBUG_STEP, False)
        self.__menu_debug.Enable(ID_DEBUG_NEXT_IMAGE_SET, False)
        self.__menu_debug.Enable(ID_DEBUG_NEXT_GROUP, False)
        self.__menu_debug.Enable(ID_DEBUG_CHOOSE_GROUP, False)
        self.__menu_debug.Enable(ID_DEBUG_CHOOSE_IMAGE_SET, False)
        self.__menu_debug.Enable(ID_DEBUG_CHOOSE_RANDOM_IMAGE_SET, False)
        self.__menu_debug.Enable(ID_DEBUG_VIEW_WORKSPACE, False)

        self.__menu_window = wx.Menu()
        self.__menu_window.Append(ID_WINDOW_CLOSE_ALL, "Close &All Open Windows\tctrl+L",
                                  "Close all open module display windows")
        self.__menu_window.Append(ID_WINDOW_SHOW_ALL_WINDOWS, "Show All Windows On Run",
                                  "Show all module display windows for all modules during analysis")
        self.__menu_window.Append(ID_WINDOW_HIDE_ALL_WINDOWS, "Hide All Windows On Run",
                                  "Hide all module display windows for all modules during analysis")
        self.__menu_window.AppendSeparator()

        self.__menu_help = wx.Menu()
        # We must add a non-submenu menu item before
        # make_help_menu adds submenus, otherwise the submenus
        # will disappear on the Mac.
        self.__menu_help.Append(ID_HELP_WELCOME, "Show Welcome Screen", "Display the welcome screen shown at startup")
        self.__menu_help.Append(ID_HELP_RELEASE_NOTES, "Release Notes", "Show the release notes in a browser")
        self.__menu_help.Append(ID_HELP_ONLINE_MANUAL, "Online Manual", "Launch the HTML help in a browser")
        self.__menu_help.AppendSeparator()
        cellprofiler.gui.help.make_help_menu(cellprofiler.gui.help.MAIN_HELP, self, self.__menu_help)
        self.__menu_help.AppendSeparator()
        self.__menu_help.AppendSubMenu(self.data_tools_help(), 'Data Tool Help',
                                       'Display documentation for available data tools')
        self.__menu_help.Append(ID_HELP_MODULE, 'Module Help', 'Display Documentation for the Current Module')
        self.__menu_help.Append(ID_HELP_SEARCH, "Search Help...",
                                "Search for help pages that match a search term.")
        self.__menu_help.AppendSeparator()
        self.__menu_help.Append(ID_HELP_DEVELOPERS_GUIDE, "Developer's Guide",
                                "Launch the developer's guide webpage")
        self.__menu_help.Append(ID_HELP_SOURCE_CODE, "Source Code",
                                "Visit CellProfiler's Github repository")
        self.__menu_help.Append(wx.ID_ABOUT, "&About CellProfiler", "About CellProfiler")

        self.__menu_bar = wx.MenuBar()
        self.__menu_bar.Append(self.__menu_file, '&File')
        self.__menu_bar.Append(self.menu_edit, '&Edit')
        self.__menu_bar.Append(self.__menu_debug, '&Test')
        if cellprofiler.preferences.get_show_sampling():
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
        wx.EVT_MENU(self, ID_FILE_EXIT, lambda event: self.Close())
        wx.EVT_MENU(self, ID_FILE_WIDGET_INSPECTOR, self.__on_widget_inspector)
        wx.EVT_MENU(self, ID_FILE_NEW_CP, self.__on_new_cp)

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
        wx.EVT_MENU(self, ID_HELP_MODULE, self.__on_help_module)
        wx.EVT_BUTTON(self, ID_HELP_MODULE, self.__on_help_module)
        wx.EVT_MENU(self, ID_HELP_RELEASE_NOTES, self.__on_help_release_notes)
        wx.EVT_MENU(self, ID_HELP_ONLINE_MANUAL, self.__on_help_online_manual)
        wx.EVT_MENU(self, ID_HELP_DEVELOPERS_GUIDE, self.__on_help_developers_guide)
        wx.EVT_MENU(self, ID_HELP_SOURCE_CODE, self.__on_help_source_code)
        wx.EVT_MENU(self, ID_HELP_SEARCH, self.__on_search_help)
        wx.EVT_MENU(self, ID_HELP_ABOUT, self.about)
        wx.EVT_MENU(self, ID_OPTIONS_PREFERENCES, self.__on_preferences)
        wx.EVT_MENU(self, ID_WINDOW_CLOSE_ALL, self.__on_close_all)
        wx.EVT_MENU(self, ID_DEBUG_PDB, self.__debug_pdb)
        accelerator_table = wx.AcceleratorTable(
                [(wx.ACCEL_CMD, ord('N'), ID_FILE_ANALYZE_IMAGES),
                 (wx.ACCEL_CMD, ord('O'), ID_FILE_LOAD),
                 (wx.ACCEL_CMD, ord('P'), ID_FILE_SAVE_PIPELINE),
                 (wx.ACCEL_CMD | wx.ACCEL_SHIFT, ord('S'), ID_FILE_SAVE),
                 (wx.ACCEL_CMD, ord('L'), ID_WINDOW_CLOSE_ALL),
                 (wx.ACCEL_CMD, ord('Q'), ID_FILE_EXIT),
                 (wx.ACCEL_CMD, ord('W'), ID_FILE_EXIT),
                 (wx.ACCEL_CMD, ord('A'), wx.ID_SELECTALL),
                 (wx.ACCEL_CMD, ord('C'), wx.ID_COPY),
                 (wx.ACCEL_CMD, ord('V'), wx.ID_PASTE),
                 (wx.ACCEL_NORMAL, wx.WXK_F5, ID_DEBUG_TOGGLE),
                 (wx.ACCEL_NORMAL, wx.WXK_F6, ID_DEBUG_STEP),
                 (wx.ACCEL_NORMAL, wx.WXK_F7, ID_DEBUG_NEXT_IMAGE_SET),
                 (wx.ACCEL_NORMAL, wx.WXK_F8, ID_DEBUG_NEXT_GROUP),
                 (wx.ACCEL_CMD, ord('Z'), ID_EDIT_UNDO)])
        self.SetAcceleratorTable(accelerator_table)
        self.enable_launch_commands()

    def data_tools_help(self):
        """Create a help menu for the data tools"""
        if not hasattr(self, "__data_tools_help_menu"):
            self.__menu_data_tools_help_menu = wx.Menu()

            def on_plate_viewer_help(event):
                import htmldialog
                dlg = htmldialog.HTMLDialog(
                        self, "Help on plate viewer", cellprofiler.gui.help.PLATEVIEWER_HELP)
                dlg.Show()

            new_id = wx.NewId()
            self.__menu_data_tools_help_menu.Append(new_id, "Plate viewer")
            wx.EVT_MENU(self, new_id, on_plate_viewer_help)

            for data_tool_name in cellprofiler.modules.get_data_tool_names():
                new_id = wx.NewId()
                self.__menu_data_tools_help_menu.Append(new_id, data_tool_name)

                def on_data_tool_help(event, data_tool_name=data_tool_name):
                    self.__on_data_tool_help(event, data_tool_name)

                wx.EVT_MENU(self, new_id, on_data_tool_help)
        return self.__menu_data_tools_help_menu

    def data_tools_menu(self):
        """Create a menu of data tools"""

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
            wx.EVT_MENU(self, new_id, on_data_tool_overview)

            self.__data_tools_menu.AppendSeparator()

            self.__data_tools_menu.Append(
                    ID_FILE_PLATEVIEWER, 'Plate Viewer',
                    'Open the plate viewer to inspect the images in the current workspace')

            self.__data_tools_menu.AppendSeparator()

            for data_tool_name in cellprofiler.modules.get_data_tool_names():
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
    @staticmethod
    def on_cut(event):
        """Handle ID_CUT"""
        focus = wx.Window.FindFocus()
        if (focus is not None and hasattr(focus, "Cut")
            and hasattr(focus, 'CanCut') and focus.CanCut()):
            focus.Cut()

    @staticmethod
    def on_update_cut_ui(event):
        focus = wx.Window.FindFocus()
        event.Enable(bool(
                focus and hasattr(focus, 'CanCut') and focus.CanCut()))

    @staticmethod
    def on_copy(event):
        """Handle ID_COPY"""
        focus = wx.Window.FindFocus()
        if focus is not None and hasattr(focus, "Copy") and \
                hasattr(focus, 'CanCopy') and focus.CanCopy():
            focus.Copy()

    @staticmethod
    def on_update_copy_ui(event):
        focus = wx.Window.FindFocus()
        event.Enable(bool(
                focus and hasattr(focus, 'CanCopy') and focus.CanCopy()))

    @staticmethod
    def on_paste(event):
        """Handle ID_PASTE"""
        focus = wx.Window.FindFocus()
        if focus is not None and hasattr(focus, "Paste") and \
                hasattr(focus, "CanPaste") and focus.CanPaste():
            focus.Paste()

    @staticmethod
    def on_update_paste_ui(event):
        focus = wx.Window.FindFocus()
        event.Enable(bool(
                focus and hasattr(focus, 'CanPaste') and focus.CanPaste()))

    @staticmethod
    def on_select_all(event):
        focus = wx.Window.FindFocus()
        if focus and hasattr(focus, "SelectAll"):
            focus.SelectAll()

    @staticmethod
    def on_update_select_all_ui(event):
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
        """Enable commands to start analysis or test mode"""
        startstop = self.__menu_debug.FindItemById(ID_DEBUG_TOGGLE)
        startstop.Text = '&Start Test Mode\tF5'
        startstop.Help = 'Start testing your pipeline'
        for cmd in self.debug_commands:
            self.__menu_debug.Enable(cmd, False)
        self.__menu_file.Enable(ID_FILE_ANALYZE_IMAGES, True)
        self.__menu_file.Enable(ID_FILE_RESTART, True)
        self.__menu_debug.Enable(ID_DEBUG_TOGGLE, True)
        self.__menu_file.Enable(ID_FILE_RUN_MULTIPLE_PIPELINES, True)

        self.__menu_file.Enable(ID_FILE_STOP_ANALYSIS, False)

    def enable_analysis_commands(self):
        """Enable commands to pause or stop analysis"""
        self.__menu_file.Enable(ID_FILE_ANALYZE_IMAGES, False)
        self.__menu_file.Enable(ID_FILE_RESTART, False)
        self.__menu_debug.Enable(ID_DEBUG_TOGGLE, False)
        self.__menu_file.Enable(ID_FILE_RUN_MULTIPLE_PIPELINES, False)

        self.__menu_file.Enable(ID_FILE_STOP_ANALYSIS, True)

    @staticmethod
    def __on_widget_inspector(evt):
        try:
            import wx.lib.inspection
            wx.lib.inspection.InspectionTool().Show()
        except:
            wx.MessageBox("Inspection tool is not available on this platform")

    @staticmethod
    def __on_preferences(event):
        dlg = cellprofiler.gui.preferencesdlg.PreferencesDlg()
        dlg.Show()

    def __on_close_all(self, event):
        cellprofiler.gui.figure.close_all(self)

    @staticmethod
    def __on_new_cp(event):
        import os
        if not hasattr(sys, 'frozen'):
            os.system('open CellProfiler_python.command')
        else:
            os.system('open -na CellProfiler.app')

    @staticmethod
    def __on_help_release_notes(event):
        import webbrowser
        webbrowser.open("http://github.com/CellProfiler/CellProfiler/wiki/CellProfiler-release-notes")

    @staticmethod
    def __on_help_online_manual(event):
        import webbrowser
        webbrowser.open("http://d1zymp9ayga15t.cloudfront.net/CPmanual/index.html")

    @staticmethod
    def __on_help_developers_guide(event):
        import webbrowser
        webbrowser.open("http://www.cellprofiler.org/wiki/index.php/Main_Page")

    @staticmethod
    def __on_help_source_code(event):
        import webbrowser
        webbrowser.open("https://github.com/CellProfiler/CellProfiler")

    def __on_help_path_list(self, event):
        import htmldialog
        dlg = htmldialog.HTMLDialog(self, "Help on file list", cellprofiler.gui.help.HELP_ON_FILE_LIST)
        dlg.Show()

    @staticmethod
    def about(event):
        info = cellprofiler.gui.dialog.AboutDialogInfo()

        wx.AboutBox(info)

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
            wx.MessageBox(cellprofiler.gui.help.HELP_ON_MODULE_BUT_NONE_SELECTED,
                          "No module selected",
                          style=wx.OK | wx.ICON_INFORMATION)

    @staticmethod
    def __debug_pdb(event):
        pdb.set_trace()

    def do_help_modules(self, modules):
        for module in modules:
            ## An attempt to place images inline with the help. However, the
            ## images will not scale properly in size (yet)
            # result = module.get_help()
            # root = os.path.split(__file__)[0]
            # if len(root) == 0:
            # root = os.curdir
            # root = os.path.split(os.path.abspath(root))[0] # Back up one level
            # absolute_image_path = os.path.join(root, 'icons','%s.png'%(module.module_name,))
            ## Check if the file that goes with this module exists on this computer
            # if os.path.exists(absolute_image_path) and os.path.isfile(absolute_image_path):
            ## If so, strip out end html tags so I can add more stuff
            # result = result.replace('</body>','').replace('</html>','')
            ## Include images specific to the module
            # result += '\n\n<div><p><img src="%s", width="50%%"></p></div>\n'%absolute_image_path
            ## Now end the help text
            # result += '</body></html>'
            # self.do_help_module(module.module_name, result)
            self.do_help_module(module.module_name, module.get_help())

    def do_help_module(self, module_name, help_text):
        helpframe = wx.Frame(self, -1, 'Help for module, "%s"' %
                             module_name, size=(640, 480))
        helpframe.MenuBar = wx.MenuBar()
        ####################################################
        #
        # Add the HTML window
        #
        ####################################################

        sizer = wx.BoxSizer()
        helpframe.SetSizer(sizer)
        window = cellprofiler.gui.html.htmlwindow.HtmlClickableWindow(helpframe)
        sizer.Add(window, 1, wx.EXPAND)
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
        helpframe.Bind(wx.EVT_MENU, on_exit, id=ID_FILE_EXIT)

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
                wx.MessageBox("Failed to copy to the clipboard", "Error",
                              wx.OK | wx.ICON_ERROR)

        helpframe.MenuBar.Append(menu, '&Edit')
        helpframe.Bind(wx.EVT_MENU, on_copy, id=ID_EDIT_COPY)
        helpframe.Bind(wx.EVT_MENU, on_edit_select_all, id=ID_EDIT_SELECT_ALL)
        helpframe.Bind(wx.EVT_IDLE, on_idle)

        ####################################################
        #
        # Build an accelerator table for some of the commands
        #
        ####################################################
        accelerator_table = wx.AcceleratorTable(
                [(wx.ACCEL_CMD, ord('Q'), ID_FILE_EXIT),
                 (wx.ACCEL_CMD, ord('P'), ID_FILE_PRINT),
                 (wx.ACCEL_CMD, ord('C'), ID_EDIT_COPY)])
        helpframe.SetAcceleratorTable(accelerator_table)
        helpframe.SetIcon(cellprofiler.gui.get_cp_icon())
        helpframe.Layout()
        helpframe.Show()

    @staticmethod
    def print_help(event, module_name, help_text):
        """Print the help text for a module"""
        printer = wx.html.HtmlEasyPrinting("Printing %s" % module_name,
                                           event.GetEventObject())
        printer.GetPrintData().SetPaperId(wx.PAPER_LETTER)
        printer.PrintText(help_text)

    @staticmethod
    def save_help(event, module_name, help_text):
        """Save the help text for a module"""
        save_dlg = wx.FileDialog(event.GetEventObject(),
                                 message="Save help for %s to file" % module_name,
                                 defaultFile="%s.html" % module_name,
                                 wildcard="*.html",
                                 style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        result = save_dlg.ShowModal()
        if result == wx.ID_OK:
            pathname = save_dlg.GetPath()
            fd = open(pathname, "wt")
            fd.write(help_text)
            fd.close()

    def on_open_image(self, event):
        dlg = wx.FileDialog(self,
                            message="Open an image file",
                            wildcard="Image file (*.tif,*.tiff,*.jpg,*.jpeg,*.png,*.gif,*.bmp)|*.tif;*.tiff;*.jpg;*.jpeg;*.png;*.gif;*.bmp|*.* (all files)|*.*",
                            style=wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            from cellprofiler.modules.loadimages import LoadImagesImageProvider
            from cellprofiler.gui.figure import Figure
            lip = LoadImagesImageProvider("dummy", "", dlg.Path)
            image = lip.provide_image(None).pixel_data
            frame = Figure(self, title=dlg.Path, subplots=(1, 1))
            if image.ndim == 3:
                frame.subplot_imshow_color(0, 0, image, title=dlg.Path)
            else:
                frame.subplot_imshow_grayscale(0, 0, image, title=dlg.Path)
            frame.panel.draw()

    def __attach_views(self):
        self.__pipeline_list_view = cellprofiler.gui.pipelinelistview.PipelineListView(self.__module_list_panel, self)
        self.__pipeline_controller = cellprofiler.gui.pipelinecontroller.PipelineController(self.__workspace, self)
        self.__pipeline_list_view.attach_to_pipeline(self.__pipeline, self.__pipeline_controller)
        self.__pipeline_controller.attach_to_test_controls_panel(self.__pipeline_test_panel)
        self.__pipeline_controller.attach_to_module_controls_panel(self.__module_controls_panel)
        self.__pipeline_controller.attach_to_path_list_ctrl(
                self.__path_list_ctrl,
                self.__path_list_filter_checkbox)
        self.__module_view = cellprofiler.gui.moduleview.ModuleView(
                self.__module_panel,
                self.__workspace,
                frame = self,
                notes_panel = self.__notes_panel)
        self.__pipeline_controller.attach_to_module_view(self.__module_view)
        self.__pipeline_list_view.attach_to_module_view(self.__module_view)
        self.__preferences_view = cellprofiler.gui.preferencesview.PreferencesView(
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

        top_left_sizer = wx.BoxSizer(wx.VERTICAL)
        top_left_sizer.Add(self.__module_list_panel, 1, wx.EXPAND | wx.ALL, 1)
        top_left_sizer.Add(self.__module_controls_panel, 0, wx.EXPAND | wx.ALL, 2)
        top_left_sizer.Add(self.__pipeline_test_panel, 0, wx.EXPAND | wx.ALL, 2)
        top_left_win.SetSizer(top_left_sizer)

        border = wx.BoxSizer()
        border.Add(splitter, 1, wx.EXPAND | wx.ALL, 1)
        self.SetSizer(border)
        self.Layout()
        right_win.Layout()
        top_left_win.Layout()

    def __set_icon(self):
        self.SetIcon(cellprofiler.gui.get_cp_icon())

    def __make_search_frame(self):
        """Make and hide the "search the help" frame"""
        background_color = cellprofiler.preferences.get_background_color()
        size = (wx.SystemSettings.GetMetric(wx.SYS_SCREEN_X) / 2,
                wx.SystemSettings.GetMetric(wx.SYS_SCREEN_Y) / 2)
        self.search_frame = wx.Frame(
                self, title = "Search CellProfiler help",
                size = size,
                style = wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL)
        self.search_frame.AutoLayout = True
        self.search_frame.SetIcon(cellprofiler.gui.get_cp_icon())
        self.search_frame.Sizer = wx.BoxSizer(wx.VERTICAL)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.search_frame.Sizer.Add(sizer, 0, wx.EXPAND | wx.ALL, 4)
        sizer.Add(wx.StaticText(self.search_frame, label="Search:"), 0,
                  wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        sizer.AddSpacer(2)
        search_text_ctrl = wx.TextCtrl(self.search_frame)
        sizer.Add(search_text_ctrl, 1, wx.EXPAND)
        search_button = wx.Button(self.search_frame, label="Search")
        search_button.SetDefault()
        sizer.AddSpacer(2)
        sizer.Add(search_button, 0, wx.EXPAND)

        html_window = cellprofiler.gui.html.htmlwindow.HtmlClickableWindow(self.search_frame)
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
            """Handle anchor clicks manually

            The HTML window (on Windows at least) jams the anchor to the
            top of the window which obscures it.
            """
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
        self.search_frame.SetIcon(cellprofiler.gui.get_cp_icon())

    def __on_search_help(self, event):
        if self.search_frame is not None:
            self.search_frame.Show()
            self.search_frame.Raise()

    def __make_startup_blurb_frame(self):
        """Make the frame surrounding the startup blurb panel"""
        background_color = cellprofiler.preferences.get_background_color()
        frame = self.startup_blurb_frame = wx.Frame(
                self, title="Welcome to CellProfiler",
                size=(640, 480),
                name=cellprofiler.gui.html.htmlwindow.WELCOME_SCREEN_FRAME)
        # frame.BackgroundColour = background_color
        frame.Sizer = wx.BoxSizer()
        content = cellprofiler.gui.html.htmlwindow.HtmlClickableWindow(frame)
        content.load_startup_blurb()
        frame.Sizer.Add(content, 1, wx.EXPAND)
        frame.SetIcon(cellprofiler.gui.get_cp_icon())

        def on_close(event):
            assert isinstance(event, wx.CloseEvent)
            event.EventObject.Hide()
            event.Veto()

        frame.Bind(wx.EVT_CLOSE, on_close)
        frame.Layout()

    def __on_data_tool(self, event, tool_name):
        module = cellprofiler.modules.instantiate_module(tool_name)
        args, varargs, varkw, vardef = inspect.getargspec(module.run_as_data_tool)
        if len(args) + (0 if varargs is None else len(varargs)) == 1:
            # Data tool doesn't need the data tool frame because it doesn't
            # take the "workspace" argument
            #
            module.run_as_data_tool()
            return
        dlg = wx.FileDialog(
                self, "Choose data output file for %s data tool" %
                      tool_name, wildcard="Measurements file(*.mat,*.h5)|*.mat;*.h5",
                style=(wx.FD_OPEN | wx.FILE_MUST_EXIST))
        if dlg.ShowModal() == wx.ID_OK:
            cellprofiler.gui.datatoolframe.DataToolFrame(self,
                                                         module_name=tool_name,
                                                         measurements_file_name=dlg.Path)

    def __on_data_tool_help(self, event, tool_name):
        module = cellprofiler.modules.instantiate_module(tool_name)
        self.do_help_module(tool_name, module.get_help())

    def display_error(self, message, error):
        """Displays an exception in a standardized way

        """
        for listener in self.__error_listeners:
            listener(message, error)
        tb = sys.exc_info()[2]
        traceback.print_tb(tb)
        text = '\n'.join(traceback.format_list(traceback.extract_tb(tb)))
        text = error.message + '\n' + text
        cellprofiler.gui.errordialog.display_error_message(self, text, "Caught exception during operation")

    def add_error_listener(self, listener):
        """Add a listener for display errors"""
        self.__error_listeners.append(listener)

    def remove_error_listener(self, listener):
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
