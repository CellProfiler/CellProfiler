# -*- Encoding: utf-8 -*-
""" CellProfiler.CellProfilerGUI.CPFrame - Cell Profiler's main window
"""

import codecs
import logging
import os
import pdb
import sys

import wx
import wx.adv
import wx.html
import wx.lib.inspection
import wx.lib.scrolledpanel

from cellprofiler_core.preferences import EXT_PIPELINE
from cellprofiler_core.preferences import EXT_PROJECT
from cellprofiler_core.preferences import get_show_sampling
from cellprofiler_core.preferences import get_startup_blurb
from cellprofiler_core.preferences import get_widget_inspector
from cellprofiler_core.utilities.core.modules import instantiate_module

import cellprofiler.gui
import cellprofiler.gui.utilities.icon
from cellprofiler import __version__ as cellprofiler_version

from .plugins_menu import PLUGIN_MENU_ENTRIES
from ._workspace_model import Workspace
from .utilities.figure import close_all
from .help.content import read_content
from .help.menu import Menu
from .html.htmlwindow import HtmlClickableWindow
from .html.utils import rst_to_html_fragment
from .imagesetctrl import ImageSetCtrl
from .module_view import ModuleView
from .pathlist import PathListCtrl
from .pipeline import Pipeline
from .pipelinecontroller import PipelineController
from .pipelinelistview import PipelineListView
from .preferences_dialog._preferences_dialog import PreferencesDialog
from .readers_dialog._readers_dialog import ReadersDialog
from .preferences_view import PreferencesView
from .utilities.module_view import stop_validation_queue_thread

LOGGER = logging.getLogger(__name__)

HELP_ON_FILE_LIST = """\
The *File List* panel displays the image files that are managed by the
**Images**, **Metadata**, **NamesAndTypes** and **Groups** modules.
You can drop files and directories into this window or use the
*Browse…* button to add files to the list. The context menu for the
window lets you display or remove files and lets you remove folders.

The buttons and checkbox along the bottom have the following
functions:

-  *Browse…*: Browse for files and folders to add.
-  *Clear*: Clear all entries from the File list
-  *Show files excluded by filters*: *(Only shown if filtered based on
   rules is selected)* Check this to see all files in the list. Uncheck
   it to see only the files that pass the rules criteria in the
   **Images** module.
-  *Expand tree*: Expand all of the folders in the tree
-  *Collapse tree*: Collapse the folders in the tree
"""

HELP_ON_MODULE_BUT_NONE_SELECTED = """\
The help button can be used to obtain help for the currently selected
module in the pipeline panel on the left side of the CellProfiler
interface.

You do not have any modules in the pipeline, yet. Add a
module to the pipeline using the “+” button or by using File > Load
Pipeline.\
"""

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
ID_FILE_EXPORT_PIPELINE_CITATIONS = wx.NewId()
ID_FILE_IMPORT_FILE_LIST = wx.NewId()
ID_FILE_ANALYZE_IMAGES = wx.NewId()
ID_FILE_STOP_ANALYSIS = wx.NewId()
ID_FILE_PRINT = wx.NewId()
ID_FILE_PLATEVIEWER = wx.NewId()
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
ID_EDIT_BROWSE_FOR_FOLDER = wx.NewId()
ID_EDIT_CLEAR_FILE_LIST = wx.NewId()
ID_EDIT_REMOVE_FROM_FILE_LIST = wx.NewId()
ID_EDIT_SHOW_FILE_LIST_IMAGE = wx.NewId()
ID_EDIT_ENABLE_MODULE = wx.NewId()
ID_EDIT_DISPLAY_MODULE = wx.NewId()
ID_EDIT_GO_TO_MODULE = wx.NewId()
ID_FIND_USAGES = wx.NewId()

ID_OPTIONS_PREFERENCES = wx.ID_PREFERENCES
ID_OPTIONS_READERS = wx.NewId()
ID_CHECK_NEW_VERSION = wx.NewId()

ID_DEBUG_TOGGLE = wx.NewId()
ID_DEBUG_STEP = wx.NewId()
ID_DEBUG_NEXT_IMAGE_SET = wx.NewId()
ID_DEBUG_NEXT_GROUP = wx.NewId()
ID_DEBUG_CHOOSE_GROUP = wx.NewId()
ID_DEBUG_CHOOSE_IMAGE_SET = wx.NewId()
ID_DEBUG_CHOOSE_RANDOM_IMAGE_SET = wx.NewId()
ID_DEBUG_CHOOSE_RANDOM_IMAGE_GROUP = wx.NewId()
ID_DEBUG_RELOAD = wx.NewId()
ID_DEBUG_PDB = wx.NewId()
ID_DEBUG_RUN_FROM_THIS_MODULE = wx.NewId()
ID_DEBUG_STEP_FROM_THIS_MODULE = wx.NewId()
ID_DEBUG_HELP = wx.NewId()
ID_DEBUG_VIEW_WORKSPACE = wx.NewId()

# ~*~
ID_SAMPLE_INIT = wx.NewId()
# ~^~

ID_WINDOW = wx.NewId()
ID_WINDOW_CLOSE_ALL = wx.NewId()
ID_WINDOW_SHOW_ALL_WINDOWS = wx.NewId()
ID_WINDOW_HIDE_ALL_WINDOWS = wx.NewId()
ID_WINDOW_ALL = (
    ID_WINDOW_CLOSE_ALL,
    ID_WINDOW_SHOW_ALL_WINDOWS,
    ID_WINDOW_HIDE_ALL_WINDOWS,
)

WINDOW_IDS = []

ID_HELP_MODULE = wx.NewId()
ID_HELP_SOURCE_CODE = wx.NewId()


class CPFrame(wx.Frame):
    def __init__(self, *args, **kwds):
        """Initialize the frame and its layout

        """
        kwds["style"] = wx.DEFAULT_FRAME_STYLE

        self.__pipeline = Pipeline()
        self.__workspace = Workspace(self.__pipeline, None, None, None, None, None)

        super(CPFrame, self).__init__(*args, **kwds)

        # background_color = cellprofiler_core.preferences.get_background_color()
        self.__splitter = wx.SplitterWindow(self, -1, style=wx.SP_BORDER)
        #
        # Screen size metrics might be used below
        #
        screen_width = wx.SystemSettings.GetMetric(wx.SYS_SCREEN_X)
        screen_height = wx.SystemSettings.GetMetric(wx.SYS_SCREEN_Y)

        # Crappy splitters leave crud on the screen because they want custom
        # background painting but fail to do it. Here, we have a fight with
        # them and beat them.
        self.__splitter.SetBackgroundStyle(0)

        self.__right_win = wx.Panel(self.__splitter, style=wx.BORDER_NONE, name="right_win")
        self.__right_win.SetAutoLayout(True)

        self.__left_win = wx.Panel(self.__splitter, style=wx.BORDER_NONE, name="left_win")
        # bottom left will be the file browser

        self.__module_list_panel = wx.Panel(self.__left_win)
        self.__module_list_panel.SetToolTip(
            "The pipeline panel contains the modules in the pipeline. Click on the '+' button below or right-click in the panel to begin adding modules."
        )
        self.__pipeline_test_panel = wx.Panel(self.__left_win, -1)
        self.__pipeline_test_panel.SetToolTip(
            "The test mode panel is used for previewing the module settings prior to an analysis run. Click the buttons or use the 'Test' menu item to begin testing your module settings."
        )
        self.__module_controls_panel = wx.Panel(
            self.__left_win, -1, style=wx.BORDER_NONE
        )
        self.__module_controls_panel.SetToolTip(
            "The module controls add, remove, move and get help for modules. Click on the '+' button to begin adding modules."
        )
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
        self.__right_win.SetSizer(wx.BoxSizer(wx.VERTICAL))
        self.__notes_panel = wx.Panel(self.__right_win)
        self.__right_win.GetSizer().Add(self.__notes_panel, 0, wx.EXPAND | wx.ALL)
        self.__right_win.GetSizer().AddSpacer(4)
        self.__path_module_imageset_panel = wx.Panel(self.__right_win, name="path_module_imageset_panel")
        self.__right_win.GetSizer().Add(
            self.__path_module_imageset_panel, 1, wx.EXPAND | wx.ALL
        )
        self.__pmi_layout_in_progress = False
        self.__path_module_imageset_panel.Bind(
            wx.EVT_SIZE, self.__on_path_module_imageset_panel_size
        )

        ########################################################################
        #
        # The path list control that holds all of the files being dealt with
        # by the pipeline
        #
        ########################################################################

        #
        # Path list sash controls path list sizing
        #
        self.__path_list_sash = wx.adv.SashLayoutWindow(
            self.__path_module_imageset_panel, style=wx.NO_BORDER, name="path_list_sash"
        )
        self.__path_list_sash.Bind(wx.adv.EVT_SASH_DRAGGED, self.__on_sash_drag)
        self.__path_list_sash.SetOrientation(wx.adv.LAYOUT_HORIZONTAL)
        self.__path_list_sash.SetAlignment(wx.adv.LAYOUT_TOP)
        self.__path_list_sash.SetDefaultSize((screen_width, screen_height / 4))
        self.__path_list_sash.SetDefaultBorderSize(4)
        self.__path_list_sash.SetSashVisible(wx.adv.SASH_BOTTOM, True)
        self.__path_list_sash.SetAutoLayout(True)
        self.__path_list_sash.Hide()
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.__path_list_sash.SetSizer(wx.BoxSizer(wx.VERTICAL))
        self.__path_list_sash.GetSizer().Add(sizer, 1, wx.EXPAND)
        # Add spacer so that group box doesn't cover sash's handle
        self.__path_list_sash.GetSizer().AddSpacer(6)
        #
        # Path list control
        #
        self.__path_list_ctrl = PathListCtrl(self.__path_list_sash, style=wx.TR_HIDE_ROOT | wx.TR_HAS_BUTTONS |
                                             wx.TR_MULTIPLE | wx.TR_FULL_ROW_HIGHLIGHT | wx.TR_LINES_AT_ROOT)
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
            self.__path_list_sash, label="Show files excluded by filters"
        )
        hsizer.Add(self.__path_list_filter_checkbox, 0, wx.EXPAND)

        def show_disabled(event):
            self.__path_list_ctrl.set_show_disabled(
                self.__path_list_filter_checkbox.GetValue()
            )

        self.__path_list_filter_checkbox.Bind(wx.EVT_CHECKBOX, show_disabled)
        hsizer.AddStretchSpacer()
        #
        # Help
        #
        hsizer.AddSpacer(5)
        self.__path_list_help_button = wx.Button(
            self.__path_list_sash, label="?", style=wx.BU_EXACTFIT
        )
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

        self.__imageset_sash = wx.adv.SashLayoutWindow(
            self.__path_module_imageset_panel, style=wx.NO_BORDER, name="imageset_sash"
        )
        self.__imageset_sash.SetOrientation(wx.adv.LAYOUT_HORIZONTAL)
        self.__imageset_sash.SetAlignment(wx.adv.LAYOUT_BOTTOM)
        self.__imageset_sash.SetDefaultSize((screen_width, screen_height / 4))
        self.__imageset_sash.SetDefaultBorderSize(4)
        self.__imageset_sash.SetExtraBorderSize(2)
        self.__imageset_sash.SetSashVisible(wx.adv.SASH_TOP, True)
        self.__imageset_sash.Bind(wx.adv.EVT_SASH_DRAGGED, self.__on_sash_drag)
        self.__imageset_sash.Hide()
        self.__imageset_panel = wx.Panel(self.__imageset_sash, name="imageset_panel")
        self.__imageset_panel.SetSizer(wx.BoxSizer())
        self.__imageset_panel.SetAutoLayout(True)

        self.__imageset_ctrl = ImageSetCtrl(
            self.__workspace, self.__imageset_panel, read_only=True
        )

        self.__imageset_panel.GetSizer().Add(self.__imageset_ctrl, 1, wx.EXPAND)
        self.__grid_ctrl = ModuleView.CornerButtonGrid(self.__imageset_panel)
        self.__imageset_panel.GetSizer().Add(self.__grid_ctrl, 1, wx.EXPAND)
        self.__right_win.GetSizer().AddSpacer(4)

        #
        # Preferences panel
        #
        self.__preferences_panel = wx.Panel(self.__right_win, -1)
        self.__right_win.GetSizer().Add(self.__preferences_panel, 1, wx.EXPAND)
        self.__preferences_panel.SetToolTip(
            "The folder panel sets/creates the input and output folders and output filename. Once your pipeline is ready and your folders set, click 'Analyze Images' to begin the analysis run."
        )

        #
        # Progress and status panels
        #
        self.__progress_panel = wx.Panel(self.__right_win)
        self.__progress_panel.SetAutoLayout(True)
        self.__right_win.GetSizer().Add(self.__progress_panel, 0, wx.EXPAND)
        self.__status_panel = wx.Panel(self.__right_win)
        self.__status_panel.SetAutoLayout(True)
        self.__right_win.GetSizer().Add(self.__status_panel, 0, wx.EXPAND)
        self.__add_menu()
        self.__attach_views()
        self.__set_properties()
        self.__set_icon()
        self.__do_layout()
        self.startup_blurb_frame = None
        self.__error_listeners = []
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        self.SetAutoLayout(True)
        if get_startup_blurb() and sys.platform != "linux":
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
        sizer = self.__imageset_panel.GetSizer()
        assert isinstance(sizer, wx.Sizer)
        if (
            sizer.IsShown(self.__imageset_ctrl) is False
            or self.__imageset_sash.IsShown() is False
        ):
            sizer.Show(self.__imageset_ctrl, True)
            sizer.Show(self.__grid_ctrl, False)
            self.show_imageset_sash(True)
            self.__imageset_panel.Layout()

    def show_grid_ctrl(self, table=None):
        if table is not None:
            self.__grid_ctrl.SetTable(table)
        sizer = self.__imageset_panel.GetSizer()
        if (
            sizer.IsShown(self.__imageset_ctrl)
            or self.__imageset_sash.IsShown() is False
        ):
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
        if (
            show == self.__path_module_imageset_panel.IsShownOnScreen()
            and show == self.__notes_panel.IsShownOnScreen()
        ):
            return
        right_sizer = self.__right_win.GetSizer()
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
                scroll_x=True, scroll_y=True, scrollToTop=False
            )
            self.__imageset_sash.Layout()

    def show_welcome_screen(self, show):
        """Show or hide the welcome screen

        show - If True, show the welcome screen and hide the preferences
               and module UI, otherwise hide the welcome screen.
        """
        if self.startup_blurb_frame is None:
            if not show:
                return
            else:
                from ._welcome_frame import WelcomeFrame
                self.startup_blurb_frame = WelcomeFrame(self)
                # Put the welcome screen over the module settings.
                r = self.__right_win.GetScreenRect()
                self.startup_blurb_frame.SetRect(r)

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
            self.__preferences_panel.GetParent().Layout()

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
            wx.adv.LayoutAlgorithm().LayoutWindow(
                self.__path_module_imageset_panel, self.__module_panel
            )
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
            LOGGER.warning(
                "Failed to flush temporary measurements file during close",
                exc_info=True,
            )
        try:
            from cellprofiler_core.constants.reader import ALL_READERS
            for reader in ALL_READERS.values():
                reader.clear_cached_readers()
        except:
            LOGGER.warning(
                "Failed to clear reader cache during close", exc_info=True,
            )
        try:
            self.__preferences_view.close()
        except:
            LOGGER.warning("Failed during close", exc_info=True)

        try:
            self.pipeline_controller.on_close()
        except:
            LOGGER.warning("Failed to close the pipeline controller", exc_info=True)

        try:
            stop_validation_queue_thread()
        except:
            LOGGER.warning("Failed to stop pipeline validation thread", exc_info=True)
        wx.GetApp().ExitMainLoop()

    def __set_properties(self):
        self.SetTitle("CellProfiler %s" % cellprofiler_version)
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
        d = dict(
            [(x, False) for x in (wx.ID_COPY, wx.ID_CUT, wx.ID_PASTE, wx.ID_SELECTALL)]
        )
        for eyedee in ids:
            d[eyedee] = True
        for k, v in list(d.items()):
            self.menu_edit.Enable(k, v)

    def __add_menu(self):
        """Add the menu to the frame

        """
        self.__menu_file = wx.Menu()
        self.__menu_file.Append(
            wx.ID_NEW, "New Project", helpString="Create an empty project"
        )
        self.__menu_file.Append(
            wx.ID_OPEN,
            "Open Project...\tctrl+O",
            helpString="Open a project from a .{} project file".format(EXT_PROJECT),
        )
        self.recent_workspace_files = wx.Menu()
        self.__menu_file.AppendSubMenu(self.recent_workspace_files, "Open Recent")
        self.__menu_file.Append(
            wx.ID_SAVE,
            "Save Project\tctrl+S",
            helpString="Save the project to the current project file",
        )
        self.__menu_file.Append(
            wx.ID_SAVEAS,
            "Save Project As...",
            helpString="Save the project to a file of your choice",
        )
        self.__menu_file.Append(
            ID_FILE_REVERT_TO_SAVED,
            "Revert to Saved",
            helpString="Reload the project file, discarding changes",
        )
        submenu = wx.Menu()
        submenu.Append(
            ID_FILE_LOAD_PIPELINE,
            "Pipeline from File...",
            "Import a pipeline into the project from a .%s file" % EXT_PIPELINE,
        )
        submenu.Append(
            ID_FILE_URL_LOAD_PIPELINE,
            "Pipeline from URL...",
            "Load a pipeline from the web",
        )
        submenu.Append(
            ID_FILE_IMPORT_FILE_LIST,
            "File List...",
            "Add files or URLs to the Images module file list",
        )
        self.__menu_file.AppendSubMenu(submenu, "Import")

        submenu = wx.Menu()
        submenu.Append(
            ID_FILE_SAVE_PIPELINE,
            "Pipeline...\tctrl+P",
            "Save the project's pipeline to a .%s file" % EXT_PIPELINE,
        )
        submenu.Append(
            ID_FILE_EXPORT_IMAGE_SETS,
            "Image Set Listing...",
            "Export the project's image sets as a CSV file suitable for LoadData",
        )
        submenu.Append(
            ID_FILE_EXPORT_PIPELINE_NOTES,
            "Pipeline notes...",
            "Save a text file outlining the pipeline's modules and module notes",
        )
        submenu.Append(
            ID_FILE_EXPORT_PIPELINE_CITATIONS,
            "Citation list for your pipeline...",
            "Save a text file bibliography listing citations for your current pipeline modules",
        )
        self.__menu_file.AppendSubMenu(submenu, "Export")
        self.__menu_file.Append(
            ID_FILE_CLEAR_PIPELINE,
            "Clear Pipeline",
            "Remove all modules from the current pipeline",
        )
        self.__menu_file.AppendSeparator()
        self.__menu_file.Append(
            ID_FILE_OPEN_IMAGE, "View Image", "Open an image file for viewing"
        )
        self.__menu_file.AppendSeparator()
        self.__menu_file.Append(
            ID_FILE_ANALYZE_IMAGES,
            "Analyze Images\tctrl+N",
            "Run the pipeline on the images in the image directory",
        )
        self.__menu_file.Append(
            ID_FILE_STOP_ANALYSIS, "Stop Analysis", "Stop running the pipeline"
        )
        self.__menu_file.AppendSeparator()
        if sys.platform == "darwin":
            self.__menu_file.Append(ID_FILE_NEW_CP, "Open A New CP Window")
            self.__menu_file.AppendSeparator()
        self.__menu_file.Append(
            ID_OPTIONS_READERS,
            "Configure Readers...",
            "Configure image file reader preferences",
        )
        self.__menu_file.Append(
            ID_OPTIONS_PREFERENCES,
            "&Preferences...",
            "Set global application preferences",
        )

        self.recent_files = wx.Menu()
        self.recent_pipeline_files = wx.Menu()
        self.__menu_file.Append(ID_FILE_EXIT, "Q&uit\tctrl+Q", "Quit the application")

        self.menu_edit = wx.Menu()
        self.menu_edit.Append(wx.ID_UNDO, helpString="Undo last action")
        self.menu_edit.AppendSeparator()

        self.menu_edit.Append(wx.ID_CUT)
        self.menu_edit.Append(wx.ID_COPY)
        self.menu_edit.Append(wx.ID_PASTE)
        self.menu_edit.Append(wx.ID_SELECTALL)

        self.menu_edit.AppendSeparator()
        self.menu_edit.Append(
            ID_EDIT_MOVE_UP,
            "Move Selected Modules &Up",
            "Move selected modules toward the start of the pipeline",
        )
        self.menu_edit.Append(
            ID_EDIT_MOVE_DOWN,
            "Move Selected Modules &Down",
            "Move selected modules toward the end of the pipeline",
        )
        self.menu_edit.Append(
            ID_EDIT_DELETE, "&Delete Selected Modules", "Delete selected modules"
        )
        self.menu_edit.Append(
            ID_EDIT_DUPLICATE,
            "Duplicate Selected Modules",
            "Duplicate selected modules",
        )
        self.menu_edit.Append(
            ID_EDIT_ENABLE_MODULE,
            "Disable Selected Modules",
            "Disable a module to skip it when running the pipeline",
        )
        self.menu_edit.Append(
            ID_EDIT_DISPLAY_MODULE,
            "Disable Display of Selected Modules",
            "Turn off module output display",
        )
        self.menu_edit_add_module = wx.Menu()
        self.menu_edit.AppendSubMenu(self.menu_edit_add_module, "&Add Module")
        self.menu_edit_goto_module = wx.Menu()
        self.menu_edit.AppendSubMenu(self.menu_edit_goto_module, "&Go to Module")

        self.menu_edit.AppendSeparator()
        self.menu_edit.Append(
            ID_EDIT_SHOW_FILE_LIST_IMAGE,
            "Show Selected Image",
            "Display the first selected image in the file list",
        )
        self.menu_edit.Append(
            ID_EDIT_REMOVE_FROM_FILE_LIST,
            "Remove From File List",
            "Remove the selected files from the file list",
        )
        self.menu_edit.Append(
            ID_EDIT_BROWSE_FOR_FILES,
            "Browse for Images",
            "Select images to add to the file list using a file browser",
        )
        self.menu_edit.Append(
            ID_EDIT_BROWSE_FOR_FOLDER,
            "Browse for Image Folder",
            "Select a folder of images to add to the file list using a file browser",
        )
        self.menu_edit.Append(
            ID_EDIT_CLEAR_FILE_LIST,
            "Clear File List",
            "Remove all files from the file list",
        )
        self.menu_edit.Append(
            ID_EDIT_EXPAND_ALL,
            "Expand All Folders",
            "Expand all folders in the file list and show all file names",
        )
        self.menu_edit.Append(
            ID_EDIT_COLLAPSE_ALL,
            "Collapse All Folders",
            "Collapse all folders in the file list, hiding all file names",
        )

        self.__menu_debug = wx.Menu()
        self.__menu_debug.Append(
            ID_DEBUG_TOGGLE, "&Start Test Mode\tF5", "Start the pipeline debugger"
        )
        self.__menu_debug.Append(
            ID_DEBUG_STEP,
            "Ste&p to Next Module\tF6",
            "Execute the currently selected module",
        )
        self.__menu_debug.Append(
            ID_DEBUG_NEXT_IMAGE_SET,
            "&Next Image Set\tF7",
            "Advance to the next image set",
        )
        self.__menu_debug.Append(
            ID_DEBUG_NEXT_GROUP,
            "Next Image &Group\tF8",
            "Advance to the next group in the image set list",
        )
        self.__menu_debug.Append(
            ID_DEBUG_CHOOSE_RANDOM_IMAGE_SET,
            "Random Image Set",
            "Advance to a random image set",
        )
        self.__menu_debug.Append(
            ID_DEBUG_CHOOSE_RANDOM_IMAGE_GROUP,
            "Random Image Group",
            "Advance to a random image group",
        )
        self.__menu_debug.Append(
            ID_DEBUG_CHOOSE_IMAGE_SET,
            "Choose Image Set",
            "Choose any of the available image sets",
        )
        self.__menu_debug.Append(
            ID_DEBUG_CHOOSE_GROUP,
            "Choose Image Group",
            "Choose which image set group to process in test-mode",
        )

        self.__menu_debug.Append(
            ID_DEBUG_VIEW_WORKSPACE, "View workspace", "View the current workspace",
        )

        if not hasattr(sys, "frozen") or os.getenv("CELLPROFILER_DEBUG"):
            self.__menu_debug.Append(ID_DEBUG_RELOAD, "Reload Modules' Source")
            self.__menu_debug.Append(ID_DEBUG_PDB, "Break Into Debugger")

            if get_widget_inspector():
                self.__menu_debug.Append(ID_FILE_WIDGET_INSPECTOR, "Widget inspector")

        self.__menu_debug.Append(ID_DEBUG_HELP, "Pipeline Testing Help")
        self.__menu_debug.Enable(ID_DEBUG_STEP, False)
        self.__menu_debug.Enable(ID_DEBUG_NEXT_IMAGE_SET, False)
        self.__menu_debug.Enable(ID_DEBUG_NEXT_GROUP, False)
        self.__menu_debug.Enable(ID_DEBUG_CHOOSE_GROUP, False)
        self.__menu_debug.Enable(ID_DEBUG_CHOOSE_IMAGE_SET, False)
        self.__menu_debug.Enable(ID_DEBUG_CHOOSE_RANDOM_IMAGE_SET, False)
        self.__menu_debug.Enable(ID_DEBUG_CHOOSE_RANDOM_IMAGE_GROUP, False)

        self.__menu_window = wx.Menu()
        self.__menu_window.Append(
            ID_WINDOW_CLOSE_ALL,
            "Close &All Open Windows\tctrl+L",
            "Close all open module display windows",
        )
        self.__menu_window.Append(
            ID_WINDOW_SHOW_ALL_WINDOWS,
            "Show All Windows On Run",
            "Show all module display windows for all modules during analysis",
        )
        self.__menu_window.Append(
            ID_WINDOW_HIDE_ALL_WINDOWS,
            "Hide All Windows On Run",
            "Hide all module display windows for all modules during analysis",
        )
        self.__menu_window.AppendSeparator()

        self.__menu_window.Append(
            ID_FILE_PLATEVIEWER,
            "Show Plate Viewer",
            "Open the plate viewer to inspect the images in the current workspace",
        )

        if sys.platform == "win32":
            self.__menu_window.AppendSeparator()

        self.__menu_plugins = None
        if PLUGIN_MENU_ENTRIES:
            # Only show the plugins menu if a plugin is using it
            self.__menu_plugins = wx.Menu()
            for callback_fn, wx_id, name, tooltip in PLUGIN_MENU_ENTRIES:
                self.__menu_plugins.Append(wx_id, name, tooltip)
                self.Bind(wx.EVT_MENU, callback_fn, id=wx_id)

        self.__menu_help = Menu(self)

        self.__menu_bar = wx.MenuBar()
        self.__menu_bar.Append(self.__menu_file, "&File")
        self.__menu_bar.Append(self.menu_edit, "&Edit")
        self.__menu_bar.Append(self.__menu_debug, "&Test")
        if get_show_sampling():
            self.__menu_sample = wx.Menu()
            self.__menu_sample.Append(
                ID_SAMPLE_INIT,
                "Initialize Sampling",
                "Initialize sampling up to current module",
            )
            self.__menu_bar.Append(self.__menu_sample, "&Sample")
        if PLUGIN_MENU_ENTRIES:
            self.__menu_bar.Append(self.__menu_plugins, "&Plugins")
        self.__menu_bar.Append(self.__menu_window, "&Windows")
        if wx.VERSION <= (2, 8, 10, 1, "") and wx.Platform == "__WXMAC__":
            self.__menu_bar.Append(self.__menu_help, "CellProfiler Help")
        else:
            self.__menu_bar.Append(self.__menu_help, "&Help")
        self.SetMenuBar(self.__menu_bar)
        self.enable_edit_commands([])

        self.Bind(wx.EVT_MENU, self.on_open_image, id=ID_FILE_OPEN_IMAGE)
        self.Bind(wx.EVT_MENU, lambda event: self.Close(), id=ID_FILE_EXIT)
        self.Bind(wx.EVT_MENU, self.__on_widget_inspector, id=ID_FILE_WIDGET_INSPECTOR)
        self.Bind(wx.EVT_MENU, self.__on_new_cp, id=ID_FILE_NEW_CP)

        self.Bind(wx.EVT_MENU, self.on_cut, id=wx.ID_CUT)
        self.Bind(wx.EVT_UPDATE_UI, self.on_update_cut_ui, id=wx.ID_CUT)

        self.Bind(wx.EVT_MENU, self.on_copy, id=wx.ID_COPY)
        self.Bind(wx.EVT_UPDATE_UI, self.on_update_copy_ui, id=wx.ID_COPY)

        self.Bind(wx.EVT_MENU, self.on_paste, id=wx.ID_PASTE)
        self.Bind(wx.EVT_UPDATE_UI, self.on_update_paste_ui, id=wx.ID_PASTE)

        self.Bind(wx.EVT_MENU, self.on_select_all, id=wx.ID_SELECTALL)
        self.Bind(wx.EVT_UPDATE_UI, self.on_update_select_all_ui, id=wx.ID_SELECTALL)

        # ID_HELP_MODULE is used in _both_ button contexts and menu contexts,
        # so it needs event bindings for either type
        self.Bind(wx.EVT_MENU, self.__on_help_module, id=ID_HELP_MODULE)
        self.Bind(wx.EVT_BUTTON, self.__on_help_module, id=ID_HELP_MODULE)

        self.Bind(wx.EVT_MENU, self.__on_readers, id=ID_OPTIONS_READERS)
        
        self.Bind(wx.EVT_MENU, self.__on_preferences, id=ID_OPTIONS_PREFERENCES)
        self.Bind(wx.EVT_MENU, self.__on_close_all, id=ID_WINDOW_CLOSE_ALL)
        self.Bind(wx.EVT_MENU, self.__debug_pdb, id=ID_DEBUG_PDB)
        self.Bind(wx.EVT_MENU, self.__on_debug_help, id=ID_DEBUG_HELP)

        accelerator_table = wx.AcceleratorTable(
            [
                (wx.ACCEL_CMD, ord("N"), ID_FILE_ANALYZE_IMAGES),
                (wx.ACCEL_CMD, ord("O"), ID_FILE_LOAD),
                (wx.ACCEL_CMD, ord("P"), ID_FILE_SAVE_PIPELINE),
                (wx.ACCEL_CMD | wx.ACCEL_SHIFT, ord("S"), ID_FILE_SAVE),
                (wx.ACCEL_CMD, ord("L"), ID_WINDOW_CLOSE_ALL),
                (wx.ACCEL_CMD, ord("Q"), ID_FILE_EXIT),
                (wx.ACCEL_CMD, ord("W"), ID_FILE_EXIT),
                (wx.ACCEL_CMD, ord("A"), wx.ID_SELECTALL),
                (wx.ACCEL_CMD, ord("C"), wx.ID_COPY),
                (wx.ACCEL_CMD, ord("V"), wx.ID_PASTE),
                (wx.ACCEL_NORMAL, wx.WXK_F5, ID_DEBUG_TOGGLE),
                (wx.ACCEL_NORMAL, wx.WXK_F6, ID_DEBUG_STEP),
                (wx.ACCEL_NORMAL, wx.WXK_F7, ID_DEBUG_NEXT_IMAGE_SET),
                (wx.ACCEL_NORMAL, wx.WXK_F8, ID_DEBUG_NEXT_GROUP),
                (wx.ACCEL_CMD, ord("Z"), ID_EDIT_UNDO),
            ]
        )
        self.SetAcceleratorTable(accelerator_table)
        self.enable_launch_commands()

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
        if (
            focus is not None
            and hasattr(focus, "Cut")
            and hasattr(focus, "CanCut")
            and focus.CanCut()
        ):
            focus.Cut()

    @staticmethod
    def on_update_cut_ui(event):
        focus = wx.Window.FindFocus()
        event.Enable(bool(focus and hasattr(focus, "CanCut") and focus.CanCut()))

    @staticmethod
    def on_copy(event):
        """Handle ID_COPY"""
        focus = wx.Window.FindFocus()
        if (
            focus is not None
            and hasattr(focus, "Copy")
            and hasattr(focus, "CanCopy")
            and focus.CanCopy()
        ):
            focus.Copy()

    @staticmethod
    def on_update_copy_ui(event):
        focus = wx.Window.FindFocus()
        event.Enable(bool(focus and hasattr(focus, "CanCopy") and focus.CanCopy()))

    @staticmethod
    def on_paste(event):
        """Handle ID_PASTE"""
        focus = wx.Window.FindFocus()
        if (
            focus is not None
            and hasattr(focus, "Paste")
            and hasattr(focus, "CanPaste")
            and focus.CanPaste()
        ):
            focus.Paste()

    @staticmethod
    def on_update_paste_ui(event):
        focus = wx.Window.FindFocus()
        event.Enable(bool(focus and hasattr(focus, "CanPaste") and focus.CanPaste()))

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

    debug_commands = (
        ID_DEBUG_STEP,
        ID_DEBUG_NEXT_IMAGE_SET,
        ID_DEBUG_NEXT_GROUP,
        ID_DEBUG_CHOOSE_GROUP,
        ID_DEBUG_CHOOSE_IMAGE_SET,
        ID_DEBUG_CHOOSE_RANDOM_IMAGE_SET,
        ID_DEBUG_CHOOSE_RANDOM_IMAGE_GROUP,
        ID_DEBUG_VIEW_WORKSPACE,
    )

    def enable_debug_commands(self):
        """Enable or disable the debug commands (like ID_DEBUG_STEP)"""
        startstop = self.__menu_debug.FindItemById(ID_DEBUG_TOGGLE)
        self.__menu_file.Enable(ID_FILE_ANALYZE_IMAGES, False)

        assert isinstance(startstop, wx.MenuItem)
        startstop.SetItemLabel("&Exit Test Mode\tF5")
        startstop.SetHelp("Stop testing your pipeline")
        for cmd in self.debug_commands:
            self.__menu_debug.Enable(cmd, True)

    def enable_launch_commands(self):
        """Enable commands to start analysis or test mode"""
        startstop = self.__menu_debug.FindItemById(ID_DEBUG_TOGGLE)
        startstop.SetItemLabel("&Start Test Mode\tF5")
        startstop.SetHelp("Start testing your pipeline")
        for cmd in self.debug_commands:
            self.__menu_debug.Enable(cmd, False)
        self.__menu_file.Enable(ID_FILE_ANALYZE_IMAGES, True)
        self.__menu_debug.Enable(ID_DEBUG_TOGGLE, True)

        self.__menu_file.Enable(ID_FILE_STOP_ANALYSIS, False)

    def enable_analysis_commands(self):
        """Enable commands to pause or stop analysis"""
        self.__menu_file.Enable(ID_FILE_ANALYZE_IMAGES, False)
        self.__menu_debug.Enable(ID_DEBUG_TOGGLE, False)

        self.__menu_file.Enable(ID_FILE_STOP_ANALYSIS, True)

    @staticmethod
    def __on_widget_inspector(evt):
        try:
            wx.lib.inspection.InspectionTool().Show()
        except:
            wx.MessageBox("Inspection tool is not available on this platform")

    @staticmethod
    def __on_readers(event):
        dlg = ReadersDialog()
        dlg.Show()

    @staticmethod
    def __on_preferences(event):
        dlg = PreferencesDialog()
        dlg.Show()

    def __on_close_all(self, event):
        close_all(self)

    @staticmethod
    def __on_new_cp(event):
        if hasattr(sys, "frozen"):
            app_path = sys.executable.split("/Contents")[0]
            os.system(f"open -na {app_path}")
        else:
            os.system("python3 -m cellprofiler")

    def __on_help_path_list(self, event):
        import cellprofiler.gui.htmldialog

        dlg = cellprofiler.gui.htmldialog.HTMLDialog(
            self, "Help on file list", rst_to_html_fragment(HELP_ON_FILE_LIST),
        )
        dlg.Show()

    def __on_debug_help(self, event):
        import cellprofiler.gui.htmldialog

        contents = read_content("navigation_test_menu.rst")
        help_dialog = cellprofiler.gui.htmldialog.HTMLDialog(
            self, "Test Mode Help", rst_to_html_fragment(contents),
        )
        help_dialog.Show()

    def __on_help_welcome(self, event):
        self.show_welcome_screen(True)

    def __on_help_module(self, event):
        modules = self.__pipeline_list_view.get_selected_modules()
        active_module = self.__pipeline_list_view.get_active_module()
        if len(modules) > 0:
            self.do_help_modules(modules)
        elif active_module is not None:
            self.do_help_module(active_module.module_name, active_module.get_help())
        else:
            wx.MessageBox(
                HELP_ON_MODULE_BUT_NONE_SELECTED,
                "No module selected",
                style=wx.OK | wx.ICON_INFORMATION,
            )

    @staticmethod
    def __debug_pdb(event):
        pdb.set_trace()

    def do_help_modules(self, modules):
        for module in modules:
            # An attempt to place images inline with the help. However, the
            # images will not scale properly in size (yet)
            # result = module.get_help()
            # root = os.path.split(__file__)[0]
            # if len(root) == 0:
            # root = os.curdir
            # root = os.path.split(os.path.abspath(root))[0] # Back up one level
            # absolute_image_path = os.path.join(root, 'icons','%s.png'%(module.module_name,))
            # Check if the file that goes with this module exists on this computer
            # if os.path.exists(absolute_image_path) and os.path.isfile(absolute_image_path):
            # If so, strip out end html tags so I can add more stuff
            # result = result.replace('</body>','').replace('</html>','')
            # Include images specific to the module
            # result += '\n\n<div><p><img src="%s", width="50%%"></p></div>\n'%absolute_image_path
            # Now end the help text
            # result += '</body></html>'
            # self.do_help_module(module.module_name, result)
            self.do_help_module(module.module_name, module.get_help())

    def do_help_module(self, module_name, help_text):
        helpframe = wx.Frame(
            self, -1, 'Help for module, "%s"' % module_name, size=(640, 480)
        )
        helpframe.SetMenuBar(wx.MenuBar())
        ####################################################
        #
        # Add the HTML window
        #
        ####################################################

        sizer = wx.BoxSizer()
        helpframe.SetSizer(sizer)
        window = HtmlClickableWindow(helpframe)
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

        helpframe.GetMenuBar().Append(menu, "&File")
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
                wx.MessageBox(
                    "Failed to copy to the clipboard", "Error", wx.OK | wx.ICON_ERROR
                )

        helpframe.GetMenuBar().Append(menu, "&Edit")
        helpframe.Bind(wx.EVT_MENU, on_copy, id=ID_EDIT_COPY)
        helpframe.Bind(wx.EVT_MENU, on_edit_select_all, id=ID_EDIT_SELECT_ALL)
        helpframe.Bind(wx.EVT_IDLE, on_idle)

        ####################################################
        #
        # Build an accelerator table for some of the commands
        #
        ####################################################
        accelerator_table = wx.AcceleratorTable(
            [
                (wx.ACCEL_CMD, ord("Q"), ID_FILE_EXIT),
                (wx.ACCEL_CMD, ord("P"), ID_FILE_PRINT),
                (wx.ACCEL_CMD, ord("C"), ID_EDIT_COPY),
            ]
        )
        helpframe.SetAcceleratorTable(accelerator_table)
        helpframe.SetIcon(cellprofiler.gui.utilities.icon.get_cp_icon())
        helpframe.Layout()
        helpframe.Show()

    @staticmethod
    def print_help(event, module_name, help_text):
        """Print the help text for a module"""
        printer = wx.html.HtmlEasyPrinting("Printing %s" % module_name)
        printer.GetPrintData().SetPaperId(wx.PAPER_LETTER)
        printer.PrintText(help_text)

    @staticmethod
    def save_help(event, module_name, help_text):
        """Save the help text for a module"""
        save_dlg = wx.FileDialog(
            event.GetEventObject().GetWindow(),
            message="Save help for %s to file" % module_name,
            defaultFile="%s.html" % module_name,
            wildcard="*.html",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        )

        result = save_dlg.ShowModal()

        if result == wx.ID_OK:
            with codecs.open(save_dlg.GetPath(), "w", encoding="utf-8") as fd:
                fd.write(
                    '<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />'
                )
                fd.write(help_text)

    def on_open_image(self, event):
        dlg = wx.FileDialog(
            self,
            message="Open an image file",
            wildcard="Image file (*.tif,*.tiff,*.jpg,*.jpeg,*.png,*.gif,*.bmp)|*.tif;*.tiff;*.jpg;*.jpeg;*.png;*.gif;*.bmp|*.* (all files)|*.*",
            style=wx.FD_OPEN,
        )
        if dlg.ShowModal() == wx.ID_OK:
            from cellprofiler_core.image import FileImage
            from .figure import Figure

            lip = FileImage("dummy", "", dlg.GetPath())
            image = lip.provide_image(None).pixel_data
            frame = Figure(self, title=dlg.GetPath(), subplots=(1, 1))
            if image.ndim == 3:
                frame.subplot_imshow_color(0, 0, image, title=dlg.GetPath())
            else:
                frame.subplot_imshow_grayscale(0, 0, image, title=dlg.GetPath())
            frame.panel.draw()

    def __attach_views(self):
        self.__pipeline_list_view = PipelineListView(self.__module_list_panel, self)
        self.__pipeline_controller = PipelineController(self.__workspace, self)
        self.__pipeline_list_view.attach_to_pipeline(
            self.__pipeline, self.__pipeline_controller
        )
        self.__pipeline_controller.attach_to_test_controls_panel(
            self.__pipeline_test_panel
        )
        self.__pipeline_controller.attach_to_module_controls_panel(
            self.__module_controls_panel
        )
        self.__pipeline_controller.attach_to_path_list_ctrl(
            self.__path_list_ctrl, self.__path_list_filter_checkbox
        )
        self.__module_view = ModuleView(
            self.__module_panel,
            self.__workspace,
            frame=self,
            notes_panel=self.__notes_panel,
        )
        self.__pipeline_controller.attach_to_module_view(self.__module_view)
        self.__pipeline_list_view.attach_to_module_view(self.__module_view)
        self.__preferences_view = PreferencesView(
            self.__right_win.GetSizer(),
            self.__preferences_panel,
            self.__progress_panel,
            self.__status_panel,
        )
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
        self.__splitter.SetMinimumPaneSize(5)

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
        self.SetIcon(cellprofiler.gui.utilities.icon.get_cp_icon())

    def __on_data_tool_help(self, event, tool_name):
        module = instantiate_module(tool_name)
        self.do_help_module(tool_name, module.get_help())

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

    def get_workspace(self):
        """Get the pipeline - mostly to drive testing"""
        return self.__workspace

    workspace = property(get_workspace)

    def get_module_view(self):
        """Return the module view window"""
        return self.__module_view

    module_view = property(get_module_view)

    def get_pipeline_list_view(self):
        return self.__pipeline_list_view

    pipeline_list_view = property(get_pipeline_list_view)

    def menu_item_exists(self, candidate_id):
        menu_item = self.__menu_bar.FindItemById(candidate_id)
        return menu_item is not None

    def remove_menu_item(self, item_id):
        menu_item = self.__menu_bar.FindItemById(item_id)
        if not menu_item:
            LOGGER.error(f"Item with id {item_id} does not exist")
            return
        parent = menu_item.GetMenu()
        removed = parent.Remove(menu_item)

    # caller responsible for making sure child does not already exist
    def _inject_menu_item(self, parent_menu, child_id, title, sibling_id=None):
        if sibling_id:
            sibling_menu_item, sibling_menu_pos = parent_menu.FindChildItem(sibling_id)
            if not sibling_menu_item:
                LOGGER.error(f"Sibling with id {sibling_id} does not exist")
                return
            parent_menu.Insert(sibling_menu_pos, child_id, title)
        else:
            parent_menu.Append(child_id, title)

    def inject_menu_item_by_id(self, parent_id, child_id, title, sibling_id=None):
        parent_menu_idx = self.__menu_bar.FindItemById(parent_id)
        if parent_menu_idx == wx.NOT_FOUND:
            LOGGER.error(f"Parent with id {parent_id} does not exist")
            return
        parent_menu = self.__menu_bar.GetMenu(parent_menu_idx)
        self._inject_menu_item(parent_menu, child_id, title, sibling_id)

    def inject_menu_item_by_title(self, parent_title, child_id, title, sibling_id=None):
        parent_menu_idx = self.__menu_bar.FindMenu(parent_title)
        if parent_menu_idx == wx.NOT_FOUND:
            LOGGER.error(f"Parent with title \"{parent_title}\" does not exist")
            return
        parent_menu = self.__menu_bar.GetMenu(parent_menu_idx)
        self._inject_menu_item(parent_menu, child_id, title, sibling_id)
