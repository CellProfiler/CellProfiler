# coding=utf-8
"""PipelineController.py - controls (modifies) a pipeline
"""


import csv
import datetime
import gc
import hashlib
import io
import logging
import os
import random
import re
import string
import sys
import threading
import h5py
import numpy

from functools import reduce, cmp_to_key
from queue import PriorityQueue, Queue, Empty
from urllib.request import urlretrieve, url2pathname
from urllib.parse import urlparse

import wx
import wx.lib.buttons
import wx.lib.mixins.listctrl

from cellprofiler_core.analysis import DEBUG
from cellprofiler_core.analysis._analysis import Analysis
from cellprofiler_core.analysis._runner import Runner
from cellprofiler_core.analysis.event import (
    Finished,
    Resumed,
    Paused,
    Started,
    Progress,
)
from cellprofiler_core.analysis.reply import (
    DebugCancel,
    ExceptionPleaseDebug,
    OmeroLogin,
    Interaction,
    Ack,
    ServerExited,
)
from cellprofiler_core.analysis.request import (
    Display,
    DisplayPostRun,
    DisplayPostGroup,
    DebugComplete,
    DebugWaiting,
    ExceptionReport,
)
from cellprofiler_core.analysis.request import Interaction as InteractionRequest
from cellprofiler_core.analysis.request import OmeroLogin as OmeroLoginRequest
from cellprofiler_core.constants.measurement import (
    EXPERIMENT,
    IMAGE,
    C_URL,
    C_METADATA,
    C_PATH_NAME,
    C_FRAME,
    C_FILE_NAME,
    GROUP_INDEX,
    GROUP_NUMBER,
)
from cellprofiler_core.constants.pipeline import (
    M_USER_PIPELINE,
    M_PIPELINE,
    DIRECTION_DOWN,
    DIRECTION_UP,
)
from cellprofiler_core.constants.reader import ALL_READERS, ZARR_FILETYPE
from cellprofiler_core.constants.workspace import DISPOSITION_SKIP
from cellprofiler_core.image import ImageSetList
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.module import Module
from cellprofiler_core.object import ObjectSet
from cellprofiler_core.pipeline import (
    PipelineLoadCancelledException,
    PostRunException,
    PrepareRunException,
    URLsAdded,
    URLsCleared,
    URLsRemoved,
    LoadException,
    ModuleAdded,
    ModuleMoved,
    ModuleRemoved,
    PipelineCleared,
    PipelineLoaded,
    RunException,
    Listener,
    PrepareRunError,
    ImageFile,
)
from cellprofiler_core.pipeline.io._v6 import load
from cellprofiler_core.preferences import (
    RECENT_FILE_COUNT,
    add_image_directory_listener,
    get_image_set_file,
    clear_image_set_file,
    get_background_color,
    get_current_workspace_path,
    EXT_PROJECT_CHOICES,
    set_workspace_file,
    set_current_workspace_path,
    add_progress_callback,
    remove_progress_callback,
    EXT_PROJECT,
    get_save_pipeline_with_project,
    SPP_PIPELINE_ONLY,
    SPP_PIPELINE_AND_FILE_LIST,
    EXT_PIPELINE,
    EXT_PIPELINE_CHOICES,
    cancel_progress,
    add_output_directory_listener,
    get_show_exiting_test_mode_dlg,
    SPP_FILE_LIST_ONLY,
    get_recent_files,
    WORKSPACE_FILE,
    set_show_exiting_test_mode_dlg,
    get_default_output_directory,
    set_choose_image_set_frame_size,
    get_choose_image_set_frame_size,
    get_show_analysis_complete_dlg,
    get_wants_pony,
    set_show_analysis_complete_dlg,
    set_default_output_directory,
    get_max_workers,
    set_default_image_directory,
    get_telemetry,
    get_conserve_memory,
)
from cellprofiler_core.setting import ValidationError
from cellprofiler_core.utilities.core.modules import (
    instantiate_module,
    get_module_class,
    get_module_names,
)
from cellprofiler_core.utilities.legacy import cmp
from cellprofiler_core.utilities.pathname import pathname2url
from cellprofiler_core.utilities.zmq import Reply
from wx.adv import Sound

import cellprofiler.gui._workspace_model
import cellprofiler.gui.addmoduleframe
import cellprofiler.gui.cpframe
import cellprofiler.gui.dialog
import cellprofiler.gui.help
import cellprofiler.gui.help.content
import cellprofiler.gui.html.utils
from cellprofiler.gui.htmldialog import HTMLDialog
import cellprofiler.gui.module_view._setting_edited_event
import cellprofiler.gui.moduleview
import cellprofiler.gui.omerologin
import cellprofiler.gui.parametersampleframe
import cellprofiler.gui.pathlist
import cellprofiler.gui.pipeline
import cellprofiler.gui.workspace_view
import cellprofiler.icons
from cellprofiler.gui.pipelinelistview import EVT_PLV_VALID_STEP_COLUMN_CLICKED
from cellprofiler import __version__ as cellprofiler_version
from .workspace_view import WorkspaceView

LOGGER = logging.getLogger(__name__)

RECENT_PIPELINE_FILE_MENU_ID = [wx.NewId() for i in range(RECENT_FILE_COUNT)]
RECENT_WORKSPACE_FILE_MENU_ID = [wx.NewId() for i in range(RECENT_FILE_COUNT)]
ED_STOP = "Stop"
ED_CONTINUE = "Continue"
ED_SKIP = "Skip"


class PipelineController(object):
    """Controls the pipeline through the UI

    """

    def __init__(self, workspace, frame):
        self.__workspace = workspace

        pipeline = self.__pipeline = workspace.pipeline
        pipeline.add_listener(self.__on_pipeline_event)
        self.__analysis = None
        self.__frame = frame
        self.__add_module_frame = cellprofiler.gui.addmoduleframe.AddModuleFrame(
            frame, -1, "Add modules"
        )
        self.__add_module_frame.add_listener(self.on_add_to_pipeline)
        # ~*~
        self.__parameter_sample_frame = None
        # ~^~
        self.__setting_errors = {}
        self.__dirty_workspace = False
        self.__debug_image_set_list = None
        self.__debug_measurements = None
        self.__debug_grids = None
        self.__keys = None
        self.__groupings = None
        self.__grouping_index = None
        self.__within_group_index = None
        self.__plate_viewer = None
        self.__locked_workspace_filename = None
        self.pipeline_list = []
        add_image_directory_listener(self.__on_image_directory_change)
        add_output_directory_listener(self.__on_output_directory_change)

        # interaction/display requests and exceptions from an Analysis
        self.interaction_request_queue = PriorityQueue()
        self.interaction_pending = False
        self.debug_request_queue = None

        self.populate_recent_files()
        self.menu_id_to_module_name = {}
        self.module_name_to_menu_id = {}
        self.populate_edit_menu(self.__frame.menu_edit_add_module)

        assert isinstance(frame, wx.Frame)

        frame.Bind(
            wx.EVT_MENU,
            self.__on_new_workspace,
            id=cellprofiler.gui.cpframe.ID_FILE_NEW_WORKSPACE,
        )

        frame.Bind(
            wx.EVT_MENU,
            self.__on_open_workspace,
            id=cellprofiler.gui.cpframe.ID_FILE_LOAD,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.__on_save_workspace,
            id=cellprofiler.gui.cpframe.ID_FILE_SAVE,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.__on_save_as_workspace,
            id=cellprofiler.gui.cpframe.ID_FILE_SAVE_AS,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.__on_load_pipeline,
            id=cellprofiler.gui.cpframe.ID_FILE_LOAD_PIPELINE,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.__on_url_load_pipeline,
            id=cellprofiler.gui.cpframe.ID_FILE_URL_LOAD_PIPELINE,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.__on_import_file_list,
            id=cellprofiler.gui.cpframe.ID_FILE_IMPORT_FILE_LIST,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.__on_save_as_pipeline,
            id=cellprofiler.gui.cpframe.ID_FILE_SAVE_PIPELINE,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.__on_export_image_sets,
            id=cellprofiler.gui.cpframe.ID_FILE_EXPORT_IMAGE_SETS,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.__on_export_pipeline_notes,
            id=cellprofiler.gui.cpframe.ID_FILE_EXPORT_PIPELINE_NOTES,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.__on_export_pipeline_citations,
            id=cellprofiler.gui.cpframe.ID_FILE_EXPORT_PIPELINE_CITATIONS,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.__on_revert_workspace,
            id=cellprofiler.gui.cpframe.ID_FILE_REVERT_TO_SAVED,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.__on_clear_pipeline,
            id=cellprofiler.gui.cpframe.ID_FILE_CLEAR_PIPELINE,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.__on_plateviewer,
            id=cellprofiler.gui.cpframe.ID_FILE_PLATEVIEWER,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.on_analyze_images,
            id=cellprofiler.gui.cpframe.ID_FILE_ANALYZE_IMAGES,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.on_stop_running,
            id=cellprofiler.gui.cpframe.ID_FILE_STOP_ANALYSIS,
        )

        frame.Bind(wx.EVT_MENU, self.on_undo, id=cellprofiler.gui.cpframe.ID_EDIT_UNDO)

        frame.Bind(
            wx.EVT_UPDATE_UI,
            self.on_update_undo_ui,
            id=cellprofiler.gui.cpframe.ID_EDIT_UNDO,
        )

        frame.Bind(
            wx.EVT_MENU, self.on_module_up, id=cellprofiler.gui.cpframe.ID_EDIT_MOVE_UP
        )
        frame.Bind(
            wx.EVT_MENU,
            self.on_module_down,
            id=cellprofiler.gui.cpframe.ID_EDIT_MOVE_DOWN,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.on_remove_module,
            id=cellprofiler.gui.cpframe.ID_EDIT_DELETE,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.on_duplicate_module,
            id=cellprofiler.gui.cpframe.ID_EDIT_DUPLICATE,
        )

        frame.Bind(
            wx.EVT_MENU,
            self.on_pathlist_browse_files,
            id=cellprofiler.gui.cpframe.ID_EDIT_BROWSE_FOR_FILES,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.on_pathlist_browse_folder,
            id=cellprofiler.gui.cpframe.ID_EDIT_BROWSE_FOR_FOLDER,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.on_pathlist_clear,
            id=cellprofiler.gui.cpframe.ID_EDIT_CLEAR_FILE_LIST,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.on_pathlist_collapse_all,
            id=cellprofiler.gui.cpframe.ID_EDIT_COLLAPSE_ALL,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.on_pathlist_expand_all,
            id=cellprofiler.gui.cpframe.ID_EDIT_EXPAND_ALL,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.on_pathlist_remove,
            id=cellprofiler.gui.cpframe.ID_EDIT_REMOVE_FROM_FILE_LIST,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.on_pathlist_show,
            id=cellprofiler.gui.cpframe.ID_EDIT_SHOW_FILE_LIST_IMAGE,
        )

        menu_identifiers = [
            cellprofiler.gui.cpframe.ID_EDIT_BROWSE_FOR_FILES,
            cellprofiler.gui.cpframe.ID_EDIT_CLEAR_FILE_LIST,
            cellprofiler.gui.cpframe.ID_EDIT_COLLAPSE_ALL,
            cellprofiler.gui.cpframe.ID_EDIT_EXPAND_ALL,
            cellprofiler.gui.cpframe.ID_EDIT_REMOVE_FROM_FILE_LIST,
            cellprofiler.gui.cpframe.ID_EDIT_SHOW_FILE_LIST_IMAGE,
        ]

        for menu_id in menu_identifiers:
            frame.Bind(wx.EVT_UPDATE_UI, self.on_update_pathlist_ui, id=menu_id)

        frame.Bind(
            wx.EVT_UPDATE_UI,
            self.on_update_module_enable,
            id=cellprofiler.gui.cpframe.ID_EDIT_ENABLE_MODULE,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.on_module_enable,
            id=cellprofiler.gui.cpframe.ID_EDIT_ENABLE_MODULE,
        )
        frame.Bind(
            wx.EVT_UPDATE_UI,
            self.on_update_module_display_toggle,
            id=cellprofiler.gui.cpframe.ID_EDIT_DISPLAY_MODULE,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.on_module_display_toggle,
            id=cellprofiler.gui.cpframe.ID_EDIT_DISPLAY_MODULE,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.on_debug_toggle,
            id=cellprofiler.gui.cpframe.ID_DEBUG_TOGGLE,
        )
        frame.Bind(
            wx.EVT_MENU, self.on_debug_step, id=cellprofiler.gui.cpframe.ID_DEBUG_STEP
        )
        frame.Bind(
            wx.EVT_MENU,
            self.on_debug_next_image_set,
            id=cellprofiler.gui.cpframe.ID_DEBUG_NEXT_IMAGE_SET,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.on_debug_next_group,
            id=cellprofiler.gui.cpframe.ID_DEBUG_NEXT_GROUP,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.on_debug_choose_group,
            id=cellprofiler.gui.cpframe.ID_DEBUG_CHOOSE_GROUP,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.on_debug_choose_image_set,
            id=cellprofiler.gui.cpframe.ID_DEBUG_CHOOSE_IMAGE_SET,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.on_debug_random_image_set,
            id=cellprofiler.gui.cpframe.ID_DEBUG_CHOOSE_RANDOM_IMAGE_SET,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.on_debug_random_image_group,
            id=cellprofiler.gui.cpframe.ID_DEBUG_CHOOSE_RANDOM_IMAGE_GROUP,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.on_debug_reload,
            id=cellprofiler.gui.cpframe.ID_DEBUG_RELOAD,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.on_run_from_this_module,
            id=cellprofiler.gui.cpframe.ID_DEBUG_RUN_FROM_THIS_MODULE,
        )

        frame.Bind(
            wx.EVT_MENU,
            self.on_step_from_this_module,
            id=cellprofiler.gui.cpframe.ID_DEBUG_STEP_FROM_THIS_MODULE,
        )

        frame.Bind(
            wx.EVT_MENU,
            self.on_view_workspace,
            id=cellprofiler.gui.cpframe.ID_DEBUG_VIEW_WORKSPACE,
        )

        frame.Bind(
            wx.EVT_MENU, self.on_sample_init, id=cellprofiler.gui.cpframe.ID_SAMPLE_INIT
        )

        frame.Bind(
            wx.EVT_MENU,
            self.on_show_all_windows,
            id=cellprofiler.gui.cpframe.ID_WINDOW_SHOW_ALL_WINDOWS,
        )
        frame.Bind(
            wx.EVT_MENU,
            self.on_hide_all_windows,
            id=cellprofiler.gui.cpframe.ID_WINDOW_HIDE_ALL_WINDOWS,
        )

        frame.Bind(
            EVT_PLV_VALID_STEP_COLUMN_CLICKED, self.on_step_from_specific_module,
        )

        #TODO: disabled until CellProfiler/CellProfiler#4684 is resolved
        # from cellprofiler_core.bioformats.formatreader import set_omero_login_hook
        # set_omero_login_hook(self.omero_login)

        self.workspace_view = None

    def start(self, workspace_file, pipeline_path):
        """Do initialization after GUI hookup

        Perform steps that need to happen after all of the user interface
        elements have been initialized.
        """
        if workspace_file is not None:
            self.do_open_workspace(
                workspace_file, load_pipeline=(pipeline_path is None)
            )
        else:
            self.do_create_workspace()
        if pipeline_path is not None:
            self.do_load_pipeline(pipeline_path)
        file_list = get_image_set_file()
        clear_image_set_file()
        if file_list is not None:
            self.__pipeline.read_file_list(file_list)

    def attach_to_pipeline_list_view(self, pipeline_list_view):
        """Glom onto events from the list box with all of the module names in it

        """
        self.__pipeline_list_view = pipeline_list_view

    def attach_to_path_list_ctrl(
        self, path_list_ctrl, path_list_filtered_files_checkbox
    ):
        """Attach the pipeline controller to the path_list_ctrl

        This lets the pipeline controller populate the path list as
        it changes.
        """
        self.__path_list_ctrl = path_list_ctrl
        self.__path_list_is_filtered = None
        self.__path_list_filter_checkbox = path_list_filtered_files_checkbox
        self.__path_list_filter_checkbox.Value = (
            self.__path_list_ctrl.get_show_disabled()
        )

        path_list_ctrl.set_context_menu_fn(
            self.get_pathlist_file_context_menu,
            self.get_pathlist_folder_context_menu,
            self.get_pathlist_empty_context_menu,
            self.on_pathlist_file_command,
            self.on_pathlist_folder_command,
            self.on_pathlist_empty_command,
        )
        path_list_ctrl.set_delete_fn(self.on_pathlist_file_delete)

        self.path_list_drop_target = FLDropTarget(
            self.on_pathlist_drop_files, self.on_pathlist_drop_text
        )

        path_list_ctrl.SetDropTarget(self.path_list_drop_target)

        path_list_ctrl.fn_add_files = self.on_pathlist_drop_files

        def show_disabled(event):
            self.__path_list_ctrl.set_show_disabled(
                self.__path_list_filter_checkbox.Value
            )

        self.__path_list_filter_checkbox.Bind(wx.EVT_CHECKBOX, show_disabled)

    def set_path_list_filtering(self, use_filter):
        """Update the path list UI according to the filter on/off state

        use_filter - True if filtering, False if all files enabled.
        """
        use_filter = bool(use_filter)
        if self.__path_list_is_filtered is not use_filter:
            sizer = self.__path_list_filter_checkbox.GetContainingSizer()
            if sizer is not None:
                sizer.Show(self.__path_list_filter_checkbox, use_filter)
            if not use_filter:
                self.__path_list_ctrl.enable_all_paths()
            self.__path_list_is_filtered = use_filter

    def attach_to_module_view(self, module_view):
        """Listen for setting changes from the module view

        """
        self.__module_view = module_view
        module_view.add_listener(self.__on_module_view_event)

    def attach_to_module_controls_panel(self, module_controls_panel):
        """Attach the pipeline controller to the module controls panel

        Attach the pipeline controller to the module controls panel.
        In addition, the PipelineController gets to add whatever buttons it wants to the
        panel.
        """
        self.__module_controls_panel = module_controls_panel
        mcp_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.__help_button = wx.Button(
            self.__module_controls_panel,
            cellprofiler.gui.cpframe.ID_HELP_MODULE,
            "?",
            (0, 0),
            (30, -1),
        )
        self.__help_button.SetToolTip("Get Help for selected module")
        self.__mcp_text = wx.StaticText(
            self.__module_controls_panel, -1, "Adjust modules:"
        )
        self.__mcp_add_module_button = wx.Button(
            self.__module_controls_panel, -1, "+", (0, 0), (30, -1)
        )
        self.__mcp_add_module_button.SetToolTip("Add a module")
        self.__mcp_remove_module_button = wx.Button(
            self.__module_controls_panel,
            cellprofiler.gui.cpframe.ID_EDIT_DELETE,
            "-",
            (0, 0),
            (30, -1),
        )
        self.__mcp_remove_module_button.SetToolTip("Remove selected module")
        self.__mcp_module_up_button = wx.Button(
            self.__module_controls_panel,
            cellprofiler.gui.cpframe.ID_EDIT_MOVE_UP,
            "^",
            (0, 0),
            (30, -1),
        )
        self.__mcp_module_up_button.SetToolTip("Move selected module up")
        self.__mcp_module_down_button = wx.Button(
            self.__module_controls_panel,
            cellprofiler.gui.cpframe.ID_EDIT_MOVE_DOWN,
            "v",
            (0, 0),
            (30, -1),
        )
        self.__mcp_module_down_button.SetToolTip("Move selected module down")
        mcp_sizer.AddMany(
            [
                (self.__help_button, 0, wx.ALIGN_CENTER | wx.ALL, 3),
                ((1, 3), 3),
                (self.__mcp_text, 0, wx.ALIGN_CENTER | wx.ALL, 3),
                (self.__mcp_add_module_button, 0, wx.ALIGN_CENTER | wx.ALL, 3),
                (self.__mcp_remove_module_button, 0, wx.ALIGN_CENTER | wx.ALL, 3),
                (self.__mcp_module_up_button, 0, wx.ALIGN_CENTER | wx.ALL, 3),
                (self.__mcp_module_down_button, 0, wx.ALIGN_CENTER | wx.ALL, 3),
            ]
        )
        self.__module_controls_panel.SetSizer(mcp_sizer)
        self.__module_controls_panel.Bind(
            wx.EVT_BUTTON, self.__on_add_module, self.__mcp_add_module_button
        )
        self.__module_controls_panel.Bind(
            wx.EVT_BUTTON, self.on_remove_module, self.__mcp_remove_module_button
        )
        self.__module_controls_panel.Bind(
            wx.EVT_BUTTON, self.on_module_up, self.__mcp_module_up_button
        )
        self.__module_controls_panel.Bind(
            wx.EVT_BUTTON, self.on_module_down, self.__mcp_module_down_button
        )

    ANALYZE_IMAGES = "Analyze Images"
    ANALYZE_IMAGES_HELP = "Start a CellProfiler analysis run"
    ENTER_TEST_MODE = "Start Test Mode"
    ENTER_TEST_MODE_HELP = "Test your pipeline settings"
    EXIT_TEST_MODE = "Exit Test Mode"
    EXIT_TEST_MODE_HELP = "Exit pipeline testing"
    PAUSE = "Pause"
    PAUSE_HELP = "Pause the analysis run"
    RESUME = "Resume"
    RESUME_HELP = "Resume the analysis run"

    def attach_to_test_controls_panel(self, panel):
        """Attach the pipeline controller to the test controls panel

        Attach the pipeline controller to the test controls panel.
        In addition, the PipelineController gets to add whatever buttons it wants to the
        panel.
        """
        bkgnd_color = get_background_color()
        assert isinstance(panel, wx.Window)
        self.__test_controls_panel = panel
        #
        # There are three sizers, one for each mode:
        # * tcp_launch_sizer - when idle, for launching analysis or test mode
        # * tcp_analysis_sizer - when in analysis mode
        # * tcp_test_sizer - when in test mode
        #
        tcp_sizer = wx.BoxSizer(wx.VERTICAL)
        self.__test_controls_panel.SetSizer(tcp_sizer)
        self.__tcp_launch_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.__tcp_analysis_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.__tcp_test_sizer = wx.BoxSizer(wx.VERTICAL)
        tcp_sizer.Add(self.__tcp_launch_sizer, 0, wx.EXPAND)
        tcp_sizer.Add(self.__tcp_analysis_sizer, 0, wx.EXPAND)
        tcp_sizer.Add(self.__tcp_test_sizer, 0, wx.EXPAND)
        #
        # Launch sizer
        #
        self.__test_bmp = wx.Bitmap(cellprofiler.icons.get_builtin_image("IMG_TEST"))
        self.__test_mode_button = wx.lib.buttons.GenBitmapTextButton(
            panel, bitmap=self.__test_bmp, label=self.ENTER_TEST_MODE
        )
        self.__test_mode_button.Bind(wx.EVT_BUTTON, self.on_debug_toggle)
        self.__test_mode_button.SetToolTip(self.ENTER_TEST_MODE_HELP)

        self.__tcp_launch_sizer.Add(self.__test_mode_button, 1, wx.EXPAND)

        analyze_bmp = wx.Bitmap(cellprofiler.icons.get_builtin_image("IMG_ANALYZE_16"))
        self.__analyze_images_button = wx.lib.buttons.GenBitmapTextButton(
            panel, bitmap=analyze_bmp, label=self.ANALYZE_IMAGES
        )
        self.__analyze_images_button.Bind(wx.EVT_BUTTON, self.on_analyze_images)
        self.__analyze_images_button.SetToolTip(self.ANALYZE_IMAGES_HELP)

        self.__tcp_launch_sizer.Add(self.__analyze_images_button, 1, wx.EXPAND)

        #
        # Analysis sizer
        #
        stop_bmp = wx.Bitmap(cellprofiler.icons.get_builtin_image("IMG_STOP"))
        pause_bmp = wx.Bitmap(cellprofiler.icons.get_builtin_image("pause"))
        self.__pause_button = wx.lib.buttons.GenBitmapTextButton(
            panel, bitmap=pause_bmp, label=self.PAUSE
        )
        self.__pause_button.Bind(wx.EVT_BUTTON, self.on_pause)
        self.__pause_button.SetToolTip(self.PAUSE_HELP)
        self.__tcp_analysis_sizer.Add(self.__pause_button, 1, wx.EXPAND)

        self.__resume_button = wx.lib.buttons.GenBitmapTextButton(
            panel, bitmap=analyze_bmp, label=self.RESUME
        )
        self.__resume_button.Bind(wx.EVT_BUTTON, self.on_resume)
        self.__resume_button.SetToolTip(self.RESUME_HELP)
        self.__tcp_analysis_sizer.Add(self.__resume_button, 1, wx.EXPAND)

        self.__stop_analysis_button = wx.lib.buttons.GenBitmapTextButton(
            panel, bitmap=stop_bmp, label="Stop Analysis"
        )
        self.__stop_analysis_button.Bind(wx.EVT_BUTTON, self.on_stop_running)
        self.__stop_analysis_button.SetToolTip("Cancel the analysis run")
        self.__tcp_analysis_sizer.Add(self.__stop_analysis_button, 1, wx.EXPAND)
        #
        # Test mode sizer
        #
        sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.__tcp_test_sizer.Add(sub_sizer, 1, wx.EXPAND)
        self.__tcp_test_sizer.AddSpacer(2)

        run_bmp = wx.Bitmap(cellprofiler.icons.get_builtin_image("IMG_RUN"))
        self.__tcp_continue = wx.lib.buttons.GenBitmapTextButton(
            panel, label="Run", bitmap=run_bmp
        )
        self.__tcp_continue.SetToolTip(wx.ToolTip("Run to next pause"))
        self.__tcp_continue.Bind(wx.EVT_BUTTON, self.on_debug_continue)
        sub_sizer.Add(self.__tcp_continue, 1, wx.EXPAND)

        self.__tcp_step = wx.lib.buttons.GenBitmapTextButton(
            panel, label="Step", bitmap=analyze_bmp
        )
        self.__tcp_step.SetToolTip(wx.ToolTip("Step to next module"))
        self.__tcp_step.Bind(wx.EVT_BUTTON, self.on_debug_step)
        sub_sizer.Add(self.__tcp_step, 1, wx.EXPAND)

        sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.__tcp_test_sizer.Add(sub_sizer, 1, wx.EXPAND)

        self.__tcp_stop_testmode = wx.lib.buttons.GenBitmapTextButton(
            panel, label="Exit Test Mode", bitmap=stop_bmp
        )
        self.__tcp_stop_testmode.SetToolTip(wx.ToolTip("Exit test mode"))
        self.__tcp_stop_testmode.Bind(wx.EVT_BUTTON, self.on_debug_stop)
        sub_sizer.Add(self.__tcp_stop_testmode, 1, wx.EXPAND)

        next_image_bmp = wx.Bitmap(cellprofiler.icons.get_builtin_image("IMG_IMAGE"))
        self.__tcp_next_imageset = wx.lib.buttons.GenBitmapTextButton(
            panel, label="Next Image Set", bitmap=next_image_bmp
        )
        self.__tcp_next_imageset.SetToolTip(wx.ToolTip("Jump to next image set"))
        self.__tcp_next_imageset.Bind(wx.EVT_BUTTON, self.on_debug_next_image_set)
        sub_sizer.Add(self.__tcp_next_imageset, 1, wx.EXPAND)

        self.show_launch_controls()

    def show_launch_controls(self):
        """Show the "Analyze images" and "Enter test mode" buttons"""
        self.__test_controls_panel.GetSizer().Hide(self.__tcp_test_sizer)
        self.__test_controls_panel.GetSizer().Hide(self.__tcp_analysis_sizer)
        self.__test_controls_panel.GetSizer().Show(self.__tcp_launch_sizer)
        self.__test_controls_panel.Layout()
        self.__test_controls_panel.GetParent().Layout()
        self.__frame.enable_launch_commands()
        self.__test_controls_panel.Update()

    def show_analysis_controls(self):
        """Show the controls that stop and pause analysis"""
        self.__test_controls_panel.GetSizer().Hide(self.__tcp_test_sizer)
        self.__test_controls_panel.GetSizer().Hide(self.__tcp_launch_sizer)
        self.__test_controls_panel.GetSizer().Show(self.__tcp_analysis_sizer)
        self.__stop_analysis_button.Enable()
        self.show_pause_button()
        self.__test_controls_panel.Layout()
        self.__test_controls_panel.GetParent().Layout()
        self.__frame.enable_analysis_commands()

    def show_pause_button(self):
        self.__pause_button.Enable()
        self.__tcp_analysis_sizer.Show(self.__pause_button)
        self.__tcp_analysis_sizer.Hide(self.__resume_button)
        self.__test_controls_panel.Layout()

    def show_resume_button(self):
        self.__resume_button.Enable()
        self.__tcp_analysis_sizer.Hide(self.__pause_button)
        self.__tcp_analysis_sizer.Show(self.__resume_button)
        self.__test_controls_panel.Layout()

    def show_test_controls(self):
        """Show the controls for dealing with test mode"""
        self.__test_controls_panel.GetSizer().Show(self.__tcp_test_sizer)
        self.__test_controls_panel.Layout()
        self.__test_controls_panel.GetSizer().Hide(self.__tcp_launch_sizer)
        self.__test_controls_panel.GetSizer().Hide(self.__tcp_analysis_sizer)
        self.__test_controls_panel.GetParent().Layout()
        self.__frame.enable_debug_commands()

    def omero_login(self):
        with cellprofiler.gui.omerologin.OmeroLoginDlg(
            self.__frame, title="Log into Omero"
        ) as dlg:
            dlg.ShowModal()

    def __on_open_workspace(self, event):
        """Handle the Open Workspace menu command"""
        path = self.do_open_workspace_dlg()
        if path is not None:
            self.do_open_workspace(path)

    def __on_revert_workspace(self, event):
        path = get_current_workspace_path()
        if path is not None:
            if (
                wx.MessageBox(
                    "Do you really want to revert all unsaved changes?",
                    "Revert to saved file",
                    wx.YES_NO | wx.ICON_QUESTION,
                    self.__frame,
                )
                == wx.YES
            ):
                self.do_open_workspace(path)

    def do_open_workspace_dlg(self):
        """Display the open workspace dialog, returning the chosen file

        returns a path or None if the user canceled. If it returns a path,
        the workspace file is locked.
        """
        wildcard = "CellProfiler project (%s)|%s|All files (*.*)|*.*" % (
            ",".join(["*.%s" % x for x in EXT_PROJECT_CHOICES]),
            ";".join(["*.%s" % x for x in EXT_PROJECT_CHOICES]),
        )
        with wx.FileDialog(
            self.__frame, "Choose a project file to open", wildcard=wildcard
        ) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                return dlg.Path
            return None

    def do_open_workspace(self, filename, load_pipeline=True):
        """Open the given workspace file

        filename - the path to the file to open. It should already be locked.
        """
        if not os.path.isfile(filename):
            wx.MessageBox(
                "Could not find project file: %s" % filename,
                caption="Error opening project file",
                parent=self.__frame,
                style=wx.OK | wx.ICON_ERROR,
            )
            return
        #
        # Meh, maybe the user loaded a pipeline file...
        #
        if not h5py.is_hdf5(filename):
            if cellprofiler.gui.pipeline.Pipeline.is_pipeline_txt_file(filename):
                message = (
                    'The file, "%s", is a pipeline file, not a project file. '
                    "Do you want to load it as a pipeline?"
                ) % os.path.split(filename)[-1]
                result = wx.MessageBox(
                    message,
                    "Cannot load as project file",
                    style=wx.YES | wx.NO | wx.YES_DEFAULT | wx.ICON_QUESTION,
                    parent=self.__frame,
                )
                if result == wx.YES:
                    self.do_load_pipeline(filename)
                return
            else:
                message = (
                    'CellProfiler cannot read the file, "%s", as a project '
                    "file. It may be damaged or corrupted or may not be in the "
                    ".cpproj format"
                ) % os.path.split(filename)[-1]
                wx.MessageBox(
                    message,
                    "Cannot read %s" % filename,
                    style=wx.OK | wx.ICON_ERROR,
                    parent=self.__frame,
                )
                return

        if self.is_running():
            # Defensive programming - the user shouldn't be able
            # to do this.
            wx.MessageBox(
                'The project file, "%s", cannot be loaded during analysis.\n\nPlease stop the analysis and try again.'
                % filename,
                caption="Cannot load project file",
                style=wx.OK | wx.ICON_INFORMATION,
                parent=self.__frame,
            )
            return
        self.exit_test_mode()
        progress_callback_fn = None
        message = "Loading %s" % filename
        with wx.ProgressDialog(
            parent=self.__frame,
            title="Opening project",
            message=message,
            style=wx.PD_CAN_ABORT | wx.PD_APP_MODAL,
        ) as dlg:
            try:
                assert isinstance(dlg, wx.ProgressDialog)
                dlg.longest_msg_len = dlg.GetFullTextExtent(message)[0]

                def progress_callback(operation_id, progress, message):
                    if progress not in (1, None):
                        proceed, skip = dlg.Pulse(message)
                        if not proceed:
                            wx.MessageBox("You cancelled opening this project.")
                            raise Exception("You cancelled opening this project.")
                        msg_len = dlg.GetTextExtent(message)[0]
                        if msg_len > dlg.longest_msg_len:
                            dlg.longest_msg_len = msg_len
                            dlg.Fit()

                add_progress_callback(progress_callback)
                progress_callback_fn = progress_callback

                self.__workspace.load(filename, load_pipeline)
                set_workspace_file(filename)
                set_current_workspace_path(filename)
                self.__pipeline.load_file_list(self.__workspace)
                self.__pipeline.turn_off_batch_mode()
                if not load_pipeline:
                    self.__workspace.measurements.clear()
                    self.__workspace.save_pipeline_to_measurements()
                self.on_update_pathlist()
                self.__dirty_workspace = False
                self.set_title()
                self.display_pipeline_message_for_user()
            except PipelineLoadCancelledException:
                # In response to user interaction, so pass
                self.__pipeline.clear()
            finally:
                remove_progress_callback(progress_callback_fn)
        # issue #1855 - apparently there is a WX bug in 3.0 that
        # disables buttons during the progress bar display (good),
        # re-enables the buttons after the progress bar is down (good)
        # but fails to repaint them, leaving them gray (bad).
        #
        # I tried "wx.CallAfter" but the progress bar UI is still up
        # even though the Python progress bar object has been destroyed.
        #
        wx.CallLater(250, self.repaint_after_progress_bar)

    def repaint_after_progress_bar(self):
        parent_stack = [self.__frame]
        while len(parent_stack) > 0:
            window = parent_stack.pop()
            for child in window.GetChildren():
                if child.IsShown():
                    child.Refresh(eraseBackground=True)
                    parent_stack.append(child)

    def display_pipeline_message_for_user(self):
        if self.__pipeline.message_for_user is not None:
            frame = wx.Frame(self.__frame, title=self.__pipeline.caption_for_user)
            frame.SetSizer(wx.BoxSizer(wx.VERTICAL))
            panel = wx.Panel(frame)
            frame.GetSizer().Add(panel, 1, wx.EXPAND)
            panel.SetSizer(wx.BoxSizer(wx.VERTICAL))
            subpanel = wx.Panel(panel)
            panel.GetSizer().Add(subpanel, 1, wx.EXPAND)
            subpanel.SetSizer(wx.BoxSizer(wx.VERTICAL))
            subpanel.GetSizer().AddSpacer(15)
            message_sizer = wx.BoxSizer(wx.HORIZONTAL)
            subpanel.GetSizer().Add(
                message_sizer, 1, wx.EXPAND | wx.RIGHT | wx.LEFT, 15
            )
            subpanel.GetSizer().AddSpacer(15)
            button_bar = wx.StdDialogButtonSizer()
            panel.GetSizer().Add(button_bar, 0, wx.EXPAND | wx.ALL, 5)

            info_bitmap = wx.ArtProvider.GetBitmap(
                wx.ART_INFORMATION, client=wx.ART_CMN_DIALOG
            )
            message_sizer.Add(
                wx.StaticBitmap(subpanel, bitmap=info_bitmap),
                0,
                wx.ALIGN_TOP | wx.ALIGN_LEFT,
            )
            message_sizer.AddSpacer(12)
            text = wx.StaticText(subpanel, label=self.__pipeline.message_for_user)
            message_sizer.Add(text, 0, wx.ALIGN_LEFT | wx.ALIGN_TOP)

            ok_button = wx.Button(panel, wx.ID_OK)
            button_bar.AddButton(ok_button)
            button_bar.Realize()
            ok_button.Bind(wx.EVT_BUTTON, lambda event: frame.Close())
            frame.Fit()
            frame.Show()

    def __on_new_workspace(self, event):
        """Handle the New Workspace menu command"""
        if self.__dirty_workspace:
            result = wx.MessageBox(
                "Do you want to save your existing project?",
                caption="Save project",
                style=wx.YES_NO | wx.CANCEL | wx.ICON_QUESTION,
                parent=self.__frame,
            )
            if result == wx.CANCEL:
                return
            elif result == wx.YES:
                path = get_current_workspace_path()
                if path is None:
                    if not self.do_save_as_workspace():
                        return
                else:
                    self.do_save_workspace(path)
        self.do_create_workspace()

    def do_create_workspace(self):
        """Create a new workspace file"""
        self.stop_debugging()
        if self.is_running():
            self.stop_running()
        self.__workspace.create()
        self.__pipeline.clear_urls()
        self.__pipeline.clear()
        self.__clear_errors()
        self.__workspace.measurements.clear()
        self.__workspace.save_pipeline_to_measurements()
        self.__dirty_workspace = False
        set_current_workspace_path(None)
        self.__pipeline_list_view.select_one_module(1)
        self.enable_module_controls_panel_buttons()
        self.set_title()
        self.__pipeline._Pipeline__undo_stack = []

    def __on_save_as_workspace(self, event):
        """Handle the Save Workspace As menu command"""
        self.do_save_as_workspace()

    def do_save_as_workspace(self):
        default_filename = get_current_workspace_path()
        if default_filename is None:
            default_filename = "project.%s" % EXT_PROJECT
            default_path = None
        else:
            default_path, default_filename = os.path.split(default_filename)
            default_filename = os.path.splitext(default_filename)[0] + "." + EXT_PROJECT

        wildcard = "CellProfiler project (*.%s)|*.%s" % (EXT_PROJECT, EXT_PROJECT,)
        with wx.FileDialog(
            self.__frame,
            "Save project file as",
            wildcard=wildcard,
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        ) as dlg:
            if default_path is not None:
                dlg.Path = os.path.join(default_path, default_filename)
            if dlg.ShowModal() == wx.ID_OK:
                pathname, filename = os.path.split(dlg.Path)
                fullname = dlg.Path
                dot_cpproj_ext = "." + EXT_PROJECT
                if sys.platform == "darwin" and not filename.endswith(dot_cpproj_ext):
                    fullname += dot_cpproj_ext
                self.do_save_workspace(fullname)
                set_current_workspace_path(fullname)
                set_workspace_file(fullname)
                self.set_title()
                return True
            return False

    def __on_save_workspace(self, event):
        """Handle the Save Project menu command"""
        path = get_current_workspace_path()
        if path is None:
            self.do_save_as_workspace()
        else:
            self.do_save_workspace(path)

    def do_save_workspace(self, filename):
        """Create a copy of the current workspace file"""
        self.__workspace.save(filename)
        set_workspace_file(filename)
        self.__dirty_workspace = False
        self.set_title()
        others = get_save_pipeline_with_project()
        if others in (SPP_PIPELINE_ONLY, SPP_PIPELINE_AND_FILE_LIST,):
            pipeline_path = os.path.splitext(filename)[0] + "." + EXT_PIPELINE
            self.__pipeline.save(pipeline_path, save_image_plane_details=False)
        if others in (SPP_FILE_LIST_ONLY, SPP_PIPELINE_AND_FILE_LIST,):
            filelist_path = os.path.splitext(filename)[0] + ".txt"
            self.do_export_text_file_list(filelist_path)

        return True

    def __on_load_pipeline(self, event):
        wildcard = "CellProfiler pipeline (%s)|%s" % (
            ",".join([".%s" % x for x in EXT_PIPELINE_CHOICES]),
            ";".join(["*.%s" % x for x in EXT_PIPELINE_CHOICES]),
        )
        dlg = wx.FileDialog(
            self.__frame, "Choose a pipeline file to open", wildcard=wildcard
        )
        if dlg.ShowModal() == wx.ID_OK:
            pathname = dlg.GetPath()
            self.do_load_pipeline(pathname)
        dlg.Destroy()

    def __on_url_load_pipeline(self, event):
        dlg = wx.TextEntryDialog(
            self.__frame,
            "Enter the pipeline's URL\n\n"
            "Example: https://svn.broadinstitute.org/"
            "CellProfiler/trunk/ExampleImages/"
            "ExampleSBSImages/ExampleSBS.cppipe",
            "Load pipeline via URL",
        )
        if dlg.ShowModal() == wx.ID_OK:
            import cellprofiler.misc

            url = cellprofiler.misc.generate_presigned_url(dlg.GetValue())
            filename, headers = urlretrieve(url)
            try:
                self.do_load_pipeline(filename)
            finally:
                os.remove(filename)
        dlg.Destroy()

    def __on_import_file_list(self, event):
        wildcard = (
            "CSV file (*.csv)|*.csv|" "Text file (*.txt)|*.txt|" "All files (*.*)|*.*"
        )
        with wx.FileDialog(self.__frame, "Import file list", wildcard=wildcard) as dlg:
            assert isinstance(dlg, wx.FileDialog)
            if dlg.ShowModal() == wx.ID_OK:
                if dlg.GetFilterIndex() == 0:
                    self.do_import_csv_file_list(dlg.GetPath())
                else:
                    self.do_import_text_file_list(dlg.GetPath())

    def do_import_csv_file_list(self, path):
        """Import path names from a CSV file

        path - path to the CSV file

        The CSV file should have no header. Each field in the CSV file
        is treated as a path. An example:
        "/images/A01_w1.tif","/images/A01_w2.tif"
        "/images/A02_w1.tif","/images/A02_w2.tif"
        """
        with open(path, mode="r") as fd:
            rdr = csv.reader(fd)
            pathnames = sum(rdr, [])
            self.__pipeline.add_pathnames_to_file_list(pathnames)

    def do_import_text_file_list(self, path):
        """Import path names from a text file

        path - path to the text file

        Each line in the text file is treated as a path. Whitespace at the
        start or end of the line is stripped. An example:
        /images/A01_w1.tif
        /images/A01_w2.tif
        /images/A02_w1.tif
        /images/A02_w2.tif

        If your file name has line feeds in it or whitespace at the start
        or end of the line, maybe you're asking too much :-)
        """
        with open(path) as fd:
            pathnames = [p.strip() for p in fd]
            self.__pipeline.add_pathnames_to_file_list(pathnames)

    def do_export_text_file_list(self, path):
        """Export pathnames to a text file

        path - path to the text file.

        The output is in the same format as for do_import_text_file_list
        """
        with open(path, mode="w") as fd:
            for url in self.__workspace.file_list.get_filelist():
                if isinstance(url, bytes):
                    url = url.decode()
                fd.write(url + "\n")

    def is_running(self):
        return self.__analysis is not None

    def do_load_pipeline(self, pathname):
        if not os.path.isfile(pathname):
            wx.MessageBox(
                "Could not find pipeline file: %s." % pathname,
                caption="Error loading pipeline file",
                parent=self.__frame,
                style=wx.OK | wx.ICON_ERROR,
            )
            return
        try:
            if self.__pipeline.test_mode:
                self.stop_debugging()
            if self.is_running():
                self.stop_running()

            if h5py.is_hdf5(pathname):
                if not self.load_hdf5_pipeline(pathname):
                    return
            else:
                self.__pipeline.load(pathname)
            self.__pipeline.turn_off_batch_mode()
            self.__clear_errors()
            self.__workspace.save_pipeline_to_measurements()
            self.display_pipeline_message_for_user()
            target_project_path = os.path.splitext(pathname)[0] + "." + EXT_PROJECT
            if (
                not os.path.exists(target_project_path)
                and get_current_workspace_path() is None
            ):
                set_current_workspace_path(target_project_path)
                self.set_title()

        except PipelineLoadCancelledException:
            self.__pipeline.clear()
        except Exception as instance:
            error = cellprofiler.gui.dialog.Error("Error", str(instance))

            if error.status is wx.ID_CANCEL:
                cancel_progress()

    def load_hdf5_pipeline(self, pathname):
        """Load a pipeline from an HDF5 measurements file or similar

        pathname - pathname to the file
        """
        assert h5py.is_hdf5(pathname)
        m = Measurements(filename=pathname, mode="r")
        has_user_pipeline = m.has_feature(EXPERIMENT, M_USER_PIPELINE,)
        has_pipeline = m.has_feature(EXPERIMENT, M_PIPELINE,)
        if has_user_pipeline:
            if has_pipeline:
                with wx.Dialog(self.__frame, title="Choose pipeline to open") as dlg:
                    dlg.Sizer = wx.BoxSizer(wx.VERTICAL)
                    sizer = wx.BoxSizer(wx.VERTICAL)
                    dlg.Sizer.Add(sizer, 1, wx.EXPAND | wx.ALL, 10)
                    message = (
                        "%s contains two pipelines, the primary pipeline and\n"
                        "the user pipeline. The primary pipeline is the one\n"
                        "that should be used for batch processing, but it\n"
                        "may have been modified to make it suitable for that\n"
                        "purpose. The user pipeline is the pipeline as\n"
                        "displayed by CellProfiler and is more suitable for\n"
                        "editing and running in test mode.\n\n"
                        "Do you want to open the primary or user pipeline?"
                    ) % os.path.split(pathname)[1]
                    sizer.Add(wx.StaticText(dlg, label=message), 0, wx.EXPAND)
                    sizer.AddSpacer(4)
                    gb_sizer = wx.BoxSizer(wx.VERTICAL)
                    sizer.Add(gb_sizer, 1, wx.EXPAND)
                    rb_primary = wx.RadioButton(dlg, label="&Primary pipeline")
                    gb_sizer.Add(rb_primary, 0, wx.ALIGN_LEFT)
                    gb_sizer.AddSpacer(2)
                    rb_user = wx.RadioButton(dlg, label="&User pipeline")
                    gb_sizer.Add(rb_user, 0, wx.ALIGN_LEFT)
                    rb_user.SetValue(True)

                    btn_sizer = wx.StdDialogButtonSizer()
                    dlg.Sizer.Add(btn_sizer, 0, wx.EXPAND)
                    btn_sizer.AddButton(wx.Button(dlg, wx.ID_OK))
                    btn_sizer.AddButton(wx.Button(dlg, wx.ID_CANCEL))
                    btn_sizer.Realize()
                    dlg.Fit()
                    if dlg.ShowModal() == wx.ID_OK:
                        ftr = M_USER_PIPELINE if rb_user.GetValue() else M_PIPELINE
                    else:
                        return False
            else:
                ftr = M_USER_PIPELINE
        else:
            ftr = M_PIPELINE
        pipeline_text = m.get_experiment_measurement(ftr)
        pipeline_text = pipeline_text
        self.__pipeline.load(io.StringIO(pipeline_text))
        return True

    def __clear_errors(self):
        for key, error in list(self.__setting_errors.items()):
            self.__frame.preferences_view.pop_error_text(error)
        self.__setting_errors = {}

    def __on_save_as_pipeline(self, event):
        try:
            self.do_save_pipeline()
        except Exception as e:
            wx.MessageBox(
                "Exception:\n%s" % e,
                "Could not save pipeline...",
                wx.ICON_ERROR | wx.OK,
                self.__frame,
            )

    def do_save_pipeline(self):
        """Save the pipeline, asking the user for the name

        return True if the user saved the pipeline
        """
        default_filename = get_current_workspace_path()
        if default_filename is None:
            default_filename = "pipeline.%s" % EXT_PIPELINE
            default_path = None
        else:
            default_path, default_filename = os.path.split(default_filename)
            default_filename = (
                os.path.splitext(default_filename)[0] + "." + EXT_PIPELINE
            )
        wildcard = (
            "CellProfiler pipeline (*.%s)|*.%s|"
            "CellProfiler pipeline and file list (*.%s)|*.%s|"
            "JSON object (*.json)|*.json"
        ) % (EXT_PIPELINE, EXT_PIPELINE, EXT_PIPELINE, EXT_PIPELINE)
        with wx.FileDialog(
            self.__frame,
            "Save pipeline",
            wildcard=wildcard,
            defaultFile=default_filename,
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        ) as dlg:
            assert isinstance(dlg, wx.FileDialog)
            if default_path is not None:
                dlg.SetPath(os.path.join(default_path, default_filename))
            if dlg.ShowModal() == wx.ID_OK:
                save_image_plane_details = dlg.GetFilterIndex() == 1
                file_name = dlg.GetFilename()
                pathname = dlg.GetPath()
                if not sys.platform.startswith("win"):
                    dot_ext_pipeline = "." + EXT_PIPELINE
                    if not file_name.endswith(dot_ext_pipeline) and not file_name.endswith(".json"):
                        # on platforms other than Windows, add the default suffix, unless user specified json export
                        pathname += dot_ext_pipeline
                self.__pipeline.save(
                    pathname, save_image_plane_details=save_image_plane_details
                )
                return True
            return False

    def __on_export_image_sets(self, event):
        """Export the pipeline's image sets to a .csv file"""
        if len(self.__pipeline.file_list) == 0:
            wx.MessageBox(
                "Unable to export image sets, file list is empty.",
                "Export Image Sets",
                wx.OK | wx.ICON_INFORMATION,
                self.__frame,
            )
            return

        dlg = wx.FileDialog(
            self.__frame,
            "Export image sets",
            wildcard="Image set file (*.csv)|*.csv",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        )
        try:
            dialog_response = dlg.ShowModal()
            path = dlg.GetPath()
        finally:
            dlg.Destroy()

        if dialog_response == wx.ID_OK:
            try:
                self.__workspace.refresh_image_set()
                self.__workspace.measurements.write_image_sets(path)

                # Show helpful message to guide in proper use (GithHub issue #688)
                frame = wx.Frame(self.__frame, title="Image set listing saved")
                frame.SetSizer(wx.BoxSizer(wx.VERTICAL))
                panel = wx.Panel(frame)
                frame.GetSizer().Add(panel, 1, wx.EXPAND)
                panel.SetSizer(wx.BoxSizer(wx.VERTICAL))
                subpanel = wx.Panel(panel)
                panel.GetSizer().Add(subpanel, 1, wx.EXPAND)
                subpanel.SetSizer(wx.BoxSizer(wx.VERTICAL))
                subpanel.GetSizer().AddSpacer(15)
                message_sizer = wx.BoxSizer(wx.HORIZONTAL)
                subpanel.GetSizer().Add(
                    message_sizer, 1, wx.EXPAND | wx.RIGHT | wx.LEFT, 15
                )
                subpanel.GetSizer().AddSpacer(15)
                button_bar = wx.StdDialogButtonSizer()
                panel.GetSizer().Add(button_bar, 0, wx.EXPAND | wx.ALL, 5)

                info_bitmap = wx.ArtProvider.GetBitmap(
                    wx.ART_INFORMATION, client=wx.ART_CMN_DIALOG
                )
                message_sizer.Add(
                    wx.StaticBitmap(subpanel, bitmap=info_bitmap),
                    0,
                    wx.ALIGN_TOP | wx.ALIGN_LEFT,
                )
                message_sizer.AddSpacer(12)
                help_text = (
                    "Your image set listing has been saved as a comma-delimited file (CSV). This file can be loaded \n"
                    "into CellProfiler using the LoadData module (located in the File Processing category). In the\n"
                    "module, specify the CSV in the input data file location, and set the base image location to 'None'.\n"
                    "\n"
                    "If you are running CellProfiler from the command line without the UI (i.e., 'headless'), you can use\n"
                    "the '--data-file' switch to use an alternate CSV file as input to LoadData rather than the one specified\n"
                    "in the LoadData module itself.\n"
                )
                text = wx.StaticText(subpanel, label=help_text)
                message_sizer.Add(text, 0, wx.ALIGN_LEFT | wx.ALIGN_TOP)

                ok_button = wx.Button(panel, wx.ID_OK)
                button_bar.AddButton(ok_button)
                button_bar.Realize()
                ok_button.Bind(wx.EVT_BUTTON, lambda event: frame.Close())
                frame.Fit()
                frame.Show()
            except Exception as e:
                error = cellprofiler.gui.dialog.Error("Error", str(e))
                if error.status is wx.ID_CANCEL:
                    cancel_progress()

    def __on_export_pipeline_notes(self, event):
        default_filename = get_current_workspace_path()
        if default_filename is None:
            default_filename = "pipeline.txt"
            default_path = None
        else:
            default_path, default_filename = os.path.split(default_filename)
            default_filename = os.path.splitext(default_filename)[0] + ".txt"

        with wx.FileDialog(
            self.__frame,
            "Export pipeline notes",
            defaultFile=default_filename,
            wildcard="Text file (*.txt)|*.txt",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        ) as dlg:
            if default_path is not None:
                dlg.Directory = default_path
            if dlg.ShowModal() == wx.ID_OK:
                with open(dlg.Path, "w") as fd:
                    self.__workspace.pipeline.save_pipeline_notes(fd)

    def __on_export_pipeline_citations(self, event):
        default_filename = get_current_workspace_path()
        if default_filename is None:
            default_filename = "citations.txt"
            default_path = None
        else:
            default_path, default_filename = os.path.split(default_filename)
            default_filename = os.path.splitext(default_filename)[0] + ".txt"
        with wx.FileDialog(
            self.__frame,
            "Export pipeline citations",
            defaultFile=default_filename,
            wildcard="Text file (*.txt)|*.txt",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        ) as dlg:
            if default_path is not None:
                dlg.Directory = default_path
            if dlg.ShowModal() == wx.ID_OK:
                with open(dlg.Path, "w", encoding='utf-8') as fd:
                    self.__workspace.pipeline.save_pipeline_citations(fd)

    def __on_plateviewer(self, event):
        import cellprofiler.gui.plateviewer as pv

        data = pv.PlateData()
        try:
            self.__workspace.refresh_image_set()
        except Exception as instance:
            extended_message = "Failed to make image sets"

            error = cellprofiler.gui.dialog.Error("Error", extended_message)

            if error.status is wx.ID_CANCEL:
                cancel_progress()

            return
        m = self.__workspace.measurements
        assert isinstance(m, Measurements)

        image_numbers = m.get_image_numbers()
        if len(image_numbers) == 0:
            self.display_plate_viewer_help(
                "Your project does not produce any image sets.\n"
                "Please configure the input modules correctly.",
                "Plate viewer: No image sets",
            )
            return
        url_features = [f for f in m.get_feature_names(IMAGE) if f.startswith(C_URL)]
        pws = []
        for feature in ("Plate", "Well", "Site"):
            measurement = C_METADATA + "_" + feature
            if m.has_feature(IMAGE, measurement):
                pws.append(m.get_measurement(IMAGE, measurement, image_numbers))
            else:
                pws.append([None] * len(image_numbers))
        plate, well, site = pws
        if pws[1][0] is None:
            self.display_plate_viewer_help(
                "Your project needs to tag every image set with well metadata\n"
                "Please use the Metadata module to define a metadata tag\n"
                "named, "
                "Well"
                ".",
                "Plate viewer: No well metadata",
            )
            return

        for url_feature in url_features:
            channel = [url_feature[(len(C_URL) + 1) :]] * len(image_numbers)
            urls = m.get_measurement(IMAGE, url_feature, image_numbers)
            data.add_files(
                [url for url in urls], plate, well, site, channel_names=channel
            )
        if self.__plate_viewer is not None:
            self.__pv_frame.Destroy()
        self.__pv_frame = wx.Frame(self.__frame, title="Plate viewer")
        self.__plate_viewer = pv.PlateViewer(self.__pv_frame, data)
        if not self.__plate_viewer.error:
            self.__pv_frame.Fit()
            self.__pv_frame.Show()

    def display_plate_viewer_help(self, message, caption):
        """Display a helpful dialog for a plate viewer config error

        message - message to display

        caption - caption on frame bar
        """
        message += "\n\nPress " "Help" " for the plate viewer manual page."
        with wx.Dialog(self.__frame, title=caption) as dlg:
            assert isinstance(dlg, wx.Dialog)
            dlg.SetSizer(wx.BoxSizer(wx.VERTICAL))
            message_sizer = wx.BoxSizer(wx.HORIZONTAL)
            dlg.GetSizer().Add(message_sizer, 0, wx.EXPAND | wx.ALL, 15)
            bmpInfo = wx.ArtProvider.GetBitmap(wx.ART_INFORMATION, wx.ART_CMN_DIALOG)
            message_sizer.Add(
                wx.StaticBitmap(dlg, bitmap=bmpInfo), 0, wx.ALIGN_TOP | wx.ALIGN_LEFT
            )
            message_sizer.AddSpacer(12)
            message_sizer.Add(
                wx.StaticText(dlg, label=message), 0, wx.ALIGN_TOP | wx.ALIGN_LEFT
            )
            button_sizer = wx.StdDialogButtonSizer()
            dlg.GetSizer().Add(button_sizer, 0, wx.EXPAND | wx.ALL, 8)
            ok_button = wx.Button(dlg, wx.ID_OK)
            help_button = wx.Button(dlg, wx.ID_HELP)
            button_sizer.AddButton(ok_button)
            button_sizer.AddButton(help_button)
            button_sizer.Realize()

            def do_ok(event):
                dlg.EndModal(0)

            def do_help(event):
                HTMLDialog(
                    self.__frame,
                    "Help for plate viewer",
                    cellprofiler.gui.html.utils.rst_to_html_fragment(
                        cellprofiler.gui.help.content.read_content(
                            "output_plateviewer.rst"
                        )
                    ),
                ).Show()

            ok_button.Bind(wx.EVT_BUTTON, do_ok)
            help_button.Bind(wx.EVT_BUTTON, do_help)
            dlg.Fit()
            dlg.ShowModal()

    def populate_recent_files(self):
        """Populate the recent files menu"""
        for menu, ids, file_names, fn in (
            (
                self.__frame.recent_pipeline_files,
                RECENT_PIPELINE_FILE_MENU_ID,
                get_recent_files(),
                self.do_load_pipeline,
            ),
            (
                self.__frame.recent_workspace_files,
                RECENT_WORKSPACE_FILE_MENU_ID,
                get_recent_files(WORKSPACE_FILE),
                self.do_open_workspace,
            ),
        ):
            assert isinstance(menu, wx.Menu)
            while len(menu.GetMenuItems()) > 0:
                self.__frame.Unbind(wx.EVT_MENU, id=menu.GetMenuItems()[0].Id)
                menu.Remove(menu.GetMenuItems()[0])
            for index, file_name in enumerate(file_names):
                menu.Append(ids[index], file_name)
                self.__frame.Bind(
                    wx.EVT_MENU,
                    lambda event, file_name=file_name, fn=fn: fn(file_name),
                    id=ids[index],
                )

    def set_title(self):
        """Set the title of the parent frame"""
        pathname = get_current_workspace_path()
        if pathname is None:
            self.__frame.Title = "CellProfiler %s" % cellprofiler_version
            return
        path, filename = os.path.split(pathname)
        if self.__dirty_workspace:
            self.__frame.Title = "CellProfiler %s: %s* (%s)" % (
                cellprofiler_version,
                filename,
                path,
            )
        else:
            self.__frame.Title = "CellProfiler %s: %s (%s)" % (
                cellprofiler_version,
                filename,
                path,
            )

    def __on_clear_pipeline(self, event):
        if (
            wx.MessageBox(
                "Do you really want to remove all modules from the pipeline?",
                "Clearing pipeline",
                wx.YES_NO | wx.ICON_QUESTION,
                self.__frame,
            )
            == wx.YES
        ):
            self.stop_debugging()
            if self.is_running():
                self.stop_running()
            self.__pipeline.clear()
            self.__clear_errors()
            self.__pipeline_list_view.select_one_module(1)
            self.enable_module_controls_panel_buttons()

    def check_close(self):
        """Return True if we are allowed to close

        Check for pipeline dirty, return false if user doesn't want to close
        """
        if self.__dirty_workspace:
            #
            # Create a dialog box asking the user what to do.
            #
            dialog = wx.Dialog(self.__frame, title="Closing CellProfiler")
            super_sizer = wx.BoxSizer(wx.VERTICAL)
            dialog.SetSizer(super_sizer)
            #
            # This is the main window with the icon and question
            #
            sizer = wx.BoxSizer(wx.HORIZONTAL)
            super_sizer.Add(sizer, 1, wx.EXPAND | wx.ALL, 5)
            question_mark = wx.ArtProvider.GetBitmap(wx.ART_HELP, wx.ART_MESSAGE_BOX)
            icon = wx.StaticBitmap(dialog, -1, question_mark)
            sizer.Add(icon, 0, wx.EXPAND | wx.ALL, 5)
            text = wx.StaticText(dialog, label="Do you want to save your project?")
            sizer.Add(text, 0, wx.EXPAND | wx.ALL, 5)
            super_sizer.Add(
                wx.StaticLine(dialog), 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 20
            )
            #
            # These are the buttons
            #
            button_sizer = wx.BoxSizer(wx.HORIZONTAL)
            super_sizer.Add(button_sizer, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALL, 5)
            SAVE_ID = wx.NewId()
            DONT_SAVE_ID = wx.NewId()
            RETURN_TO_CP_ID = wx.NewId()
            answer = [RETURN_TO_CP_ID]
            for button_id, text, set_default in (
                (SAVE_ID, "Save", True),
                (RETURN_TO_CP_ID, "Return to CellProfiler", False),
                (DONT_SAVE_ID, "Don't Save", False),
            ):
                button = wx.Button(dialog, button_id, text)
                if set_default:
                    button.SetDefault()
                button_sizer.Add(button, 0, wx.EXPAND | wx.ALL, 5)

                def on_button(event, button_id=button_id):
                    dialog.SetReturnCode(button_id)
                    answer[0] = button_id
                    dialog.Close()

                dialog.Bind(wx.EVT_BUTTON, on_button, button, button_id)
            dialog.Fit()
            dialog.CentreOnParent()
            try:
                dialog.ShowModal()
                if answer[0] == SAVE_ID:
                    workspace_path = get_current_workspace_path()
                    if workspace_path is None:
                        return self.do_save_as_workspace()
                    if not self.do_save_workspace(workspace_path):
                        # Cancel the closing if the user fails to save
                        return False
                elif answer[0] == RETURN_TO_CP_ID:
                    return False
            finally:
                dialog.Destroy()
        return True

    def on_close(self):
        self.close_debug_measurements()
        if self.is_running():
            self.stop_running()
        self.__workspace.close()

    def __on_pipeline_event(self, caller, event):
        if not wx.IsMainThread():
            wx.CallAfter(self.__on_pipeline_event, caller, event)
        if isinstance(event, RunException):
            error_msg = None
            self.__pipeline_list_view.select_one_module(event.module.module_num)
            try:
                import MySQLdb

                if (
                    isinstance(event.error, MySQLdb.OperationalError)
                    and len(event.error.args) > 1
                ):
                    #
                    # The informative error is in args[1] for MySQL
                    #
                    error_msg = event.error.args[1]
            except:
                pass
            if error_msg is None:
                if isinstance(event.error, EnvironmentError) and event.error.strerror is not None:
                    error_msg = event.error.strerror
                else:
                    error_msg = str(event.error)
            if isinstance(event, PrepareRunException):
                message = (
                    "Encountered unrecoverable error in %s during startup:\n%s"
                    % (event.module.module_name, error_msg)
                )
                continue_only = True
            elif isinstance(event, PostRunException):
                message = (
                    "Encountered uncrecoverable error in %s during post-processing:\n%s"
                    % (event.module.module_name, error_msg)
                )
                continue_only = True
            else:
                message = (
                    "Error while processing %s:\n"
                    "%s\n\nDo you want to stop processing?"
                ) % (event.module.module_name, error_msg)
                continue_only = False

            error = cellprofiler.gui.dialog.Error("Error", message)

            if error.status is wx.ID_CANCEL:
                cancel_progress()

        elif isinstance(event, LoadException):
            self.on_load_exception_event(event)
        elif isinstance(event, URLsAdded):
            self.on_urls_added(event)
        elif isinstance(event, URLsRemoved):
            self.on_urls_removed(event)
        elif isinstance(event, URLsCleared):
            self.on_urls_cleared(event)
        elif event.is_pipeline_modification:
            self.__dirty_workspace = True
            self.set_title()
            needs_default_image_folder = self.__pipeline.needs_default_image_folder()
            self.__frame.get_preferences_view().show_default_image_folder(
                needs_default_image_folder
            )
            if event.is_image_set_modification:
                self.on_image_set_modification()
            self.__workspace.save_pipeline_to_measurements()
            if isinstance(
                event,
                (
                    ModuleAdded,
                    ModuleMoved,
                    ModuleRemoved,
                    PipelineCleared,
                    PipelineLoaded,
                ),
            ):
                self.populate_goto_menu()

    def on_image_set_modification(self):
        self.__workspace.invalidate_image_set()
        self.exit_test_mode()

    def __on_image_directory_change(self, event):
        self.on_image_set_modification()

    def __on_output_directory_change(self, event):
        self.on_image_set_modification()

    def on_workspace_event(self, event):
        """Workspace's file list changed. Invalidate the workspace cache."""
        if isinstance(
            event,
            cellprofiler.gui._workspace_model.Workspace.WorkspaceFileListNotification,
        ):
            self.on_image_set_modification()
            self.__dirty_workspace = True

    @staticmethod
    def on_load_exception_event(event):
        """Handle a pipeline load exception"""
        if event.module is None:
            module_name = event.module_name
        else:
            module_name = event.module.module_name
        if event.settings is None or len(event.settings) == 0:
            message = "Error while loading %s: %s\nDo you want to stop processing?" % (
                module_name,
                str(event.error),
            )
        else:
            message = (
                "Error while loading %s: %s\n"
                "Do you want to stop processing?\n\n"
                "Module settings:\n"
                "\t%s"
            ) % (module_name, str(event.error), "\n\t".join(event.settings))
        error = cellprofiler.gui.dialog.Error("Error", message)

        if error.status is wx.ID_CANCEL:
            cancel_progress()

            event.cancel_run = False

    def on_urls_added(self, event):
        """Callback from pipeline when paths are added to the pipeline"""
        urls = event.urls
        self.__workspace.file_list.add_files_to_filelist(urls)
        urlset = set(urls)
        file_objects = [image_file for image_file in self.__pipeline.file_list if image_file.url in urlset]
        self.__path_list_ctrl.add_files(file_objects)
        self.__pipeline_list_view.notify_has_file_list(
            len(self.__pipeline.file_list) > 0
        )
        self.exit_test_mode()

    def on_urls_removed(self, event):
        """Callback from pipeline when paths are removed from the pipeline"""
        urls = event.urls
        if isinstance(urls[0], ImageFile):
            urls = [imfile.url for imfile in urls]
        self.__path_list_ctrl.remove_files(urls)
        self.__workspace.file_list.remove_files_from_filelist(urls)
        self.__pipeline_list_view.notify_has_file_list(
            len(self.__pipeline.file_list) > 0
        )
        self.exit_test_mode()

    def on_urls_cleared(self, event):
        """Callback from pipeline when all paths are cleared from the pipeline"""
        self.__path_list_ctrl.clear_files()
        self.__workspace.file_list.clear_filelist()
        self.__pipeline_list_view.notify_has_file_list(False)
        self.exit_test_mode()

    def on_update_pathlist(self, event=None):
        enabled_urls = set([f.url for f in self.__pipeline.get_filtered_file_list(self.__workspace)])
        disabled_urls = set(self.__path_list_ctrl.get_paths())
        disabled_urls.difference_update(enabled_urls)
        self.__path_list_ctrl.enable_paths(enabled_urls, True)
        self.__path_list_ctrl.enable_paths(disabled_urls, False)

    def on_extract_metadata(self, event=None):
        if self.__path_list_ctrl.metadata_already_extracted:
            return
        file_objects = self.__pipeline.get_filtered_file_list(self.__workspace)
        cap = len(file_objects)

        if cap == 0:
            LOGGER.info("Metadata extraction not completed, no file objects available")
            return

        with wx.ProgressDialog(
            parent=self.__frame,
            title="Extracting metadata",
            message="Extracting metadata from file headers...",
            maximum=cap,
            style=wx.PD_CAN_ABORT | wx.PD_APP_MODAL,
        ) as dlg:
            urls = []
            for idx, file_object in enumerate(file_objects):
                if dlg.WasCancelled():
                    break
                if not file_object.extracted:
                    dlg.Update(idx, f"Working on {file_object.filename}")
                    file_object.extract_planes(self.__workspace)
                    urls.append(file_object.url)
            self.__path_list_ctrl.update_metadata(urls)
            self.__path_list_ctrl.set_metadata_extracted(not dlg.WasCancelled())

    def on_update_pathlist_ui(self, event):
        """Called with an UpdateUIEvent for a pathlist command ID"""
        assert isinstance(event, wx.UpdateUIEvent)
        event.Enable(True)
        if not self.__path_list_ctrl.IsShownOnScreen():
            event.Enable(False)
        elif event.GetId() == cellprofiler.gui.cpframe.ID_EDIT_REMOVE_FROM_FILE_LIST:
            if not self.__path_list_ctrl.has_selections():
                event.Enable(False)
        elif event.GetId() == cellprofiler.gui.cpframe.ID_EDIT_SHOW_FILE_LIST_IMAGE:
            if not self.__path_list_ctrl.has_focus_item():
                event.Enable(False)

    def on_pathlist_browse_folder(self, event, default_dir=wx.EmptyString):
        """Handle request for browsing for pathlist folder"""
        with wx.DirDialog(self.__path_list_ctrl, "Select image folder") as dlg:
            assert isinstance(dlg, wx.DirDialog)
            if dlg.ShowModal() == wx.ID_OK:
                path = dlg.GetPath()
                self.add_paths_to_pathlist([path])

    def on_pathlist_browse_files(self, event, default_dir=wx.EmptyString):
        """Handle request for browsing for pathlist files"""
        with wx.FileDialog(
            self.__path_list_ctrl,
            "Select image files",
            defaultDir=default_dir,
            wildcard=(
                "Image files (*.tif,*.tiff,*.png,*.jpg,*.gif,*.jpg)|"
                "*.tif;*.tiff;*.jpg;*.jpeg;*.png;*.gif;*.bmp|"
                "All files (*.*)|*.*"
            ),
            style=wx.FD_DEFAULT_STYLE | wx.FD_MULTIPLE | wx.FD_OPEN,
        ) as dlg:
            assert isinstance(dlg, wx.FileDialog)
            if dlg.ShowModal() == wx.ID_OK:
                paths = dlg.GetPaths()
                self.add_paths_to_pathlist(paths)

    PATHLIST_CMD_SHOW = "Show Selected Image"
    PATHLIST_CMD_BROWSE_FILES = "Browse For Images"
    PATHLIST_CMD_BROWSE_FOLDER = "Browse For Folder"
    PATHLIST_CMD_REMOVE = "Remove From File List"
    PATHLIST_CMD_REFRESH = "Refresh File List"
    PATHLIST_TEXT_REFRESH = "Remove Unavailable Files"
    PATHLIST_CMD_EXPAND_ALL = "Expand All Folders"
    PATHLIST_CMD_COLLAPSE_ALL = "Collapse All Folders"
    PATHLIST_CMD_CLEAR = "Clear File List"

    def get_pathlist_file_context_menu(self, paths):
        return (
            (self.PATHLIST_CMD_SHOW, self.PATHLIST_CMD_SHOW),
            (self.PATHLIST_CMD_REMOVE, self.PATHLIST_CMD_REMOVE),
            (self.PATHLIST_CMD_REFRESH, self.PATHLIST_TEXT_REFRESH),
            (self.PATHLIST_CMD_BROWSE_FILES, self.PATHLIST_CMD_BROWSE_FILES),
            (self.PATHLIST_CMD_BROWSE_FOLDER, self.PATHLIST_CMD_BROWSE_FOLDER),
            (self.PATHLIST_CMD_EXPAND_ALL, self.PATHLIST_CMD_EXPAND_ALL),
            (self.PATHLIST_CMD_COLLAPSE_ALL, self.PATHLIST_CMD_COLLAPSE_ALL),
            (self.PATHLIST_CMD_CLEAR, self.PATHLIST_CMD_CLEAR),
        )

    def on_pathlist_file_command(self, paths, cmd):
        if cmd == self.PATHLIST_CMD_SHOW or cmd is None:
            if len(paths) == 0:
                self.on_pathlist_browse_files(None)
                return
            self.on_pathlist_show()
        elif cmd == self.PATHLIST_CMD_REMOVE:
            self.on_pathlist_file_delete(paths)
        elif cmd == self.PATHLIST_CMD_REFRESH:
            self.on_pathlist_refresh(paths)
        elif cmd in [self.PATHLIST_CMD_BROWSE_FILES, self.PATHLIST_CMD_BROWSE_FOLDER]:
            if cmd == self.PATHLIST_CMD_BROWSE_FOLDER:
                browse_function = self.on_pathlist_browse_folder
            else:
                browse_function = self.on_pathlist_browse_files
            if len(paths) == 0 or not paths[0].startswith("file:"):
                browse_function(None)
            else:
                path = url2pathname(paths[0][5:])
                path = os.path.split(path)[0]
                browse_function(None, default_dir=path)
        else:
            self.on_pathlist_command(cmd)

    def on_pathlist_command(self, cmd):
        if cmd == self.PATHLIST_CMD_EXPAND_ALL:
            self.on_pathlist_expand_all()
        elif cmd == self.PATHLIST_CMD_COLLAPSE_ALL:
            self.on_pathlist_collapse_all()
        elif cmd == self.PATHLIST_CMD_CLEAR:
            self.on_pathlist_clear(None)

    def get_pathlist_folder_context_menu(self, path):
        return (
            (self.PATHLIST_CMD_REMOVE, self.PATHLIST_CMD_REMOVE),
            (self.PATHLIST_CMD_REFRESH, self.PATHLIST_TEXT_REFRESH),
            (self.PATHLIST_CMD_BROWSE_FILES, self.PATHLIST_CMD_BROWSE_FILES),
            (self.PATHLIST_CMD_BROWSE_FOLDER, self.PATHLIST_CMD_BROWSE_FOLDER),
            (self.PATHLIST_CMD_EXPAND_ALL, self.PATHLIST_CMD_EXPAND_ALL),
            (self.PATHLIST_CMD_COLLAPSE_ALL, self.PATHLIST_CMD_COLLAPSE_ALL),
            (self.PATHLIST_CMD_CLEAR, self.PATHLIST_CMD_CLEAR),
        )

    def on_pathlist_folder_command(self, path, cmd):
        if cmd == self.PATHLIST_CMD_REMOVE:
            paths = self.__path_list_ctrl.get_folder(
                path, self.__path_list_ctrl.FLAG_RECURSE
            )
            self.on_pathlist_file_delete(paths)
        elif cmd == self.PATHLIST_CMD_REFRESH:
            paths = self.__path_list_ctrl.get_folder(
                path, self.__path_list_ctrl.FLAG_RECURSE
            )
            self.on_pathlist_refresh(paths)
        elif cmd in [self.PATHLIST_CMD_BROWSE_FILES, self.PATHLIST_CMD_BROWSE_FOLDER]:
            if cmd == self.PATHLIST_CMD_BROWSE_FOLDER:
                browse_function = self.on_pathlist_browse_folder
            else:
                browse_function = self.on_pathlist_browse_files
            if path.startswith("file:"):
                path = url2pathname(path[5:])
                browse_function(None, default_dir=path)
            else:
                browse_function(None)
        else:
            self.on_pathlist_command(cmd)

    def get_pathlist_empty_context_menu(self, path):
        return (
            (self.PATHLIST_CMD_BROWSE_FILES, self.PATHLIST_CMD_BROWSE_FILES),
            (self.PATHLIST_CMD_BROWSE_FOLDER, self.PATHLIST_CMD_BROWSE_FOLDER),
        )

    def on_pathlist_empty_command(self, path, cmd):
        if cmd == self.PATHLIST_CMD_BROWSE_FILES:
            self.on_pathlist_browse_files(None)
        if cmd == self.PATHLIST_CMD_BROWSE_FOLDER:
            self.on_pathlist_browse_folder(None)

    def on_pathlist_expand_all(self, event=None):
        self.__path_list_ctrl.expand_all()

    def on_pathlist_collapse_all(self, event=None):
        self.__path_list_ctrl.collapse_all()

    def on_pathlist_remove(self, event=None):
        """Remove selected files from the path list"""
        paths = self.__path_list_ctrl.get_paths(
            self.__path_list_ctrl.FLAG_SELECTED_ONLY
        )
        self.on_pathlist_file_delete(paths)

    def on_pathlist_show(self, event=None):
        """Show the focused item's image"""
        from .utilities.figure import show_image
        from cellprofiler_core.utilities.pathname import url2pathname

        paths = self.__path_list_ctrl.get_paths(
            self.__path_list_ctrl.FLAG_FOCUS_ITEM_ONLY
        )
        if len(paths) == 0:
            wx.MessageBox(
                "No image selected.", caption="No image selected", parent=self.__frame
            )
            return
        path = url2pathname(paths[0])
        ext = os.path.splitext(path)[1]
        if len(ext) > 1 and ext[1:] in EXT_PROJECT_CHOICES:
            result = wx.MessageBox(
                "Do you want to load the project, \n"
                '"%s", into your project?' % os.path.split(path)[1],
                caption="Load project",
                style=wx.YES_NO | wx.ICON_QUESTION,
                parent=self.__path_list_ctrl,
            )
            if result == wx.YES:
                self.do_open_workspace(path)
            return

        if len(ext) > 1 and ext[1:] in EXT_PIPELINE_CHOICES:
            result = wx.MessageBox(
                "Do you want to import the pipeline, \n"
                '"%s", into your project?' % os.path.split(path)[1],
                caption="Import pipeline",
                style=wx.YES_NO | wx.ICON_QUESTION,
                parent=self.__path_list_ctrl,
            )
            if result == wx.YES:
                self.do_load_pipeline(path)
            return
        series = self.__path_list_ctrl.get_selected_series()
        show_image(
            path, self.__frame, dimensions=3 if self.__pipeline.volumetric() else 2, series=series
        )

    def on_pathlist_file_delete(self, paths):
        self.__pipeline.remove_urls(paths)
        self.__workspace.file_list.remove_files_from_filelist(paths)
        self.__path_list_ctrl.remove_files(paths)
        self.__workspace.invalidate_image_set()

    def on_pathlist_refresh(self, urls):
        """Refresh the pathlist by checking for existence of file URLs"""

        urls = list(filter((lambda url: url.startswith("file:")), urls))

        def refresh_msg(idx):
            return "Checked %d of %d" % (idx, len(urls))

        with wx.ProgressDialog(
            parent=self.__frame,
            title="Refreshing file list",
            message=refresh_msg(0),
            maximum=len(urls) + 1,
            style=wx.PD_CAN_ABORT | wx.PD_APP_MODAL,
        ) as dlg:
            assert isinstance(dlg, wx.ProgressDialog)
            to_remove = []
            for idx, url in enumerate(urls):
                path = url2pathname(url[5:])
                if not os.path.isfile(path):
                    to_remove.append(url)
                if idx % 100 == 0:
                    keep_going, skip = dlg.Update(idx, refresh_msg(idx))
                    if not keep_going:
                        return
            if len(to_remove) > 0:
                dlg.Update(len(urls), "Removing %d missing files" % len(to_remove))
                self.__pipeline.remove_urls(to_remove)

    def on_pathlist_clear(self, event):
        """Remove all files from the path list"""
        result = wx.MessageBox(
            "Are you sure you want to clear all files from your project?\n\n"
            "Clearing will remove the files from your project,\n"
            "but will not delete or modify the files themselves.\n"
            'Choose "Yes" to delete or "No" to cancel this operation.',
            caption="Confirm file list clear operation",
            style=wx.YES_NO,
            parent=self.__frame,
        )
        if result == wx.YES:
            self.__pipeline.clear_urls()
            self.__workspace.file_list.clear_filelist()
            self.__path_list_ctrl.hidden_files = []
            self.__path_list_ctrl.disabled_urls = set()
            self.__workspace.invalidate_image_set()

    def on_pathlist_drop_files(self, x, y, filenames):
        wx.CallAfter(self.add_paths_to_pathlist, filenames)

    def add_paths_to_pathlist(self, filenames):
        t0 = datetime.datetime.now()
        with wx.ProgressDialog(
            "Processing files",
            "Initializing",
            parent=self.__frame,
            style=wx.PD_APP_MODAL | wx.PD_CAN_ABORT,
        ) as dlg:
            assert isinstance(dlg, wx.ProgressDialog)

            # Set dialog width to at least 480
            min_width = 480
            h, w = dlg.GetSize()
            if w < min_width:
                dlg.SetSize(min_width, h)

            # Content needs to be in containers to be shared between threads
            queue = Queue()
            interrupt = [False]
            message = ["Initializing"]

            def fn(
                filenames=filenames, interrupt=[True], message=["Default"], queue=queue
            ):
                urls = []
                if len(filenames) > 100 and os.path.exists(filenames[0]):
                    # If we have many files to process, it's faster to just scan the whole parent folder.
                    desired = set(filenames)
                    # All files added in 1 operation should come from the same parent directory.
                    parent = os.path.dirname(filenames[0])
                    # Scandir produces a generator that'll automatically distinguish files from folders
                    options = os.scandir(parent)
                    for path in options:
                        if not interrupt or interrupt[0]:
                            break
                        if path.path in desired:
                            message[0] = "\nProcessing " + path.path
                            desired.remove(path.path)
                            if path.is_file():
                                urls.append(pathname2url(path.path))
                                if len(urls) > 100:
                                    queue.put(urls)
                                    urls = []
                            elif path.is_dir():
                                for dirpath, dirnames, filenames in os.walk(path.path):
                                    for filename in filenames:
                                        if not interrupt or interrupt[0]:
                                            break
                                        path = os.path.join(dirpath, filename)
                                        urls.append(pathname2url(path))
                                        message[0] = "\nProcessing " + path
                                        if len(urls) > 100:
                                            queue.put(urls)
                                            urls = []
                                    else:
                                        continue
                                    break
                            if not desired:
                                # Stop the generator once all paths are found
                                break
                    queue.put(urls)
                    urls = []
                    # In case of missing files or mixed parent directories, we'll use the old method to fetch the rest.
                    filenames = list(desired)

                for pathname in filenames:
                    if not interrupt or interrupt[0]:
                        break

                    message[0] = "\nProcessing " + pathname
                    if ZARR_FILETYPE.search(pathname):
                        pathname, _ = ZARR_FILETYPE.split(pathname)
                        urls.append(pathname2url(pathname))
                        if len(urls) > 100:
                            queue.put(urls)
                            urls = []
                    elif os.path.isfile(pathname):
                        urls.append(pathname2url(pathname))
                        if len(urls) > 100:
                            queue.put(urls)
                            urls = []
                    elif os.path.isdir(pathname):
                        for dirpath, dirnames, files in os.walk(pathname):
                            if ZARR_FILETYPE.search(dirpath):
                                zarr_dir, _ = ZARR_FILETYPE.split(dirpath)
                                urls.append(pathname2url(zarr_dir))
                                continue
                            for filename in files:
                                if not interrupt or interrupt[0]:
                                    break
                                path = os.path.join(dirpath, filename)
                                urls.append(pathname2url(path))
                                message[0] = "\nProcessing " + path
                                if len(urls) > 100:
                                    queue.put(urls)
                                    urls = []
                            else:
                                continue
                            break
                    else:
                        # Allow URL objects through without change
                        parsed = urlparse(pathname)
                        if parsed.scheme and parsed.scheme != 'file':
                            urls.append(pathname)
                            if len(urls) > 100:
                                queue.put(urls)
                                urls = []
                queue.put(urls)

            thread = threading.Thread(
                target=fn, name="File Processing thread", daemon=True, args=(filenames, interrupt, message, queue)
            )
            thread.start()

            def update_pulse(msg):
                waiting_for = int((datetime.datetime.now() - t0).total_seconds())
                if waiting_for > 60:
                    minutes = int(waiting_for) / 60
                    seconds = waiting_for % 60
                    msg += "\n\nElapsed time: %d minutes, %d seconds" % (minutes, seconds)
                    msg += "\nConsider using the LoadData module for loading large numbers of images."
                keep_going, skip = dlg.Pulse(msg)
                dlg.Fit()
                return keep_going

            while not interrupt[0]:
                try:
                    urls = queue.get(timeout=0.1)
                    try:
                        while True:
                            urls += queue.get(block=False)
                    except Empty:
                        keep_going = update_pulse(
                            "\nAdding %d files to file list" % len(urls)
                        )
                        self.add_urls(urls)
                except Empty as err:
                    if not thread.is_alive():
                        try:
                            self.add_urls(queue.get(block=False))
                        except:
                            pass
                        break
                    keep_going = update_pulse(message[0])
                interrupt[0] = not keep_going
            interrupt[0] = True
        self.__workspace.invalidate_image_set()

    def on_pathlist_drop_text(self, x, y, text):
        pathnames = [p.strip() for p in re.split("[\r\n]+", text.strip())]
        self.__pipeline.add_pathnames_to_file_list(pathnames)

    def pick_from_pathlist(self, selected_plane, title=None, instructions=None):
        """Pick a file from the pathlist control

        This function displays the pathlist control within a dialog box. The
        single pathlist control is reparented to the dialog box during its
        modal display.

        selected_url - select this URL in the pathlist control.

        returns the URL or None if the user cancelled.
        """
        if title is None:
            title = "Select an image file"
        with wx.Dialog(
            self.__frame,
            title=title,
            size=(640, 480),
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
        ) as dlg:
            dlg.Sizer = wx.BoxSizer(wx.VERTICAL)
            dlg.Sizer.AddSpacer(3)
            sizer = wx.BoxSizer(wx.VERTICAL)
            dlg.Sizer.Add(sizer, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 3)
            if instructions is not None:
                sizer.Add(wx.StaticText(dlg, label=instructions), 0, wx.EXPAND)
                sizer.AddSpacer(2)
            planes = self.__pipeline.get_image_plane_list(self.__workspace)
            treectrl = wx.TreeCtrl(dlg, style=wx.TR_HIDE_ROOT | wx.TR_HAS_BUTTONS |
                                              wx.TR_FULL_ROW_HIGHLIGHT | wx.TR_LINES_AT_ROOT)
            root_id = treectrl.AddRoot("File List")
            folder_id_map = {}
            FOLDER_COLOR = wx.SystemSettings.GetColour(wx.SYS_COLOUR_MENUHILIGHT)
            for i, plane_object in enumerate(planes):
                folder = plane_object.file.dirname
                filename = plane_object.file.filename
                if folder in folder_id_map:
                    folder_id = folder_id_map[folder]
                else:
                    folder_id = treectrl.AppendItem(root_id, folder)
                    treectrl.SetItemTextColour(folder_id, FOLDER_COLOR)
                    folder_id_map[folder] = folder_id
                plane_id = treectrl.AppendItem(folder_id, str(plane_object), data=plane_object)
                treectrl.Expand(folder_id)
                if plane_object == selected_plane:
                    LOGGER.info("Selected plane found")
                    treectrl.SelectItem(plane_id)
            sizer.Add(treectrl, 1, wx.EXPAND)
            button_sizer = wx.StdDialogButtonSizer()
            ok_button = wx.Button(dlg, wx.ID_OK)
            button_sizer.AddButton(ok_button)
            button_sizer.AddButton(wx.Button(dlg, wx.ID_CANCEL))
            button_sizer.Realize()
            dlg.Sizer.Add(button_sizer, 0, wx.ALIGN_CENTER_HORIZONTAL)
            any_selected = treectrl.GetSelection().IsOk()
            ok_button.Enable(any_selected)

            def on_selection_change(event=None):
                item_id = treectrl.GetSelection()
                if item_id.IsOk() and treectrl.GetItemParent(item_id) != root_id:
                    ok_button.Enable(True)
                else:
                    ok_button.Enable(False)

            treectrl.Bind(
                wx.EVT_TREE_SEL_CHANGED, on_selection_change
            )
            result = dlg.ShowModal()
            if result == wx.ID_OK:
                plane_id = treectrl.GetSelection()
                if not plane_id.IsOk():
                    return None
                return treectrl.GetItemData(plane_id)

            return None

    def add_urls(self, urls):
        """Add URLS to the pipeline"""
        # The pipeline's notification callback will add them to the workspace
        self.__pipeline.add_urls(urls)

    def on_walk_callback(self, dirpath, dirnames, filenames):
        """Handle an iteration of file walking"""

        hdf_file_list = self.__workspace.get_file_list()
        file_list = [
            pathname2url(os.path.join(dirpath, filename)) for filename in filenames
        ]
        hdf_file_list.add_files_to_filelist(file_list)
        self.__pipeline.add_urls(file_list)

    def on_walk_completed(self):
        pass

    def enable_module_controls_panel_buttons(self):
        #
        # Enable/disable the movement buttons
        #
        selected_modules = self.__get_selected_modules()
        active_module = self.__pipeline_list_view.get_active_module()
        if active_module is None or active_module.is_input_module():
            enable_up = False
            enable_down = False
        else:
            # The module_num of the first module that's not an input module
            first_module_num, last_module_num = [
                reduce(
                    fn,
                    [
                        module.module_num
                        for module in self.__pipeline.modules()
                        if not module.is_input_module()
                    ],
                    initial,
                )
                for fn, initial in ((min, len(self.__pipeline.modules())), (max, -1))
            ]
            enable_up = active_module.module_num > first_module_num
            enable_down = active_module.module_num < last_module_num
        enable_delete = True
        enable_duplicate = True
        if len(selected_modules) == 0:
            enable_delete = enable_duplicate = False

        for menu_id, control, state in (
            (
                cellprofiler.gui.cpframe.ID_EDIT_MOVE_DOWN,
                self.__mcp_module_down_button,
                enable_down,
            ),
            (
                cellprofiler.gui.cpframe.ID_EDIT_MOVE_UP,
                self.__mcp_module_up_button,
                enable_up,
            ),
            (
                cellprofiler.gui.cpframe.ID_EDIT_DELETE,
                self.__mcp_remove_module_button,
                enable_delete,
            ),
            (cellprofiler.gui.cpframe.ID_EDIT_DUPLICATE, None, enable_duplicate),
        ):
            state = state and not self.is_running()
            if control is not None:
                control.Enable(state)
            menu_item = self.__frame.menu_edit.FindItemById(menu_id)
            if menu_item is not None:
                menu_item.Enable(state)

    def __on_add_module(self, event):
        if not self.__add_module_frame.IsShownOnScreen():
            x, y = self.__frame.GetPosition()
            if x > self.__add_module_frame.GetSize().width:
                x -= self.__add_module_frame.GetSize().width
            else:
                x += self.__module_controls_panel.GetSize().width
            self.__add_module_frame.SetPosition((x, y))
        self.__add_module_frame.Show()
        self.__add_module_frame.Raise()

    def open_add_modules(self):
        self.__on_add_module(None)

    def populate_edit_menu(self, menu):
        """Display a menu of modules to add"""

        #
        # Get a two-level dictionary of categories and names
        #
        d = {"All": []}
        for module_name in get_module_names():
            try:
                module = get_module_class(module_name)
                if module.is_input_module():
                    continue
                category = module.category
                if isinstance(category, str):
                    categories = [category, "All"]
                else:
                    categories = list(category) + ["All"]
                for category in categories:
                    if category not in d:
                        d[category] = []
                    d[category].append(module_name)
            except:
                LOGGER.error(
                    "Unable to instantiate module %s.\n\n" % module_name, exc_info=True
                )

        for category in sorted(d.keys()):
            sub_menu = wx.Menu()
            for module_name in sorted(d[category]):
                if module_name in self.module_name_to_menu_id:
                    menu_id = self.module_name_to_menu_id[module_name]
                else:
                    menu_id = wx.NewId()
                    self.module_name_to_menu_id[module_name] = menu_id
                    self.menu_id_to_module_name[menu_id] = module_name
                    self.__frame.Bind(wx.EVT_MENU, self.on_menu_add_module, id=menu_id)
                sub_menu.Append(menu_id, module_name)
            menu.AppendSubMenu(sub_menu, category)

    def populate_goto_menu(self, menu=None):
        """Populate the menu items in the edit->goto menu"""
        if menu is None:
            menu = self.__frame.menu_edit_goto_module
        assert isinstance(menu, wx.Menu)
        ids = []
        for item in menu.GetMenuItems():
            assert isinstance(item, wx.MenuItem)
            self.__frame.Unbind(wx.EVT_MENU, id=item.GetId())
            ids.append(item.GetId())
        for item_id in ids:
            menu.Delete(item_id)
        modules = self.__pipeline.modules(exclude_disabled=False)
        if len(ids) < len(modules):
            ids += [wx.NewId() for _ in range(len(ids), len(modules))]
        for item_id, module in zip(ids, modules):
            item = menu.Append(
                item_id, "#%02d %s" % (module.module_num, module.module_name)
            )

            def on_goto_module(event, module_num=module.module_num):
                self.on_goto_module(event, module_num)

            self.__frame.Bind(wx.EVT_MENU, on_goto_module, id=item_id)

    def on_goto_module(self, event, module_num):
        self.__module_view.set_selection(module_num)

    def on_menu_add_module(self, event):
        from cellprofiler.gui.addmoduleframe import AddToPipelineEvent

        assert isinstance(event, wx.CommandEvent)
        if event.GetId() in self.menu_id_to_module_name:
            module_name = self.menu_id_to_module_name[event.GetId()]

            def loader(module_num, module_name=module_name):
                module = instantiate_module(module_name)
                module.set_module_num(module_num)
                return module

            self.on_add_to_pipeline(self, AddToPipelineEvent(module_name, loader))
        else:
            LOGGER.warning(
                "Could not find module associated with ID = %d, module = %s"
                % (event.GetId(), event.GetString())
            )

    def __get_selected_modules(self):
        """Get the modules selected in the GUI, but not input modules"""
        return list(
            filter(
                lambda x: not x.is_input_module(),
                self.__pipeline_list_view.get_selected_modules(),
            )
        )

    def ok_to_edit_pipeline(self):
        """Return True if ok to edit pipeline

        Warns user if not OK (is_running)
        """
        if self.is_running():
            wx.MessageBox(
                "Pipeline modification is disabled during analysis.\n"
                "Please stop the analysis before editing your pipeline.",
                caption="Error: Pipeline editing disabled during analysis",
                style=wx.OK | wx.ICON_INFORMATION,
                parent=self.__frame,
            )
            return False
        return True

    def on_remove_module(self, event):
        self.remove_selected_modules()

    def remove_selected_modules(self):
        if not self.ok_to_edit_pipeline():
            return
        with self.__pipeline.undoable_action("Remove modules"):
            selected_modules = self.__get_selected_modules()
            for module in selected_modules:
                for setting in module.settings():
                    if setting.key() in self.__setting_errors:
                        self.__frame.preferences_view.pop_error_text(
                            self.__setting_errors.pop(setting.key())
                        )
                self.__pipeline.remove_module(module.module_num)
            has_input_modules = any(
                [m.is_input_module() for m in self.__pipeline.modules()]
            )
            has_legacy_modules = any(
                [m.is_load_module() for m in self.__pipeline.modules()]
            )
            if (not has_input_modules) and (not has_legacy_modules):
                #
                # We need input modules if legacy modules have been deleted
                #
                self.__pipeline.init_modules()
            self.exit_test_mode()

    def exit_test_mode(self):
        """Exit test mode with all the bells and whistles

        This is safe to call if not in test mode
        """
        if self.is_in_debug_mode():
            self.stop_debugging()
            if get_show_exiting_test_mode_dlg():
                self.show_exiting_test_mode()

    def on_duplicate_module(self, event):
        self.duplicate_modules(self.__get_selected_modules())

    def duplicate_modules(self, modules):
        if not self.ok_to_edit_pipeline():
            return

        selected_modules = self.__get_selected_modules()
        if len(selected_modules):
            module_num = selected_modules[-1].module_num + 1
        else:
            # insert module last if nothing selected
            module_num = len(self.__pipeline.modules()) + 1
        for m in modules:
            module = self.__pipeline.instantiate_module(m.module_name)
            module.set_module_num(module_num)
            module.set_settings_from_values(
                cellprofiler.gui.pipeline.Pipeline.capture_module_settings(m),
                m.variable_revision_number,
                m.module_name,
            )
            module.show_window = m.show_window  # copy visibility
            self.__pipeline.add_module(module)
            module_num += 1

    def on_module_up(self, event):
        """Move the currently selected modules up"""
        if not self.ok_to_edit_pipeline():
            return
        selected_modules = list(self.__get_selected_modules())
        if len(selected_modules) == 0:
            active_module = self.__pipeline_list_view.get_active_module()
            if active_module is None:
                return
            selected_modules = [active_module]
        for module in selected_modules:
            self.__pipeline.move_module(module.module_num, DIRECTION_UP)

    def on_module_down(self, event):
        """Move the currently selected modules down"""
        if not self.ok_to_edit_pipeline():
            return
        selected_modules = list(self.__get_selected_modules())
        selected_modules.reverse()
        if len(selected_modules) == 0:
            active_module = self.__pipeline_list_view.get_active_module()
            if active_module is None:
                return
            selected_modules = [active_module]
        for module in selected_modules:
            self.__pipeline.move_module(module.module_num, DIRECTION_DOWN)

    def on_update_module_enable(self, event):
        """Update the UI for the ENABLE_MODULE menu item / button

        event - an UpdateUIEvent for the item
        """
        num_modules = len(self.__pipeline_list_view.get_selected_modules())
        active_module = self.__pipeline_list_view.get_active_module()
        if num_modules > 1:
            event.SetText(
                "Disable selected modules ({})".format(num_modules)
                if self.__pipeline_list_view.get_selected_modules()[0].enabled
                else "Enable selected modules ({})".format(num_modules)
            )
        elif active_module is not None:
            event.SetText(
                "Disable {} (#{})".format(
                    active_module.module_name, active_module.module_num
                )
                if active_module.enabled
                else "Enable {} (#{})".format(
                    active_module.module_name, active_module.module_num
                )
            )
        if active_module is None or active_module.is_input_module():
            event.Enable(False)
        else:
            event.Enable(True)

    def on_module_enable(self, event):
        """Toggle the active module's enable state"""
        active_module = self.__pipeline_list_view.get_active_module()
        selected_modules = self.__pipeline_list_view.get_selected_modules()
        if len(selected_modules) > 1:
            if selected_modules[0].enabled:
                for module in selected_modules:
                    self.__pipeline.disable_module(module)
            else:
                for module in selected_modules:
                    self.__pipeline.enable_module(module)
            return

        if active_module is None:
            LOGGER.warning(
                "User managed to fire the enable/disable module event and no module was active"
            )
            return
        if active_module.is_input_module():
            LOGGER.warning(
                "User managed to fire the enable/disable module event when an input module was active"
            )
            return
        if active_module.enabled:
            self.__pipeline.disable_module(active_module)
        else:
            self.__pipeline.enable_module(active_module)

    def on_update_module_display_toggle(self, event):
        """Update the UI for the DISPLAY_MODULE menu item / button

        event - an UpdateUIEvent for the item
        """
        num_modules = len(self.__pipeline_list_view.get_selected_modules())
        active_module = self.__pipeline_list_view.get_active_module()
        if num_modules > 1:
            event.SetText(
                "Disable Display of selected modules ({})".format(num_modules)
                if self.__pipeline_list_view.get_selected_modules()[0].show_window
                else "Enable Display of selected modules ({})".format(num_modules)
            )
        elif active_module is not None:
            event.SetText(
                "Disable Display of {} (#{})".format(
                    active_module.module_name, active_module.module_num
                )
                if active_module.show_window
                else "Enable Display of {} (#{})".format(
                    active_module.module_name, active_module.module_num
                )
            )
        if active_module is None or active_module.is_input_module():
            event.Enable(False)
        else:
            event.Enable(True)

    def on_module_display_toggle(self, event):
        """Toggle the active module's display state"""
        active_module = self.__pipeline_list_view.get_active_module()
        selected_modules = self.__pipeline_list_view.get_selected_modules()
        if len(selected_modules) > 1:
            for module in selected_modules:
                self.__pipeline.show_module_window(module, state=not module.show_window)
            return

        if active_module is None:
            LOGGER.warning(
                "User managed to fire the enable/disable module event and no module was active"
            )
            return
        if active_module.is_input_module():
            LOGGER.warning(
                "User managed to fire the enable/disable module event when an input module was active"
            )
            return
        self.__pipeline.show_module_window(active_module, state=not active_module.show_window)

    def on_undo(self, event):
        wx.BeginBusyCursor()
        try:
            if self.__pipeline.has_undo():
                self.__pipeline.undo()
        finally:
            wx.EndBusyCursor()

    def on_update_undo_ui(self, event):
        event.Enable(self.__pipeline.has_undo() and not self.is_running())

    def on_add_to_pipeline(self, caller, event):
        """Add a module to the pipeline using the event's module loader

        caller - ignored

        event - an AddToPipeline event
        """
        if not self.ok_to_edit_pipeline():
            return
        active_module = self.__pipeline_list_view.get_active_module()
        if active_module is None:
            # insert module last if nothing selected
            module_num = len(self.__pipeline.modules(False)) + 1
        else:
            last_input_module_num = 0
            for module in self.__pipeline.modules(False):
                if module.is_input_module():
                    last_input_module_num = module.module_num
                else:
                    break
            module_num = max(active_module.module_num, last_input_module_num) + 1
        module = event.module_loader(module_num)
        module.show_window = True  # default to show in GUI
        remove_input_modules = False
        if module.is_load_module() and any(
            [m.is_input_module() for m in self.__pipeline.modules()]
        ):
            #
            # A legacy load module, ask user if they want to convert to a
            # legacy pipeline
            #
            message = (
                "%s is an alternative input module that is incompatible\n"
                "with the Images, Metadata, NamesAndTypes, and Groups\n"
                "input modules. Do you want to remove these input\n"
                "modules and use %s instead?"
            ) % (module.module_name, module.module_name)
            if (
                wx.MessageBox(
                    message,
                    caption="Use alternative input module, %s" % module.module_name,
                    style=wx.YES_NO | wx.YES_DEFAULT | wx.ICON_QUESTION,
                    parent=self.__frame,
                )
                != wx.YES
            ):
                return
            remove_input_modules = True

        if self.__pipeline.volumetric() and not module.volumetric():
            message = "{} does not support processing 3D data and will not be added to the pipeline.".format(
                module.module_name
            )

            wx.MessageBox(message, caption="Warning", style=wx.OK)

            return

        self.__pipeline.add_module(module)
        if remove_input_modules:
            while True:
                for m in self.__pipeline.modules():
                    if m.is_input_module():
                        self.__pipeline.remove_module(m.module_num)
                        break
                else:
                    break
            self.__pipeline_list_view.select_one_module(module.module_num)

            #
            # Major event - restart from scratch
            #
            # if self.is_in_debug_mode():
            #    self.stop_debugging()

    def __on_module_view_event(self, caller, event):
        assert isinstance(
            event,
            cellprofiler.gui.module_view._setting_edited_event.SettingEditedEvent,
        ), (
            "%s is not an instance of CellProfiler.CellProfilerGUI.ModuleView.SettingEditedEvent"
            % (str(event))
        )
        setting = event.get_setting()
        if isinstance(setting, str) and setting == "module_notes":
            self.__dirty_workspace = True
            self.set_title()
            self.__pipeline.edit_module(event.get_module().module_num, False)
            return
        proposed_value = event.get_proposed_value()
        setting.set_value_text(proposed_value)
        module = event.get_module()
        module.on_setting_changed(setting, self.__pipeline)
        is_image_set_modification = module.change_causes_prepare_run(setting)
        self.__pipeline.edit_module(
            event.get_module().module_num, is_image_set_modification
        )
        if self.is_in_debug_mode() and is_image_set_modification:
            #
            # If someone edits a really important setting in debug mode,
            # then you want to reset the debugger to reprocess the image set
            # list.
            #
            self.stop_debugging()
            if get_show_exiting_test_mode_dlg():
                self.show_exiting_test_mode()

    def status_callback(self, *args):
        self.__frame.preferences_view.on_pipeline_progress(*args)

    def run_next_pipeline(self, event):
        if len(self.pipeline_list) == 0:
            return
        pipeline_details = self.pipeline_list.pop(0)
        self.do_load_pipeline(pipeline_details.path)
        set_default_image_directory(pipeline_details.default_input_folder)
        set_default_output_directory(pipeline_details.default_output_folder)
        self.on_analyze_images(event)

    def on_analyze_images(self, event):
        """Handle a user request to start running the pipeline"""
        self.do_analyze_images()

    def do_analyze_images(self):
        """Analyze images using the current workspace and pipeline"""
        ##################################
        #
        # Preconditions:
        # * Pipeline has no errors
        # * Default input and output directories are valid
        #
        ##################################

        ok, reason = self.__frame.preferences_view.check_preferences()

        if ok:
            try:
                self.__pipeline.test_valid()
            except ValidationError as v:
                ok = False
                reason = v
        if not ok:
            if (
                wx.MessageBox(
                    "%s\nAre you sure you want to continue?" % reason,
                    "Problems with pipeline",
                    wx.YES_NO,
                )
                != wx.YES
            ):
                self.pipeline_list = []
                return
        ##################################
        #
        # Start the pipeline
        #
        ##################################

        try:
            self.__module_view.disable()

            self.__pipeline_list_view.allow_editing(False)

            self.__frame.preferences_view.on_analyze_images()

            with Listener(self.__pipeline, self.on_prepare_run_error_event):
                if not self.__pipeline.prepare_run(self.__workspace):
                    self.stop_running()
                    return

            num_workers = min(
                len(self.__workspace.measurements.get_image_numbers()),
                get_max_workers(),
            )

            self.__analysis = Analysis(
                self.__pipeline, initial_measurements=self.__workspace.measurements,
            )

            self.__analysis.start(self.analysis_event_handler, num_workers)

            self.__frame.preferences_view.update_worker_count_info(num_workers)

            self.enable_module_controls_panel_buttons()

            self.populate_goto_menu()
        except Exception as error:
            LOGGER.error(error)
            extended_message = "Failure in analysis startup"

            error = cellprofiler.gui.dialog.Error("Error", extended_message)

            if error.status is wx.ID_CANCEL:
                cancel_progress()

                self.stop_running()

        return

    def on_prepare_run_error_event(self, pipeline, event):
        """Display an error message box on error during prepare_run

        This is called if the pipeline is misconfigured - an unrecoverable
        error that's the user's fault.
        """
        if isinstance(event, PrepareRunError):
            if event.module is None:
                caption = "Cannot run pipeline"
                message = (
                    "The pipeline cannot be started because of\n"
                    "a configuration problem:\n\n%s"
                ) % event.message
            else:
                caption = (
                    "Cannot run pipeline: misconfiguration in %s"
                    % event.module.module_name
                )
                message = (
                    "The pipeline cannot be started because of\n"
                    "a configuration problem in the %s module:\n\n%s"
                ) % (event.module.module_name, event.message)
            wx.MessageBox(
                message=message,
                caption=caption,
                parent=self.__frame,
                style=wx.ICON_ERROR | wx.OK,
            )

    def analysis_event_handler(self, evt):
        PRI_EXCEPTION, PRI_INTERACTION, PRI_DISPLAY = list(range(3))

        if isinstance(evt, Started):
            wx.CallAfter(self.show_analysis_controls)
        elif isinstance(evt, Progress):
            LOGGER.info("Progress", evt.counts)
            total_jobs = sum(evt.counts.values())
            completed = sum(
                map(
                    (lambda status: evt.counts.get(status, 0)),
                    (Runner.STATUS_DONE, Runner.STATUS_FINISHED_WAITING,),
                )
            )
            wx.CallAfter(
                self.__frame.preferences_view.on_pipeline_progress,
                total_jobs,
                completed,
            )
        elif isinstance(evt, Finished):
            LOGGER.info(("Cancelled!" if evt.cancelled else "Finished!"))
            # drop any interaction/display requests or exceptions
            while True:
                try:
                    self.interaction_request_queue.get_nowait()  # in case the queue's been emptied
                except Empty:
                    break
            if evt.cancelled:
                self.pipeline_list = []
            wx.CallAfter(self.on_stop_analysis, evt)
        elif isinstance(evt, Display):
            wx.CallAfter(self.module_display_request, evt)
        elif isinstance(evt, DisplayPostRun):
            wx.CallAfter(self.module_display_post_run_request, evt)
        elif isinstance(evt, DisplayPostGroup):
            wx.CallAfter(self.module_display_post_group_request, evt)
        elif isinstance(evt, InteractionRequest):
            self.interaction_request_queue.put(
                (PRI_INTERACTION, self.module_interaction_request, evt)
            )
            wx.CallAfter(self.handle_analysis_feedback)
        elif isinstance(evt, OmeroLoginRequest):
            self.interaction_request_queue.put(
                (PRI_INTERACTION, self.omero_login_request, evt)
            )
            wx.CallAfter(self.handle_analysis_feedback)
        elif isinstance(evt, ExceptionReport):
            self.interaction_request_queue.put(
                (PRI_EXCEPTION, self.analysis_exception, evt)
            )
            wx.CallAfter(self.handle_analysis_feedback)
        elif isinstance(evt, (DebugWaiting, DebugComplete,),):
            # These are handled by the dialog reading the debug
            # request queue
            if self.debug_request_queue is None:
                # Things are in a bad state here, possibly because the
                # user hasn't properly run the debugger. Chances are that
                # the user knows that something is going wrong.
                evt.reply(ServerExited())
            else:
                self.debug_request_queue.put(evt)
        elif isinstance(evt, Paused):
            wx.CallAfter(self.show_resume_button)
        elif isinstance(evt, Resumed):
            wx.CallAfter(self.show_pause_button)
        elif isinstance(evt, RunException):
            # exception in (prepare/post)_(run/group)
            wx.CallAfter(self.__on_pipeline_event, self.__pipeline, evt)
        else:
            raise ValueError("Unknown event type %s %s" % (type(evt), evt))

    def handle_analysis_feedback(self):
        """Process any pending exception or interaction requests from the
        pipeline.  This function guards against multiple modal dialogs being
        opened, which can overwhelm the user and cause UI hangs.
        """
        # just in case.
        assert (
            wx.IsMainThread()
        ), "PipelineController.handle_analysis_feedback() must be called from main thread!"

        # only one window at a time
        if self.interaction_pending:
            return

        try:
            pri_func_args = (
                self.interaction_request_queue.get_nowait()
            )  # in case the queue's been emptied
        except Empty:
            return

        self.interaction_pending = True
        try:
            pri_func_args[1](*pri_func_args[2:])
            if not self.interaction_request_queue.empty():
                wx.CallAfter(self.handle_analysis_feedback)
        finally:
            self.interaction_pending = False

    def module_display_request(self, evt):
        """
        """
        assert (
            wx.IsMainThread()
        ), "PipelineController.module_display_request() must be called from main thread!"

        module_num = evt.module_num
        if module_num <= 0 or module_num > len(
            self.__pipeline.modules(exclude_disabled=False)
        ):
            # Defensive coding: module was deleted?
            LOGGER.warning(
                "Failed to display module # %d. The pipeline may have been edited during analysis"
                % module_num
            )
            evt.reply(Ack())
            return

        # use our shared workspace
        self.__workspace.display_data.__dict__.update(evt.display_data_dict)
        try:
            module = self.__pipeline.modules(exclude_disabled=False)[module_num - 1]
            if module.display != Module.display:
                fig = self.__workspace.get_module_figure(
                    module, evt.image_set_number, self.__frame
                )
                module.display(self.__workspace, fig)
                if hasattr(fig.figure.canvas, "_isDrawn"):
                    fig.figure.canvas._isDrawn = False
                fig.figure.canvas.Refresh()
        except Exception as exc:
            LOGGER.exception(exc.__traceback__)
            error = cellprofiler.gui.dialog.Error("Error", str(exc))
            if error.status == wx.ID_CANCEL:
                cancel_progress()
        finally:
            # we need to ensure that the reply_cb gets a reply
            evt.reply(Ack())

    def module_display_post_run_request(self, evt):
        assert (
            wx.IsMainThread()
        ), "PipelineController.module_post_run_display_request() must be called from main thread!"
        module_num = evt.module_num
        # use our shared workspace
        self.__workspace.display_data.__dict__.update(evt.display_data.__dict__)
        try:
            module = self.__pipeline.modules(exclude_disabled=False)[module_num - 1]
            if module.display_post_run != Module.display_post_run:
                image_number = self.__workspace.measurements.image_set_count
                fig = self.__workspace.get_module_figure(
                    module, image_number, self.__frame
                )
                module.display_post_run(self.__workspace, fig)
                fig.Refresh()
        except Exception as exc:
            LOGGER.exception(exc.__traceback__)
            error = cellprofiler.gui.dialog.Error("Error", str(exc))
            if error.status == wx.ID_CANCEL:
                cancel_progress()

    def module_display_post_group_request(self, evt):
        assert (
            wx.IsMainThread()
        ), "PipelineController.module_post_group_display_request() must be called from main thread!"
        module_num = evt.module_num
        # use our shared workspace
        self.__workspace.display_data.__dict__.update(evt.display_data)
        try:
            module = self.__pipeline.modules(exclude_disabled=False)[module_num - 1]
            if module.display_post_group != Module.display_post_group:
                image_number = evt.image_set_number
                fig = self.__workspace.get_module_figure(
                    module, image_number, self.__frame
                )
                module.display_post_group(self.__workspace, fig)
                fig.Refresh()
        except Exception as exc:
            LOGGER.exception(exc.__traceback__)
            error = cellprofiler.gui.dialog.Error("Error", str(exc))
            if error.status == wx.ID_CANCEL:
                cancel_progress()
        finally:
            evt.reply(Ack())

    def module_interaction_request(self, evt):
        """forward a module interaction request from the running pipeline to
        our own pipeline's instance of the module, and reply with the result.
        """
        module_num = evt.module_num
        # extract args and kwargs from the request.
        # see main().interaction_handler() in worker.py
        args = [evt.__dict__["arg_%d" % idx] for idx in range(evt.num_args)]
        kwargs = dict(
            (name, evt.__dict__["kwarg_%s" % name]) for name in evt.kwargs_names
        )
        result = ""
        try:
            module = self.__pipeline.modules(exclude_disabled=False)[module_num - 1]
            result = module.handle_interaction(*args, **kwargs)
        except Exception as exc:
            LOGGER.exception(exc.__traceback__)
            error = cellprofiler.gui.dialog.Error("Error", str(exc))
            if error.status == wx.ID_CANCEL:
                cancel_progress()
        finally:
            # we need to ensure that the reply_cb gets a reply (even if it
            # being empty causes futher exceptions).
            evt.reply(Interaction(result=result))

    @staticmethod
    def omero_login_request(evt):
        """Handle retrieval of the Omero credentials"""
        from cellprofiler_core.bioformats.formatreader import get_omero_credentials

        evt.reply(OmeroLogin(get_omero_credentials()))

    def analysis_exception(self, evt):
        """Report an error in analysis to the user, giving options for
        skipping, aborting, and debugging."""

        assert (
            wx.IsMainThread()
        ), "PipelineController.analysis_exception() must be called from main thread!"

        self.debug_request_queue = Queue()

        evtlist = [evt]

        def remote_debug(evtlist=None):
            # choose a random string for verification
            if evtlist is None:
                evtlist = evtlist
            verification = "".join(
                random.choice(string.ascii_letters) for x in range(5)
            )
            evt = evtlist[0]
            # Request debugging.  We get back a port.
            evt.reply(
                ExceptionPleaseDebug(DEBUG, hashlib.sha1(verification).hexdigest(),)
            )
            evt = self.debug_request_queue.get()
            port = evt.port
            result = wx.MessageBox(
                "Remote PDB waiting on port %d\nUse '%s' for verification"
                % (port, verification),
                "Remote debugging started.",
                wx.OK | wx.CANCEL | wx.ICON_INFORMATION,
            )
            if result == wx.ID_CANCEL:
                evt.reply(DebugCancel())
                return False
            # Acknowledge the port request, and we'll get back a
            # DebugComplete(), which we use as a new evt to reply with the
            # eventual CONTINUE/STOP choice.
            with wx.ProgressDialog(
                "Remote debugging on port %d" % port,
                "Debugging remotely, Cancel to abandon",
                style=wx.PD_APP_MODAL | wx.PD_CAN_ABORT,
            ) as dlg:
                while True:
                    try:
                        evtlist[0] = self.debug_request_queue.get(timeout=0.25)
                        return True
                    except Empty:
                        keep_going, skip = dlg.UpdatePulse(
                            "Debugging remotely, Cancel to abandon"
                        )
                        if not keep_going:
                            self.debug_request_queue = None
                            return False

        if evt.module_name is not None:
            message = (
                "Error while processing %s:\n" "%s\n\nDo you want to stop processing?"
            ) % (evt.module_name, evt)
        else:
            message = (
                "Error while processing (remote worker):\n"
                "%s\n\nDo you want to stop processing?"
            ) % evt

        error = cellprofiler.gui.dialog.Error("Error", message)

        if error.status == wx.ID_CANCEL:
            cancel_progress()

            self.stop_running()

            disposition = ED_STOP
        else:
            disposition = ED_CONTINUE

        evtlist[0].reply(Reply(disposition=disposition))

        wx.Yield()  # This allows cancel events to remove other exceptions from the queue.

    def sentry_pack_pipeline(self, err_module_name="", err_module_num=0):
        from sentry_sdk import utils, serializer, push_scope
        utils.MAX_STRING_LENGTH = 15000
        serializer.MAX_DATABAG_BREADTH = 25
        # push_scope will attach data to the next event only
        with push_scope() as scope:
            with io.StringIO() as fp:
                self.__pipeline.dump(fp, save_image_plane_details=False, sanitize=True)
                pipeline_parts = fp.getvalue().split("\n\n")
                pipeline_info = {
                    "error_module": f"{err_module_num:02d}_{err_module_name}",
                    "header": pipeline_parts.pop(0),
                    "modules": [module.module_name for module in self.__pipeline.modules(exclude_disabled=False)]
                }
                scope.set_context("Pipeline_Info", pipeline_info)
                index = 1
                buffer = 0
                buffer_dict = {}
                # Sentry hard limits context content to >~10000 characters per container.
                # To avoid this we send in chunks if the pipeline is large.
                for idx, part in enumerate(pipeline_parts):
                    idx += 1
                    keyname = f"Pipeline_{index:02d}"
                    buffer += len(part)
                    module_name = f"{idx:02d}_{part.split(':', 1)[0]}"
                    buffer_dict[module_name] = part
                    if buffer > 7500 or idx == len(pipeline_parts):
                        scope.set_context(f"{keyname}", buffer_dict)
                        buffer = 0
                        buffer_dict = {}
                        index += 1
            # Log step needs to be here since this triggers sentry
            LOGGER.error("Failed to run module %s", err_module_name, exc_info=True)

    def on_pause(self, event):
        self.__frame.preferences_view.pause(True)
        self.__pause_pipeline = True
        self.__analysis.pause()
        self.__pause_button.Enable(False)

    def on_resume(self, event):
        self.__frame.preferences_view.pause(False)
        self.__pause_pipeline = False
        self.__analysis.resume()
        self.__resume_button.Enable(False)

    def on_stop_running(self, event):
        """Handle a user interface request to stop running"""
        result = wx.MessageBox(
            "Are you sure you want to abort this analysis run?",
            "Stop analysis run",
            wx.YES_NO | wx.ICON_QUESTION,
            self.__frame,
        )
        if result == wx.NO:
            return
        self.__stop_analysis_button.Enable(False)
        self.pipeline_list = []
        if (self.__analysis is not None) and self.__analysis.check_running():
            self.__analysis.cancel()
            return  # self.stop_running() will be called when we receive the
            # AnalysisCancelled event in self.analysis_event_handler.
        self.stop_running()

    def on_stop_analysis(self, event):
        """Stop an analysis run.

        Handle chores that need completing after an analysis is cancelled
        or finished, like closing the measurements file or writing the .MAT
        file.

        event - a cpanalysis.AnalysisFinished event
        """
        m = event.measurements
        status = m[
            IMAGE, Runner.STATUS, m.get_image_numbers(),
        ]
        n_image_sets = sum([x == Runner.STATUS_DONE for x in status])
        self.stop_running()
        if get_wants_pony():
            Sound(os.path.join(cellprofiler.icons.resources, "wantpony.wav")).Play()
        if get_show_analysis_complete_dlg():
            self.show_analysis_complete(n_image_sets)
        m.close()
        self.run_next_pipeline(None)

    def stop_running(self):
        if self.is_running():
            self.__analysis.cancel()
            self.__analysis = None
        self.__frame.preferences_view.on_stop_analysis()
        self.__module_view.enable()
        self.__pipeline_list_view.allow_editing(True)
        self.show_launch_controls()
        self.enable_module_controls_panel_buttons()

    def is_in_debug_mode(self):
        """True if there's some sort of debugging in progress"""
        return self.__debug_image_set_list is not None

    def on_debug_toggle(self, event):
        if self.is_in_debug_mode():
            self.on_debug_stop(event)
        else:
            self.on_debug_start(event)

    def on_debug_start(self, event):
        module = self.__pipeline_list_view.reset_debug_module()
        if module is None:
            wx.MessageBox(
                "Test mode is disabled because this pipeline\n"
                "does not have any modules to run.",
                "Test mode is disabled",
                style=wx.OK | wx.ICON_ERROR,
                parent=self.__frame,
            )
            return
        self.start_debugging()

    def start_debugging(self):
        self.__pipeline.test_mode = True
        self.__pipeline_list_view.set_debug_mode(True)
        self.__test_controls_panel.GetParent().GetSizer().Layout()
        self.show_test_controls()
        with Listener(self.__pipeline, self.on_prepare_run_error_event):
            if not self.__workspace.refresh_image_set():
                self.stop_debugging()
                return False

        self.close_debug_measurements()
        self.__debug_measurements = Measurements(
            copy=self.__workspace.measurements, mode="memory"
        )
        self.__debug_object_set = ObjectSet(can_overwrite=True)
        self.__frame.enable_debug_commands()
        assert isinstance(self.__pipeline, cellprofiler.gui.pipeline.Pipeline)
        self.__debug_image_set_list = ImageSetList(True)
        workspace_model = cellprofiler.gui._workspace_model.Workspace(
            self.__pipeline,
            None,
            None,
            None,
            self.__debug_measurements,
            self.__debug_image_set_list,
            self.__frame,
        )
        try:
            workspace_model.set_file_list(self.__workspace.file_list)
            self.__keys, self.__groupings = self.__pipeline.get_groupings(
                workspace_model
            )

            self.__grouping_index = 0
            self.__within_group_index = 0
            self.__pipeline.prepare_group(
                workspace_model, self.__groupings[0][0], self.__groupings[0][1]
            )
        finally:
            workspace_model.set_file_list(None)
        self.__debug_outlines = {}
        if not self.debug_init_imageset():
            self.stop_debugging()
            return False
        return True

    def close_debug_measurements(self):
        del self.__debug_measurements
        self.__debug_measurements = None

    def on_debug_stop(self, event):
        self.stop_debugging()

    def stop_debugging(self):
        for reader in ALL_READERS.values():
            reader.clear_cached_readers()
        self.__pipeline.test_mode = False
        self.__pipeline_list_view.set_debug_mode(False)
        self.__test_controls_panel.GetParent().GetSizer().Layout()
        self.__frame.enable_launch_commands()
        self.__debug_image_set_list = None
        self.close_debug_measurements()
        self.__debug_object_set = None
        self.__debug_outlines = None
        self.__debug_grids = None
        self.__pipeline_list_view.on_stop_debugging()
        self.__pipeline.end_run()
        self.show_launch_controls()
        if get_conserve_memory():
            gc.collect()

    def do_step(self, module, select_next_module=False):
        """Do a debugging step by running a module
        """
        failure = 1
        old_cursor = self.__frame.GetCursor()
        self.__frame.SetCursor(wx.Cursor(wx.CURSOR_WAIT))
        try:
            image_set_number = self.__debug_measurements.image_set_number
            self.__debug_measurements.add_image_measurement(
                GROUP_NUMBER, self.__grouping_index + 1
            )  # in the analysis mode version, it starts at 1
            self.__debug_measurements.add_image_measurement(
                GROUP_INDEX, self.__within_group_index + 1
            )  # in the analysis mode version, it starts at 1
            self.__debug_measurements.add_image_measurement(
                "Group_Length", len(self.__groupings[self.__grouping_index][1])
            )
            workspace_model = cellprofiler.gui._workspace_model.Workspace(
                self.__pipeline,
                module,
                self.__debug_measurements,
                self.__debug_object_set,
                self.__debug_measurements,
                self.__debug_image_set_list,
                self.__frame if module.show_window else None,
                outlines=self.__debug_outlines,
            )
            self.__debug_grids = workspace_model.set_grids(self.__debug_grids)
            cancelled = [False]

            def cancel_handler(cancelled=None):
                if cancelled is None:
                    cancelled = []
                cancelled[0] = True

            workspace_model.cancel_handler = cancel_handler
            self.__pipeline.run_module(module, workspace_model)
            if cancelled[0]:
                self.__frame.SetCursor(old_cursor)
                return False

            if module.show_window:
                fig = workspace_model.get_module_figure(module, image_set_number)
                module.display(workspace_model, fig)
                fig.Refresh()
            workspace_model.refresh()
            if workspace_model.disposition == DISPOSITION_SKIP:
                self.last_debug_module()
            elif (
                module.module_num < len(self.__pipeline.modules())
                and select_next_module
            ):
                self.__pipeline_list_view.select_one_module(module.module_num + 1)
            failure = 0
            if self.workspace_view is not None:
                self.workspace_view.set_workspace(workspace_model)
        except Exception as instance:
            if get_telemetry():
                self.sentry_pack_pipeline(module.module_name, module.module_num)
            else:
                LOGGER.error("Failed to run module %s", module.module_name, exc_info=True)
            event = RunException(instance, module)
            self.__pipeline.notify_listeners(event)
            self.__pipeline_list_view.select_one_module(module.module_num)
            if event.cancel_run:
                self.on_debug_stop(event)
                failure = -1
            failure = 1
        self.__frame.SetCursor(old_cursor)
        if (
            module.module_name != "Restart" or failure == -1
        ) and self.__debug_measurements is not None:
            module_error_measurement = "ModuleError_%02d%s" % (
                module.module_num,
                module.module_name,
            )
            self.__debug_measurements.add_measurement(
                "Image", module_error_measurement, failure
            )
        return failure == 0

    def current_debug_module(self):
        assert self.is_in_debug_mode()
        return self.__pipeline_list_view.get_current_debug_module()

    def next_debug_module(self):
        return self.__pipeline_list_view.advance_debug_module() is not None

    def last_debug_module(self):
        for module in reversed(self.__pipeline.modules()):
            if not module.is_input_module():
                self.__pipeline_list_view.set_current_debug_module(module)
                return module
        self.__pipeline_list_view.reset_debug_module()
        return None

    def on_debug_step(self, event):
        module = self.current_debug_module()
        if module is None:
            return
        success = self.do_step(module)
        if success:
            self.next_debug_module()

    def on_debug_continue(self, event):
        first_module = self.current_debug_module()
        if first_module is None:
            return
        count = 1
        for module in self.__pipeline.modules()[first_module.module_num :]:
            if module.wants_pause:
                break
            count += 1
        message_format = "Running module %d of %d: %s"
        index = 0
        with wx.ProgressDialog(
            "Running modules in test mode",
            message_format % (index + 1, count, first_module.module_name),
            maximum=count,
            parent=self.__frame,
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_CAN_ABORT,
        ) as dlg:
            dlg.Show()
            max_message_width = None
            while True:
                assert isinstance(dlg, wx.ProgressDialog)
                module = self.current_debug_module()
                message = message_format % (index + 1, count, module.module_name)
                message_width = dlg.GetFullTextExtent(message)[0]
                if max_message_width is None:
                    max_message_width = message_width
                elif max_message_width < message_width:
                    diff = message_width - max_message_width
                    max_message_width = message_width
                    width, height = dlg.GetSize()
                    width += diff
                    dlg.SetSize(wx.Size(width, height))
                wants_continue, wants_skip = dlg.Update(index, message)
                if not wants_continue:
                    return
                index += 1
                success = self.do_step(module)
                if not success:
                    return
                if not self.next_debug_module():
                    return
                if self.current_debug_module().wants_pause:
                    return

    def on_debug_next_image_set(self, event):
        #
        # We have two indices, one into the groups and one into
        # the image indexes within the groups
        #
        keys, image_numbers = self.__groupings[self.__grouping_index]
        if len(image_numbers) == 0:
            return
        self.__within_group_index = (self.__within_group_index + 1) % len(image_numbers)
        image_number = image_numbers[self.__within_group_index]
        self.__debug_measurements.next_image_set(image_number)
        self.debug_init_imageset()
        self.__debug_outlines = {}

    def on_debug_prev_image_set(self, event):
        keys, image_numbers = self.__groupings[self.__grouping_index]
        self.__within_group_index = (
            self.__within_group_index + len(image_numbers) - 1
        ) % len(image_numbers)
        image_number = image_numbers[self.__within_group_index]
        self.__debug_measurements.next_image_set(image_number)
        self.__pipeline_list_view.reset_debug_module()
        self.__debug_outlines = {}

    def on_debug_next_group(self, event):
        if self.__grouping_index is not None:
            self.debug_choose_group(
                ((self.__grouping_index + 1) % len(self.__groupings))
            )

    def on_debug_prev_group(self, event):
        if self.__grouping_index is not None:
            self.debug_choose_group(
                (
                    (self.__grouping_index + len(self.__groupings) - 1)
                    % len(self.__groupings)
                )
            )

    def on_debug_random_image_set(self, event):
        keys, image_numbers = self.__groupings[self.__grouping_index]
        numpy.random.seed()
        if len(image_numbers) == 0:
            return
        elif len(image_numbers) == 1:
            self.__within_group_index = 0
        else:
            self.__within_group_index = numpy.random.randint(0, len(image_numbers))
        image_number = image_numbers[self.__within_group_index]
        self.__debug_measurements.next_image_set(image_number)
        self.debug_init_imageset()
        self.__debug_outlines = {}

    def on_debug_random_image_group(self, event):
        """Choose a group"""
        if len(self.__groupings) < 2:
            wx.MessageBox(
                "There is only one group and it is currently running in test mode",
                "Random image group",
            )
            return
        numpy.random.seed()
        group_index = numpy.random.randint(0, len(self.__groupings))
        self.debug_choose_group(group_index)

    def debug_choose_group(self, index):
        self.__grouping_index = index
        self.__within_group_index = 0
        workspace_model = cellprofiler.gui._workspace_model.Workspace(
            self.__pipeline,
            None,
            None,
            None,
            self.__debug_measurements,
            self.__debug_image_set_list,
            self.__frame,
        )

        self.__pipeline.prepare_group(
            workspace_model,
            self.__groupings[self.__grouping_index][0],
            self.__groupings[self.__grouping_index][1],
        )
        key, image_numbers = self.__groupings[self.__grouping_index]
        image_number = image_numbers[self.__within_group_index]
        self.__debug_measurements.next_image_set(image_number)
        self.debug_init_imageset()
        self.__debug_outlines = {}

    def on_debug_choose_group(self, event):
        """Choose a group"""
        if len(self.__groupings) < 2:
            wx.MessageBox(
                "There is only one group and it is currently running in test mode",
                "Choose image group",
            )
            return
        dialog = wx.Dialog(
            self.__frame,
            title="Choose an image group",
            style=wx.RESIZE_BORDER | wx.DEFAULT_DIALOG_STYLE,
        )
        super_sizer = wx.BoxSizer(wx.VERTICAL)
        dialog.SetSizer(super_sizer)
        super_sizer.Add(
            wx.StaticText(dialog, label="Select a group set for testing:"),
            0,
            wx.EXPAND | wx.ALL,
            5,
        )
        choices = []

        for grouping, image_numbers in self.__groupings:
            text = ["%s=%s" % (k, v) for k, v in list(grouping.items())]
            text = ", ".join(text)
            choices.append(text)
        lb = wx.ListBox(dialog, choices=choices)
        if self.__grouping_index < len(choices):
            lb.Select(self.__grouping_index)
            lb.EnsureVisible(self.__grouping_index)
        else:
            lb.Select(0)
        super_sizer.Add(lb, 1, wx.EXPAND | wx.ALL, 10)
        super_sizer.Add(wx.StaticLine(dialog), 0, wx.EXPAND | wx.ALL, 5)
        btnsizer = wx.StdDialogButtonSizer()
        btnsizer.AddButton(wx.Button(dialog, wx.ID_OK))
        btnsizer.AddButton(wx.Button(dialog, wx.ID_CANCEL))
        btnsizer.Realize()
        super_sizer.Add(btnsizer)
        super_sizer.Add((2, 2))
        dialog.Fit()
        dialog.CenterOnParent()
        try:
            if dialog.ShowModal() == wx.ID_OK:
                self.debug_choose_group(lb.GetSelection())
        finally:
            dialog.Destroy()

    def on_debug_choose_image_set(self, event):
        """Choose one of the current image sets

        """

        def feature_cmp(x, y):
            if "_" not in x or "_" not in y:
                return cmp(x, y)
            (cx, fx), (cy, fy) = [z.split("_", 1) for z in (x, y)]
            #
            # For image names, group image file, path and frame consecutively
            #
            # Put metadata first.
            #
            file_md_order = (
                C_FILE_NAME,
                C_PATH_NAME,
                C_FRAME,
            )
            cx_is_file_md, cy_is_file_md = [cz in file_md_order for cz in (cx, cy)]
            if cx_is_file_md:
                if not cy_is_file_md:
                    return 1
                elif fx != fy:
                    return cmp(fx, fy)
                else:
                    cx_priority, cy_priority = [
                        file_md_order.index(cz) for cz in (cx, cy)
                    ]
                    return cmp(cx_priority, cy_priority)
            elif cy_is_file_md:
                return -1
            else:
                return cmp(x, y)

        m = self.__debug_measurements
        features = sorted(
            [
                f
                for f in m.get_feature_names(IMAGE)
                if f.split("_")[0] in (C_METADATA, C_FILE_NAME, C_PATH_NAME, C_FRAME,)
            ],
            key=cmp_to_key(feature_cmp),
        )
        image_numbers = numpy.array(self.__groupings[self.__grouping_index][1], int)
        columns = dict([(f, m[IMAGE, f, image_numbers]) for f in features])
        choices = {}
        for i, image_number in enumerate(image_numbers):
            choices[image_number] = [columns[f][i] for f in features]

        if len(choices) == 0:
            wx.MessageBox(
                "Sorry, there are no available images. Check your LoadImages module's settings",
                "Can't choose image",
            )
            return
        if len(choices) > 1:
            # Get rid of columns with redundant info
            useless_columns = []
            cvalues = list(choices.values())
            for i, f in enumerate(features):
                if all([cv[i] == cvalues[0][i] for cv in cvalues[1:]]):
                    useless_columns.insert(0, i)
            for i in useless_columns:
                for cv in cvalues:
                    del cv[i]
                del features[i]

        class ListCtrlAndWidthMixin(
            wx.ListCtrl, wx.lib.mixins.listctrl.ListCtrlAutoWidthMixin
        ):
            pass

        class ChooseImageSetDialog(wx.Dialog, wx.lib.mixins.listctrl.ColumnSorterMixin):
            def __init__(self, parent):
                dlg_size = get_choose_image_set_frame_size()
                if dlg_size is None:
                    dlg_size = wx.DefaultSize
                wx.Dialog.__init__(
                    self,
                    parent,
                    title="Choose an image cycle",
                    size=dlg_size,
                    style=wx.RESIZE_BORDER | wx.DEFAULT_DIALOG_STYLE,
                )
                super_sizer = wx.BoxSizer(wx.VERTICAL)
                self.SetSizer(super_sizer)
                super_sizer.Add(
                    wx.StaticText(self, label="Select an image cycle for testing:"),
                    0,
                    wx.EXPAND | wx.ALL,
                    5,
                )
                self.list_ctrl = ListCtrlAndWidthMixin(self, style=wx.LC_REPORT)
                self.list_ctrl.InsertColumn(0, "Image #")
                total_width = self.list_ctrl.GetFullTextExtent("Image #")[0]
                for i, f in enumerate(features):
                    if f.startswith(C_METADATA):
                        name = f[(len(C_METADATA) + 1) :]
                    elif f.startswith(C_FILE_NAME):
                        name = f[(len(C_FILE_NAME) + 1) :]
                    elif f.startswith(C_FRAME):
                        name = f[(len(C_FRAME) + 1) :] + " frame"
                    else:
                        name = f[(len(C_PATH_NAME) + 1) :] + " folder"
                    self.list_ctrl.InsertColumn(i + 1, name)
                    width = 0
                    for row in list(choices.values()):
                        w, h, _, _ = self.list_ctrl.GetFullTextExtent(str(row[i]))
                        if w > width:
                            width = w
                    self.list_ctrl.SetColumnWidth(i + 1, width + 15)
                    total_width += width + 25
                total_width += 30
                self.list_ctrl.SetMinSize(
                    wx.Size(min(total_width, 640), self.list_ctrl.GetMinHeight())
                )
                self.itemDataMap = dict(
                    [
                        (
                            k,
                            [
                                "%06d" % v
                                if isinstance(v, int)
                                else "%020.10f" % v
                                if isinstance(v, float)
                                else str(v)
                                for v in [k] + choices[k]
                            ],
                        )
                        for k in choices
                    ]
                )

                for image_number in sorted(choices.keys()):
                    row = [str(image_number)] + [str(x) for x in choices[image_number]]
                    pos = self.list_ctrl.Append(row)
                    self.list_ctrl.SetItemData(pos, image_number)
                wx.lib.mixins.listctrl.ColumnSorterMixin.__init__(
                    self, self.list_ctrl.GetColumnCount()
                )
                super_sizer.Add(self.list_ctrl, 1, wx.EXPAND | wx.ALL, 10)
                super_sizer.Add(wx.StaticLine(self), 0, wx.EXPAND | wx.ALL, 5)
                btnsizer = wx.StdDialogButtonSizer()
                btnsizer.AddButton(wx.Button(self, wx.ID_OK))
                btnsizer.AddButton(wx.Button(self, wx.ID_CANCEL))
                btnsizer.Realize()
                super_sizer.Add(btnsizer)
                super_sizer.Add((2, 2))
                self.Layout()
                self.CenterOnParent()
                self.Bind(wx.EVT_SIZE, self.on_size)

            @staticmethod
            def on_size(event):
                size = event.GetSize()

                assert isinstance(event, wx.SizeEvent)

                set_choose_image_set_frame_size(size.width, size.height)

                event.Skip(True)

            def GetListCtrl(self):
                return self.list_ctrl

        with ChooseImageSetDialog(self.__frame) as dialog:
            if self.__within_group_index < len(choices):
                dialog.list_ctrl.Select(self.__within_group_index)
                dialog.list_ctrl.EnsureVisible(self.__within_group_index)
            else:
                dialog.list_ctrl.Select(0)
            if dialog.ShowModal() == wx.ID_OK:
                selection = dialog.list_ctrl.GetFirstSelected()
                if selection == -1:
                    return
                image_number = dialog.list_ctrl.GetItemData(selection)
                self.__debug_measurements.next_image_set(image_number)
                self.__pipeline_list_view.reset_debug_module()
                for i, (grouping, image_numbers) in enumerate(self.__groupings):
                    if image_number in image_numbers:
                        self.__grouping_index = i
                        self.__within_group_index = list(image_numbers).index(
                            image_number
                        )
                        break
                self.debug_init_imageset()

    def debug_init_imageset(self):
        """Initialize the current image set by running the input modules"""
        for module in self.__pipeline.modules():
            if module.is_input_module():
                if not self.do_step(module, False):
                    return False
        modules = list(
            filter((lambda m: not m.is_input_module()), self.__pipeline.modules())
        )
        #
        # Select the first executable module
        #
        if len(modules) > 0:
            self.__pipeline_list_view.reset_debug_module()
        return True

    def on_debug_reload(self, event):
        """Reload modules from source, warning the user if the pipeline could
        not be reinstantiated with the new versions.

        """
        success = self.__pipeline.reload_modules()
        if not success:
            wx.MessageBox(
                (
                    "CellProfiler has reloaded modules from source, but "
                    "couldn't reinstantiate the pipeline with the new modules.\n"
                    "See the log for details."
                ),
                "Error reloading modules.",
                wx.ICON_ERROR | wx.OK,
            )

    def on_run_from_this_module(self, event):
        active_module = self.__pipeline_list_view.get_active_module()
        self.__pipeline_list_view.set_current_debug_module(active_module)
        self.on_debug_continue(event)

    def on_step_from_this_module(self, event):
        active_module = self.__pipeline_list_view.get_active_module()
        self.__pipeline_list_view.set_current_debug_module(active_module)
        success = self.do_step(active_module)
        if success:
            self.next_debug_module()

    def on_step_from_specific_module(self, event):
        target_module = event.module
        self.__pipeline_list_view.set_current_debug_module(target_module)
        success = self.do_step(target_module)
        if success:
            self.next_debug_module()

    def on_view_workspace(self, event):
        if not self.is_in_debug_mode():
            wx.MessageBox(
                "Workspace Viewer is only available in Test Mode",
                style=wx.OK | wx.ICON_ERROR,
                parent=self.__frame,
            )
            return
        elif self.__pipeline.volumetric():
            wx.MessageBox(
                "Workspace Viewer is currently only available with 2D Pipelines",
                style=wx.OK | wx.ICON_ERROR,
                parent=self.__frame,
            )
            return
        # Need to pack the measurements into the workspace
        workspace = cellprofiler.gui._workspace_model.Workspace(
            self.__pipeline,
            None,
            self.__debug_measurements,
            self.__debug_object_set,
            self.__debug_measurements,
            self.__debug_image_set_list,
        )
        if self.workspace_view is None:
            self.workspace_view = WorkspaceView(self.__frame, workspace)
        else:
            self.workspace_view.set_workspace(workspace)
            self.workspace_view.frame.Show()
            self.workspace_view.frame.Raise()

    def on_sample_init(self, event):
        if self.__module_view is not None:
            if self.__module_view.get_current_module() is not None:
                self.show_parameter_sample_options(
                    self.__module_view.get_current_module().get_module_num(), event
                )
            else:
                LOGGER.info("No current module")

    def show_parameter_sample_options(self, module_num, event):
        if self.__parameter_sample_frame is None:
            selected_module = self.__pipeline.module(module_num)
            selected_module.test_valid(self.__pipeline)

            top_level_frame = self.__frame
            self.parameter_sample_frame = cellprofiler.gui.parametersampleframe.ParameterSampleFrame(
                top_level_frame, selected_module, self.__pipeline, -1
            )
            self.parameter_sample_frame.Bind(
                wx.EVT_CLOSE, self.on_parameter_sample_frame_close
            )
            self.parameter_sample_frame.Show(True)

    def on_parameter_sample_frame_close(self, event):
        event.Skip()
        self.__parameter_sample_frame = None

    # ~^~

    def show_analysis_complete(self, n_image_sets):
        """Show the "Analysis complete" dialog"""
        dlg = wx.Dialog(self.__frame, -1, "Analysis complete")
        sizer = wx.BoxSizer(wx.VERTICAL)
        dlg.SetSizer(sizer)
        sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(sub_sizer, 1, wx.EXPAND)
        text_ctrl = wx.StaticText(
            dlg,
            label="Finished processing %d image sets. Any saved images\n"
            "or exported output files specified by your pipeline\n"
            "have been saved in your designated locations.\n\n"
            "Note that the module display windows may not show\n"
            "the final image cycle on computers with multiple\n"
            "processing cores." % n_image_sets,
        )
        sub_sizer.Add(
            text_ctrl, 1, wx.EXPAND | wx.ALL, 10,
        )
        bitmap = wx.ArtProvider.GetBitmap(
            wx.ART_INFORMATION, wx.ART_CMN_DIALOG, size=(32, 32)
        )
        sub_sizer.Add(wx.StaticBitmap(dlg, -1, bitmap), 0, wx.EXPAND | wx.ALL, 10)
        dont_show_again = wx.CheckBox(dlg, -1, "Don't show this again")
        dont_show_again.SetValue(False)
        sizer.Add(dont_show_again, 0, wx.ALIGN_LEFT | wx.LEFT, 10)
        button_sizer = wx.StdDialogButtonSizer()
        save_pipeline_button = wx.Button(dlg, -1, "Save project")
        button_sizer.AddButton(save_pipeline_button)
        if sys.platform in ("darwin", "win32"):
            open_default_output_folder_button = wx.Button(
                dlg, -1, "Open default output folder"
            )
            button_sizer.SetNegativeButton(open_default_output_folder_button)
            button_sizer.AddButton(open_default_output_folder_button)

            def on_open_default_output_folder(event):
                import subprocess

                if sys.platform == "darwin":
                    subprocess.call(
                        ["open", get_default_output_directory(),]
                    )
                elif sys.platform == "win32":
                    subprocess.call(
                        ["explorer", os.path.abspath(get_default_output_directory())]
                    )

            open_default_output_folder_button.Bind(
                wx.EVT_BUTTON, on_open_default_output_folder
            )
        button_sizer.SetCancelButton(save_pipeline_button)
        button_sizer.AddButton(wx.Button(dlg, wx.ID_OK))
        sizer.Add(button_sizer, 0, wx.EXPAND | wx.ALL, 10)

        def on_save_workspace(event):
            self.__on_save_workspace(event)
            wx.MessageBox(
                "Saved project %s" % get_current_workspace_path(),
                caption="Saved project",
                parent=self.__frame,
            )

        save_pipeline_button.Bind(wx.EVT_BUTTON, on_save_workspace)

        button_sizer.Realize()
        dlg.Fit()
        dlg.CenterOnParent()
        try:
            dlg.ShowModal()
            if dont_show_again.GetValue():
                set_show_analysis_complete_dlg(False)
        finally:
            dlg.Destroy()

    def show_exiting_test_mode(self):
        """Show the "Analysis complete" dialog"""
        dlg = wx.Dialog(self.__frame, -1, "Exiting test mode")
        sizer = wx.BoxSizer(wx.VERTICAL)
        dlg.SetSizer(sizer)
        sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(sub_sizer, 1, wx.EXPAND)
        text_ctrl = wx.StaticText(
            dlg,
            label=(
                "You have changed the pipeline so\n" "that test mode will now exit.\n"
            ),
        )
        sub_sizer.Add(
            text_ctrl, 1, wx.EXPAND | wx.ALL, 10,
        )
        bitmap = wx.ArtProvider.GetBitmap(
            wx.ART_INFORMATION, wx.ART_CMN_DIALOG, size=(32, 32)
        )
        sub_sizer.Add(wx.StaticBitmap(dlg, -1, bitmap), 0, wx.EXPAND | wx.ALL, 10)
        dont_show_again = wx.CheckBox(dlg, -1, "Don't show this again")
        dont_show_again.SetValue(False)
        sizer.Add(dont_show_again, 0, wx.ALIGN_LEFT | wx.LEFT, 10)
        button_sizer = wx.StdDialogButtonSizer()
        button_sizer.AddButton(wx.Button(dlg, wx.ID_OK))
        sizer.Add(button_sizer, 0, wx.EXPAND | wx.ALL, 10)
        button_sizer.Realize()
        dlg.Fit()
        dlg.CenterOnParent()
        try:
            dlg.ShowModal()
            if dont_show_again.GetValue():
                set_show_exiting_test_mode_dlg(False)
        finally:
            dlg.Destroy()

    def on_show_all_windows(self, event):
        """Turn "show_window" on for every module in the pipeline"""
        with self.__pipeline.undoable_action("Show all windows"):
            for module in self.__pipeline.modules(exclude_disabled=False):
                self.__pipeline.show_module_window(module, True)

    def on_hide_all_windows(self, event):
        """Turn "show_window" off for every module in the pipeline"""
        with self.__pipeline.undoable_action("Hide all windows"):
            for module in self.__pipeline.modules(exclude_disabled=False):
                self.__pipeline.show_module_window(module, False)

    def run_pipeline(self):
        """Run the current pipeline, returning the measurements
        """
        return self.__pipeline.Run(self.__frame)


# TODO: wx 3.0 seems to have broken the composite drop target functionality
#       so I am reverting to only allowing files, hence the commented-out code
#
class FLDropTarget(wx.FileDropTarget):
    """A generic drop target (for the path list)"""

    def __init__(self, file_callback_fn, text_callback_fn):
        super(self.__class__, self).__init__()
        self.file_callback_fn = file_callback_fn
        self.text_callback_fn = text_callback_fn
        self.file_data_object = wx.FileDataObject()
        self.text_data_object = wx.TextDataObject()
        self.composite_data_object = wx.DataObjectComposite()
        self.composite_data_object.Add(self.file_data_object, True)
        self.composite_data_object.Add(self.text_data_object)
        # self.SetDataObject(self.composite_data_object)

    def OnDropFiles(self, x, y, filenames):
        self.file_callback_fn(x, y, filenames)
        return True

    def OnDropText(self, x, y, text):
        self.text_callback_fn(x, y, text)
        return True

    def OnEnter(self, x, y, d):
        return wx.DragCopy

    def OnDragOver(self, x, y, d):
        return wx.DragCopy

    # def OnData(self, x, y, d):
    #    if self.GetData():
    #        df = self.composite_data_object.GetReceivedFormat().GetType()
    #        if df in (wx.DF_TEXT, wx.DF_UNICODETEXT):
    #            self.OnDropText(x, y, self.text_data_object.GetText())
    #        elif df == wx.DF_FILENAME:
    #            self.OnDropFiles(x, y,
    #                             self.file_data_object.GetFilenames())
    #    return wx.DragCopy
    #
    # @staticmethod
    # def OnDrop(x, y):
    #    return True
