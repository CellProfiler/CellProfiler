"""PipelineController.py - controls (modifies) a pipeline

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2013 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

import csv
import h5py
import logging
import math
import numpy
import wx
import os
import re
import shutil
import sys
import Queue
import cpframe
import random
import string
import hashlib
from cStringIO import StringIO
import threading
import urllib

import cellprofiler.pipeline as cpp
import cellprofiler.preferences as cpprefs
import cellprofiler.cpimage as cpi
import cellprofiler.measurements as cpm
import cellprofiler.workspace as cpw
import cellprofiler.objects as cpo
from cellprofiler.gui.addmoduleframe import AddModuleFrame
from cellprofiler.gui import get_cp_bitmap
import cellprofiler.gui.moduleview
from cellprofiler.gui.movieslider import EVT_TAKE_STEP
from cellprofiler.gui.help import HELP_ON_MODULE_BUT_NONE_SELECTED, PLATEVIEWER_HELP
from cellprofiler.gui.bitmaplabelbutton import BitmapLabelButton
import cellprofiler.utilities.version as version
from errordialog import display_error_dialog, ED_CONTINUE, ED_STOP, ED_SKIP
from errordialog import display_error_message, clear_old_errors
from runmultiplepipelinesdialog import RunMultplePipelinesDialog
from cellprofiler.modules.loadimages import C_FILE_NAME, C_PATH_NAME, C_FRAME
from cellprofiler.modules.loadimages import pathname2url
import cellprofiler.gui.parametersampleframe as psf
import cellprofiler.analysis as cpanalysis
import cellprofiler.cpmodule as cpmodule
import cellprofiler.gui.loadsavedlg as cplsdlg
import cellprofiler.utilities.walk_in_background as W
from cellprofiler.gui.omerologin import OmeroLoginDlg
from cellprofiler.icons import get_builtin_image
from cellprofiler.gui.htmldialog import HTMLDialog

logger = logging.getLogger(__name__)
RECENT_PIPELINE_FILE_MENU_ID = [wx.NewId() for i in range(cpprefs.RECENT_FILE_COUNT)]
RECENT_WORKSPACE_FILE_MENU_ID = [wx.NewId() for i in range(cpprefs.RECENT_FILE_COUNT)]
WRITING_MAT_FILE = "Writing .MAT measurements file..."
WROTE_MAT_FILE = ".MAT measurements file has been saved"

class PipelineController:
    """Controls the pipeline through the UI
    
    """
    def __init__(self, workspace, frame):
        self.__workspace = workspace
        pipeline = self.__pipeline = workspace.pipeline
        pipeline.add_listener(self.__on_pipeline_event)
        self.__analysis = None
        self.__frame = frame
        self.__add_module_frame = AddModuleFrame(frame,-1,"Add modules")
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
        cpprefs.add_image_directory_listener(self.__on_image_directory_change)
        cpprefs.add_output_directory_listener(self.__on_output_directory_change)

        # interaction/display requests and exceptions from an Analysis
        self.interaction_request_queue = Queue.PriorityQueue()
        self.interaction_pending = False
        self.debug_request_queue = None

        self.populate_recent_files()
        self.menu_id_to_module_name = {}
        self.module_name_to_menu_id = {}
        self.populate_edit_menu(self.__frame.menu_edit_add_module)
        assert isinstance(frame, wx.Frame)
        frame.Bind(wx.EVT_MENU, self.__on_new_workspace,
                   id = cpframe.ID_FILE_NEW_WORKSPACE)
        wx.EVT_MENU(frame, cpframe.ID_FILE_LOAD,
                    self.__on_open_workspace)
        wx.EVT_MENU(frame, cpframe.ID_FILE_SAVE,
                    self.__on_save_workspace)
        wx.EVT_MENU(frame, cpframe.ID_FILE_SAVE_AS, 
                    self.__on_save_as_workspace)
        wx.EVT_MENU(frame, cpframe.ID_FILE_LOAD_PIPELINE,
                    self.__on_load_pipeline)
        wx.EVT_MENU(frame, cpframe.ID_FILE_URL_LOAD_PIPELINE, self.__on_url_load_pipeline)
        wx.EVT_MENU(frame, cpframe.ID_FILE_SAVE_PIPELINE,self.__on_save_as_pipeline)
        wx.EVT_MENU(frame, cpframe.ID_FILE_EXPORT_IMAGE_SETS,
                    self.__on_export_image_sets)
        wx.EVT_MENU(frame, cpframe.ID_FILE_REVERT_TO_SAVED, 
                    self.__on_revert_workspace)
        wx.EVT_MENU(frame, cpframe.ID_FILE_CLEAR_PIPELINE,self.__on_clear_pipeline)
        wx.EVT_MENU(frame, cpframe.ID_FILE_PLATEVIEWER, self.__on_plateviewer)
        wx.EVT_MENU(frame, cpframe.ID_FILE_ANALYZE_IMAGES,self.on_analyze_images)
        wx.EVT_MENU(frame, cpframe.ID_FILE_STOP_ANALYSIS,self.on_stop_running)
        wx.EVT_MENU(frame, cpframe.ID_FILE_RUN_MULTIPLE_PIPELINES, self.on_run_multiple_pipelines)
        wx.EVT_MENU(frame, cpframe.ID_FILE_RESTART, self.on_restart)
        
        wx.EVT_MENU(frame, cpframe.ID_EDIT_UNDO, self.on_undo)
        frame.Bind(wx.EVT_UPDATE_UI, self.on_update_undo_ui, id=cpframe.ID_EDIT_UNDO)
        wx.EVT_MENU(frame, cpframe.ID_EDIT_MOVE_UP, self.on_module_up)
        wx.EVT_MENU(frame, cpframe.ID_EDIT_MOVE_DOWN, self.on_module_down)
        wx.EVT_MENU(frame, cpframe.ID_EDIT_DELETE, self.on_remove_module)
        wx.EVT_MENU(frame, cpframe.ID_EDIT_DUPLICATE, self.on_duplicate_module)
        
        wx.EVT_MENU(frame, cpframe.ID_EDIT_BROWSE_FOR_FILES, 
                    self.on_pathlist_browse)
        wx.EVT_MENU(frame, cpframe.ID_EDIT_CLEAR_FILE_LIST,
                    self.on_pathlist_clear)
        wx.EVT_MENU(frame, cpframe.ID_EDIT_COLLAPSE_ALL,
                    self.on_pathlist_collapse_all)
        wx.EVT_MENU(frame, cpframe.ID_EDIT_EXPAND_ALL,
                    self.on_pathlist_expand_all)
        wx.EVT_MENU(frame, cpframe.ID_EDIT_REMOVE_FROM_FILE_LIST,
                    self.on_pathlist_remove)
        wx.EVT_MENU(frame, cpframe.ID_EDIT_SHOW_FILE_LIST_IMAGE,
                    self.on_pathlist_show)
        for menu_id in (cpframe.ID_EDIT_BROWSE_FOR_FILES, 
                        cpframe.ID_EDIT_CLEAR_FILE_LIST,
                        cpframe.ID_EDIT_COLLAPSE_ALL,
                        cpframe.ID_EDIT_EXPAND_ALL,
                        cpframe.ID_EDIT_REMOVE_FROM_FILE_LIST,
                        cpframe.ID_EDIT_SHOW_FILE_LIST_IMAGE):
            frame.Bind(wx.EVT_UPDATE_UI, self.on_update_pathlist_ui,
                       id = menu_id)
        
        wx.EVT_MENU(frame,cpframe.ID_DEBUG_TOGGLE,self.on_debug_toggle)
        wx.EVT_MENU(frame,cpframe.ID_DEBUG_STEP,self.on_debug_step)
        wx.EVT_MENU(frame,cpframe.ID_DEBUG_NEXT_IMAGE_SET,self.on_debug_next_image_set)
        wx.EVT_MENU(frame,cpframe.ID_DEBUG_NEXT_GROUP, self.on_debug_next_group)
        wx.EVT_MENU(frame,cpframe.ID_DEBUG_CHOOSE_GROUP, self.on_debug_choose_group)
        wx.EVT_MENU(frame,cpframe.ID_DEBUG_CHOOSE_IMAGE_SET, self.on_debug_choose_image_set)
        wx.EVT_MENU(frame,cpframe.ID_DEBUG_CHOOSE_RANDOM_IMAGE_SET, self.on_debug_random_image_set)
        wx.EVT_MENU(frame,cpframe.ID_DEBUG_RELOAD, self.on_debug_reload)

        # ~*~
        wx.EVT_MENU(frame, cpframe.ID_SAMPLE_INIT, self.on_sample_init)
        # ~^~
        
        wx.EVT_MENU(frame,cpframe.ID_WINDOW_SHOW_ALL_WINDOWS, self.on_show_all_windows)
        wx.EVT_MENU(frame,cpframe.ID_WINDOW_HIDE_ALL_WINDOWS, self.on_hide_all_windows)
        
        from bioformats.formatreader import set_omero_login_hook
        set_omero_login_hook(self.omero_login)
        
    def start(self, workspace_file, pipeline_path):
        '''Do initialization after GUI hookup
        
        Perform steps that need to happen after all of the user interface
        elements have been initialized.
        '''
        if workspace_file is not None:
            self.do_open_workspace(workspace_file, 
                                   load_pipeline=(pipeline_path is None))
        else:
            self.do_create_workspace()
        if pipeline_path is not None:
            self.do_load_pipeline(pipeline_path)
            
    def attach_to_pipeline_list_view(self, pipeline_list_view):
        """Glom onto events from the list box with all of the module names in it
        
        """
        self.__pipeline_list_view = pipeline_list_view
        
    def attach_to_path_list_ctrl(self, 
                                 path_list_ctrl, 
                                 path_list_filtered_files_checkbox):
        '''Attach the pipeline controller to the path_list_ctrl
        
        This lets the pipeline controller populate the path list as
        it changes.
        '''
        self.__path_list_ctrl = path_list_ctrl
        self.__path_list_is_filtered = None
        self.__path_list_filter_checkbox = path_list_filtered_files_checkbox
        self.__path_list_filter_checkbox.Value = \
            self.__path_list_ctrl.get_show_disabled()
        
        path_list_ctrl.set_context_menu_fn(
            self.get_pathlist_file_context_menu,
            self.get_pathlist_folder_context_menu,
            self.on_pathlist_file_command,
            self.on_pathlist_folder_command)
        path_list_ctrl.set_delete_fn(self.on_pathlist_file_delete)
        self.path_list_drop_target = FLDropTarget(
            self.on_pathlist_drop_files,
            self.on_pathlist_drop_text)
        path_list_ctrl.SetDropTarget(self.path_list_drop_target)
        
        def show_disabled(event):
            self.__path_list_ctrl.set_show_disabled(
                self.__path_list_filter_checkbox.Value)
        self.__path_list_filter_checkbox.Bind(wx.EVT_CHECKBOX, show_disabled)
        
    def set_path_list_filtering(self, use_filter):
        '''Update the path list UI according to the filter on/off state
        
        use_filter - True if filtering, False if all files enabled.
        '''
        use_filter = bool(use_filter)
        if self.__path_list_is_filtered is not use_filter:
            sizer = self.__path_list_filter_checkbox.GetContainingSizer()
            if sizer is not None:
                sizer.Show(self.__path_list_filter_checkbox, use_filter)
            if not use_filter:
                self.__path_list_ctrl.enable_all_paths()
            self.__path_list_is_filtered = use_filter
        
    def attach_to_module_view(self,module_view):
        """Listen for setting changes from the module view
        
        """
        self.__module_view = module_view
        module_view.add_listener(self.__on_module_view_event)
    
    def attach_to_module_controls_panel(self,module_controls_panel):
        """Attach the pipeline controller to the module controls panel
        
        Attach the pipeline controller to the module controls panel.
        In addition, the PipelineController gets to add whatever buttons it wants to the
        panel.
        """
        self.__module_controls_panel = module_controls_panel
        mcp_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.__help_button = wx.Button(self.__module_controls_panel,-1,"?",(0,0), (30, -1))
        self.__help_button.SetToolTipString("Get Help for selected module")
        self.__mcp_text = wx.StaticText(self.__module_controls_panel,-1,"Adjust modules:")
        self.__mcp_add_module_button = wx.Button(self.__module_controls_panel,-1,"+",(0,0), (30, -1))
        self.__mcp_add_module_button.SetToolTipString("Add a module")
        self.__mcp_remove_module_button = wx.Button(
            self.__module_controls_panel, cpframe.ID_EDIT_DELETE, 
            "-",(0,0), (30, -1))
        self.__mcp_remove_module_button.SetToolTipString("Remove selected module")
        self.__mcp_module_up_button = wx.Button(
            self.__module_controls_panel, cpframe.ID_EDIT_MOVE_UP,
            "^",(0,0), (30, -1))
        self.__mcp_module_up_button.SetToolTipString("Move selected module up")
        self.__mcp_module_down_button = wx.Button(
            self.__module_controls_panel, cpframe.ID_EDIT_MOVE_DOWN,
            "v", (0,0), (30, -1))
        self.__mcp_module_down_button.SetToolTipString("Move selected module down")
        mcp_sizer.AddMany([(self.__help_button, 0, wx.ALIGN_CENTER | wx.ALL, 3),
                           ((1, 3), 3),
                           (self.__mcp_text, 0, wx.ALIGN_CENTER | wx.ALL, 3),
                           (self.__mcp_add_module_button, 0, wx.ALIGN_CENTER | wx.ALL, 3),
                           (self.__mcp_remove_module_button, 0, wx.ALIGN_CENTER | wx.ALL, 3),
                           (self.__mcp_module_up_button, 0, wx.ALIGN_CENTER | wx.ALL, 3),
                           (self.__mcp_module_down_button, 0, wx.ALIGN_CENTER | wx.ALL, 3)])
        self.__module_controls_panel.SetSizer(mcp_sizer)
        self.__module_controls_panel.Bind(wx.EVT_BUTTON, self.__on_help, self.__help_button)
        self.__module_controls_panel.Bind(wx.EVT_BUTTON, self.__on_add_module,self.__mcp_add_module_button)
        self.__module_controls_panel.Bind(wx.EVT_BUTTON, self.on_remove_module,self.__mcp_remove_module_button)
        self.__module_controls_panel.Bind(wx.EVT_BUTTON, self.on_module_up,self.__mcp_module_up_button)
        self.__module_controls_panel.Bind(wx.EVT_BUTTON, self.on_module_down,self.__mcp_module_down_button)

            
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
        bkgnd_color = cpprefs.get_background_color()
        assert isinstance(panel, wx.Window)
        self.__test_controls_panel = panel
        panel.SetBackgroundColour(bkgnd_color)
        #
        # There are three sizers, one for each mode:
        # * tcp_launch_sizer - when idle, for launching analysis or test mode
        # * tcp_analysis_sizer - when in analysis mode
        # * tcp_test_sizer - when in test mode
        #
        tcp_sizer = wx.BoxSizer(wx.VERTICAL)
        self.__test_controls_panel.Sizer = tcp_sizer
        self.__tcp_launch_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.__tcp_analysis_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.__tcp_test_sizer = wx.BoxSizer(wx.VERTICAL)
        tcp_sizer.Add(self.__tcp_launch_sizer, 0, wx.EXPAND)
        tcp_sizer.Add(self.__tcp_analysis_sizer, 0, wx.EXPAND)
        tcp_sizer.Add(self.__tcp_test_sizer, 0, wx.EXPAND)
        #
        # Launch sizer
        #
        self.__test_bmp = wx.BitmapFromImage(get_builtin_image("IMG_TEST"))
        self.__test_mode_button = BitmapLabelButton(
                panel, bitmap = self.__test_bmp, label = self.ENTER_TEST_MODE)
        self.__test_mode_button.Bind(wx.EVT_BUTTON, self.on_debug_toggle)
        self.__test_mode_button.SetToolTipString(self.ENTER_TEST_MODE_HELP)         
        
        self.__tcp_launch_sizer.Add(self.__test_mode_button, 1, wx.EXPAND)        

        analyze_bmp = wx.BitmapFromImage(get_builtin_image("IMG_ANALYZE_16"))
        self.__analyze_images_button = BitmapLabelButton(
            panel, bitmap = analyze_bmp, label = self.ANALYZE_IMAGES)
        self.__analyze_images_button.Bind(wx.EVT_BUTTON, self.on_analyze_images)
        self.__analyze_images_button.SetToolTipString(self.ANALYZE_IMAGES_HELP)
        
        self.__tcp_launch_sizer.Add(self.__analyze_images_button, 1, wx.EXPAND)
        
               
        #
        # Analysis sizer
        #
        stop_bmp = wx.BitmapFromImage(get_builtin_image("IMG_STOP"))
        pause_bmp = wx.BitmapFromImage(get_builtin_image("IMG_PAUSE"))
        self.__pause_button = BitmapLabelButton(
            panel, bitmap = pause_bmp, label = self.PAUSE)
        self.__pause_button.Bind(wx.EVT_BUTTON, self.on_pause)
        self.__pause_button.SetToolTipString(self.PAUSE_HELP)
        self.__tcp_analysis_sizer.Add(self.__pause_button, 1, wx.EXPAND)
        
        self.__resume_button = BitmapLabelButton(
            panel, bitmap = analyze_bmp, label = self.RESUME)
        self.__resume_button.Bind(wx.EVT_BUTTON, self.on_resume)
        self.__resume_button.SetToolTipString(self.RESUME_HELP)
        self.__tcp_analysis_sizer.Add(self.__resume_button, 1, wx.EXPAND)
        
        self.__stop_analysis_button = BitmapLabelButton(
            panel, bitmap = stop_bmp, label = 'Stop analysis')
        self.__stop_analysis_button.Bind(wx.EVT_BUTTON, self.on_stop_running)
        self.__stop_analysis_button.SetToolTipString(
            "Cancel the analysis run")
        self.__tcp_analysis_sizer.Add(self.__stop_analysis_button, 1, wx.EXPAND)
        #
        # Test mode sizer
        #
        sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.__tcp_test_sizer.Add(sub_sizer, 1, wx.EXPAND)
        self.__tcp_test_sizer.AddSpacer(2)

        run_bmp = wx.BitmapFromImage(get_builtin_image("IMG_RUN"))
        self.__tcp_continue = BitmapLabelButton(
            panel, label="Run", bitmap = run_bmp)
        self.__tcp_continue.SetToolTip(wx.ToolTip("Run to next pause"))
        self.__tcp_continue.Bind(
            wx.EVT_BUTTON, self.on_debug_continue)
        sub_sizer.Add(self.__tcp_continue, 1, wx.EXPAND)
        
        self.__tcp_step = BitmapLabelButton(
            panel, label = "Step", bitmap = analyze_bmp)
        self.__tcp_step.SetToolTip(wx.ToolTip("Step to next module"))
        self.__tcp_step.Bind(wx.EVT_BUTTON, self.on_debug_step)
        sub_sizer.Add(self.__tcp_step, 1, wx.EXPAND)

        sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.__tcp_test_sizer.Add(sub_sizer, 1, wx.EXPAND)
        
        self.__tcp_stop_testmode = BitmapLabelButton(
            panel, label = "Exit Test Mode", bitmap = stop_bmp)
        self.__tcp_stop_testmode.SetToolTip(wx.ToolTip("Exit test mode"))
        self.__tcp_stop_testmode.Bind(wx.EVT_BUTTON, self.on_debug_stop)        
        sub_sizer.Add(self.__tcp_stop_testmode, 1, wx.EXPAND)        

        next_image_bmp = wx.BitmapFromImage(get_builtin_image("IMG_IMAGE"))
        self.__tcp_next_imageset = BitmapLabelButton(
            panel, label = "Next Image Set", bitmap = next_image_bmp)
        self.__tcp_next_imageset.SetToolTip(wx.ToolTip("Jump to next image set"))        
        self.__tcp_next_imageset.Bind(wx.EVT_BUTTON, self.on_debug_next_image_set)
        sub_sizer.Add(self.__tcp_next_imageset, 1, wx.EXPAND)        

        for child in panel.GetChildren():
            child.SetBackgroundColour(bkgnd_color)

        self.show_launch_controls()
        
    def show_launch_controls(self):
        '''Show the "Analyze images" and "Enter test mode" buttons'''
        self.__test_controls_panel.Sizer.Hide(self.__tcp_test_sizer)
        self.__test_controls_panel.Sizer.Hide(self.__tcp_analysis_sizer)
        self.__test_controls_panel.Sizer.Show(self.__tcp_launch_sizer)
        self.__test_controls_panel.Layout()
        self.__test_controls_panel.Parent.Layout()
        self.__frame.enable_launch_commands()

    def show_analysis_controls(self):
        '''Show the controls that stop and pause analysis'''
        self.__test_controls_panel.Sizer.Hide(self.__tcp_test_sizer)
        self.__test_controls_panel.Sizer.Hide(self.__tcp_launch_sizer)
        self.__test_controls_panel.Sizer.Show(self.__tcp_analysis_sizer)
        self.__stop_analysis_button.Enable(True)
        self.show_pause_button()
        self.__test_controls_panel.Layout()
        self.__test_controls_panel.Parent.Layout()
        self.__frame.enable_analysis_commands()
        
    def show_pause_button(self):
        self.__pause_button.Enable(True)
        self.__tcp_analysis_sizer.Show(self.__pause_button)
        self.__tcp_analysis_sizer.Hide(self.__resume_button)
        self.__test_controls_panel.Layout()
        
    def show_resume_button(self):
        self.__resume_button.Enable(True)
        self.__tcp_analysis_sizer.Hide(self.__pause_button)
        self.__tcp_analysis_sizer.Show(self.__resume_button)
        self.__test_controls_panel.Layout()
        
    def show_test_controls(self):
        '''Show the controls for dealing with test mode'''
        self.__test_controls_panel.Sizer.Show(self.__tcp_test_sizer)
        self.__test_controls_panel.Sizer.Hide(self.__tcp_launch_sizer)
        self.__test_controls_panel.Sizer.Hide(self.__tcp_analysis_sizer)
        self.__test_controls_panel.Layout()
        self.__test_controls_panel.Parent.Layout()
        self.__frame.enable_debug_commands()
        
    def omero_login(self):
        with OmeroLoginDlg(self.__frame, title = "Log into Omero") as dlg:
            dlg.ShowModal()

    def __on_open_workspace(self, event):
        '''Handle the Open Workspace menu command'''
        path = self.do_open_workspace_dlg()
        if path is not None:
            self.do_open_workspace(path, True)
    
    def __on_revert_workspace(self, event):
        path = cpprefs.get_current_workspace_path()
        if path is not None:
            self.do_open_workspace(path, True)
    
    def do_open_workspace_dlg(self):
        '''Display the open workspace dialog, returning the chosen file
        
        returns a path or None if the user canceled. If it returns a path,
        the workspace file is locked.
        '''
        with wx.FileDialog(
            self.__frame,
            "Choose a workspace file to open",
            wildcard = "CellProfiler workspace (*.cpi)|*.cpi|All files (*.*)|*.*") as dlg:
            dlg.Directory = cpprefs.get_default_output_directory()
            if dlg.ShowModal() == wx.ID_OK:
                return dlg.Path
            return None
    
    def do_open_workspace(self, filename, load_pipeline=True):
        '''Open the given workspace file
        
        filename - the path to the file to open. It should already be locked.
        '''
        progress_callback_fn = None
        message = "Loading %s" % filename
        with wx.ProgressDialog(
                parent = self.__frame,
                title = "Opening project",
                message= message,
                style=wx.PD_CAN_ABORT|wx.PD_APP_MODAL) as dlg:
            try:
                assert isinstance(dlg, wx.ProgressDialog)
                dlg.longest_msg_len = dlg.GetTextExtent(message)[0]
                    
                def progress_callback(operation_id, progress, message):
                    if progress != 1:
                        proceed, skip = dlg.Pulse(message)
                        if not proceed:
                            raise Exception("User cancelled opening workspace")
                        msg_len = dlg.GetTextExtent(message)[0]
                        if msg_len > dlg.longest_msg_len:
                            dlg.longest_msg_len = msg_len
                            dlg.Fit()
                cpprefs.add_progress_callback(progress_callback)
                progress_callback_fn = progress_callback
                    
                self.__workspace.load(filename, load_pipeline)
                cpprefs.set_workspace_file(filename)
                cpprefs.set_current_workspace_path(filename)
                self.__pipeline.load_image_plane_details(self.__workspace)
                if not load_pipeline:
                    self.__workspace.measurements.clear()
                    self.__workspace.save_pipeline_to_measurements()
                self.__dirty_workspace = False
                self.set_title()
            finally:
                cpprefs.remove_progress_callback(progress_callback_fn)
            
    def __on_new_workspace(self, event):
        '''Handle the New Workspace menu command'''
        if self.__dirty_workspace:
            result =  wx.MessageBox(
                "Do you want to save your existing project?",
                caption = "Save project",
                style = wx.YES_NO | wx.CANCEL | wx.ICON_QUESTION,
                parent = self.__frame)
            if result == wx.ID_CANCEL:
                return
            elif result == wx.ID_YES:
                if cpprefs.get_current_workspace_path() is None:
                    if not self.do_save_as_workspace():
                        return
                else:
                    self.do_save_workspace()
        self.do_create_workspace()
        
    def do_create_workspace(self):
        '''Create a new workspace file'''
        self.stop_debugging()
        if self.is_running():
            self.stop_running()            
        self.__workspace.create()
        self.__pipeline.clear_image_plane_details()
        self.__pipeline.clear()
        self.__clear_errors()
        self.__workspace.measurements.clear()
        self.__workspace.save_pipeline_to_measurements()
        self.__dirty_workspace = False
        cpprefs.set_current_workspace_path(None)
        self.__pipeline_list_view.select_one_module(1)
        self.enable_module_controls_panel_buttons()
        self.set_title()
        
    def __on_save_as_workspace(self, event):
        '''Handle the Save Workspace As menu command'''
        self.do_save_as_workspace()
        
    def do_save_as_workspace(self):
        with wx.FileDialog(
            self.__frame,
            "Save project file as",
            wildcard = "CellProfiler project (*.cpi)|*.cpi",
            style = wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            dlg.Directory = cpprefs.get_default_output_directory()
            if dlg.ShowModal() == wx.ID_OK:
                self.do_save_workspace(dlg.Path)
                cpprefs.set_current_workspace_path(dlg.Path)
                cpprefs.set_workspace_file(dlg.Path)
                self.set_title()
                return True
            return False
                
    def __on_save_workspace(self, event):
        '''Handle the Save Workspace menu command'''
        path = cpprefs.get_current_workspace_path()
        if path is None:
            self.do_save_as_workspace()
        else:
            self.do_save_workspace(path)
                
    def do_save_workspace(self, filename):
        '''Create a copy of the current workspace file'''
        self.__workspace.save(filename)
        cpprefs.set_workspace_file(filename)
        self.__dirty_workspace = False
        self.set_title()
        return True

    def __on_load_pipeline(self, event):
        dlg = wx.FileDialog(self.__frame,
                            "Choose a pipeline file to open",
                            wildcard = ("CellProfiler pipeline (*.cp,*.cpi,*.mat,*.h5)|*.cp;*.cpi;*.mat;*.h5"))
        dlg.Directory = cpprefs.get_default_output_directory()
        if dlg.ShowModal()==wx.ID_OK:
            pathname = os.path.join(dlg.GetDirectory(),dlg.GetFilename())
            self.do_load_pipeline(pathname)
        dlg.Destroy()
            
    def __on_url_load_pipeline(self, event):
        dlg = wx.TextEntryDialog(self.__frame,
                                 "Enter the pipeline's URL\n\n"
                                 "Example: https://svn.broadinstitute.org/"
                                 "CellProfiler/trunk/ExampleImages/"
                                 "ExampleSBSImages/ExampleSBS.cp",
                                 "Load pipeline via URL")
        if dlg.ShowModal() == wx.ID_OK:
            import urllib2
            self.do_load_pipeline(urllib2.urlopen(dlg.Value))
        dlg.Destroy()
    
    def is_running(self):
        return self.__analysis is not None
    
    def do_load_pipeline(self, pathname):
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
            if self.__pipeline.can_convert_legacy_input_modules():
                # Note: the length of the longest line of text also
                #       controls the size of the directory entry text box
                text = (
"Your pipeline contains the legacy module LoadImages, and legacy references\n"
"to the Default Input Folder. CellProfiler can convert this pipeline by:\n\n"
"(1) Using the new input modules (Images, Metadata, NamesAndTypes, Groups).\n"
"(2) Using an existing folder instead of the Default Input Folder.\n\n"
"If you choose to convert the pipeline, you should then make sure to provide\n"
"your original images to the Images module as input, and confirm that your\n"
"metadata (if any) is provided to the Metadata module.")
                CANCEL = 0
                CONVERT = 1
                DONT_CONVERT = 2
                
                with wx.Dialog(self.__frame,
                               title = "Convert legacy pipeline?") as dlg:
                    import wx.lib.filebrowsebutton as filebrowse
                    #
                    # Structure:
                    # 
                    # dialog sizer
                    #    vsizer
                    #       sizer (horizontal)
                    #           static bitmap
                    #           static text
                    #       static box
                    #       static box sizer (vertical)
                    #           rb_convert
                    #           dir_ctrl
                    #           rb_dont_convert
                    #    standard dialog button sizer
                    #       ID_OK button
                    #       ID_CANCEL button
                    #
                    dlg.Sizer = wx.BoxSizer(wx.VERTICAL)
                    vsizer = wx.BoxSizer(wx.VERTICAL)
                    dlg.Sizer.Add(vsizer, 0, wx.EXPAND | wx.ALL, 10)
                    sizer = wx.BoxSizer(wx.HORIZONTAL)
                    vsizer.Add(sizer, 0, wx.EXPAND | wx.ALL)
                    bmp = wx.ArtProvider.GetBitmap(wx.ART_QUESTION,
                                                   wx.ART_CMN_DIALOG)
                    sizer.Add(wx.StaticBitmap(dlg, bitmap = bmp), 0,
                              wx.ALIGN_LEFT | wx.ALIGN_TOP)
                    sizer.AddSpacer(8)
                    sizer.Add(wx.StaticText(dlg, label=text), 
                              0, wx.ALIGN_LEFT | wx.ALIGN_TOP)
                    vsizer.AddSpacer(8)
                    lmargin = bmp.GetSize()[0] + 8
                    group_ctrl = wx.StaticBox(dlg, label="Conversion choices")
                    sizer = wx.StaticBoxSizer(group_ctrl, wx.VERTICAL)
                    vsizer.Add(sizer, 0, wx.EXPAND | wx.LEFT, lmargin)
                    rb_convert = wx.RadioButton(
                        dlg, 
                        label = "Convert legacy modules and the default input folder",
                        style = wx.RB_GROUP)
                    sizer.Add(rb_convert, 0, wx.ALIGN_LEFT)
                    sizer.AddSpacer(4)
                    dir_ctrl = filebrowse.DirBrowseButton(
                        dlg, labelText = "Folder",
                        dialogTitle = "Browse for default input folder",
                        startDirectory = cpprefs.get_default_image_directory())
                    dir_ctrl.SetValue(cpprefs.get_default_image_directory())
                    sizer.Add(dir_ctrl, 1, wx.EXPAND | wx.LEFT, 10)
                    sizer.AddSpacer(4)
                    rb_dont_convert = wx.RadioButton(
                        dlg, label = "Do not convert")
                    sizer.Add(rb_dont_convert, 0, wx.ALIGN_LEFT)
                    vsizer.AddSpacer(8)
                    btn_sizer = wx.StdDialogButtonSizer()
                    dlg.Sizer.Add(btn_sizer, 0, wx.ALIGN_RIGHT)
                    btn_sizer.AddButton(wx.Button(dlg, wx.ID_OK))
                    btn_sizer.AddButton(wx.Button(dlg, wx.ID_CANCEL))
                    btn_sizer.Realize()
                    #
                    dlg.action = CONVERT
                    rb_convert.Value = True
                    rb_dont_convert.Value = False
                    for rb, action in ((rb_convert, CONVERT),
                                       (rb_dont_convert, DONT_CONVERT)):
                        def fn(event, action=action, dlg=dlg):
                            dlg.action = action
                            dir_ctrl.Enable(action == CONVERT)
                        rb.Bind(wx.EVT_RADIOBUTTON, fn)
                    dlg.Fit()
                    result = dlg.ShowModal()
                    if result == wx.ID_OK:
                        if dlg.action == CONVERT:
                            self.__pipeline.convert_legacy_input_modules()
                            self.__pipeline.convert_default_input_folder(
                                dir_ctrl.GetValue())
                
            self.__workspace.save_pipeline_to_measurements()
            
        except Exception,instance:
            from cellprofiler.gui.errordialog import display_error_dialog
            display_error_dialog(self.__frame, instance, self.__pipeline,
                                 continue_only=True)
            
    def load_hdf5_pipeline(self, pathname):
        '''Load a pipeline from an HDF5 measurements file or similar
        
        pathname - pathname to the file
        '''
        assert h5py.is_hdf5(pathname)
        m = cpm.Measurements(
            filename = pathname,
            mode = "r")
        has_user_pipeline = m.has_feature(
            cpm.EXPERIMENT, cpp.M_USER_PIPELINE)
        has_pipeline = m.has_feature(
            cpm.EXPERIMENT, cpp.M_PIPELINE)
        if has_user_pipeline:
            if has_pipeline:
                with wx.Dialog(
                    self.__frame,
                    title = "Choose pipeline to open") as dlg:
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
                        "Do you want to open the primary or user pipeline?") %\
                        os.path.split(pathname)[1]
                    sizer.Add(wx.StaticText(dlg, label = message), 0, wx.EXPAND)
                    sizer.AddSpacer(4)
                    groupbox = wx.StaticBox(dlg, label="Pipeline choice")
                    gb_sizer = wx.StaticBoxSizer(groupbox, wx.VERTICAL)
                    sizer.Add(gb_sizer, 1, wx.EXPAND)
                    rb_primary = wx.RadioButton(dlg, label = "&Primary pipeline")
                    gb_sizer.Add(rb_primary, 0, wx.ALIGN_LEFT)
                    gb_sizer.AddSpacer(2)
                    rb_user = wx.RadioButton(dlg, label = "&User pipeline")
                    gb_sizer.Add(rb_user, 0, wx.ALIGN_LEFT)
                    rb_user.SetValue(True)
                    
                    btn_sizer = wx.StdDialogButtonSizer()
                    dlg.Sizer.Add(btn_sizer, 0, wx.EXPAND)
                    btn_sizer.AddButton(wx.Button(dlg, wx.ID_OK))
                    btn_sizer.AddButton(wx.Button(dlg, wx.ID_CANCEL))
                    btn_sizer.Realize()
                    dlg.Fit()
                    if dlg.ShowModal() == wx.ID_OK:
                        ftr = (cpp.M_USER_PIPELINE if rb_user.Value else
                               cpp.M_PIPELINE)
                    else:
                        return False
            else:
                ftr = cpp.M_USER_PIPELINE
        else:
            ftr = cpp.M_PIPELINE
        pipeline_text = m.get_experiment_measurement(ftr)
        pipeline_text = pipeline_text.encode('us-ascii')
        self.__pipeline.load(StringIO(pipeline_text))
        return True

    def __clear_errors(self):
        for key,error in self.__setting_errors.iteritems():
            self.__frame.preferences_view.pop_error_text(error)
        self.__setting_errors = {}
        
    def __on_save_as_pipeline(self, event):
        try:
            self.do_save_pipeline()
        except Exception, e:
            wx.MessageBox('Exception:\n%s'%(e), 'Could not save pipeline...', wx.ICON_ERROR|wx.OK, self.__frame)
            
    def do_save_pipeline(self):
        '''Save the pipeline, asking the user for the name

        return True if the user saved the pipeline
        '''
        wildcard="CellProfiler pipeline (*.cp)|*.cp"
        dlg = wx.FileDialog(self.__frame,
                            "Save pipeline",
                            wildcard=wildcard,
                            style=wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT)
        try:
            if dlg.ShowModal() == wx.ID_OK:
                file_name = dlg.GetFilename()
                if not sys.platform.startswith("win"):
                    if file_name.find('.') == -1:
                        # on platforms other than Windows, add the default suffix
                        file_name += ".cp"
                pathname = os.path.join(dlg.GetDirectory(), file_name)
                self.__pipeline.save(pathname)
                return True
            return False
        finally:
            dlg.Destroy()
    
    def __on_export_image_sets(self, event):
        '''Export the pipeline's image sets to a .csv file'''
        dlg = wx.FileDialog(self.__frame, "Export image sets",
                            wildcard = "Image set file (*.csv)|*.csv",
                            style = wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        try:
            if dlg.ShowModal() == wx.ID_OK:
                try:
                    self.__workspace.refresh_image_set()
                    self.__workspace.measurements.write_image_sets(dlg.Path)
                except Exception, e:
                    display_error_dialog(self.__frame, e, self.__pipeline,
                                         "Failed to export image sets",
                                         continue_only=True)
        finally:
            dlg.Destroy()
            
    def __on_plateviewer(self, event):
        import cellprofiler.gui.plateviewer as pv
        
        data = pv.PlateData()
        try:
            self.__workspace.refresh_image_set()
        except Exception, e:
            display_error_dialog(self.__frame, e, self.__pipeline,
                                 "Failed to make image sets",
                                 continue_only=True)
            return
        m = self.__workspace.measurements
        assert isinstance(m, cpm.Measurements)
        
        image_numbers = m.get_image_numbers()
        if len(image_numbers) == 0:
            self.display_plate_viewer_help(
                "Your project does not produce any image sets.\n"
                "Please configure the input modules correctly.",
                "Plate viewer: No image sets")
            return
        url_features = [f for f in m.get_feature_names(cpm.IMAGE)
                        if f.startswith(cpm.C_URL)]
        pws = []
        for feature in ("Plate", "Well", "Site"):
            measurement = cpm.C_METADATA + "_" + feature
            if m.has_feature(cpm.IMAGE, measurement):
                pws.append(
                    m.get_measurement(cpm.IMAGE, measurement, image_numbers))
            else:
                pws.append([None] * len(image_numbers))
        plate, well, site = pws
        if pws[1][0] is None:
            self.display_plate_viewer_help(
                "Your project needs to tag every image set with well metadata\n"
                "Please use the Metadata module to define a metadata tag\n"
                "named, ""Well"".",
                "Plate viewer: No well metadata")
            return
        
        for url_feature in url_features:
            channel = [url_feature[(len(cpm.C_URL)+1):]] * len(image_numbers)
            urls = m.get_measurement(cpm.IMAGE, url_feature, image_numbers)
            data.add_files(
                [url.encode('utf-8') for url in urls],
                plate, well, site, channel_names = channel)
        if self.__plate_viewer is None:
            self.__pv_frame = wx.Frame(self.__frame, title = "Plate viewer")
        else:
            self.__pv_frame.DestroyChildren()
        self.__plate_viewer = pv.PlateViewer(self.__pv_frame, data)
        self.__pv_frame.Fit()
        self.__pv_frame.Show()
        
    def display_plate_viewer_help(self, message, caption):
        '''Display a helpful dialog for a plate viewer config error
        
        message - message to display
        
        caption - caption on frame bar
        '''
        message = message + "\n\nPress ""Help"" for the plate viewer manual page."
        with wx.Dialog(self.__frame, title = caption) as dlg:
            assert isinstance(dlg, wx.Dialog)
            dlg.Sizer = wx.BoxSizer(wx.VERTICAL)
            message_sizer = wx.BoxSizer(wx.HORIZONTAL)
            dlg.Sizer.Add(message_sizer, 0, wx.EXPAND | wx.ALL, 15)
            bmpInfo = wx.ArtProvider.GetBitmap(wx.ART_INFORMATION, wx.ART_CMN_DIALOG)
            message_sizer.Add(wx.StaticBitmap(dlg, bitmap=bmpInfo), 0, 
                              wx.ALIGN_TOP | wx.ALIGN_LEFT)
            message_sizer.AddSpacer(12)
            message_sizer.Add(wx.StaticText(dlg, label=message), 0, 
                              wx.ALIGN_TOP | wx.ALIGN_LEFT)
            button_sizer = wx.StdDialogButtonSizer()
            dlg.Sizer.Add(button_sizer, 0, wx.EXPAND | wx.ALL, 8)
            ok_button = wx.Button(dlg, wx.ID_OK)
            help_button = wx.Button(dlg, wx.ID_HELP)
            button_sizer.AddButton(ok_button)
            button_sizer.AddButton(help_button)
            button_sizer.Realize()
            def do_ok(event):
                dlg.EndModal(0)
            def do_help(event):
                HTMLDialog(self.__frame, "Help for plate viewer",
                           PLATEVIEWER_HELP).Show()
            ok_button.Bind(wx.EVT_BUTTON, do_ok)
            help_button.Bind(wx.EVT_BUTTON, do_help)
            dlg.Fit()
            dlg.ShowModal()
    
    def set_current_pipeline_path(self, pathname):
        cpprefs.set_current_pipeline_path(pathname)
        cpprefs.add_recent_file(pathname)
        self.populate_recent_files()
        
    def populate_recent_files(self):
        '''Populate the recent files menu'''
        for menu, ids, file_names, fn in (
            (self.__frame.recent_pipeline_files, 
             RECENT_PIPELINE_FILE_MENU_ID,
             cpprefs.get_recent_files(),
             self.do_load_pipeline),
            (self.__frame.recent_workspace_files,
             RECENT_WORKSPACE_FILE_MENU_ID,
             cpprefs.get_recent_files(cpprefs.WORKSPACE_FILE),
             self.do_open_workspace)):
            assert isinstance(menu, wx.Menu)
            while len(menu.MenuItems) > 0:
                self.__frame.Unbind(wx.EVT_MENU, id = menu.MenuItems[0].Id)
                menu.RemoveItem(menu.MenuItems[0])
            for index, file_name in enumerate(file_names):
                menu.Append(ids[index], file_name)
                self.__frame.Bind(
                    wx.EVT_MENU,
                    lambda event, file_name=file_name, fn=fn: fn(file_name),
                    id = ids[index])
        
    def set_title(self):
        '''Set the title of the parent frame'''
        pathname = cpprefs.get_current_workspace_path()
        if pathname is None:
            self.__frame.Title = "CellProfiler %s" % (version.title_string)
            return
        path, file = os.path.split(pathname)
        if self.__dirty_workspace:
            self.__frame.Title = "CellProfiler %s: %s* (%s)" % (version.title_string, file, path)
        else:
            self.__frame.Title = "CellProfiler %s: %s (%s)" % (version.title_string, file, path)
            
    def __on_clear_pipeline(self,event):
        if wx.MessageBox("Do you really want to remove all modules from the pipeline?",
                         "Clearing pipeline",
                         wx.YES_NO | wx.ICON_QUESTION, self.__frame) == wx.YES:
            self.stop_debugging()
            if self.is_running():
                self.stop_running()            
            self.__pipeline.clear()
            self.__clear_errors()
            self.__pipeline_list_view.select_one_module(1)
            self.enable_module_controls_panel_buttons()
    
    def check_close(self):
        '''Return True if we are allowed to close
        
        Check for pipeline dirty, return false if user doesn't want to close
        '''
        if self.__dirty_workspace:
            #
            # Create a dialog box asking the user what to do.
            #
            dialog = wx.Dialog(self.__frame,
                               title = "Closing CellProfiler")
            super_sizer = wx.BoxSizer(wx.VERTICAL)
            dialog.SetSizer(super_sizer)
            #
            # This is the main window with the icon and question
            #
            sizer = wx.BoxSizer(wx.HORIZONTAL)
            super_sizer.Add(sizer, 1, wx.EXPAND|wx.ALL, 5)
            question_mark = wx.ArtProvider.GetBitmap(wx.ART_HELP,
                                                     wx.ART_MESSAGE_BOX)
            icon = wx.StaticBitmap(dialog, -1, question_mark)
            sizer.Add(icon, 0, wx.EXPAND | wx.ALL, 5)
            text = wx.StaticText(dialog, label = "Do you want to save your project?")
            sizer.Add(text, 0, wx.EXPAND | wx.ALL, 5)
            super_sizer.Add(wx.StaticLine(dialog), 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 20)
            #
            # These are the buttons
            #
            button_sizer = wx.BoxSizer(wx.HORIZONTAL)
            super_sizer.Add(button_sizer, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALL, 5)
            SAVE_ID = wx.NewId()
            DONT_SAVE_ID = wx.NewId()
            RETURN_TO_CP_ID = wx.NewId()
            answer = [RETURN_TO_CP_ID]
            for button_id, text, set_default in (
                (SAVE_ID, "Save", True),
                (RETURN_TO_CP_ID, "Return to CellProfiler", False),
                (DONT_SAVE_ID, "Don't Save", False)):
                button = wx.Button(dialog, button_id, text)
                if set_default:
                    button.SetDefault()
                button_sizer.Add(button, 0, wx.EXPAND | wx.ALL, 5)
                def on_button(event, button_id = button_id):
                    dialog.SetReturnCode(button_id)
                    answer[0] = button_id
                    dialog.Close()
                dialog.Bind(wx.EVT_BUTTON, on_button, button,button_id)
            dialog.Fit()
            dialog.CentreOnParent()
            try:
                dialog.ShowModal()
                if answer[0] == SAVE_ID:
                    workspace_path = cpprefs.get_current_workspace_path()
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
    
    def __on_pipeline_event(self,caller,event):
        if isinstance(event,cpp.RunExceptionEvent):
            error_msg = None
            self.__pipeline_list_view.select_one_module(event.module.module_num)
            try:
                import MySQLdb
                if (isinstance(event.error, MySQLdb.OperationalError) and
                    len(event.error.args) > 1):
                    #
                    # The informative error is in args[1] for MySQL
                    #
                    error_msg = event.error.args[1]
            except:
                pass
            if error_msg is None:
                error_msg = str(event.error)
            message = (("Error while processing %s:\n"
                        "%s\n\nDo you want to stop processing?") %
                       (event.module.module_name,error_msg))
            result = display_error_dialog(self.__frame,
                                          event.error,
                                          self.__pipeline,
                                          message,
                                          event.tb)
            event.cancel_run = result == ED_STOP
            event.skip_thisset = result == ED_SKIP
                
        elif isinstance(event, cpp.LoadExceptionEvent):
            self.on_load_exception_event(event)
        elif isinstance(event, cpp.ImagePlaneDetailsAddedEvent):
            self.on_image_plane_details_added(event)
        elif isinstance(event, cpp.ImagePlaneDetailsRemovedEvent):
            self.on_image_plane_details_removed(event)
        elif event.is_pipeline_modification:
            self.__dirty_workspace = True
            self.set_title()
            needs_default_image_folder = \
                self.__pipeline.needs_default_image_folder()
            self.__frame.get_preferences_view().show_default_image_folder(
                needs_default_image_folder)
            if event.is_image_set_modification:
                self.on_image_set_modification()
            self.__workspace.save_pipeline_to_measurements()
            
    def on_image_set_modification(self):
        self.__workspace.invalidate_image_set()
        self.exit_test_mode()
        
    def __on_image_directory_change(self, event):
        self.on_image_set_modification()
        
    def __on_output_directory_change(self, event):
        self.on_image_set_modification()
        
    def on_workspace_event(self, event):
        '''Workspace's file list changed. Invalidate the workspace cache.'''
        if isinstance(event, cpw.Workspace.WorkspaceFileListNotification):
            self.on_image_set_modification()
            self.__dirty_workspace = True
        
    def on_load_exception_event(self, event):
        '''Handle a pipeline load exception'''
        if event.module is None:
            module_name = event.module_name
        else:
            module_name = event.module.module_name
        if event.settings is None or len(event.settings) == 0:
            message = ("Error while loading %s: %s\nDo you want to stop processing?"%
                       (module_name, event.error.message))
        else:
            message = ("Error while loading %s: %s\n"
                       "Do you want to stop processing?\n\n"
                       "Module settings:\n"
                       "\t%s") % ( module_name,
                                   event.error.message,
                                   '\n\t'.join(event.settings))
        if display_error_message(
            self.__frame, message, 
            "Pipeline error", 
            buttons = [wx.ID_YES, wx.ID_NO]) == wx.NO:
            event.cancel_run = False
            
    def on_image_plane_details_added(self, event):
        '''Callback from pipeline when paths are added to the pipeline'''
        urls = [ipd.url for ipd in event.image_plane_details]
        self.__path_list_ctrl.add_paths(urls)
        self.__workspace.file_list.add_files_to_filelist(urls)
        
    def on_image_plane_details_removed(self, event):
        '''Callback from pipeline when paths are removed from the pipeline'''
        urls = [ipd.url for ipd in event.image_plane_details]
        self.__path_list_ctrl.remove_paths(urls)
        self.__workspace.file_list.remove_files_from_filelist(urls)
        
    def on_update_pathlist(self, event=None):
        ipds = self.__pipeline.get_filtered_image_plane_details(self.__workspace)
        enabled_urls = set([ipd.url for ipd in ipds])
        disabled_urls = set(self.__path_list_ctrl.get_paths())
        disabled_urls.difference_update(enabled_urls)
        self.__path_list_ctrl.enable_paths(enabled_urls, True)
        self.__path_list_ctrl.enable_paths(disabled_urls, False)
        
    def on_update_pathlist_ui(self, event):
        '''Called with an UpdateUIEvent for a pathlist command ID'''
        assert isinstance(event, wx.UpdateUIEvent)
        event.Enable(True)
        if not self.__path_list_ctrl.IsShownOnScreen():
            event.Enable(False)
        elif event.Id == cpframe.ID_EDIT_REMOVE_FROM_FILE_LIST:
            if not self.__path_list_ctrl.has_selections():
                event.Enable(False)
        elif event.Id == cpframe.ID_EDIT_SHOW_FILE_LIST_IMAGE:
            if not self.__path_list_ctrl.has_focus_item():
                event.Enable(False)
        
    def on_pathlist_browse(self, event, default_dir = wx.EmptyString):
        '''Handle request for browsing for pathlist files'''
        with wx.FileDialog(
            self.__path_list_ctrl,
            "Select image files",
            defaultDir = default_dir,
            wildcard = ("Image files (*.tif,*.tiff,*.png,*.jpg,*.gif,*.jpg)|"
                        "*.tif;*.tiff;*.jpg;*.jpeg;*.png;*.gif;*.bmp|"
                        "All files (*.*)|*.*"),
            style = wx.FD_DEFAULT_STYLE | wx.FD_MULTIPLE | wx.FD_OPEN) as dlg:
            assert isinstance(dlg, wx.FileDialog)
            if dlg.ShowModal() == wx.ID_OK:
                paths = dlg.GetPaths()
                self.add_paths_to_pathlist(paths)
        
        
    PATHLIST_CMD_SHOW = "Show selected image"
    PATHLIST_CMD_BROWSE = "Browse for images"
    PATHLIST_CMD_REMOVE = "Remove from list"
    PATHLIST_CMD_REFRESH = "Refresh"
    PATHLIST_CMD_EXPAND_ALL = "Expand all folders"
    PATHLIST_CMD_COLLAPSE_ALL = "Collapse all folders"
    PATHLIST_CMD_CLEAR = "Clear list"
    
    def get_pathlist_file_context_menu(self, paths):
        return ((self.PATHLIST_CMD_SHOW, self.PATHLIST_CMD_SHOW),
                (self.PATHLIST_CMD_REMOVE, self.PATHLIST_CMD_REMOVE),
                (self.PATHLIST_CMD_BROWSE, self.PATHLIST_CMD_BROWSE),
                (self.PATHLIST_CMD_EXPAND_ALL, self.PATHLIST_CMD_EXPAND_ALL),
                (self.PATHLIST_CMD_COLLAPSE_ALL, self.PATHLIST_CMD_COLLAPSE_ALL),
                (self.PATHLIST_CMD_CLEAR, self.PATHLIST_CMD_CLEAR))
    
    def on_pathlist_file_command(self, paths, cmd):
        if cmd == self.PATHLIST_CMD_SHOW or cmd is None:
            if len(paths) == 0:
                return
            self.on_pathlist_show()
        elif cmd == self.PATHLIST_CMD_REMOVE:
            self.on_pathlist_file_delete(paths)
        elif cmd == self.PATHLIST_CMD_BROWSE:
            if len(paths) == 0 or not paths[0].startswith("file:"):
                self.on_pathlist_browse(None)
            else:
                path = urllib.url2pathname(paths[0][5:])
                path = os.path.split(path)[0]
                self.on_pathlist_browse(
                    None,
                    default_dir=path)
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
        return ((self.PATHLIST_CMD_REMOVE, self.PATHLIST_CMD_REMOVE),
                (self.PATHLIST_CMD_REFRESH, self.PATHLIST_CMD_REFRESH),
                (self.PATHLIST_CMD_BROWSE, self.PATHLIST_CMD_BROWSE),
                (self.PATHLIST_CMD_EXPAND_ALL, self.PATHLIST_CMD_EXPAND_ALL),
                (self.PATHLIST_CMD_COLLAPSE_ALL, self.PATHLIST_CMD_COLLAPSE_ALL),
                (self.PATHLIST_CMD_CLEAR, self.PATHLIST_CMD_CLEAR))
    
    def on_pathlist_folder_command(self, path, cmd):
        if cmd == self.PATHLIST_CMD_REMOVE:
            paths = self.__path_list_ctrl.get_folder(
                path, self.__path_list_ctrl.FLAG_RECURSE)
            self.on_pathlist_file_delete(paths)
        elif cmd == self.PATHLIST_CMD_REFRESH:
            W.walk_in_background(path, 
                                 self.on_walk_callback, 
                                 self.on_walk_completed)
        elif cmd == self.PATHLIST_CMD_BROWSE:
            if path.startswith("file:"):
                path = urllib.url2pathname(path[5:])
                self.on_pathlist_browse(None, default_dir=path)
            else:
                self.on_pathlist_browse(None)
        else:
            self.on_pathlist_command(cmd)
            
    def on_pathlist_expand_all(self, event=None):
        self.__path_list_ctrl.expand_all()
        
    def on_pathlist_collapse_all(self, event=None):
        self.__path_list_ctrl.collapse_all()
        
    def on_pathlist_remove(self, event=None):
        '''Remove selected files from the path list'''
        paths = self.__path_list_ctrl.get_paths(
            self.__path_list_ctrl.FLAG_SELECTED_ONLY)
        self.on_pathlist_file_delete(paths)
        
    def on_pathlist_show(self, event=None):
        '''Show the focused item's image'''
        from cellprofiler.gui.cpfigure import show_image
        paths = self.__path_list_ctrl.get_paths(
            self.__path_list_ctrl.FLAG_FOCUS_ITEM_ONLY)
        if len(paths) == 0:
            wx.MessageBox("No image selected.", caption = "No image selected", parent = self.__frame)
            return
        show_image(paths[0], self.__frame)
                
    def on_pathlist_file_delete(self, paths):
        self.__pipeline.remove_image_plane_details(
            [ cpp.ImagePlaneDetails(url, None, None, None)
              for url in paths])
        self.__workspace.file_list.remove_files_from_filelist(paths)
        
    def on_pathlist_clear(self, event):
        '''Remove all files from the path list'''
        result = wx.MessageBox(
            "Are you sure you want to clear all files from your project?\n\n"
            "Clearing will remove the files from your project,\n"
            "but will not delete or modify the files themselves.\n"
            'Choose "Yes" to delete or "No" to cancel this operation.',
            caption = "Confirm file list clear operation",
            style = wx.YES_NO,
            parent = self.__frame)
        if result == wx.YES:
            self.__pipeline.clear_image_plane_details()
            self.__workspace.file_list.clear_filelist()
            
    def on_pathlist_drop_files(self, x, y, filenames):
        self.add_paths_to_pathlist(filenames)
        
    def add_paths_to_pathlist(self, filenames):
        with wx.ProgressDialog("Processing files",
                               "Initializing",
                               parent = self.__frame,
                               style = wx.PD_APP_MODAL | wx.PD_CAN_ABORT) as dlg:
            assert isinstance(dlg, wx.ProgressDialog)
            queue = Queue.Queue()
            interrupt = [False]
            message = ["Initializing"]
            def fn(filenames=filenames, 
                   interrupt=interrupt, 
                   message=message,
                   queue = queue):
                urls = []
                for pathname in filenames:
                    if interrupt[0]:
                        break
                    # Hack - convert drive names to lower case in
                    #        Windows to normalize them.
                    if (sys.platform == 'win32' and pathname[0].isalpha()
                        and pathname[1] == ":"):
                        pathname = os.path.normpath(pathname[:2]) + pathname[2:]
                    message[0] = "Processing " + pathname
                    
                    if os.path.isfile(pathname):
                        urls.append(pathname2url(pathname))
                        if len(urls) > 100:
                            queue.put(urls)
                            urls = []
                    elif os.path.isdir(pathname):
                        for dirpath, dirnames, filenames in os.walk(pathname):
                            for filename in filenames:
                                if interrupt[0]:
                                    break
                                path = os.path.join(dirpath, filename)
                                urls.append(pathname2url(path))
                                message[0] = "Processing " + path
                                if len(urls) > 100:
                                    queue.put(urls)
                                    urls = []
                            else:
                                continue
                            break
                queue.put(urls)
                
            thread = threading.Thread(target=fn)
            thread.setDaemon(True)
            thread.start()
            while not interrupt[0]:
                try:
                    urls = queue.get(block=True, timeout=0.1)
                    try:
                        while True:
                            urls += queue.get(block=False)
                    except:
                        keep_going, skip = dlg.UpdatePulse(
                            "Adding %d files to file list" %len(urls))
                        self.add_urls(urls)
                except:
                    if not thread.is_alive():
                        try:
                            self.add_urls(queue.get(block=False))
                        except:
                            pass
                        break
                    keep_going, skip = dlg.UpdatePulse(message[0])
                dlg.Fit()
                interrupt[0] = not keep_going
            interrupt[0] = True
    
    def on_pathlist_drop_text(self, x, y, text):
        pathnames = [p.strip() for p in text.split("\n")]
        urls = []
        for pathname in pathnames:
            if len(pathname) == 0:
                continue
            if (pathname.startswith("http:") or 
                pathname.startswith("https:") or
                pathname.startswith("ftp:") or
                pathname.startswith("omero:")):
                urls.append(pathname)
            else:
                urls.append(pathname2url(pathname))
        self.add_urls(urls)
                
    def add_urls(self, urls):
        '''Add URLS to the pipeline and file list'''
        self.__pipeline.add_image_plane_details([
            cpp.ImagePlaneDetails(url, None, None, None)
            for url in urls])
        self.__workspace.file_list.add_files_to_filelist(urls)
        
    def on_walk_callback(self, dirpath, dirnames, filenames):
        '''Handle an iteration of file walking'''
        
        hdf_file_list = self.__workspace.get_file_list()
        file_list = [pathname2url(os.path.join(dirpath, filename))
                     for filename in filenames]
        hdf_file_list.add_files_to_filelist(file_list)
        self.__pipeline.add_image_plane_details([
            cpp.ImagePlaneDetails(url, None, None, None)
            for url in file_list])
        
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
                    [module.module_num for module in self.__pipeline.modules()
                     if not module.is_input_module()], initial)
                for fn, initial in ((min, len(self.__pipeline.modules())),
                                    (max, -1))]
            enable_up = active_module.module_num > first_module_num
            enable_down = active_module.module_num < last_module_num
        enable_delete = True
        enable_duplicate = True
        if len(selected_modules) == 0:
            enable_delete = enable_duplicate = False
        
        for menu_id, control, state in (
            (cpframe.ID_EDIT_MOVE_DOWN, self.__mcp_module_down_button, enable_down),
            (cpframe.ID_EDIT_MOVE_UP, self.__mcp_module_up_button, enable_up),
            (cpframe.ID_EDIT_DELETE, self.__mcp_remove_module_button, enable_delete),
            (cpframe.ID_EDIT_DUPLICATE, None, enable_duplicate)):
            if control is not None:
                control.Enable(state)
            menu_item = self.__frame.menu_edit.FindItemById(menu_id)
            if menu_item is not None:
                menu_item.Enable(state)
        
    def __on_help(self, event):
        modules = self.__pipeline_list_view.get_selected_modules()
        if len(modules) > 0:
            self.__frame.do_help_modules(modules)
        else:
            wx.MessageBox(HELP_ON_MODULE_BUT_NONE_SELECTED, 
                          "No module selected",
                          style=wx.OK|wx.ICON_INFORMATION)
        
    def __on_add_module(self,event):
        if not self.__add_module_frame.IsShownOnScreen():
            x, y = self.__frame.GetPositionTuple()
            x = max(x - self.__add_module_frame.GetSize().width, 0)
            self.__add_module_frame.SetPosition((x, y))
        self.__add_module_frame.Show()
        self.__add_module_frame.Raise()
    
    def populate_edit_menu(self, menu):
        '''Display a menu of modules to add'''
        from cellprofiler.modules import get_module_names
        #
        # Get a two-level dictionary of categories and names
        #
        d = { "All": [] }
        for module_name in get_module_names():
            try:
                module = cellprofiler.modules.instantiate_module(module_name)
                if module.is_input_module():
                    continue
                category = module.category
                if isinstance(category, (str,unicode)):
                    categories = [category, "All"]
                else:
                    categories = list(category) + ["All"]
                for category in categories:
                    if not d.has_key(category):
                        d[category] = []
                    d[category].append(module_name)
            except:
                logger.error("Unable to instantiate module %s.\n\n" % 
                             module_name, exc_info=True)
         
        for category in sorted(d.keys()):
            sub_menu = wx.Menu()
            for module_name in sorted(d[category]):
                if self.module_name_to_menu_id.has_key(module_name):
                    menu_id = self.module_name_to_menu_id[module_name]
                else:
                    menu_id = wx.NewId()
                    self.module_name_to_menu_id[module_name] = menu_id
                    self.menu_id_to_module_name[menu_id] = module_name
                    self.__frame.Bind(wx.EVT_MENU, 
                                      self.on_menu_add_module, 
                                      id = menu_id)
                sub_menu.Append(menu_id, module_name)
            menu.AppendSubMenu(sub_menu, category)
            
    def on_menu_add_module(self, event):
        from cellprofiler.modules import instantiate_module
        from cellprofiler.gui.addmoduleframe import AddToPipelineEvent
        assert isinstance(event, wx.CommandEvent)
        if self.menu_id_to_module_name.has_key(event.Id):
            module_name = self.menu_id_to_module_name[event.Id]
            def loader(module_num, module_name=module_name):
                module = instantiate_module(module_name)
                module.set_module_num(module_num)
                return module
            self.on_add_to_pipeline(
                self, AddToPipelineEvent(module_name, loader))
        else:
            logger.warn("Could not find module associated with ID = %d, module = %s" % (
                event.Id, event.GetString()))
            
        
    def __get_selected_modules(self):
        '''Get the modules selected in the GUI, but not input modules'''
        return filter(lambda x: not x.is_input_module(),
                      self.__pipeline_list_view.get_selected_modules())
    
    def on_remove_module(self,event):
        self.remove_selected_modules()
    
    def remove_selected_modules(self):
        with self.__pipeline.undoable_action("Remove modules"):
            selected_modules = self.__get_selected_modules()
            for module in selected_modules:
                for setting in module.settings():
                    if self.__setting_errors.has_key(setting.key()):
                        self.__frame.preferences_view.pop_error_text(self.__setting_errors.pop(setting.key()))                    
                self.__pipeline.remove_module(module.module_num)
            has_input_modules = any([m.is_input_module()
                                     for m in self.__pipeline.modules()])
            has_legacy_modules = any([m.is_load_module()
                                      for m in self.__pipeline.modules()])
            if (not has_input_modules) and (not has_legacy_modules):
                #
                # We need input modules if legacy modules have been deleted
                #
                self.__pipeline.init_modules()
            self.exit_test_mode()
        
    def exit_test_mode(self):
        '''Exit test mode with all the bells and whistles
        
        This is safe to call if not in test mode
        '''
        if self.is_in_debug_mode():
            self.stop_debugging()
            if cpprefs.get_show_exiting_test_mode_dlg():
                self.show_exiting_test_mode()

    def on_duplicate_module(self, event):
        self.duplicate_modules(self.__get_selected_modules())
        
    def duplicate_modules(self, modules):
        selected_modules = self.__get_selected_modules()
        if len(selected_modules):
            module_num=selected_modules[-1].module_num+1
        else:
            # insert module last if nothing selected
            module_num = len(self.__pipeline.modules())+1
        for m in modules:
            module = self.__pipeline.instantiate_module(m.module_name)
            module.module_num = module_num
            module.set_settings_from_values(
                cpp.Pipeline.capture_module_settings(m),
                m.variable_revision_number, m.module_name, False)
            module.show_window = m.show_window  # copy visibility
            self.__pipeline.add_module(module)
            module_num += 1
            
            
    def on_module_up(self,event):
        """Move the currently selected modules up"""
        active_module = self.__pipeline_list_view.get_active_module()
        if active_module is not None:
            self.__pipeline.move_module(
                active_module.module_num, cpp.DIRECTION_UP)
            #
            # Major event - restart from scratch
            #
            if self.is_in_debug_mode():
                self.stop_debugging()
                if cpprefs.get_show_exiting_test_mode_dlg():
                    self.show_exiting_test_mode()
        
    def on_module_down(self,event):
        """Move the currently selected modules down"""
        active_module = self.__pipeline_list_view.get_active_module()
        if active_module is not None:
            self.__pipeline.move_module(
                active_module.module_num, cpp.DIRECTION_DOWN)
            #
            # Major event - restart from scratch
            #
            if self.is_in_debug_mode():
                self.stop_debugging()
                if cpprefs.get_show_exiting_test_mode_dlg():
                    self.show_exiting_test_mode()
            
    def on_undo(self, event):
        wx.BeginBusyCursor()
        try:
            if self.__pipeline.has_undo():
                self.__pipeline.undo()
        finally:
            wx.EndBusyCursor()
            
    def on_update_undo_ui(self, event):
        event.Enable(self.__pipeline.has_undo())
    
    def on_add_to_pipeline(self, caller, event):
        """Add a module to the pipeline using the event's module loader
        
        caller - ignored
        
        event - an AddToPipeline event
        """
        active_module = self.__pipeline_list_view.get_active_module()
        if active_module is None:
            # insert module last if nothing selected
            module_num = len(self.__pipeline.modules(False))+1 
        else:
            last_input_module_num = 0
            for module in self.__pipeline.modules(False):
                if module.is_input_module():
                    last_input_module_num = module.module_num
                else:
                    break
            module_num = max(active_module.module_num, last_input_module_num)+1
        module = event.module_loader(module_num)
        module.show_window = True  # default to show in GUI
        remove_input_modules = False
        if (module.is_load_module() and 
            any([m.is_input_module() for m in self.__pipeline.modules()])):
            #
            # A legacy load module, ask user if they want to convert to a
            # legacy pipeline
            #
            message = ("%s is a legacy input module that is incompatible\n"
                       "with the Images, Metadata, NamesAndTypes, and Groups\n"
                       "input modules. Do you want to remove these input\n"
                       "modules and use %s instead?") % (
                           module.module_name, module.module_name)
            if wx.MessageBox(
                message, 
                caption = "Use legacy input module, %s" %module.module_name,
                style = wx.YES_NO | wx.YES_DEFAULT | wx.ICON_QUESTION,
                parent = self.__frame) != wx.YES:
                return
            remove_input_modules = True
            
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
        #if self.is_in_debug_mode():
        #    self.stop_debugging()
        
    def __on_module_view_event(self,caller,event):
        assert isinstance(event,cellprofiler.gui.moduleview.SettingEditedEvent), '%s is not an instance of CellProfiler.CellProfilerGUI.ModuleView.SettingEditedEvent'%(str(event))
        setting = event.get_setting()
        proposed_value = event.get_proposed_value()
        setting.value = proposed_value
        module = event.get_module()
        is_image_set_modification = module.change_causes_prepare_run(setting)
        self.__pipeline.edit_module(event.get_module().module_num,
                                    is_image_set_modification)
        if self.is_in_debug_mode() and is_image_set_modification:
            #
            # If someone edits a really important setting in debug mode,
            # then you want to reset the debugger to reprocess the image set
            # list.
            #
            self.stop_debugging()
            if cpprefs.get_show_exiting_test_mode_dlg():
                self.show_exiting_test_mode()

    def status_callback(self, *args):
        self.__frame.preferences_view.on_pipeline_progress(*args)

    def on_run_multiple_pipelines(self, event):
        '''Menu handler for run multiple pipelines'''
        dlg = RunMultplePipelinesDialog(
                parent = self.__frame, 
                title = "Run multiple pipelines",
                style = wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER |wx.THICK_FRAME,
                size = (640,480))
        try:
            if dlg.ShowModal() == wx.ID_OK:
                self.pipeline_list = dlg.get_pipelines()
                self.run_next_pipeline(event)
        except:
            dlg.Destroy()
            
    def run_next_pipeline(self, event):
        if len(self.pipeline_list) == 0:
            return
        pipeline_details = self.pipeline_list.pop(0)
        self.do_load_pipeline(pipeline_details.path)
        cpprefs.set_default_image_directory(pipeline_details.default_input_folder)
        cpprefs.set_default_output_directory(pipeline_details.default_output_folder)
        cpprefs.set_output_file_name(pipeline_details.measurements_file)
        self.on_analyze_images(event)
        
    def on_analyze_images(self, event):
        '''Handle a user request to start running the pipeline'''
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
            except cellprofiler.settings.ValidationError, v:
                ok = False
                reason = v.message
        if not ok:
            if wx.MessageBox("%s\nAre you sure you want to continue?" % reason,
                             "Problems with pipeline", wx.YES_NO) != wx.YES:
                self.pipeline_list = []
                return
        ##################################
        #
        # Start the pipeline
        #
        ##################################

        try:
            self.__module_view.disable()
            self.__frame.preferences_view.on_analyze_images()
            clear_old_errors()
            with cpp.Pipeline.PipelineListener(
                self.__pipeline, self.on_prepare_run_error_event):
                if not self.__pipeline.prepare_run(self.__workspace):
                    self.stop_running()
                    return
            measurements_file_path = None
            if cpprefs.get_write_MAT_files() == cpprefs.WRITE_HDF5:
                measurements_file_path = self.get_output_file_path()
                
            num_workers = min(
                len(self.__workspace.measurements.get_image_numbers()),
                cpprefs.get_max_workers())
            self.__analysis = cpanalysis.Analysis(
                self.__pipeline, 
                measurements_file_path,
                initial_measurements=self.__workspace.measurements)
            self.__analysis.start(self.analysis_event_handler,
                                  num_workers)

        except Exception, e:
            # Catastrophic failure
            display_error_dialog(self.__frame,
                                 e,
                                 self.__pipeline,
                                 "Failure in analysis startup.",
                                 sys.exc_info()[2],
                                 continue_only=True)
            self.stop_running()
        return
    
    def on_prepare_run_error_event(self, pipeline, event):
        '''Display an error message box on error during prepare_run
        
        This is called if the pipeline is misconfigured - an unrecoverable
        error that's the user's fault.
        '''
        if isinstance(event, cpp.PrepareRunErrorEvent):
            if event.module is None:
                caption = "Cannot run pipeline"
                message = ("The pipeline cannot be started because of\n"
                           "a configuration problem:\n\n%s") % event.message
            else:
                caption = "Cannot run pipeline: misconfiguration in %s" %\
                    event.module.module_name
                message = ("The pipeline cannot be started because of\n"
                           "a configuration problem in the %s module:\n\n%s") %\
                    (event.module.module_name, event.message)
            wx.MessageBox(
                message = message,
                caption = caption,
                parent = self.__frame,
                style = wx.ICON_ERROR | wx.OK)

    def analysis_event_handler(self, evt):
        PRI_EXCEPTION, PRI_INTERACTION, PRI_DISPLAY = range(3)

        if isinstance(evt, cpanalysis.AnalysisStarted):
            wx.CallAfter(self.show_analysis_controls)
        elif isinstance(evt, cpanalysis.AnalysisProgress):
            print "Progress", evt.counts
            total_jobs = sum(evt.counts.values())
            completed = evt.counts.get(cpanalysis.AnalysisRunner.STATUS_DONE, 0)
            wx.CallAfter(self.__frame.preferences_view.on_pipeline_progress, 
                         total_jobs, completed)
        elif isinstance(evt, cpanalysis.AnalysisFinished):
            print ("Cancelled!" if evt.cancelled else "Finished!")
            # drop any interaction/display requests or exceptions
            while True:
                try:
                    self.interaction_request_queue.get_nowait()  # in case the queue's been emptied
                except Queue.Empty:
                    break
            if evt.cancelled:
                self.pipeline_list = []
                
            wx.CallAfter(self.on_stop_analysis, evt)
        elif isinstance(evt, cpanalysis.DisplayRequest):
            wx.CallAfter(self.module_display_request, evt)
        elif isinstance(evt, cpanalysis.DisplayPostRunRequest):
            wx.CallAfter(self.module_display_post_run_request, evt)
        elif isinstance(evt, cpanalysis.InteractionRequest):
            self.interaction_request_queue.put((PRI_INTERACTION, self.module_interaction_request, evt))
            wx.CallAfter(self.handle_analysis_feedback)
        elif isinstance(evt, cpanalysis.OmeroLoginRequest):
            self.interaction_request_queue.put((PRI_INTERACTION, self.omero_login_request, evt))
            wx.CallAfter(self.handle_analysis_feedback)
        elif isinstance(evt, cpanalysis.ExceptionReport):
            self.interaction_request_queue.put((PRI_EXCEPTION, self.analysis_exception, evt))
            wx.CallAfter(self.handle_analysis_feedback)
        elif isinstance(evt, (cpanalysis.DebugWaiting, cpanalysis.DebugComplete)):
            # These are handled by the dialog reading the debug
            # request queue
            if self.debug_request_queue is None:
                # Things are in a bad state here, possibly because the
                # user hasn't properly run the debugger. Chances are that
                # the user knows that something is going wrong.
                evt.reply(cpanalysis.ServerExited())
            else:
                self.debug_request_queue.put(evt)
        elif isinstance(evt, cpanalysis.AnalysisPaused):
            wx.CallAfter(self.show_resume_button)
        elif isinstance(evt, cpanalysis.AnalysisResumed):
            wx.CallAfter(self.show_pause_button)
        elif isinstance(evt, cellprofiler.pipeline.RunExceptionEvent):
            # exception in (prepare/post)_(run/group)
            import pdb
            pdb.post_mortem(evt.tb)
        else:
            raise ValueError("Unknown event type %s %s" % (type(evt), evt))

    def handle_analysis_feedback(self):
        '''Process any pending exception or interaction requests from the
        pipeline.  This function guards against multiple modal dialogs being
        opened, which can overwhelm the user and cause UI hangs.
        '''
        # just in case.
        assert wx.Thread_IsMain(), "PipelineController.handle_analysis_feedback() must be called from main thread!"

        # only one window at a time
        if self.interaction_pending:
            return

        try:
            pri_func_args = self.interaction_request_queue.get_nowait()  # in case the queue's been emptied
        except Queue.Empty:
            return

        self.interaction_pending = True
        try:
            pri_func_args[1](*pri_func_args[2:])
            if not self.interaction_request_queue.empty():
                wx.CallAfter(self.handle_analysis_feedback)
        finally:
            self.interaction_pending = False

    def module_display_request(self, evt):
        '''
        '''
        assert wx.Thread_IsMain(), "PipelineController.module_display_request() must be called from main thread!"

        module_num = evt.module_num
        # use our shared workspace
        self.__workspace.display_data.__dict__.update(evt.display_data_dict)
        try:
            module = self.__pipeline.modules()[module_num - 1]
            if module.display != cpmodule.CPModule.display:
                fig = self.__workspace.get_module_figure(module,
                                                         evt.image_set_number,
                                                         self.__frame)
                module.display(self.__workspace, fig)
                if hasattr(fig.figure.canvas, "_isDrawn"):
                    fig.figure.canvas._isDrawn = False
                fig.figure.canvas.Refresh()
        except:
            _, exc, tb = sys.exc_info()
            display_error_dialog(None, exc, self.__pipeline, tb=tb, continue_only=True,
                                 message="Exception in handling display request for module %s #%d" \
                                     % (module.module_name, module_num))
        finally:
            # we need to ensure that the reply_cb gets a reply
            evt.reply(cpanalysis.Ack())
            
    def module_display_post_run_request(self, evt):
        assert wx.Thread_IsMain(), "PipelineController.module_post_run_display_request() must be called from main thread!"
        module_num = evt.module_num
        # use our shared workspace
        self.__workspace.display_data.__dict__.update(evt.display_data.__dict__)
        try:
            module = self.__pipeline.modules()[module_num - 1]
            if module.display_post_run != cpmodule.CPModule.display_post_run:
                image_number = self.__workspace.measurements.image_set_count+1
                fig = self.__workspace.get_module_figure(module,
                                                         image_number,
                                                         self.__frame)
                module.display_post_run(self.__workspace, fig)
                fig.Refresh()
        except:
            _, exc, tb = sys.exc_info()
            display_error_dialog(None, exc, self.__pipeline, tb=tb, continue_only=True,
                                 message="Exception in handling display request for module %s #%d" \
                                     % (module.module_name, module_num))
        

    def module_interaction_request(self, evt):
        '''forward a module interaction request from the running pipeline to
        our own pipeline's instance of the module, and reply with the result.
        '''
        module_num = evt.module_num
        # extract args and kwargs from the request.
        # see main().interaction_handler() in analysis_worker.py
        args = [evt.__dict__['arg_%d' % idx] for idx in range(evt.num_args)]
        kwargs = dict((name, evt.__dict__['kwarg_%s' % name]) for name in evt.kwargs_names)
        result = ""
        try:
            module = self.__pipeline.modules()[module_num - 1]
            result = module.handle_interaction(*args, **kwargs)
        except:
            _, exc, tb = sys.exc_info()
            display_error_dialog(None, exc, self.__pipeline, tb=tb, continue_only=True,
                                 message="Exception in handling interaction request for module %s(#%d)" \
                                     % (module.module_name, module_num))
        finally:
            # we need to ensure that the reply_cb gets a reply (even if it
            # being empty causes futher exceptions).
            evt.reply(cpanalysis.InteractionReply(result=result))
            
    def omero_login_request(self, evt):
        '''Handle retrieval of the Omero credentials'''
        from bioformats.formatreader import get_omero_credentials
        evt.reply(cpanalysis.OmeroLoginReply(get_omero_credentials()))

    def analysis_exception(self, evt):
        '''Report an error in analysis to the user, giving options for
        skipping, aborting, and debugging.'''


        assert wx.Thread_IsMain(), "PipelineController.analysis_exception() must be called from main thread!"

        self.debug_request_queue = Queue.Queue()

        evtlist = [evt]
        def remote_debug(evtlist = evtlist):
            # choose a random string for verification
            verification = ''.join(random.choice(string.ascii_letters) for x in range(5))
            evt = evtlist[0]
            # Request debugging.  We get back a port.
            evt.reply(
                cpanalysis.ExceptionPleaseDebugReply(
                    cpanalysis.DEBUG,
                    hashlib.sha1(verification).hexdigest()))
            evt = self.debug_request_queue.get()
            port = evt.port
            result = wx.MessageBox(
                "Remote PDB waiting on port %d\nUse '%s' for verification" % 
                (port, verification),
                "Remote debugging started.",
                wx.OK | wx.CANCEL | wx.ICON_INFORMATION)
            if result == wx.ID_CANCEL:
                evt.reply(cpanalysis.DebugCancel())
                return False
            # Acknowledge the port request, and we'll get back a
            # DebugComplete(), which we use as a new evt to reply with the
            # eventual CONTINUE/STOP choice.
            with wx.ProgressDialog(
                "Remote debugging on port %d" % port,
                "Debugging remotely, Cancel to abandon",
                style = wx.PD_APP_MODAL | wx.PD_CAN_ABORT) as dlg:
                while True:
                    try:
                        evtlist[0] = self.debug_request_queue.get(timeout = .25)
                        return True
                    except Queue.Empty:
                        keep_going, skip = dlg.UpdatePulse(
                            "Debugging remotely, Cancel to abandon")
                        if not keep_going:
                            self.debug_request_queue = None
                            return False

        if evt.module_name is not None:
            message = (("Error while processing %s:\n"
                        "%s\n\nDo you want to stop processing?") %
                       (evt.module_name, evt))
        else:
            message = (("Error while processing (remote worker):\n"
                        "%s\n\nDo you want to stop processing?") %
                       (evt))

        disposition = display_error_dialog(
            None, evt.exc_type, self.__pipeline, message,
            remote_exc_info=(evt.exc_type, evt.exc_message, evt.exc_traceback,
                             evt.filename, evt.line_number, 
                             remote_debug))
        if disposition == ED_STOP:
            self.__analysis.cancel()

        evtlist[0].reply(cpanalysis.Reply(disposition=disposition))

        wx.Yield()  # This allows cancel events to remove other exceptions from the queue.

    def on_restart(self, event):
        '''Restart a pipeline from a measurements file'''
        dlg = wx.FileDialog(self.__frame, "Select measurements file",
                            wildcard = "Measurements file (*.mat, *.h5)|*.mat;*.h5",
                            style = wx.FD_OPEN)
        try:
            if dlg.ShowModal() != wx.ID_OK:
                return
            path = dlg.Path
        finally:
            dlg.Destroy()
        
        ##################################
        #
        # Start the pipeline
        #
        ##################################
        
        try:
            measurements = cpm.load_measurements(path)
            pipeline_txt = measurements.get_experiment_measurement(
                cpp.M_PIPELINE)
            self.__pipeline.loadtxt(StringIO(pipeline_txt.encode("utf-8")))
            self.__module_view.disable()
            self.__frame.preferences_view.on_analyze_images()
            measurements_file_path = None
            if cpprefs.get_write_MAT_files() == cpprefs.WRITE_HDF5:
                measurements_file_path = self.get_output_file_path()
                
            self.__analysis = cpanalysis.Analysis(
                self.__pipeline, 
                measurements_file_path,
                initial_measurements=measurements)
            self.__analysis.start(self.analysis_event_handler,
                                  overwrite = False)
            
        except Exception, e:
            # Catastrophic failure
            display_error_dialog(self.__frame,
                                 e,
                                 self.__pipeline,
                                 "Failure in analysis startup.",
                                 sys.exc_info()[2],
                                 continue_only=True)
            self.stop_running()
                
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
        
    def on_stop_running(self,event):
        '''Handle a user interface request to stop running'''
        self.__stop_analysis_button.Enable(False)
        self.pipeline_list = []
        if (self.__analysis is not None) and self.__analysis.check_running():
            self.__analysis.cancel()
            return  # self.stop_running() will be called when we receive the
                    # AnalysisCancelled event in self.analysis_event_handler.
        self.stop_running()
    
    def on_stop_analysis(self, event):
        '''Stop an analysis run.
        
        Handle chores that need completing after an analysis is cancelled
        or finished, like closing the measurements file or writing the .MAT
        file.
        
        event - a cpanalysis.AnalysisFinished event
        '''
        try:
            if cpprefs.get_write_MAT_files() is True:
                # The user wants to write a .mat file.
                if event.cancelled:
                    if event.measurements is None:
                        return
                    with wx.FileDialog(
                        self.__frame,
                        "Save measurements to a file",
                        wildcard="CellProfiler measurements (*.mat)|*.mat",
                        style = wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
                        if dlg.ShowModal() == wx.ID_OK:
                            path = dlg.Path
                        else:
                            return
                else:
                    path = self.get_output_file_path()
                self.__pipeline.save_measurements(path, event.measurements)
        finally:
            event.measurements.close()
            self.stop_running()
            if cpprefs.get_show_analysis_complete_dlg():
                self.show_analysis_complete()
            self.run_next_pipeline(None)
        
    def stop_running(self):
        if self.is_running():
            self.__analysis.cancel()
            self.__analysis = None
        self.__frame.preferences_view.on_stop_analysis()
        self.__module_view.enable()
        self.show_launch_controls()
    
    def is_in_debug_mode(self):
        """True if there's some sort of debugging in progress"""
        return self.__debug_image_set_list != None
    
    def on_debug_toggle(self, event):
        if self.is_in_debug_mode():
            self.on_debug_stop(event)
        else:
            self.on_debug_start(event)
            
    def on_debug_start(self, event):
        module = self.__pipeline_list_view.reset_debug_module()
        if module is None:
            wx.MessageBox("Test mode is disabled because this pipeline\n"
                          "does not have any modules to run.",
                          "Test mode is disabled",
                          style = wx.OK | wx.ICON_ERROR,
                          parent = self.__frame)
            return
        self.start_debugging()
    
    def start_debugging(self):
        self.__pipeline_list_view.set_debug_mode(True)
        self.__test_controls_panel.GetParent().GetSizer().Layout()
        self.__pipeline.test_mode = True
        self.show_test_controls()
        with cpp.Pipeline.PipelineListener(
            self.__pipeline, self.on_prepare_run_error_event):
            if not self.__workspace.refresh_image_set():
                self.stop_debugging()
                return False
        
        self.close_debug_measurements()
        self.__debug_measurements = cellprofiler.measurements.Measurements(
            copy = self.__workspace.measurements,
            mode="memory")
        self.__debug_object_set = cpo.ObjectSet(can_overwrite=True)
        self.__frame.enable_debug_commands()
        assert isinstance(self.__pipeline, cpp.Pipeline)
        self.__debug_image_set_list = cpi.ImageSetList(True)
        workspace = cpw.Workspace(self.__pipeline, None, None, None,
                                  self.__debug_measurements,
                                  self.__debug_image_set_list,
                                  self.__frame)
        workspace.set_file_list(self.__workspace.file_list)
        self.__keys, self.__groupings = self.__pipeline.get_groupings(
            workspace)

        self.__grouping_index = 0
        self.__within_group_index = 0
        self.__pipeline.prepare_group(workspace,
                                      self.__groupings[0][0],
                                      self.__groupings[0][1])
        self.__debug_outlines = {}
        return self.debug_init_imageset()
    
    def close_debug_measurements(self):
        del self.__debug_measurements
        self.__debug_measurements = None
        
    def on_debug_stop(self, event):
        self.stop_debugging()

    def stop_debugging(self):
        self.__pipeline_list_view.set_debug_mode(False)
        self.__test_controls_panel.GetParent().GetSizer().Layout()
        self.__frame.enable_launch_commands()
        self.__debug_image_set_list = None
        self.close_debug_measurements()
        self.__debug_object_set = None
        self.__debug_outlines = None
        self.__debug_grids = None
        self.__pipeline_list_view.on_stop_debugging()
        self.__pipeline.test_mode = False
        self.__pipeline.end_run()
        self.show_launch_controls()

    def do_step(self, module, select_next_module=True):
        """Do a debugging step by running a module
        """
        failure = 1
        old_cursor = self.__frame.GetCursor()
        self.__frame.SetCursor(wx.StockCursor(wx.CURSOR_WAIT))
        try:
            image_set_number = self.__debug_measurements.image_set_number
            self.__debug_measurements.add_image_measurement(
                cpp.GROUP_NUMBER, self.__grouping_index)
            self.__debug_measurements.add_image_measurement(
                cpp.GROUP_INDEX, self.__within_group_index)
            workspace = cpw.Workspace(self.__pipeline,
                                      module,
                                      self.__debug_measurements,
                                      self.__debug_object_set,
                                      self.__debug_measurements,
                                      self.__debug_image_set_list,
                                      self.__frame if module.show_window else None,
                                      outlines = self.__debug_outlines)
            self.__debug_grids = workspace.set_grids(self.__debug_grids)
            module.run(workspace)
            if module.show_window:
                fig = workspace.get_module_figure(module, image_set_number)
                module.display(workspace, fig)
                fig.Refresh()
            workspace.refresh()
            if workspace.disposition == cpw.DISPOSITION_SKIP:
                self.last_debug_module()
            elif (module.module_num < len(self.__pipeline.modules()) and
                  select_next_module):
                self.__pipeline_list_view.select_one_module(module.module_num+1)
            failure=0
        except Exception,instance:
            logger.error("Failed to run module %s", module.module_name,
                         exc_info=True)
            event = cpp.RunExceptionEvent(instance,module)
            self.__pipeline.notify_listeners(event)
            if event.cancel_run:
                self.on_debug_stop(event)
                failure=-1
            failure=1
        self.__frame.SetCursor(old_cursor)
        if ((module.module_name != 'Restart' or failure==-1) and
            self.__debug_measurements != None):
            module_error_measurement = 'ModuleError_%02d%s'%(module.module_num,module.module_name)
            self.__debug_measurements.add_measurement('Image',
                                                      module_error_measurement,
                                                      failure);
        return failure==0
    
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
        for module in self.__pipeline.modules()[first_module.module_num:]:
            if module.wants_pause:
                break
            count += 1
        message_format = "Running module %d of %d: %s"
        index = 0
        with wx.ProgressDialog(
            "Running modules in test mode",
            message_format % (index+1, count, first_module.module_name),
            maximum = count,
            parent = self.__frame,
            style = wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_CAN_ABORT) as dlg:
            dlg.Show()
            while True:
                assert isinstance(dlg, wx.ProgressDialog)
                module = self.current_debug_module()
                message = message_format % (
                    index+1, count, module.module_name)
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
        self.__within_group_index = ((self.__within_group_index + 1) % 
                                     len(image_numbers))
        image_number = image_numbers[self.__within_group_index]
        self.__debug_measurements.next_image_set(image_number)
        self.debug_init_imageset()
        self.__debug_outlines = {}

    def on_debug_prev_image_set(self, event):
        keys, image_numbers = self.__groupings[self.__grouping_index]
        self.__within_group_index = ((self.__within_group_index + len(image_numbers) - 1) % 
                                     len(image_numbers))
        image_number = image_numbers[self.__within_group_index]
        self.__debug_measurements.next_image_set(image_number)
        self.__pipeline_list_view.reset_debug_module()
        self.__debug_outlines = {}

    def on_debug_next_group(self, event):
        if self.__grouping_index is not None:
            self.debug_choose_group(((self.__grouping_index + 1) % 
                               len(self.__groupings)))
    
    def on_debug_prev_group(self, event):
        if self.__grouping_index is not None:
            self.debug_choose_group(((self.__grouping_index + len(self.__groupings) - 1) % 
                               len(self.__groupings)))
            
    def on_debug_random_image_set(self,event):
        group_index = 0 if len(self.__groupings) == 1 else numpy.random.randint(0,len(self.__groupings)-1,size=1)
        keys, image_numbers = self.__groupings[group_index]
        if len(image_numbers) == 0:
            return
        numpy.random.seed()
        image_number_index = numpy.random.randint(1,len(image_numbers),size=1)[0]
        self.__within_group_index = ((image_number_index-1) % len(image_numbers))
        image_number = image_numbers[self.__within_group_index]
        self.__debug_measurements.next_image_set(image_number)
        self.__pipeline_list_view.reset_debug_module()
        self.__debug_outlines = {}
        
    def debug_choose_group(self, index):
        self.__grouping_index = index
        self.__within_group_index = 0
        workspace = cpw.Workspace(self.__pipeline, None, None, None,
                                  self.__debug_measurements,
                                  self.__debug_image_set_list,
                                  self.__frame)
        
        self.__pipeline.prepare_group(workspace,
                                      self.__groupings[self.__grouping_index][0],
                                      self.__groupings[self.__grouping_index][1])
        key, image_numbers = self.__groupings[self.__grouping_index]
        image_number = image_numbers[self.__within_group_index]
        self.__debug_measurements.next_image_set(image_number)
        self.__pipeline_list_view.reset_debug_module()
        self.__debug_outlines = {}
            
    def on_debug_choose_group(self, event):
        '''Choose a group'''
        if len(self.__groupings) < 2:
            wx.MessageBox("There is only one group and it is currently running in test mode","Choose image group")
            return
        dialog = wx.Dialog(self.__frame, title="Choose an image group", style=wx.RESIZE_BORDER|wx.DEFAULT_DIALOG_STYLE)
        super_sizer = wx.BoxSizer(wx.VERTICAL)
        dialog.SetSizer(super_sizer)
        super_sizer.Add(wx.StaticText(dialog, label = "Select a group set for testing:"),0,wx.EXPAND|wx.ALL,5)
        choices = []
        
        for grouping, image_numbers in self.__groupings:
            text = ["%s=%s"%(k,v) for k,v in grouping.iteritems()]
            text = ', '.join(text)
            choices.append(text)
        lb = wx.ListBox(dialog, choices=choices)
        if self.__grouping_index < len(choices):
            lb.Select(self.__grouping_index)
        else:
            lb.Select(0)
        super_sizer.Add(lb, 1, wx.EXPAND|wx.ALL, 10)
        super_sizer.Add(wx.StaticLine(dialog),0,wx.EXPAND|wx.ALL,5)
        btnsizer = wx.StdDialogButtonSizer()
        btnsizer.AddButton(wx.Button(dialog, wx.ID_OK))
        btnsizer.AddButton(wx.Button(dialog, wx.ID_CANCEL))
        btnsizer.Realize()
        super_sizer.Add(btnsizer)
        super_sizer.Add((2,2))
        dialog.Fit()
        dialog.CenterOnParent()
        try:
            if dialog.ShowModal() == wx.ID_OK:
                self.debug_choose_group(lb.Selection)
        finally:
            dialog.Destroy()
    
    def on_debug_choose_image_set(self, event):
        '''Choose one of the current image sets
        
        '''
        dialog = wx.Dialog(self.__frame, title="Choose an image cycle", style=wx.RESIZE_BORDER|wx.DEFAULT_DIALOG_STYLE)
        super_sizer = wx.BoxSizer(wx.VERTICAL)
        dialog.SetSizer(super_sizer)
        super_sizer.Add(wx.StaticText(dialog, label = "Select an image cycle for testing:"),0,wx.EXPAND|wx.ALL,5)
        choices = []
        indexes = []
        m = self.__debug_measurements
        features = [f for f in 
                    m.get_feature_names(cpm.IMAGE)
                    if f.split("_")[0] in (cpm.C_METADATA, C_FILE_NAME,
                                           C_PATH_NAME, C_FRAME)]
        for image_number in self.__groupings[self.__grouping_index][1]:
            indexes.append(image_number)
            text = ', '.join([
                "%s=%s" % (f, m.get_measurement(cpm.IMAGE, f, 
                                                image_set_number = image_number))
                for f in features])
                                                              
            choices.append(text)
        if len(choices) == 0:
            wx.MessageBox("Sorry, there are no available images. Check your LoadImages module's settings",
                          "Can't choose image")
            return
        lb = wx.ListBox(dialog, -1, choices=choices)
        if self.__within_group_index < len(choices):
            lb.Select(self.__within_group_index)
        else:
            lb.Select(0)
        super_sizer.Add(lb, 1, wx.EXPAND|wx.ALL, 10)
        super_sizer.Add(wx.StaticLine(dialog),0,wx.EXPAND|wx.ALL,5)
        btnsizer = wx.StdDialogButtonSizer()
        btnsizer.AddButton(wx.Button(dialog, wx.ID_OK))
        btnsizer.AddButton(wx.Button(dialog, wx.ID_CANCEL))
        btnsizer.Realize()
        super_sizer.Add(btnsizer)
        super_sizer.Add((2,2))
        dialog.Fit()
        dialog.CenterOnParent()
        try:
            if dialog.ShowModal() == wx.ID_OK:
                image_number = indexes[lb.Selection]
                self.__debug_measurements.next_image_set(image_number)
                self.__pipeline_list_view.reset_debug_module()
                for i, (grouping, image_numbers) in enumerate(self.__groupings):
                    if image_number in image_numbers:
                        self.__grouping_index = i
                        self.__within_group_index = \
                            list(image_numbers).index(image_number)
                        break
                self.debug_init_imageset()
        finally:
            dialog.Destroy()
            
    def debug_init_imageset(self):
        '''Initialize the current image set by running the input modules'''
        for module in self.__pipeline.modules():
            if module.is_input_module():
                if not self.do_step(module, False):
                    return False
        modules = filter((lambda m:not m.is_input_module()),
                         self.__pipeline.modules())
        #
        # Select the first executable module
        #
        if len(modules) > 0:
            self.__pipeline_list_view.reset_debug_module()
        return True

    def on_debug_reload(self, event):
        '''Reload modules from source, warning the user if the pipeline could
        not be reinstantiated with the new versions.

        '''
        success = self.__pipeline.reload_modules()
        if not success:
            wx.MessageBox(("CellProfiler has reloaded modules from source, but "
                           "couldn't reinstantiate the pipeline with the new modules.\n"
                           "See the log for details."),
                          "Error reloading modules.",
                          wx.ICON_ERROR | wx.OK)

    def on_sample_init(self, event):
        if self.__module_view != None:
            if self.__module_view.get_current_module() != None:
                self.show_parameter_sample_options(
                    self.__module_view.get_current_module().get_module_num(), event)
            else:
                print "No current module"

    def show_parameter_sample_options(self, module_num, event):
        if self.__parameter_sample_frame == None:
            selected_module = self.__pipeline.module(module_num)
            selected_module.test_valid(self.__pipeline)

            top_level_frame = self.__frame
            self.parameter_sample_frame = psf.ParameterSampleFrame(
                top_level_frame, selected_module, self.__pipeline, -1)
            self.parameter_sample_frame.Bind(
                wx.EVT_CLOSE, self.on_parameter_sample_frame_close)
            self.parameter_sample_frame.Show(True)

    def on_parameter_sample_frame_close(self, event):
        event.Skip()
        self.__parameter_sample_frame = None

    # ~^~
                    
    def show_analysis_complete(self):
        '''Show the "Analysis complete" dialog'''
        dlg = wx.Dialog(self.__frame, -1, "Analysis complete")
        sizer = wx.BoxSizer(wx.VERTICAL)
        dlg.SetSizer(sizer)
        sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(sub_sizer, 1, wx.EXPAND)
        font = wx.SystemSettings.GetFont(wx.SYS_SYSTEM_FONT)
        text_ctrl = wx.StaticText(dlg, 
                                  label="Finished processing pipeline.")
        text_ctrl.Font = font
        sub_sizer.Add(
            text_ctrl,
            1, wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_CENTER_VERTICAL | 
            wx.EXPAND | wx.ALL, 10)
        bitmap = wx.ArtProvider.GetBitmap(wx.ART_INFORMATION,
                                          wx.ART_CMN_DIALOG,
                                          size=(32,32))
        sub_sizer.Add(wx.StaticBitmap(dlg, -1, bitmap), 0,
                      wx.EXPAND | wx.ALL, 10)
        dont_show_again = wx.CheckBox(dlg, -1, "Don't show this again")
        dont_show_again.Value = False
        sizer.Add(dont_show_again, 0, 
                  wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
        button_sizer = wx.StdDialogButtonSizer()
        save_pipeline_button = wx.Button(dlg, -1, "Save project")
        button_sizer.AddButton(save_pipeline_button)
        button_sizer.SetCancelButton(save_pipeline_button)
        button_sizer.AddButton(wx.Button(dlg, wx.ID_OK))
        sizer.Add(button_sizer, 0, 
                  wx.ALIGN_CENTER_HORIZONTAL | wx.EXPAND | wx.ALL, 10)
        def on_save_workspace(event):
            self.__on_save_workspace(event)
            wx.MessageBox(
                "Saved workspace %s" % cpprefs.get_current_workspace_path(),
                caption = "Saved workspace",
                parent = self.__frame)
        save_pipeline_button.Bind(wx.EVT_BUTTON, on_save_workspace)
        button_sizer.Realize()
        dlg.Fit()
        dlg.CenterOnParent()
        try:
            dlg.ShowModal()
            if dont_show_again.Value:
                cpprefs.set_show_analysis_complete_dlg(False)
        finally:
            dlg.Destroy()
            
    def show_exiting_test_mode(self):
        '''Show the "Analysis complete" dialog'''
        dlg = wx.Dialog(self.__frame, -1, "Exiting test mode")
        sizer = wx.BoxSizer(wx.VERTICAL)
        dlg.SetSizer(sizer)
        sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(sub_sizer, 1, wx.EXPAND)
        text_ctrl = wx.StaticText(dlg, 
                                  label=("You have changed the pipeline so\n"
                                         "that test mode will now exit.\n"))
        sub_sizer.Add(
            text_ctrl,
            1, wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_CENTER_VERTICAL | 
            wx.EXPAND | wx.ALL, 10)
        bitmap = wx.ArtProvider.GetBitmap(wx.ART_INFORMATION,
                                          wx.ART_CMN_DIALOG,
                                          size=(32,32))
        sub_sizer.Add(wx.StaticBitmap(dlg, -1, bitmap), 0,
                      wx.EXPAND | wx.ALL, 10)
        dont_show_again = wx.CheckBox(dlg, -1, "Don't show this again")
        dont_show_again.Value = False
        sizer.Add(dont_show_again, 0, 
                  wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
        button_sizer = wx.StdDialogButtonSizer()
        button_sizer.AddButton(wx.Button(dlg, wx.ID_OK))
        sizer.Add(button_sizer, 0, 
                  wx.ALIGN_CENTER_HORIZONTAL | wx.EXPAND | wx.ALL, 10)
        button_sizer.Realize()
        dlg.Fit()
        dlg.CenterOnParent()
        try:
            dlg.ShowModal()
            if dont_show_again.Value:
                cpprefs.set_show_exiting_test_mode_dlg(False)
        finally:
            dlg.Destroy()
            
    def get_output_file_path(self):
        path = os.path.join(cpprefs.get_default_output_directory(),
                            cpprefs.get_output_file_name())
        if os.path.exists(path) and not cpprefs.get_allow_output_file_overwrite():
            (first_part,ext)=os.path.splitext(path)
            start = 1
            match = re.match('^(.+)__([0-9]+)$',first_part)
            if match:
                first_part = match.groups()[0]
                start = int(match.groups()[1])
            for i in range(start,1000):
                alternate_name = '%(first_part)s__%(i)d%(ext)s'%(locals())
                if not os.path.exists(alternate_name):
                    break
            result = wx.MessageDialog(parent=self.__frame,
                                message='%s already exists. Would you like to create %s instead?'%(path, alternate_name),
                                caption='Output file exists',
                                style = wx.YES_NO+wx.ICON_QUESTION)
            user_choice = result.ShowModal()
            result.Destroy()
            if user_choice & wx.YES:
                path = alternate_name
                cpprefs.set_output_file_name(os.path.split(alternate_name)[1])
            else:
                return None
        return path
    
    def on_show_all_windows(self, event):
        '''Turn "show_window" on for every module in the pipeline'''
        with self.__pipeline.undoable_action("Show all windows"):
            for module in self.__pipeline.modules():
                self.__pipeline.show_module_window(module, True)
        
    def on_hide_all_windows(self, event):
        '''Turn "show_window" off for every module in the pipeline'''
        with self.__pipeline.undoable_action("Hide all windows"):
            for module in self.__pipeline.modules():
                self.__pipeline.show_module_window(module, False)
            
    def run_pipeline(self):
        """Run the current pipeline, returning the measurements
        """
        return self.__pipeline.Run(self.__frame)
    
class FLDropTarget(wx.PyDropTarget):
    '''A generic drop target (for the path list)'''
    def __init__(self, file_callback_fn, text_callback_fn):
        super(self.__class__, self).__init__()
        self.file_callback_fn = file_callback_fn
        self.text_callback_fn = text_callback_fn
        self.file_data_object = wx.FileDataObject()
        self.text_data_object = wx.TextDataObject()
        self.composite_data_object = wx.DataObjectComposite()
        self.composite_data_object.Add(self.file_data_object, True)
        self.composite_data_object.Add(self.text_data_object)
        self.SetDataObject(self.composite_data_object)
        
    def OnDropFiles(self, x, y, filenames):
        self.file_callback_fn(x, y, filenames)
        
    def OnDropText(self, x, y, text):
        self.text_callback_fn(x, y, text)
        
    def OnEnter(self, x, y, d):
        return wx.DragCopy
        
    def OnDragOver(self, x, y, d):
        return wx.DragCopy
    
    def OnData(self, x, y, d):
        if self.GetData():
            df = self.composite_data_object.GetReceivedFormat().GetType()
            if  df in (wx.DF_TEXT, wx.DF_UNICODETEXT):
                self.OnDropText(x, y, self.text_data_object.GetText())
            elif df == wx.DF_FILENAME:
                self.OnDropFiles(x, y,
                                 self.file_data_object.GetFilenames())
        return wx.DragCopy
        
    def OnDrop(self, x, y):
        return True
        
