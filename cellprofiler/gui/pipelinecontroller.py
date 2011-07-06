"""PipelineController.py - controls (modifies) a pipeline

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"
import logging
import math
import numpy
import wx
import os
import re
import sys
import traceback
import scipy.io.matlab.mio
import cpframe
import cellprofiler.pipeline
import cellprofiler.preferences as cpprefs
import cellprofiler.cpimage as cpi
import cellprofiler.measurements as cpm
import cellprofiler.workspace as cpw
import cellprofiler.objects as cpo
from cellprofiler.gui.addmoduleframe import AddModuleFrame
import cellprofiler.gui.moduleview
from cellprofiler.gui.movieslider import EVT_TAKE_STEP
from cellprofiler.gui.help import HELP_ON_MODULE_BUT_NONE_SELECTED
import cellprofiler.utilities.get_revision as get_revision
from errordialog import display_error_dialog, ED_CONTINUE, ED_STOP, ED_SKIP
from runmultiplepipelinesdialog import RunMultplePipelinesDialog
import cellprofiler.gui.parametersampleframe as psf
import cellprofiler.distributed as cpdistributed

logger = logging.getLogger(__name__)
RECENT_FILE_MENU_ID = [wx.NewId() for i in range(cpprefs.RECENT_FILE_COUNT)]
WRITING_MAT_FILE = "Writing .MAT measurements file..."
WROTE_MAT_FILE = ".MAT measurements file has been saved"

class PipelineController:
    """Controls the pipeline through the UI
    
    """
    def __init__(self,pipeline,frame):
        self.__pipeline =pipeline
        pipeline.add_listener(self.__on_pipeline_event)
        self.__frame = frame
        self.__add_module_frame = AddModuleFrame(frame,-1,"Add modules")
        self.__add_module_frame.add_listener(self.on_add_to_pipeline)
        # ~*~
        self.__parameter_sample_frame = None
        # ~^~
        self.__setting_errors = {}
        self.__running_pipeline = None
        self.__dirty_pipeline = False
        self.__inside_running_pipeline = False 
        self.__pause_pipeline = False
        self.__pipeline_measurements = None
        self.__debug_image_set_list = None
        self.__debug_measurements = None
        self.__debug_grids = None
        self.__keys = None
        self.__groupings = None
        self.__grouping_index = None
        self.__within_group_index = None
        self.__distributor = None
        self.pipeline_list = []
        self.populate_recent_files()
        self.populate_edit_menu(self.__frame.menu_edit_add_module)
        wx.EVT_MENU(frame, cpframe.ID_FILE_LOAD_PIPELINE,self.__on_load_pipeline)
        wx.EVT_MENU(frame, cpframe.ID_FILE_URL_LOAD_PIPELINE, self.__on_url_load_pipeline)
        wx.EVT_MENU(frame, cpframe.ID_FILE_SAVE_PIPELINE,self.__on_save_pipeline)
        wx.EVT_MENU(frame, cpframe.ID_FILE_SAVE_AS_PIPELINE, self.__on_save_as_pipeline)
        wx.EVT_MENU(frame, cpframe.ID_FILE_CLEAR_PIPELINE,self.__on_clear_pipeline)
        wx.EVT_MENU(frame, cpframe.ID_FILE_ANALYZE_IMAGES,self.on_analyze_images)
        wx.EVT_MENU(frame, cpframe.ID_FILE_STOP_ANALYSIS,self.on_stop_running)
        wx.EVT_MENU(frame, cpframe.ID_FILE_RUN_MULTIPLE_PIPELINES, self.on_run_multiple_pipelines)
        wx.EVT_MENU(frame, cpframe.ID_FILE_RESTART, self.on_restart)
        
        wx.EVT_MENU(frame, cpframe.ID_EDIT_MOVE_UP, self.on_module_up)
        wx.EVT_MENU(frame, cpframe.ID_EDIT_MOVE_DOWN, self.on_module_down)
        wx.EVT_MENU(frame, cpframe.ID_EDIT_UNDO, self.on_undo)
        wx.EVT_MENU(frame, cpframe.ID_EDIT_DELETE, self.on_remove_module)
        wx.EVT_MENU(frame, cpframe.ID_EDIT_DUPLICATE, self.on_duplicate_module)
        
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
        
        wx.EVT_MENU_OPEN(frame, self.on_frame_menu_open)
        
        cellprofiler.pipeline.evt_modulerunner_done(frame,
                                                    self.on_module_runner_done)
    
    def attach_to_pipeline_list_view(self,pipeline_list_view, movie_viewer):
        """Glom onto events from the list box with all of the module names in it
        
        """
        self.__pipeline_list_view = pipeline_list_view
        self.__movie_viewer = movie_viewer
        
    def attach_to_module_view(self,module_view):
        """Listen for setting changes from the module view
        
        """
        self.__module_view = module_view
        module_view.add_listener(self.__on_module_view_event)
    
    def attach_to_directory_view(self,directory_view):
        """Listen for requests to load pipelines
        
        """
        self.__directory_view = directory_view
        directory_view.add_pipeline_listener(self.__on_dir_load_pipeline)
    
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
        self.__mcp_remove_module_button = wx.Button(self.__module_controls_panel,-1,"-",(0,0), (30, -1))
        self.__mcp_remove_module_button.SetToolTipString("Remove selected module")
        self.__mcp_module_up_button = wx.Button(self.__module_controls_panel,-1,"^",(0,0), (30, -1))
        self.__mcp_module_up_button.SetToolTipString("Move selected module up")
        self.__mcp_module_down_button = wx.Button(self.__module_controls_panel,-1,"v",(0,0), (30, -1))
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

    def attach_to_test_controls_panel(self, test_controls_panel):
        """Attach the pipeline controller to the test controls panel
        
        Attach the pipeline controller to the test controls panel.
        In addition, the PipelineController gets to add whatever buttons it wants to the
        panel.
        """
        self.__test_controls_panel = test_controls_panel
        self.__tcp_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.__tcp_continue = wx.Button(test_controls_panel, -1, "Run", (0,0))
        self.__tcp_step = wx.Button(test_controls_panel, -1, "Step", (0,0))
        self.__tcp_next_imageset = wx.Button(test_controls_panel, -1, "Next Image", (0,0))
        self.__tcp_sizer.AddMany([(self.__tcp_continue, 0, wx.ALL | wx.EXPAND, 2),
                                  ((1, 1), 1),
                                  (self.__tcp_step, 0, wx.ALL | wx.EXPAND, 2),
                                  ((1, 1), 1),
                                  (self.__tcp_next_imageset, 0, wx.ALL | wx.EXPAND, 2)])
        self.__test_controls_panel.SetSizer(self.__tcp_sizer)
        self.__tcp_continue.SetToolTip(wx.ToolTip("Run to next pause"))
        self.__tcp_step.SetToolTip(wx.ToolTip("Step to next module"))
        self.__tcp_next_imageset.SetToolTip(wx.ToolTip("Jump to next image cycle"))
        self.__test_controls_panel.Bind(wx.EVT_BUTTON, self.on_debug_continue, self.__tcp_continue)
        self.__test_controls_panel.Bind(wx.EVT_BUTTON, self.on_debug_step, self.__tcp_step)
        self.__test_controls_panel.Bind(wx.EVT_BUTTON, self.on_debug_next_image_set, self.__tcp_next_imageset)

    def __on_load_pipeline(self,event):
        if self.__dirty_pipeline:
            if wx.MessageBox('Do you want to save your current pipeline\n'
                             'before loading?', 'Save modified pipeline',
                             wx.YES_NO|wx.ICON_QUESTION, self.__frame) & wx.YES:
                self.do_save_pipeline()
        dlg = wx.FileDialog(self.__frame,
                            "Choose a pipeline file to open",
                            wildcard = ("CellProfiler pipeline (*.cp,*.mat)|*.cp;*.mat"))
        dlg.Directory = cpprefs.get_default_output_directory()
        if dlg.ShowModal()==wx.ID_OK:
            pathname = os.path.join(dlg.GetDirectory(),dlg.GetFilename())
            self.do_load_pipeline(pathname)
            
    def __on_url_load_pipeline(self, event):
        if self.__dirty_pipeline:
            if wx.MessageBox('Do you want to save your current pipeline\n'
                             'before loading?', 'Save modified pipeline',
                             wx.YES_NO|wx.ICON_QUESTION, self.__frame) & wx.YES:
                self.do_save_pipeline()
        dlg = wx.TextEntryDialog(self.__frame,
                                 "Enter the pipeline's URL\n\n"
                                 "Example: https://svn.broadinstitute.org/"
                                 "CellProfiler/trunk/ExampleImages/"
                                 "ExampleSBSImages/ExampleSBS.cp",
                                 "Load pipeline via URL")
        if dlg.ShowModal() == wx.ID_OK:
            import urllib2
            self.do_load_pipeline(urllib2.urlopen(dlg.Value))
    
    def __on_dir_load_pipeline(self,caller,event):
        if wx.MessageBox('Do you want to load the pipeline, "%s"?'%(os.path.split(event.Path)[1]),
                         'Load path', wx.YES_NO|wx.ICON_QUESTION ,self.__frame) & wx.YES:
            self.do_load_pipeline(event.Path)
    
    def do_load_pipeline(self,pathname):
        try:
            self.stop_debugging()
            if self.__running_pipeline:
                self.stop_running()
            self.__pipeline.load(pathname)
            self.__pipeline.turn_off_batch_mode()
            self.__clear_errors()
            if isinstance(pathname, (str, unicode)):
                self.set_current_pipeline_path(pathname)
            self.__dirty_pipeline = False
            self.set_title()
            
        except Exception,instance:
            self.__frame.display_error('Failed during loading of %s'%(pathname),instance)

    def __clear_errors(self):
        for key,error in self.__setting_errors.iteritems():
            self.__frame.preferences_view.pop_error_text(error)
        self.__setting_errors = {}
        
    def __on_save_pipeline(self, event):
        path = cpprefs.get_current_pipeline_path()
        if path is None:
            self.do_save_pipeline()
        else:
            self.__pipeline.save(path)
            self.__dirty_pipeline = False
            self.set_title()
            self.__frame.preferences_view.set_message_text(
                "Saved pipeline to " + path)
            
    def __on_save_as_pipeline(self,event):
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
        path = cpprefs.get_current_pipeline_path()
        if path is not None:
            dlg.Path = path
        else:
            dlg.Directory = cpprefs.get_default_output_directory()
        if dlg.ShowModal() == wx.ID_OK:
            file_name = dlg.GetFilename()
            if not sys.platform.startswith("win"):
                if file_name.find('.') == -1:
                    # on platforms other than Windows, add the default suffix
                    file_name += ".cp"
            pathname = os.path.join(dlg.GetDirectory(), file_name)
            self.__pipeline.save(pathname)
            self.set_current_pipeline_path(pathname)
            self.__dirty_pipeline = False
            self.set_title()
            return True
        return False
    
    def set_current_pipeline_path(self, pathname):
        cpprefs.set_current_pipeline_path(pathname)
        cpprefs.add_recent_file(pathname)
        self.populate_recent_files()
        
    def populate_recent_files(self):
        '''Populate the recent files menu'''
        recent_files = self.__frame.recent_files
        assert isinstance(recent_files, wx.Menu)
        while len(recent_files.MenuItems) > 0:
            self.__frame.Unbind(wx.EVT_MENU, id = recent_files.MenuItems[0].Id)
            recent_files.RemoveItem(recent_files.MenuItems[0])
        for index, file_name in enumerate(cpprefs.get_recent_files()):
            recent_files.Append(RECENT_FILE_MENU_ID[index], file_name)
            def on_recent_file(event, file_name = file_name):
                self.do_load_pipeline(file_name)
            self.__frame.Bind(wx.EVT_MENU,
                              on_recent_file,
                              id = RECENT_FILE_MENU_ID[index])
        
    def set_title(self):
        '''Set the title of the parent frame'''
        pathname = cpprefs.get_current_pipeline_path()
        if pathname is None:
            self.__frame.Title = "CellProfiler (r%d)"%(get_revision.version)
            return
        path, file = os.path.split(pathname)
        if self.__dirty_pipeline:
            self.__frame.Title = "CellProfiler (r%d): %s* (%s)"%(get_revision.version, file, path)
        else:
            self.__frame.Title = "CellProfiler (r%d): %s (%s)"%(get_revision.version, file, path)
            
    def __on_clear_pipeline(self,event):
        if wx.MessageBox("Do you really want to remove all modules from the pipeline?",
                         "Clearing pipeline",
                         wx.YES_NO | wx.ICON_QUESTION, self.__frame) == wx.YES:
            self.stop_debugging()
            if self.__running_pipeline:
                self.stop_running()            
            self.__pipeline.clear()
            self.__clear_errors()
            cpprefs.set_current_pipeline_path(None)
            self.__dirty_pipeline = False
            self.set_title()
            self.enable_module_controls_panel_buttons()
    
    def check_close(self):
        '''Return True if we are allowed to close
        
        Check for pipeline dirty, return false if user doesn't want to close
        '''
        if self.__dirty_pipeline:
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
            text = wx.StaticText(dialog, label = "Do you want to save the current pipeline?")
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
            dialog.ShowModal()
            if answer[0] == SAVE_ID:
                if not self.do_save_pipeline():
                    '''Cancel the closing if the user fails to save'''
                    return False
            elif answer[0] == RETURN_TO_CP_ID:
                return False
        return True
    
    def __on_pipeline_event(self,caller,event):
        if isinstance(event,cellprofiler.pipeline.RunExceptionEvent):
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
                
        elif isinstance(event, cellprofiler.pipeline.LoadExceptionEvent):
            self.on_load_exception_event(event)
        elif any([isinstance(event, x) for x in
                  (cellprofiler.pipeline.ModuleAddedPipelineEvent,
                   cellprofiler.pipeline.ModuleEditedPipelineEvent,
                   cellprofiler.pipeline.ModuleMovedPipelineEvent,
                   cellprofiler.pipeline.ModuleRemovedPipelineEvent)]):
            self.__dirty_pipeline = True
            self.set_title()
            
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
        if wx.MessageBox(message, "Pipeline error",
                         wx.YES_NO | wx.ICON_ERROR, 
                         self.__frame) == wx.NO:
            event.cancel_run = False
        
    def enable_module_controls_panel_buttons(self):
        #
        # Enable/disable the movement buttons
        #
        selected_modules = self.__pipeline_list_view.get_selected_modules()
        enable_up = True
        enable_down = True
        enable_delete = True
        enable_duplicate = True
        if len(selected_modules) == 0:
            enable_up = enable_down = enable_delete = enable_duplicate = False
        else:
            if any([m.module_num == 1 for m in selected_modules]):
                enable_up = False
            if any([m.module_num == len(self.__pipeline.modules())
                    for m in selected_modules]):
                enable_down = False
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
        
    def __on_help(self,event):
        modules = self.__get_selected_modules()
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
        from cellprofiler.modules import get_module_names, instantiate_module
        #
        # Get a two-level dictionary of categories and names
        #
        d = { "All": [] }
        for module_name in get_module_names():
            try:
                module = cellprofiler.modules.instantiate_module(module_name)
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
                menu_id = wx.NewId()
                sub_menu.Append(menu_id, module_name)
                ########################
                #
                # on_menu - add a module to the pipeline
                #
                ########################
                def on_menu(event, module_name = module_name):
                    module = instantiate_module(module_name)
                    selected_modules = self.__get_selected_modules()
                    if len(selected_modules) == 0:
                        module.module_num = len(self.__pipeline.modules())+1
                    else:
                        module.module_num = selected_modules[0].module_num + 1
                    self.__pipeline.add_module(module)
                self.__frame.Bind(wx.EVT_MENU, on_menu, id = menu_id)
            menu.AppendSubMenu(sub_menu, category)
        
    def __get_selected_modules(self):
        return self.__pipeline_list_view.get_selected_modules()
    
    def on_remove_module(self,event):
        self.remove_selected_modules()
    
    def remove_selected_modules(self):
        selected_modules = self.__get_selected_modules()
        for module in selected_modules:
            for setting in module.settings():
                if self.__setting_errors.has_key(setting.key()):
                    self.__frame.preferences_view.pop_error_text(self.__setting_errors.pop(setting.key()))                    
            self.__pipeline.remove_module(module.module_num)
        #
        # Major event - restart from scratch
        #
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
            module.set_settings_from_values([str(s) for s in m.settings()], m.variable_revision_number, m.module_name, False)
            self.__pipeline.add_module(module)
            module_num += 1
            
            
    def on_module_up(self,event):
        """Move the currently selected modules up"""
        selected_modules = self.__get_selected_modules()
        for module in selected_modules:
            self.__pipeline.move_module(module.module_num,cellprofiler.pipeline.DIRECTION_UP);
        #
        # Major event - restart from scratch
        #
        if self.is_in_debug_mode():
            self.stop_debugging()
            if cpprefs.get_show_exiting_test_mode_dlg():
                self.show_exiting_test_mode()
        
    def on_module_down(self,event):
        """Move the currently selected modules down"""
        selected_modules = self.__get_selected_modules()
        selected_modules.reverse()
        for module in selected_modules:
            self.__pipeline.move_module(module.module_num,cellprofiler.pipeline.DIRECTION_DOWN);
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
    
    def on_add_to_pipeline(self,caller,event):
        """Add a module to the pipeline using the event's module loader"""
        selected_modules = self.__get_selected_modules()
        if len(selected_modules):
            module_num=selected_modules[-1].module_num+1
        else:
            # insert module last if nothing selected
            module_num = len(self.__pipeline.modules())+1 
        self.__pipeline.add_module(event.module_loader(module_num))
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
        self.__pipeline.edit_module(event.get_module().module_num)
        if self.is_in_debug_mode():
            #
            # If someone edits a really important setting in debug mode,
            # then you want to reset the debugger to reprocess the image set
            # list.
            #
            for module in self.__pipeline.modules():
                setting = event.get_setting()
                if setting.key() in [x.key() for x in module.settings()]:
                    if module.change_causes_prepare_run(setting):
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
        if dlg.ShowModal() == wx.ID_OK:
            self.pipeline_list = dlg.get_pipelines()
            self.run_next_pipeline(event)
            
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

        if cpdistributed.run_distributed():
            try:
                self.__module_view.disable()
                self.__frame.preferences_view.on_analyze_images()
                self.__distributor = cpdistributed.Distributor(self.__frame)
                self.__distributor.start_serving(self.__pipeline, 8567, self.get_output_file_path(), self.status_callback)
                print "serving at ", self.__distributor.server_URL
                if self.__running_pipeline:
                    self.__running_pipeline.close()
                self.__running_pipeline = self.__distributor.run_with_yield()
                self.__pipeline_measurements = self.__running_pipeline.next()
            except Exception, e:
                # Catastrophic failure
                display_error_dialog(self.__frame,
                                     e,
                                     self.__pipeline,
                                     "Failure in distributed work startup",
                                     sys.exc_info()[2])
                self.stop_running()
            return

        output_path = self.get_output_file_path()
        if output_path:
            self.__module_view.disable()
            self.__frame.preferences_view.pause_button.Bind(wx.EVT_BUTTON,
                                                            self.on_pause)
            if self.__running_pipeline:
                self.__running_pipeline.close()
            self.__output_path = output_path
            self.__frame.preferences_view.on_analyze_images()
            self.__running_pipeline = self.__pipeline.run_with_yield(self.__frame,
                                                                     status_callback=self.status_callback)
            try:
                # Start the first module.
                self.__pipeline_measurements = self.__running_pipeline.next()
            except StopIteration:
                #
                # Pipeline finished on the first go (typical for something
                # like CreateBatchFiles)
                #
                self.stop_running()
                if self.__pipeline_measurements is not None and cpprefs.get_write_MAT_files():
                    self.__pipeline.save_measurements(self.__output_path,self.__pipeline_measurements)
                    self.__pipeline_measurements = None
                    self.__output_path = None
                    message = "Finished processing pipeline"
                    title = "Analysis complete"
                else:
                    message = "Pipeline processing finished, no measurements taken"
                    title = "Analysis complete"
                if len(self.pipeline_list) > 0:
                    self.run_next_pipeline(event)
                    return
                wx.MessageBox(message,title)
            except Exception, e:
                # Catastrophic failure on start
                display_error_dialog(self.__frame,
                                     e,
                                     self.__pipeline,
                                     "Failed to initialize pipeline",
                                     sys.exc_info()[2])
                self.stop_running()
    
    def on_restart(self, event):
        '''Restart a pipeline from a measurements file'''
        dlg = wx.FileDialog(self.__frame, "Select measurements file",
                            wildcard = "Measurements file (*.mat)|*.mat",
                            style = wx.FD_OPEN)
        if dlg.ShowModal() != wx.ID_OK:
            return
        
        ##################################
        #
        # Start the pipeline
        #
        ##################################
        output_path = self.get_output_file_path()
        if output_path:
            self.__module_view.disable()
            self.__frame.preferences_view.pause_button.Bind(wx.EVT_BUTTON,
                                                            self.on_pause)
            if self.__running_pipeline:
                self.__running_pipeline.close()
            self.__output_path = output_path
            self.__frame.preferences_view.on_analyze_images()
            self.__running_pipeline = \
                self.__pipeline.restart_with_yield(dlg.Path, self.__frame,
                                                   self.status_callback)
            try:
                # Start the first module.
                self.__pipeline_measurements = self.__running_pipeline.next()
            except StopIteration:
                #
                # Pipeline finished on the first go (typical for something
                # like CreateBatchFiles)
                #
                self.stop_running()
                if self.__pipeline_measurements is not None and cpprefs.get_write_MAT_files():
                    self.__pipeline.save_measurements(self.__output_path,self.__pipeline_measurements)
                    self.__pipeline_measurements = None
                    self.__output_path = None
                    message = "Finished processing pipeline"
                    title = "Analysis complete"
                else:
                    message = "Pipeline processing finished, no measurements taken"
                    title = "Analysis complete"
                if len(self.pipeline_list) > 0:
                    self.run_next_pipeline(event)
                    return
                wx.MessageBox(message,title)
                
    def on_pause(self, event):
        if not self.__pause_pipeline:
            self.__frame.preferences_view.pause(True)
            self.__pause_pipeline = True
            # This is necessary for the case where the user hits pause
            # then resume during the time a module is executing, which
            # results in two calls to __running_pipeline.next() trying
            # to execute simultaneously if the resume causes a
            # ModuleRunnerDoneEvent.
            self.__need_unpause_event = False 
        else:
            self.__frame.preferences_view.pause(False)
            self.__pause_pipeline = False
            if self.__need_unpause_event:
                # see note above
                cellprofiler.pipeline.post_module_runner_done_event(self.__frame)
        
    def on_frame_menu_open(self, event):
        pass
    
    def on_stop_running(self,event):
        self.pipeline_list = []
        self.stop_running()
        if self.__pipeline_measurements is not None:
            self.save_measurements()
        self.__pipeline_measurements = None
    
    def on_save_measurements(self, event):
        if self.__pipeline_measurements is not None:
            self.save_measurements()
        
    def save_measurements(self):
        dlg = wx.FileDialog(self.__frame,
                            "Save measurements to a file",
                            wildcard="CellProfiler measurements (*.mat)|*.mat",
                            style = wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dlg.ShowModal() == wx.ID_OK:
            pathname = os.path.join(dlg.GetDirectory(), dlg.GetFilename())
            self.__pipeline.save_measurements(pathname, 
                                              self.__pipeline_measurements)
        
    def stop_running(self):
        if self.__distributor:
            self.__distributor.stop_serving()
            self.__distributor = None
        self.__running_pipeline = False
        self.__pause_pipeline = False
        self.__frame.preferences_view.on_stop_analysis()
        self.__module_view.enable()
    
    def is_in_debug_mode(self):
        """True if there's some sort of debugging in progress"""
        return self.__debug_image_set_list != None
    
    def on_debug_toggle(self, event):
        if self.is_in_debug_mode():
            self.on_debug_stop(event)
        else:
            self.on_debug_start(event)
            
    def on_debug_start(self, event):
        self.__pipeline_list_view.select_one_module(1)
        self.__movie_viewer.Value = 0
        self.start_debugging()
    
    def start_debugging(self):
        self.__pipeline_list_view.set_debug_mode(True)
        self.__frame.preferences_view.start_debugging()
        self.__test_controls_panel.Show()
        self.__test_controls_panel.GetParent().GetSizer().Layout()
        self.__debug_measurements = cpm.Measurements(can_overwrite=True)
        self.__debug_object_set = cpo.ObjectSet(can_overwrite=True)
        self.__frame.enable_debug_commands()
        assert isinstance(self.__pipeline, cellprofiler.pipeline.Pipeline)
        self.__pipeline.test_mode = True
        try:
            self.__debug_image_set_list = \
                self.__pipeline.prepare_run(self.__frame)
            if self.__debug_image_set_list is None:
                raise ValueError("Failed to get image sets")
            self.__keys, self.__groupings = self.__pipeline.get_groupings(
                self.__debug_image_set_list)
        except ValueError, v:
            message = "Error while preparing for run:\n%s"%(v)
            wx.MessageBox(message, "Pipeline error", wx.OK | wx.ICON_ERROR, self.__frame)
            self.stop_debugging()
            return False

        self.__grouping_index = 0
        self.__within_group_index = 0
        self.__pipeline.prepare_group(self.__debug_image_set_list,
                                      self.__groupings[0][0],
                                      self.__groupings[0][1])
        self.__debug_outlines = {}
        if self.__debug_image_set_list == None:
            self.stop_debugging()
            return False
        return True
    
    def on_debug_stop(self, event):
        self.stop_debugging()

    def stop_debugging(self):
        self.__pipeline_list_view.set_debug_mode(False)
        self.__frame.preferences_view.stop_debugging()
        self.__test_controls_panel.Hide()
        self.__test_controls_panel.GetParent().GetSizer().Layout()
        self.__frame.enable_debug_commands(False)
        self.__debug_image_set_list = None
        self.__debug_measurements = None
        self.__debug_object_set = None
        self.__debug_outlines = None
        self.__debug_grids = None
        self.__pipeline_list_view.on_stop_debugging()
        self.__pipeline.test_mode = False
        self.__pipeline.end_run()
    
    def on_debug_step(self, event):
        
        modules = self.__pipeline_list_view.get_selected_modules()
        module = modules[0]
        self.do_step(module)
    
    def do_step(self, module):
        """Do a debugging step by running a module
        """
        failure = 1
        old_cursor = self.__frame.GetCursor()
        self.__frame.SetCursor(wx.StockCursor(wx.CURSOR_WAIT))
        try:
            image_set_number = self.__debug_measurements.image_set_number
            self.__debug_measurements.add_image_measurement(
                cellprofiler.pipeline.GROUP_NUMBER, self.__grouping_index)
            self.__debug_measurements.add_image_measurement(
                cellprofiler.pipeline.GROUP_INDEX, self.__within_group_index)
            image_set = self.__debug_image_set_list.get_image_set(image_set_number-1)
            workspace = cpw.Workspace(self.__pipeline,
                                      module,
                                      image_set,
                                      self.__debug_object_set,
                                      self.__debug_measurements,
                                      self.__debug_image_set_list,
                                      self.__frame if module.show_window else None,
                                      outlines = self.__debug_outlines)
            self.__debug_grids = workspace.set_grids(self.__debug_grids)
            module.run(workspace)
            if (not module.is_interactive()) and module.show_window:
                module.display(workspace)
            workspace.refresh()
            if workspace.disposition == cpw.DISPOSITION_SKIP:
                last_module_num = self.__pipeline.modules()[-1].module_num
                self.__pipeline_list_view.select_one_module(last_module_num)
                self.last_debug_module()
            elif module.module_num < len(self.__pipeline.modules()):
                self.__pipeline_list_view.select_one_module(module.module_num+1)
            failure=0
        except Exception,instance:
            logger.error("Failed to run module %s", module.module_name,
                         exc_info=True)
            event = cellprofiler.pipeline.RunExceptionEvent(instance,module)
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
        module_idx = self.__movie_viewer.Value
        return self.__pipeline.modules()[module_idx]

    def next_debug_module(self):
        if self.__movie_viewer.Value < len(self.__pipeline.modules()) - 1:
            self.__movie_viewer.Value += 1
            self.__movie_viewer.Refresh()
            return True
        else:
            return False
        
    def last_debug_module(self):
        self.__movie_viewer.Value = len(self.__pipeline.modules()) - 1

    def on_debug_step(self, event):
        if len(self.__pipeline.modules()) == 0:
            return
        success = self.do_step(self.current_debug_module())
        if success:
            self.next_debug_module()
        
    def on_debug_continue(self, event):
        if len(self.__pipeline.modules()) == 0:
            return
        while True:
            module = self.current_debug_module()
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
        self.__pipeline_list_view.select_one_module(1)
        self.__movie_viewer.Value = 0
        self.__debug_outlines = {}

    def on_debug_prev_image_set(self, event):
        keys, image_numbers = self.__groupings[self.__grouping_index]
        self.__within_group_index = ((self.__within_group_index + len(image_numbers) - 1) % 
                                     len(image_numbers))
        image_number = image_numbers[self.__within_group_index]
        self.__debug_measurements.next_image_set(image_number)
        self.__pipeline_list_view.select_one_module(1)
        self.__movie_viewer.Value = 0
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
        image_number_index = numpy.random.randint(1,len(image_numbers),size=1)
        self.__within_group_index = ((image_number_index-1) % len(image_numbers))
        image_number = image_numbers[self.__within_group_index]
        self.__debug_measurements.next_image_set(image_number)
        self.__pipeline_list_view.select_one_module(1)
        self.__movie_viewer.Value = 0
        self.__debug_outlines = {}
        
    def debug_choose_group(self, index):
        self.__grouping_index = index
        self.__within_group_index = 0
        self.__pipeline.prepare_group(self.__debug_image_set_list,
                                      self.__groupings[self.__grouping_index][0],
                                      self.__groupings[self.__grouping_index][1])
        key, image_numbers = self.__groupings[self.__grouping_index]
        image_number = image_numbers[self.__within_group_index]
        self.__debug_measurements.next_image_set(image_number)
        self.__pipeline_list_view.select_one_module(1)
        self.__movie_viewer.Value = 0
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
        lb = wx.ListBox(dialog, -1, choices=choices)
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
        if dialog.ShowModal() == wx.ID_OK:
            self.debug_choose_group(lb.Selection)
    
    def on_debug_choose_image_set(self, event):
        '''Choose one of the current image sets
        
        '''
        dialog = wx.Dialog(self.__frame, title="Choose an image cycle", style=wx.RESIZE_BORDER|wx.DEFAULT_DIALOG_STYLE)
        super_sizer = wx.BoxSizer(wx.VERTICAL)
        dialog.SetSizer(super_sizer)
        super_sizer.Add(wx.StaticText(dialog, label = "Select an image cycle for testing:"),0,wx.EXPAND|wx.ALL,5)
        choices = []
        indexes = []
        for image_number in self.__groupings[self.__grouping_index][1]:
            indexes.append(image_number)
            image_set = self.__debug_image_set_list.get_image_set(image_number-1)
            assert isinstance(image_set, cpi.ImageSet)
            text = []
            for provider in image_set.providers:
                if hasattr(provider, "get_filename"):
                    text.append(provider.get_name()+":"+provider.get_filename())
                # Add frame info for movie files
                if hasattr(provider, "get_series"):
                    text[-1] += ": Z="+str(provider.get_z())+": T="+str(provider.get_t())
                if hasattr(provider, "get_frame"):
                    text[-1] += ": Frame="+str(provider.get_frame())
            text = ', '.join(text)
            choices.append(text)
        if len(choices) == 0:
            wx.MessageBox("Sorry, there are no available images. Check your LoadImages module's settings",
                          "Can't choose image")
            return
        lb = wx.ListBox(dialog, -1, choices=choices)
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
        if dialog.ShowModal() == wx.ID_OK:
            image_number = indexes[lb.Selection]
            self.__debug_measurements.next_image_set(image_number)
            self.__pipeline_list_view.select_one_module(1)
            self.__movie_viewer.Value = 0
            for i, (grouping, image_numbers) in enumerate(self.__groupings):
                if image_number in image_numbers:
                    self.__grouping_index = i
                    self.__within_group_index = image_numbers.index(image_number)
                    break
            
    def on_debug_reload(self, event):
        self.__pipeline.reload_modules()

    # ~*~
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
    def on_module_runner_done(self,event):
        '''Run one iteration of the pipeline
        
        Called in response to a
        cellprofiler.pipeline.ModuleRunnerDoneEvent whenever a module
        is done running.
        '''
        if self.__pause_pipeline:
            # see note in self.on_pause()
            self.__need_unpause_event = True
        elif self.__running_pipeline:
            try:
                wx.Yield()
                # if the user hits "Stop", self.__running_pipeline can go away
                if self.__running_pipeline:
                    self.__pipeline_measurements = self.__running_pipeline.next()
                    event.RequestMore()
            except StopIteration:
                self.stop_running()
                if self.__pipeline_measurements != None and cpprefs.get_write_MAT_files():
                    self.__frame.preferences_view.set_message_text(
                        WRITING_MAT_FILE)
                    try:
                        self.__pipeline.save_measurements(self.__output_path,
                                                          self.__pipeline_measurements)
                        self.__frame.preferences_view.set_message_text(WROTE_MAT_FILE)
                    except IOError, err:
                        while True:
                            result = wx.MessageBox(
                                ("CellProfiler could not save your measurements. "
                                 "Do you want to try saving it using a different name?\n"
                                 "The error was:\n%s") % (err), 
                                "Error saving measurements.", 
                                wx.ICON_ERROR|wx.YES_NO)
                            if result == wx.YES:
                                try:
                                    self.save_measurements()
                                    self.__frame.preferences_view.set_message_text(WROTE_MAT_FILE)
                                    break
                                except IOError, err:
                                    self.__frame.preferences_view.set_message_text("")
                                    pass
                            else:
                                self.__frame.preferences_view.set_message_text("")
                                break
                    self.__pipeline_measurements = None
                    self.__output_path = None
                if len(self.pipeline_list) > 0:
                    self.run_next_pipeline(event)
                    return
                #
                # A little dialog with a "save pipeline" button in addition
                # to the "OK" button.
                #
                if cpprefs.get_show_analysis_complete_dlg():
                    self.show_analysis_complete()
                    
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
        save_pipeline_button = wx.Button(dlg, -1, "Save pipeline")
        button_sizer.AddButton(save_pipeline_button)
        button_sizer.SetCancelButton(save_pipeline_button)
        button_sizer.AddButton(wx.Button(dlg, wx.ID_OK))
        sizer.Add(button_sizer, 0, 
                  wx.ALIGN_CENTER_HORIZONTAL | wx.EXPAND | wx.ALL, 10)
        dlg.Bind(wx.EVT_BUTTON, self.__on_save_pipeline, 
                 save_pipeline_button)
        button_sizer.Realize()
        dlg.Fit()
        dlg.CenterOnParent()
        dlg.ShowModal()
        if dont_show_again.Value:
            cpprefs.set_show_analysis_complete_dlg(False)
            
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
        dlg.ShowModal()
        if dont_show_again.Value:
            cpprefs.set_show_exiting_test_mode_dlg(False)
            
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
            if user_choice & wx.YES:
                path = alternate_name
                cpprefs.set_output_file_name(os.path.split(alternate_name)[1])
            else:
                return None
        return path
    
    def on_show_all_windows(self, event):
        '''Turn "show_window" on for every module in the pipeline'''
        for module in self.__pipeline.modules():
            module.show_window = True
        self.__dirty_pipeline = True
        self.set_title()
        
    def on_hide_all_windows(self, event):
        '''Turn "show_window" off for every module in the pipeline'''
        for module in self.__pipeline.modules():
            module.show_window = False
        self.__dirty_pipeline = True
        self.set_title()
            
    def run_pipeline(self):
        """Run the current pipeline, returning the measurements
        """
        return self.__pipeline.Run(self.__frame)
    
