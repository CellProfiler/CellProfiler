"""PreferencesView.py - displays the default preferences in the lower right corner

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import os
import string
import time
import numpy as np
import wx
import cellprofiler.preferences as cpprefs
import cellprofiler.distributed as cpdistributed
import cellprofiler.multiprocess_server as cpmultiprocess

from cellprofiler.gui.htmldialog import HTMLDialog
from cellprofiler.gui.help import \
     DEFAULT_IMAGE_FOLDER_HELP, DEFAULT_OUTPUT_FOLDER_HELP, OUTPUT_FILENAME_HELP

WELCOME_MESSAGE = 'Welcome to CellProfiler'


ANALYZE_IMAGES = 'Analyze Images'
START_WORK_SERVER = 'Start Distributed Computation'

class PreferencesView:
    """View / controller for the preferences that get displayed in the main window
    
    """
    def __init__(self,panel):
        self.__panel = panel
        self.__sizer = wx.BoxSizer(wx.VERTICAL)
        self.__image_folder_panel = wx.Panel(panel,-1)
        self.__image_edit_box = self.__make_folder_panel(
            self.__image_folder_panel,
            cpprefs.get_default_image_directory(),
            lambda : cpprefs.get_recent_files(cpprefs.DEFAULT_IMAGE_DIRECTORY),
            'Default Input Folder',
            DEFAULT_IMAGE_FOLDER_HELP,
            [cpprefs.set_default_image_directory,
             self.__notify_pipeline_list_view_directory_change],
            refresh_action = self.refresh_input_directory)
        self.__output_folder_panel = wx.Panel(panel,-1)
        self.__output_edit_box = self.__make_folder_panel(
            self.__output_folder_panel,
            cpprefs.get_default_output_directory(),
            lambda : cpprefs.get_recent_files(cpprefs.DEFAULT_OUTPUT_DIRECTORY),
            'Default Output Folder',
            DEFAULT_OUTPUT_FOLDER_HELP,
            [cpprefs.set_default_output_directory,
             self.__notify_pipeline_list_view_directory_change])
        self.__odds_and_ends_panel = wx.Panel(panel,-1)
        self.__make_odds_and_ends_panel()
        self.__status_text = wx.StaticText(panel,-1,style=wx.SUNKEN_BORDER,label=WELCOME_MESSAGE)
        self.__progress_panel = wx.Panel(panel, -1)
        self.__make_progress_panel()
        self.__sizer.AddMany([(self.__image_folder_panel,0,wx.EXPAND|wx.ALL,1),
                              (self.__output_folder_panel,0,wx.EXPAND|wx.ALL,1),
                              (self.__odds_and_ends_panel,0,wx.EXPAND|wx.ALL,1),
                              (self.__status_text,0,wx.EXPAND|wx.ALL, 4),
                              (self.__progress_panel, 0, wx.EXPAND | wx.BOTTOM, 2)])
        panel.SetSizer(self.__sizer)
        self.__sizer.Hide(self.__progress_panel)
        self.__errors = set()
        self.__pipeline_list_view = None
        
    def close(self):
        cpprefs.remove_output_file_name_listener(self.__on_preferences_output_filename_event)
        cpprefs.remove_image_directory_listener(self.__on_preferences_image_directory_event)
        cpprefs.remove_output_directory_listener(self.__on_preferences_output_directory_event)
        
    def __make_folder_panel(self, panel, value, list_fn, text, help_text, 
                            actions, refresh_action = None):
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        help_button = wx.Button(panel,-1,'?',(0,0), (30,-1))
        text_static = wx.StaticText(panel,-1,text+':')
        edit_box = wx.ComboBox(panel, -1, value, choices=list_fn())
        browse_bmp = wx.ArtProvider.GetBitmap(wx.ART_FOLDER_OPEN,
                                              wx.ART_CMN_DIALOG,
                                              (16,16))
        browse_button = wx.BitmapButton(panel,-1,bitmap = browse_bmp)
        browse_button.SetToolTipString("Browse for %s folder" % text)
        
        new_bmp = wx.ArtProvider.GetBitmap(wx.ART_NEW_DIR,
                                           wx.ART_CMN_DIALOG,
                                           (16,16))
        new_button = wx.BitmapButton(panel,-1,bitmap=new_bmp)
        new_button.SetToolTipString("Make a new sub-folder")
        if os.path.isdir(value):
            new_button.Disable()
        sizer.AddMany([(help_button,0,wx.ALL | wx.ALIGN_CENTER, 1),
                       (text_static,0,wx.ALIGN_CENTER, 1),
                       (edit_box,3,wx.EXPAND|wx.ALL,1)])
        if refresh_action is not None:
            refresh_bitmap = wx.ArtProvider.GetBitmap(wx.ART_REDO,
                                                      wx.ART_CMN_DIALOG,
                                                      (16,16))
            refresh_button = wx.BitmapButton(panel, -1, bitmap = refresh_bitmap)
            sizer.Add(refresh_button, 0, wx.ALIGN_CENTER, 1)
            refresh_button.SetToolTipString("Refresh the Default Input Folder list")
            def on_refresh(event):
                refresh_action()
            refresh_button.Bind(wx.EVT_BUTTON, on_refresh)
        sizer.AddMany([(browse_button,0,0|wx.ALL,1),
                       (new_button,0,0|wx.ALL,1)])
        panel.SetSizer(sizer)
        def on_new_folder(event):
            if os.path.exists(edit_box.Value):
                return
            if wx.MessageBox("Do you really want to create the %s folder?" %
                             edit_box.Value,style=wx.YES_NO) == wx.YES:
                os.makedirs(edit_box.Value)
                self.__on_edit_box_change(event, edit_box, text, actions)
            
        def on_edit_box_change(event):
            if os.path.isdir(edit_box.Value):
                new_button.Disable()
                new_button.SetToolTipString("%s is a directory" % 
                                            edit_box.Value)
            else:
                new_button.Enable()
                new_button.SetToolTipString("Press button to create the %s folder" %
                                            edit_box.Value)
            self.__on_edit_box_change(event, edit_box, text, actions)
            event.Skip()
            
        panel.Bind(wx.EVT_BUTTON, lambda event: self.__on_help(event, help_text),
                   help_button)
        panel.Bind(wx.EVT_BUTTON, lambda event: self.__on_browse(event, edit_box, text), browse_button)
        panel.Bind(wx.EVT_TEXT, on_edit_box_change, edit_box)
        panel.Bind(wx.EVT_COMBOBOX, on_edit_box_change, edit_box)
        panel.Bind(wx.EVT_BUTTON, on_new_folder, new_button)
        return edit_box

    def __show_output_filename(self, show):
        for ctrl in (self.__output_filename_text,
                     self.__output_filename_edit_box,
                     self.__allow_output_filename_overwrite_check_box):
            ctrl.Show(show)
            
    def __make_odds_and_ends_panel(self):
        panel = self.__odds_and_ends_panel
        self.__output_filename_text = wx.StaticText(panel,-1,'Output Filename:')
        self.__output_filename_edit_box = wx.TextCtrl(panel,-1,'DefaultOUT.mat')
        self.__allow_output_filename_overwrite_check_box = \
            wx.CheckBox(panel, label = "Allow overwrite?")
        self.__allow_output_filename_overwrite_check_box.Value = \
            cpprefs.get_allow_output_file_overwrite()
        def on_allow_checkbox(event):
            cpprefs.set_allow_output_file_overwrite(
                self.__allow_output_filename_overwrite_check_box.Value)
        self.__allow_output_filename_overwrite_check_box.Bind(
            wx.EVT_CHECKBOX, on_allow_checkbox)
        self.__write_measurements_check_box = \
            wx.CheckBox(panel, label = "Write output file?")
        self.__write_measurements_check_box.Value = \
            cpprefs.get_write_MAT_files()
        self.__show_output_filename(cpprefs.get_write_MAT_files())
        def on_write_MAT_files_checkbox(event):
            wants_write = self.__write_measurements_check_box.Value
            cpprefs.set_write_MAT_files(wants_write)
            self.__show_output_filename(wants_write)
            panel.Layout()
            
        self.__write_measurements_check_box.Bind(
            wx.EVT_CHECKBOX, on_write_MAT_files_checkbox)
        output_filename_help_button = wx.Button(panel,-1,'?', (0,0), (30,-1))
        if not cpdistributed.run_distributed():
            self.__analyze_images_button = wx.Button(panel, -1, ANALYZE_IMAGES)
        else:
            self.__analyze_images_button = wx.Button(panel, -1, START_WORK_SERVER)
        self.__stop_analysis_button = wx.Button(panel,-1,'Stop analysis')
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.AddMany([(output_filename_help_button,0,wx.ALIGN_CENTER|wx.ALL,1),
                       (self.__output_filename_text,0,wx.ALIGN_CENTER,1),
                       (self.__output_filename_edit_box,3,wx.ALL,1),
                       (self.__allow_output_filename_overwrite_check_box, 0, wx.ALIGN_CENTER | wx.ALL, 1),
                       (self.__write_measurements_check_box, 0, wx.ALIGN_CENTER | wx.ALL, 1),
                       (self.__analyze_images_button,0,wx.ALL,1),
                       (self.__stop_analysis_button, 0, wx.ALL,1)])
        sizer.Hide(self.__stop_analysis_button)
        panel.SetSizer(sizer)
        panel.Bind(wx.EVT_BUTTON,
                   lambda event: self.__on_help(event, OUTPUT_FILENAME_HELP),
                   output_filename_help_button)
        panel.Bind(wx.EVT_TEXT, self.__on_output_filename_changed, self.__output_filename_edit_box)
        cpprefs.add_output_file_name_listener(self.__on_preferences_output_filename_event)
        cpprefs.add_image_directory_listener(self.__on_preferences_image_directory_event)
        cpprefs.add_output_directory_listener(self.__on_preferences_output_directory_event)
        cpprefs.add_run_distributed_listener(self.__on_preferences_run_distributed_event)
        panel.Bind(wx.EVT_WINDOW_DESTROY, self.__on_destroy, panel)
    
    def __make_progress_panel(self):
        panel = self.__progress_panel
        self.__current_status= wx.StaticText(panel, -1, label="LoadImages, Image Set 1/19")
        self.__progress_bar = wx.Gauge(panel, -1, size=(100, -1))
        self.__progress_bar.Value = 25
        self.__timer = wx.StaticText(panel, -1, label="1:45/3:50")
        self.pause_button = wx.Button(panel, -1, 'Pause')
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.AddMany([((1,1), 1),
                       (self.__current_status, 0, wx.ALIGN_BOTTOM),
                       ((10, 0), 0),
                       (self.__progress_bar, 0, wx.ALIGN_BOTTOM),
                       ((10, 0), 0),
                       (self.__timer, 0, wx.ALIGN_BOTTOM),
                       ((5, 0), 0),
                       (self.pause_button, 0, wx.BOTTOM|wx.ALIGN_CENTER, 2)])
        panel.SetSizer(sizer)
        panel.Layout()

    def check_preferences(self):
        '''Return True if preferences are OK (e.g. directories exist)'''
        path = self.__image_edit_box.Value
        if not os.path.isdir(path):
            if wx.MessageBox(('The Default Input Folder is "%s", but '
                              'the directory does not exist. Do you want to '
                              'create it?') % path, 
                             "Warning, cannot run pipeline",
                             style = wx.YES_NO) == wx.NO:
                return False, "Image directory does not exist"
            os.makedirs(path)
            cpprefs.set_default_image_directory(path)
        path = self.__output_edit_box.Value
        if not os.path.isdir(path):
            if wx.MessageBox(('The Default Output Folder is "%s", but '
                              'the directory does not exist. Do you want to '
                              'create it?') % path, 
                             "Warning, cannot run pipeline",
                             style = wx.YES_NO) == wx.NO:
                return False, "Output directory does not exist"
            os.makedirs(path)
            cpprefs.set_default_output_directory(path)
        return True, "OK"
                          
    def __on_destroy(self, event):
        cpprefs.remove_image_directory_listener(self.__on_preferences_image_directory_event)
        cpprefs.remove_output_directory_listener(self.__on_preferences_output_directory_event)
        cpprefs.remove_output_file_name_listener(self.__on_preferences_output_filename_event)
        cpprefs.remove_run_distributed_listener(self.__on_preferences_run_distributed_event)

    def attach_to_pipeline_controller(self, pipeline_controller):
        self.__panel.Bind(wx.EVT_BUTTON,
                          pipeline_controller.on_analyze_images, 
                          self.__analyze_images_button)
        self.__panel.Bind(wx.EVT_BUTTON,
                          pipeline_controller.on_stop_running,
                          self.__stop_analysis_button)
    
    def attach_to_pipeline_list_view(self, pipeline_list_view):
        self.__pipeline_list_view = pipeline_list_view
    
    def start_debugging(self):
        self.__analyze_images_button.Disable()
        self.__stop_analysis_button.Disable()
        
    def stop_debugging(self):
        self.__analyze_images_button.Enable()
        self.__stop_analysis_button.Enable()
        
    def on_analyze_images(self):
        self.__odds_and_ends_panel.Sizer.Hide(self.__analyze_images_button)
        self.__odds_and_ends_panel.Sizer.Show(self.__stop_analysis_button)
        self.__odds_and_ends_panel.Layout()
        self.pause_button.SetLabel('Pause')
        self.__panel.Sizer.Hide(self.__status_text)
        self.__panel.Sizer.Show(self.__progress_panel)
        self.__panel.Parent.Layout()
        self.__panel.Layout()
        # begin tracking progress
        self.__progress_watcher = ProgressWatcher(self.__progress_panel,
                                                  self.update_progress,
                                                  distributed=cpdistributed.run_distributed()
                                                  or cpmultiprocess.run_multiprocess())
        
    def on_pipeline_progress(self, *args):
        self.__progress_watcher.on_pipeline_progress(*args)

    def pause(self, do_pause):
        self.__progress_watcher.pause(do_pause)
        if do_pause:
            self.pause_button.SetLabel('Resume')
        else:
            self.pause_button.SetLabel('Pause')
        self.pause_button.Update()
        self.__progress_panel.Layout()

    def update_progress(self, message, elapsed_time, remaining_time):
        self.__current_status.SetLabel(message)
        self.__progress_bar.Value = (100 * elapsed_time) / (elapsed_time + remaining_time + .00001)
        self.__timer.SetLabel('Time %s/%s'%(secs_to_timestr(elapsed_time), secs_to_timestr(elapsed_time + remaining_time)))
        self.__progress_panel.Layout()
    
    def on_stop_analysis(self):
        self.__progress_watcher.stop()
        self.__progress_watcher = None
        self.__odds_and_ends_panel.Sizer.Show(self.__analyze_images_button)
        self.__odds_and_ends_panel.Sizer.Hide(self.__stop_analysis_button)
        self.__odds_and_ends_panel.Layout()
        self.__panel.Sizer.Hide(self.__progress_panel)
        self.__panel.Sizer.Show(self.__status_text)
        self.__panel.Parent.Layout()
        self.__panel.Layout()

    def set_message_text(self,text):
        saved_size = self.__status_text.GetSize()
        self.__status_text.SetLabel(text)
        self.__status_text.SetSize(saved_size)
        self.__status_text.Update()
    
    def pop_error_text(self,error_text):
        if error_text in self.__errors:
            self.__errors.remove(error_text)
            if len(self.__errors) == 0:
                self.set_message_color(wx.Colour(0,0,0))
                self.set_message_text(WELCOME_MESSAGE)
            else:
                self.set_message_text(self.__errors.__iter__().next())
        
    def set_message_color(self,color):
        self.__status_text.SetForegroundColour(color)
    
    def set_error_text(self,error_text):
        self.set_message_text(error_text)
        self.set_message_color(wx.Colour(255,0,0))
        self.__errors.add(error_text)
        
    def __on_browse(self, event, edit_box, text):
        dir_dialog = wx.DirDialog(self.__panel,string.capitalize(text), edit_box.GetValue())
        if dir_dialog.ShowModal() == wx.ID_OK:
            edit_box.SetValue(dir_dialog.GetPath())
            fake_event = wx.CommandEvent(wx.wxEVT_COMMAND_TEXT_UPDATED)
            fake_event.EventObject = edit_box
            fake_event.Id = edit_box.Id
            edit_box.GetEventHandler().ProcessEvent(fake_event)

    def __on_edit_box_change(self, event, edit_box, text, actions):
        path = edit_box.GetValue()
        error_text = 'The %s is not a directory'%(text)
        if os.path.isdir(path):
            for action in actions:
                action(path)
            items = edit_box.GetItems()
            if len(items) < 1 or items[0] != path:
                ins = edit_box.GetInsertionPoint()
                edit_box.Insert(edit_box.Value, 0, path)
                edit_box.Select(0)
                edit_box.SetInsertionPoint(ins)
                abspath = os.path.abspath(path)
                for i, item in enumerate(items):
                    if os.path.abspath(item) == abspath:
                        edit_box.Delete(i+1)
            self.pop_error_text(error_text)
        else:
            self.set_error_text(error_text)
    
    def __on_help(self,event, help_text):
        dlg = HTMLDialog(self.__panel, "Help", help_text)
        dlg.Show()
    
    def __on_pixel_size_changed(self,event):
        error_text = 'Pixel size must be a number'
        text = self.__pixel_size_edit_box.Value
        if text.isdigit():
            cpprefs.set_pixel_size(int(text))
            self.pop_error_text(error_text)
        else:
            self.set_error_text(error_text)
    
    def __on_output_filename_changed(self,event):
        cpprefs.set_output_file_name(self.__output_filename_edit_box.Value)
    
    def __on_preferences_output_filename_event(self,event):
        if self.__output_filename_edit_box.Value != cpprefs.get_output_file_name():
            self.__output_filename_edit_box.Value = cpprefs.get_output_file_name()
        
    def __on_preferences_output_directory_event(self,event):
        old_selection = self.__output_edit_box.Selection
        if self.__output_edit_box.Value != cpprefs.get_default_output_directory():
            self.__output_edit_box.Value = cpprefs.get_default_output_directory()
    
    def __on_preferences_image_directory_event(self, event):
        if self.__image_edit_box.Value != cpprefs.get_default_image_directory():
            self.__image_edit_box.Value = cpprefs.get_default_image_directory()

    def __on_preferences_run_distributed_event(self, event):
        self.__analyze_images_button.Label = START_WORK_SERVER if cpdistributed.run_distributed() else ANALYZE_IMAGES
        self.__analyze_images_button.Size = self.__analyze_images_button.BestSize
        self.__odds_and_ends_panel.Layout()

    def __notify_pipeline_list_view_directory_change(self, path):
        # modules may need revalidation
        if self.__pipeline_list_view is not None:
            self.__pipeline_list_view.notify_directory_change()

    def refresh_input_directory(self):
        cpprefs.fire_image_directory_changed_event()

class ProgressWatcher:
    """ Tracks pipeline progress and estimates time to completion """
    def __init__(self, parent, update_callback, distributed=False):
        self.update_callback = update_callback

        # start tracking progress
        self.start_time = time.time()
        self.end_times = None
        self.current_module_name = ''
        self.pause_start_time = None
        self.previous_pauses_duration = 0.0
        self.image_set_index = 0
        self.num_image_sets = 1

        # for distributed computation
        self.num_jobs = 1
        self.num_received = 0

        self.distributed = distributed

        timer_id = wx.NewId()
        self.timer = wx.Timer(parent, timer_id)
        self.timer.Start(500)
        if not distributed:
            wx.EVT_TIMER(parent, timer_id, self.update)
            self.update()
        else:
            wx.EVT_TIMER(parent, timer_id, self.update_distributed)
            self.update_distributed()

    def stop(self):
        self.timer.Stop()

    def update(self, event=None):
        status = '%s, Image Set %d/%d'%(self.current_module_name, self.image_set_index + 1, self.num_image_sets)
        self.update_callback(status,
                             self.elapsed_time(),
                             self.remaining_time())

    def update_distributed(self, event=None):
        status = 'Distributed work: %d/%d completed'%(self.num_received, self.num_jobs)
        self.update_callback(status,
                             self.elapsed_time(),
                             self.remaining_time_distributed())

    def on_pipeline_progress(self, *args):
        if not self.distributed:
            self.on_start_module(*args)
        else:
            self.on_receive_work(*args)

    def on_start_module(self, module, num_modules, image_set_index, 
                        num_image_sets):
        """
        Update the historical execution times, which are used as the
        bases for projecting the time that remains.  Also update the
        labels that show the current module and image set.  This
        method is called by the pipelinecontroller at the beginning of
        every module execution to update the progress bar.
        """
        self.current_module = module
        self.current_module_name = module.module_name
        self.num_modules = num_modules
        self.image_set_index = image_set_index
        self.num_image_sets = num_image_sets

        if self.end_times is None:
            # One extra element at the beginning for the start time
            self.end_times = np.zeros(1 + num_modules * num_image_sets)
        module_index = module.module_num - 1  # make it zero-based
        index = image_set_index * num_modules + (module_index - 1)
        self.end_times[1 + index] = self.elapsed_time()

        self.update()

    def on_receive_work(self, num_jobs, num_received):
        self.num_jobs = num_jobs
        self.num_received = num_received

        if self.end_times is None:
            # One extra element at the beginning for the start time
            self.end_times = np.zeros(1 + num_jobs)
        self.end_times[num_received] = self.elapsed_time()
        self.update_distributed()

    def pause(self, do_pause):
        if do_pause:
            self.pause_start_time = time.time()
        else:
            self.previous_pauses_duration += time.time() - self.pause_start_time
            self.pause_start_time = None
        
    def adjusted_time(self):
        """Current time minus the duration spent in pauses."""
        pauses_duration = self.previous_pauses_duration
        if self.pause_start_time:
            pauses_duration += time.time() - self.pause_start_time
        return time.time() - pauses_duration

    def elapsed_time(self):
        '''Return the number of seconds that have elapsed since start
           as a float.  Pauses are taken into account.
        '''
        return self.adjusted_time() - self.start_time

    def remaining_time(self):
        """Return our best estimate of the remaining duration, or None
        if we have no bases for guessing."""
        if self.end_times is None:
            return 2 * self.elapsed_time() # We have not started the first module yet
        else:
            module_index = self.current_module.module_num - 1
            index = self.image_set_index * self.num_modules + module_index
            durations = (self.end_times[1:] - self.end_times[:-1]).reshape(self.num_image_sets, self.num_modules)
            per_module_estimates = np.zeros(self.num_modules)
            per_module_estimates[:module_index] = np.median(durations[:self.image_set_index+1,:module_index], 0)
            current_module_so_far = self.elapsed_time() - self.end_times[1 + index - 1]
            if self.image_set_index > 0:
                per_module_estimates[module_index:] = np.median(durations[:self.image_set_index,module_index:], 0)
                per_module_estimates[module_index] = max(per_module_estimates[module_index], current_module_so_far)
            else:
                # Guess that the modules that haven't finished yet are
                # as slow as the slowest one we've seen so far.
                per_module_estimates[module_index] = current_module_so_far
                per_module_estimates[module_index:] = per_module_estimates[:module_index+1].max()
            per_module_estimates[:module_index] *= self.num_image_sets - self.image_set_index - 1
            per_module_estimates[module_index:] *= self.num_image_sets - self.image_set_index
            per_module_estimates[module_index] -= current_module_so_far
            return per_module_estimates.sum()

    def remaining_time_distributed(self):
        """Return our best estimate of the remaining duration, or None
        if we have no bases for guessing."""
        if self.end_times is None:
            return 2 * self.elapsed_time() # We have not started the first module yet
        else:
            expected_per_job = np.median(np.diff(self.end_times[:self.num_received + 1]))
            return expected_per_job * (self.num_jobs - self.num_received)


def secs_to_timestr(duration):
    dur = int(round(duration))
    hours = dur // (60 * 60)
    rest = dur % (60 * 60)
    minutes = rest // 60
    rest = rest % 60
    seconds = rest
    minutes = ("%02d:" if hours > 0 else "%d:")%(minutes)
    hours = "%d:"%(hours,) if hours > 0 else ""
    seconds = "%02d"%(seconds)
    return hours + minutes + seconds
            
