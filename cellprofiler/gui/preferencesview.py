# coding=utf-8
"""PreferencesView.py - displays the default preferences in the lower right corner
"""

import cellprofiler.analysis
import cellprofiler.gui.help
import cellprofiler.gui.help.content
import cellprofiler.gui.htmldialog
import cellprofiler.gui.html.utils
import cellprofiler.preferences
import numpy
import os
import string
import time
import wx

WELCOME_MESSAGE = ""

WRITE_MAT_FILE = "MATLAB"
WRITE_HDF_FILE = "HDF5"
DO_NOT_WRITE_MEASUREMENTS = "Do not write measurements"

WRITE_MAT_FILE_TEXT = WRITE_MAT_FILE
WRITE_HDF_FILE_TEXT = WRITE_HDF_FILE
DO_NOT_WRITE_MEASUREMENTS_TEXT = "Do not write MATLAB or HDF5 files"


class PreferencesView(object):
    """View / controller for the preferences that get displayed in the main window

    """

    def __init__(self, parent_sizer, panel, progress_panel, status_panel):
        self.__panel = panel
        self.__parent_sizer = parent_sizer
        panel.AutoLayout = True
        panel.SetSizer(wx.BoxSizer(wx.VERTICAL))
        static_box_sizer = wx.BoxSizer(wx.VERTICAL)
        panel.Sizer.Add(static_box_sizer, 1, wx.EXPAND)
        self.__sizer = static_box_sizer
        self.__image_folder_panel = wx.Panel(panel)
        self.__image_folder_panel.AutoLayout = True
        self.__image_edit_box = self.__make_folder_panel(
            self.__image_folder_panel,
            cellprofiler.preferences.get_default_image_directory(),
            lambda: cellprofiler.preferences.get_recent_files(cellprofiler.preferences.DEFAULT_IMAGE_DIRECTORY),
            'Default Input Folder',
            cellprofiler.preferences.DEFAULT_IMAGE_FOLDER_HELP,
            [cellprofiler.preferences.set_default_image_directory,
             self.__notify_pipeline_list_view_directory_change],
            refresh_action=self.refresh_input_directory)
        self.__output_folder_panel = wx.Panel(panel)
        self.__output_folder_panel.AutoLayout = True
        self.__output_edit_box = self.__make_folder_panel(
            self.__output_folder_panel,
            cellprofiler.preferences.get_default_output_directory(),
            lambda: cellprofiler.preferences.get_recent_files(cellprofiler.preferences.DEFAULT_OUTPUT_DIRECTORY),
            'Default Output Folder',
            cellprofiler.preferences.DEFAULT_OUTPUT_FOLDER_HELP,
            [cellprofiler.preferences.set_default_output_directory,
             self.__notify_pipeline_list_view_directory_change])
        self.__odds_and_ends_panel = wx.Panel(panel)
        self.__odds_and_ends_panel.AutoLayout = True
        self.__make_odds_and_ends_panel()
        self.__status_panel = status_panel
        status_panel.Sizer = wx.BoxSizer()
        self.__status_text = wx.StaticText(
            status_panel, style=wx.SUNKEN_BORDER, label=WELCOME_MESSAGE)
        status_panel.Sizer.Add(self.__status_text, 1, wx.EXPAND)
        self.__progress_panel = progress_panel
        self.__progress_panel.AutoLayout = True
        self.__make_progress_panel()
        self.__sizer.AddMany([(self.__image_folder_panel, 0, wx.EXPAND | wx.ALL, 1),
                              (self.__output_folder_panel, 0, wx.EXPAND | wx.ALL, 1),
                              (self.__odds_and_ends_panel, 0, wx.EXPAND | wx.ALL, 1)])
        self.show_status_text()
        self.__errors = set()
        self.__pipeline_list_view = None
        self.__progress_watcher = None

    def show_default_image_folder(self, show):
        if self.__sizer.IsShown(self.__image_folder_panel) == show:
            return
        self.__sizer.Show(self.__image_folder_panel, show)
        parent = self.__image_folder_panel.GetParent()
        while parent is not None:
            parent.Layout()
            if parent == self.__image_folder_panel.GetTopLevelParent():
                break
            parent = parent.GetParent()

    def show_progress_panel(self):
        """Show the pipeline progress panel and hide the status text"""
        self.__parent_sizer.Hide(self.__status_panel)
        self.__parent_sizer.Show(self.__progress_panel)
        self.__parent_sizer.Layout()
        self.__progress_panel.Layout()

    def show_status_text(self):
        """Show the status text and hide the pipeline progress panel"""
        self.__parent_sizer.Show(self.__status_panel)
        self.__parent_sizer.Hide(self.__progress_panel)
        self.__parent_sizer.Layout()
        self.__status_panel.Layout()

    def close(self):
        cellprofiler.preferences.remove_output_file_name_listener(self.__on_preferences_output_filename_event)
        cellprofiler.preferences.remove_image_directory_listener(self.__on_preferences_image_directory_event)
        cellprofiler.preferences.remove_output_directory_listener(self.__on_preferences_output_directory_event)

    def __make_folder_panel(self, panel, value, list_fn, text, help_text,
                            actions, refresh_action=None):
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        help_button = wx.Button(panel, label='?', style=wx.BU_EXACTFIT)
        sizer.Add(help_button, 0, wx.ALIGN_CENTER)
        sizer.AddSpacer(2)
        text_static = wx.StaticText(panel, -1, text + ':')
        sizer.Add(text_static, 0, wx.ALIGN_CENTER)
        choices = list_fn()
        if value not in choices:
            choices.insert(0, value)
        edit_box = wx.ComboBox(panel, -1, value, choices=choices)
        sizer.Add(edit_box, 1, wx.ALIGN_CENTER)
        sizer.AddSpacer(2)
        browse_bmp = wx.ArtProvider.GetBitmap(wx.ART_FOLDER_OPEN,
                                              wx.ART_CMN_DIALOG,
                                              (16, 16))
        browse_button = wx.BitmapButton(panel, -1, bitmap=browse_bmp)
        browse_button.SetToolTipString("Browse for %s folder" % text)
        sizer.Add(browse_button, 0, wx.ALIGN_CENTER)
        sizer.AddSpacer(2)

        new_bmp = wx.ArtProvider.GetBitmap(wx.ART_NEW_DIR,
                                           wx.ART_CMN_DIALOG,
                                           (16, 16))
        new_button = wx.BitmapButton(panel, -1, bitmap=new_bmp)
        new_button.SetToolTipString("Make a new sub-folder")
        if os.path.isdir(value):
            new_button.Disable()
        sizer.Add(new_button, 0, wx.ALIGN_CENTER)
        if refresh_action is not None:
            refresh_bitmap = wx.ArtProvider.GetBitmap(wx.ART_REDO,
                                                      wx.ART_CMN_DIALOG,
                                                      (16, 16))
            refresh_button = wx.BitmapButton(panel, -1, bitmap=refresh_bitmap)
            sizer.AddSpacer(2)
            sizer.Add(refresh_button, 0, wx.ALIGN_CENTER, 1)
            refresh_button.SetToolTipString("Refresh the Default Input Folder list")

            def on_refresh(event):
                refresh_action()

            refresh_button.Bind(wx.EVT_BUTTON, on_refresh)
        panel.SetSizer(sizer)

        def on_new_folder(event):
            if os.path.exists(edit_box.Value):
                return
            if wx.MessageBox("Do you really want to create the %s folder?" %
                                     edit_box.Value, style=wx.YES_NO) == wx.YES:
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

        help_button.Bind(wx.EVT_BUTTON, lambda event: self.__on_help(event, help_text))
        panel.Bind(wx.EVT_BUTTON, lambda event: self.__on_browse(event, edit_box, text), browse_button)
        panel.Bind(wx.EVT_TEXT, on_edit_box_change, edit_box)
        panel.Bind(wx.EVT_COMBOBOX, on_edit_box_change, edit_box)
        panel.Bind(wx.EVT_BUTTON, on_new_folder, new_button)
        return edit_box

    def __make_odds_and_ends_panel(self):
        panel = self.__odds_and_ends_panel
        output_filename_text = wx.StaticText(panel, -1, 'Output Filename:')
        output_filename_edit_box = wx.TextCtrl(
            panel, value=cellprofiler.preferences.get_output_file_name())
        self.__output_filename_edit_box = output_filename_edit_box
        allow_output_filename_overwrite_check_box = \
            wx.CheckBox(panel, label="Allow overwrite?")
        allow_output_filename_overwrite_check_box.Value = \
            cellprofiler.preferences.get_allow_output_file_overwrite()
        write_measurements_combo_box = wx.Choice(
            panel, choices=
            [WRITE_HDF_FILE_TEXT, WRITE_MAT_FILE_TEXT,
             DO_NOT_WRITE_MEASUREMENTS_TEXT])
        # set measurements mode, then fake an event to update output
        # filename and which controls are shown.
        measurements_mode_idx = [cellprofiler.preferences.WRITE_HDF5, True, False].index(
            cellprofiler.preferences.get_write_MAT_files())
        write_measurements_combo_box.SetSelection(measurements_mode_idx)
        output_filename_help_button = wx.Button(
            panel, label='?', style=wx.BU_EXACTFIT)
        output_file_format_text = wx.StaticText(
            panel, label="Output file format:")
        cellprofiler.preferences.add_output_file_name_listener(
            self.__on_preferences_output_filename_event)
        cellprofiler.preferences.add_image_directory_listener(
            self.__on_preferences_image_directory_event)
        cellprofiler.preferences.add_output_directory_listener(
            self.__on_preferences_output_directory_event)
        self.__hold_a_reference_to_progress_callback = self.progress_callback
        cellprofiler.preferences.add_progress_callback(
            self.__hold_a_reference_to_progress_callback)

        def on_output_filename_changed(event):
            cellprofiler.preferences.set_output_file_name(output_filename_edit_box.Value)

        def on_allow_checkbox(event):
            cellprofiler.preferences.set_allow_output_file_overwrite(
                allow_output_filename_overwrite_check_box.Value)

        def on_write_MAT_files_combo_box(event):
            #
            # Update the state to reflect the new measurement choice
            #
            sel = write_measurements_combo_box.GetStringSelection()
            output_filename = output_filename_edit_box.Value
            if sel == WRITE_HDF_FILE_TEXT:
                cellprofiler.preferences.set_write_MAT_files(cellprofiler.preferences.WRITE_HDF5)
                if output_filename.lower().endswith('.mat'):
                    output_filename = output_filename[:-4] + u".h5"
            elif sel == WRITE_MAT_FILE_TEXT:
                cellprofiler.preferences.set_write_MAT_files(True)
                if output_filename.lower().endswith('.h5'):
                    output_filename = output_filename[:-3] + u".mat"
            else:
                cellprofiler.preferences.set_write_MAT_files(False)

            if output_filename != output_filename_edit_box.Value:
                output_filename_edit_box.Value = output_filename
                cellprofiler.preferences.set_output_file_name(
                    output_filename_edit_box.Value)
            #
            # Reconstruct the sizers depending on whether we have one or two rows
            #
            if sel == DO_NOT_WRITE_MEASUREMENTS_TEXT:
                show = False
                output_sizer = wx.BoxSizer(wx.HORIZONTAL)
                output_sizer.Add(output_filename_help_button, 0, wx.EXPAND)
                output_sizer.Add(output_file_format_text, 0, wx.ALIGN_RIGHT)
                output_sizer.Add(write_measurements_combo_box, 0, wx.ALIGN_LEFT)
            else:
                show = True
                output_sizer = wx.FlexGridSizer(2, 3, 2, 2)
                output_sizer.SetFlexibleDirection(wx.HORIZONTAL)
                output_sizer.AddGrowableCol(2)
                output_filename_edit_box_sizer = wx.BoxSizer(wx.HORIZONTAL)
                output_filename_edit_box_sizer.Add(
                    output_filename_edit_box, 1, wx.EXPAND)
                output_filename_edit_box_sizer.AddSpacer(2)
                output_filename_edit_box_sizer.Add(
                    allow_output_filename_overwrite_check_box, 0,
                    wx.ALIGN_CENTER)
                output_sizer.Add(output_filename_help_button, 0, wx.EXPAND)
                output_sizer.Add(output_filename_text, 0, wx.ALIGN_RIGHT)
                output_sizer.Add(output_filename_edit_box_sizer, 1, wx.EXPAND)

                output_sizer.Add(wx.BoxSizer(), 0, wx.EXPAND)
                output_sizer.Add(output_file_format_text, 0, wx.ALIGN_RIGHT)
                output_sizer.Add(write_measurements_combo_box, 0, wx.ALIGN_LEFT)

            panel.SetSizer(output_sizer)
            for ctrl in (output_filename_text,
                         output_filename_edit_box,
                         allow_output_filename_overwrite_check_box):
                ctrl.Show(show)

            panel.Parent.Layout()
            panel.Layout()

        write_measurements_combo_box.Bind(
            wx.EVT_CHOICE, on_write_MAT_files_combo_box)
        allow_output_filename_overwrite_check_box.Bind(
            wx.EVT_CHECKBOX, on_allow_checkbox)
        output_filename_help_button.Bind(
            wx.EVT_BUTTON,
            lambda event: self.__on_help(event, cellprofiler.gui.help.content.read_content("legacy_output_file.rst")))
        output_filename_edit_box.Bind(wx.EVT_TEXT, on_output_filename_changed)
        panel.Bind(wx.EVT_WINDOW_DESTROY, self.__on_destroy, panel)
        on_write_MAT_files_combo_box(None)

    def update_worker_count_info(self, n_workers):
        """Update the # of running workers in the progress UI

        n_workers - # of workers running
        """
        if n_workers == 1:
            label = "Running 1 worker."
        else:
            label = "Running %d workers." % n_workers
        self.__worker_count_ctrl.Label = label

    def __make_progress_panel(self):
        panel = self.__progress_panel
        self.__progress_msg_ctrl = wx.StaticText(panel)
        self.__worker_count_ctrl = wx.StaticText(panel)
        self.__progress_bar = wx.Gauge(panel, -1, size=(100, -1))
        self.__progress_bar.Value = 25
        self.__timer = wx.StaticText(panel)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.AddMany([((1, 1), 1),
                       (self.__progress_msg_ctrl, 0, wx.ALIGN_BOTTOM),
                       ((10, 0), 0),
                       (self.__worker_count_ctrl, 0, wx.ALIGN_BOTTOM),
                       ((10, 0), 0),
                       (self.__progress_bar, 0, wx.ALIGN_BOTTOM),
                       ((10, 0), 0),
                       (self.__timer, 0, wx.ALIGN_BOTTOM)])
        panel.SetSizer(sizer)
        panel.Layout()
        #
        # The progress model is that one operation might invoke sub-operations
        # which might report their own progress. We model this as a stack,
        # tracking the innermost operation until done, then popping.
        #
        # The dictionary has as its key the operation ID and as it's value
        # a tuple of amount done and current message
        #
        self.__progress_stack = []
        self.__progress_dictionary = {}
        self.__progress_dialog = None

    def progress_callback(self, operation_id, progress, message):
        """Monitor progress events in the UI thread

        operation_id - a unique id identifying an instance of an operation
        progress - a number from 0 to 1 where 0 is the start of the operation
                   and 1 is its end.
        message - the message to display to the user.
        """
        if progress is None:
            if message is None:
                message = WELCOME_MESSAGE
            self.set_message_text(message)
            return

        def reset_progress():
            self.__progress_stack = []
            self.__progress_dictionary = {}
            if self.__progress_dialog is not None:
                self.__progress_dialog.Destroy()
                self.__progress_dialog = None
            wx.SetCursor(wx.NullCursor)
            self.set_message_text(WELCOME_MESSAGE)
            wx.SafeYield(None, True)

        if operation_id is None:
            reset_progress()
            return

        if operation_id not in self.__progress_stack:
            self.__progress_stack.append(operation_id)
        else:
            loc = self.__progress_stack.index(operation_id)
            if loc == 0 and progress == 1:
                reset_progress()
                return
            if progress == 1:
                loc -= 1
            for operation_id in self.__progress_stack[(loc + 1):]:
                del self.__progress_dictionary[operation_id]
            self.__progress_stack = self.__progress_stack[:(loc + 1)]
        self.__progress_dictionary[operation_id] = (progress, message)
        wx.SetCursor(wx.StockCursor(wx.CURSOR_WAIT))
        message = ", ".join(
            ["%s (%d %%)" % (message, int(progress * 100))
             for progress, message in [
                 self.__progress_dictionary[o] for o in self.__progress_stack]])
        self.set_message_text(message)
        wx.SafeYield(None, True)  # ouch, can't repaint without it.

    def check_preferences(self):
        """Return True if preferences are OK (e.g., directories exist)"""
        path = self.__image_edit_box.Value
        if not os.path.isdir(path):
            if wx.MessageBox(('The Default Input Folder is "%s", but '
                              'the directory does not exist. Do you want to '
                              'create it?') % path,
                             "Warning, cannot run pipeline",
                             style=wx.YES_NO) == wx.NO:
                return False, "Image directory does not exist"
            os.makedirs(path)
            cellprofiler.preferences.set_default_image_directory(path)
        path = self.__output_edit_box.Value
        if not os.path.isdir(path):
            if wx.MessageBox(('The Default Output Folder is "%s", but '
                              'the directory does not exist. Do you want to '
                              'create it?') % path,
                             "Warning, cannot run pipeline",
                             style=wx.YES_NO) == wx.NO:
                return False, "Output directory does not exist"
            os.makedirs(path)
            cellprofiler.preferences.set_default_output_directory(path)
        return True, "OK"

    def __on_destroy(self, event):
        cellprofiler.preferences.remove_image_directory_listener(self.__on_preferences_image_directory_event)
        cellprofiler.preferences.remove_output_directory_listener(self.__on_preferences_output_directory_event)
        cellprofiler.preferences.remove_output_file_name_listener(self.__on_preferences_output_filename_event)

    def attach_to_pipeline_list_view(self, pipeline_list_view):
        self.__pipeline_list_view = pipeline_list_view

    def on_analyze_images(self):
        # begin tracking progress
        self.__progress_watcher = ProgressWatcher(
            self.__progress_panel,
            self.update_progress,
            multiprocessing=cellprofiler.analysis.use_analysis)
        self.show_progress_panel()

    def on_pipeline_progress(self, *args):
        if self.__progress_watcher is not None:
            self.__progress_watcher.on_pipeline_progress(*args)

    def pause(self, do_pause):
        self.__progress_watcher.pause(do_pause)

    def update_progress(self, message, elapsed_time, remaining_time=None):
        #
        # Disable everything if in a modal state. The progress bar
        # seems to eat memory in huge chunks if allowed to draw its
        # oh so shiny self while modal in the ultra awesome Mac
        # interface. But you have to admit, the design is disgustingly
        # elegant even if it does cause my ugly application to
        # crash horribly.
        #
        # Taken from Cody Precord's post
        # https://groups.google.com/forum/#!topic/wxpython-users/s8AQ64ptyCg
        #
        for win in wx.GetTopLevelWindows():
            if isinstance(win, wx.Dialog):
                if win.IsModal():
                    self.__progress_bar.Show(False)
                    return
        self.__progress_bar.Show(True)
        self.__progress_msg_ctrl.Label = message
        if remaining_time is not None:
            self.__progress_bar.Value = \
                (100 * elapsed_time) / (elapsed_time + remaining_time + .00001)
            timestr = 'Time %s/%s' % (
                secs_to_timestr(elapsed_time),
                secs_to_timestr(elapsed_time + remaining_time))
        else:
            self.__progress_bar.Pulse()
            timestr = "Elapsed time: %s" % secs_to_timestr(elapsed_time)
        self.__timer.SetLabel(timestr)
        self.__progress_panel.Layout()

    def on_stop_analysis(self):
        if self.__progress_watcher is not None:
            self.__progress_watcher.stop()
        self.__progress_watcher = None
        self.show_status_text()

    def set_message_text(self, text):
        if self.__status_text.Label != text:
            saved_size = self.__status_text.GetSize()
            self.__status_text.SetLabel(text)
            self.__status_text.SetSize(saved_size)
            self.__status_text.Update()

    def pop_error_text(self, error_text):
        if error_text in self.__errors:
            self.__errors.remove(error_text)
            if len(self.__errors) == 0:
                self.set_message_text(WELCOME_MESSAGE)
            else:
                self.set_message_text(next(self.__errors.__iter__()))

    def set_error_text(self, error_text):
        self.set_message_text(error_text)
        self.__errors.add(error_text)

    def __on_browse(self, event, edit_box, text):
        dir_dialog = wx.DirDialog(self.__panel, string.capitalize(text), edit_box.GetValue())
        if dir_dialog.ShowModal() == wx.ID_OK:
            edit_box.SetValue(dir_dialog.GetPath())
            fake_event = wx.CommandEvent(wx.wxEVT_COMMAND_TEXT_UPDATED)
            fake_event.EventObject = edit_box
            fake_event.Id = edit_box.Id
            edit_box.GetEventHandler().ProcessEvent(fake_event)

    def __on_edit_box_change(self, event, edit_box, text, actions):
        path = edit_box.GetValue()
        error_text = 'The %s is not a directory' % text
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
                        edit_box.Delete(i + 1)
            self.pop_error_text(error_text)
        else:
            self.set_error_text(error_text)

    def __on_help(self, event, help_text):
        dlg = cellprofiler.gui.htmldialog.HTMLDialog(
            self.__panel,
            "Help",
            cellprofiler.gui.html.utils.rst_to_html_fragment(help_text)
        )
        dlg.Show()

    def __on_pixel_size_changed(self, event):
        error_text = 'Pixel size must be a number'
        text = self.__pixel_size_edit_box.Value
        if text.isdigit():
            cellprofiler.preferences.set_pixel_size(int(text))
            self.pop_error_text(error_text)
        else:
            self.set_error_text(error_text)

    def __on_preferences_output_filename_event(self, event):
        if self.__output_filename_edit_box.Value != cellprofiler.preferences.get_output_file_name():
            self.__output_filename_edit_box.Value = cellprofiler.preferences.get_output_file_name()

    def __on_preferences_output_directory_event(self, event):
        old_selection = self.__output_edit_box.Selection
        if self.__output_edit_box.Value != cellprofiler.preferences.get_default_output_directory():
            self.__output_edit_box.Value = cellprofiler.preferences.get_default_output_directory()

    def __on_preferences_image_directory_event(self, event):
        if self.__image_edit_box.Value != cellprofiler.preferences.get_default_image_directory():
            self.__image_edit_box.Value = cellprofiler.preferences.get_default_image_directory()

    def __notify_pipeline_list_view_directory_change(self, path):
        # modules may need revalidation
        if self.__pipeline_list_view is not None:
            self.__pipeline_list_view.notify_directory_change()

    @staticmethod
    def refresh_input_directory():
        cellprofiler.preferences.fire_image_directory_changed_event()


class ProgressWatcher(object):
    """ Tracks pipeline progress and estimates time to completion """

    def __init__(self, parent, update_callback, multiprocessing=False):
        self.update_callback = update_callback

        # start tracking progress
        self.start_time = time.time()
        self.end_times = None
        self.current_module_name = ''
        self.pause_start_time = None
        self.previous_pauses_duration = 0.0
        self.image_set_index = 0
        self.num_image_sets = 1

        # for multiprocessing computation
        self.num_jobs = 1
        self.num_received = 0

        self.multiprocessing = multiprocessing

        timer_id = wx.NewId()
        self.timer = wx.Timer(parent, timer_id)
        self.timer.Start(500)
        if not multiprocessing:
            wx.EVT_TIMER(parent, timer_id, self.update)
            self.update()
        else:
            wx.EVT_TIMER(parent, timer_id, self.update_multiprocessing)
            self.update_multiprocessing()

    def stop(self):
        self.timer.Stop()

    def update(self, event=None):
        status = '%s, Image Set %d/%d' % (self.current_module_name, self.image_set_index + 1, self.num_image_sets)
        self.update_callback(status,
                             self.elapsed_time(),
                             self.remaining_time())

    def update_multiprocessing(self, event=None):
        if self.num_jobs > self.num_received:
            status = 'Processing: %d of %d image sets completed' % \
                     (self.num_received, self.num_jobs)
            self.update_callback(
                status,
                self.elapsed_time(),
                self.remaining_time_multiprocessing())
        else:
            status = "Post-processing, please wait"
            self.update_callback(status, self.elapsed_time())

    def on_pipeline_progress(self, *args):
        if not self.multiprocessing:
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
            self.end_times = numpy.zeros(1 + num_modules * num_image_sets)
        module_index = module.module_num - 1  # make it zero-based
        index = image_set_index * num_modules + (module_index - 1)
        self.end_times[1 + index] = self.elapsed_time()

        self.update()

    def on_receive_work(self, num_jobs, num_received):
        self.num_jobs = num_jobs
        self.num_received = num_received
        if self.end_times is None:
            # One extra element at the beginning for the start time
            self.end_times = numpy.zeros(1 + num_jobs)
            self.end_times[0] = self.elapsed_time()
        self.end_times[num_received] = self.elapsed_time()
        self.update_multiprocessing()

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
        """Return the number of seconds that have elapsed since start
           as a float.  Pauses are taken into account.
        """
        return self.adjusted_time() - self.start_time

    def remaining_time(self):
        """Return our best estimate of the remaining duration, or None
        if we have no bases for guessing."""
        if self.end_times is None:
            return 2 * self.elapsed_time()  # We have not started the first module yet
        else:
            module_index = self.current_module.module_num - 1
            index = self.image_set_index * self.num_modules + module_index
            durations = (self.end_times[1:] - self.end_times[:-1]).reshape(self.num_image_sets, self.num_modules)
            per_module_estimates = numpy.zeros(self.num_modules)
            per_module_estimates[:module_index] = numpy.median(durations[:self.image_set_index + 1, :module_index], 0)
            current_module_so_far = self.elapsed_time() - self.end_times[1 + index - 1]
            if self.image_set_index > 0:
                per_module_estimates[module_index:] = numpy.median(durations[:self.image_set_index, module_index:], 0)
                per_module_estimates[module_index] = max(per_module_estimates[module_index], current_module_so_far)
            else:
                # Guess that the modules that haven't finished yet are
                # as slow as the slowest one we've seen so far.
                per_module_estimates[module_index] = current_module_so_far
                per_module_estimates[module_index:] = per_module_estimates[:module_index + 1].max()
            per_module_estimates[:module_index] *= self.num_image_sets - self.image_set_index - 1
            per_module_estimates[module_index:] *= self.num_image_sets - self.image_set_index
            per_module_estimates[module_index] -= current_module_so_far
            return per_module_estimates.sum()

    def remaining_time_multiprocessing(self):
        """Return our best estimate of the remaining duration, or None
        if we have no bases for guessing."""
        if (self.end_times is None) or (self.num_received == 0):
            return 2 * self.elapsed_time()  # We have not started the first module yet
        else:
            expected_per_job = self.end_times[self.num_received] / self.num_received
            return expected_per_job * (self.num_jobs - self.num_received)


def secs_to_timestr(duration):
    dur = int(round(duration))
    hours = dur // (60 * 60)
    rest = dur % (60 * 60)
    minutes = rest // 60
    rest %= 60
    seconds = rest
    minutes = ("%02d:" if hours > 0 else "%d:") % minutes
    hours = "%d:" % (hours,) if hours > 0 else ""
    seconds = "%02d" % seconds
    return hours + minutes + seconds
