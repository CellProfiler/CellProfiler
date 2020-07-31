"""
displays the default preferences in the lower right corner
"""

import os

import wx
from cellprofiler_core.analysis import use_analysis
from cellprofiler_core.preferences import DEFAULT_IMAGE_DIRECTORY
from cellprofiler_core.preferences import DEFAULT_IMAGE_FOLDER_HELP
from cellprofiler_core.preferences import DEFAULT_OUTPUT_DIRECTORY
from cellprofiler_core.preferences import DEFAULT_OUTPUT_FOLDER_HELP
from cellprofiler_core.preferences import add_image_directory_listener
from cellprofiler_core.preferences import add_output_directory_listener
from cellprofiler_core.preferences import add_progress_callback
from cellprofiler_core.preferences import fire_image_directory_changed_event
from cellprofiler_core.preferences import get_default_image_directory
from cellprofiler_core.preferences import get_default_output_directory
from cellprofiler_core.preferences import get_recent_files
from cellprofiler_core.preferences import remove_image_directory_listener
from cellprofiler_core.preferences import remove_output_directory_listener
from cellprofiler_core.preferences import set_default_image_directory
from cellprofiler_core.preferences import set_default_output_directory
from cellprofiler_core.preferences import set_pixel_size

from ._progress_watcher import ProgressWatcher
from ..constants.preferences_view import WELCOME_MESSAGE
from ..html.utils import rst_to_html_fragment
from ..htmldialog import HTMLDialog
from ..utilities.preferences_view import secs_to_timestr


class PreferencesView:
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
        self.__image_folder_panel.SetAutoLayout(True)
        self.__image_edit_box = self.__make_folder_panel(
            self.__image_folder_panel,
            get_default_image_directory(),
            lambda: get_recent_files(DEFAULT_IMAGE_DIRECTORY),
            "Default Input Folder",
            DEFAULT_IMAGE_FOLDER_HELP,
            [
                set_default_image_directory,
                self.__notify_pipeline_list_view_directory_change,
            ],
            refresh_action=self.refresh_input_directory,
        )
        self.__output_folder_panel = wx.Panel(panel)
        self.__output_folder_panel.SetAutoLayout(True)
        self.__output_edit_box = self.__make_folder_panel(
            self.__output_folder_panel,
            get_default_output_directory(),
            lambda: get_recent_files(DEFAULT_OUTPUT_DIRECTORY),
            "Default Output Folder",
            DEFAULT_OUTPUT_FOLDER_HELP,
            [
                set_default_output_directory,
                self.__notify_pipeline_list_view_directory_change,
            ],
        )
        self.__odds_and_ends_panel = wx.Panel(panel)
        self.__odds_and_ends_panel.SetAutoLayout(True)
        self.__make_odds_and_ends_panel()
        self.__status_panel = status_panel
        status_panel.Sizer = wx.BoxSizer()
        self.__status_text = wx.StaticText(
            status_panel, style=wx.SUNKEN_BORDER, label=WELCOME_MESSAGE
        )
        status_panel.Sizer.Add(self.__status_text, 1, wx.EXPAND)
        self.__progress_panel = progress_panel
        self.__progress_panel.AutoLayout = True
        self.__make_progress_panel()
        self.__sizer.AddMany(
            [
                (self.__image_folder_panel, 0, wx.EXPAND | wx.ALL, 1),
                (self.__output_folder_panel, 0, wx.EXPAND | wx.ALL, 1),
                (self.__odds_and_ends_panel, 0, wx.EXPAND | wx.ALL, 1),
            ]
        )
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
        remove_image_directory_listener(self.__on_preferences_image_directory_event)
        remove_output_directory_listener(self.__on_preferences_output_directory_event)

    def __make_folder_panel(
        self, panel, value, list_fn, text, help_text, actions, refresh_action=None
    ):
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        help_button = wx.Button(panel, label="?", style=wx.BU_EXACTFIT)
        sizer.Add(help_button, 0, wx.ALIGN_CENTER)
        sizer.AddSpacer(2)
        text_static = wx.StaticText(panel, -1, text + ":")
        sizer.Add(text_static, 0, wx.ALIGN_CENTER)
        choices = list(set(list_fn()))
        if value not in choices:
            choices.insert(0, value)
        edit_box = wx.ComboBox(panel, -1, value, choices=choices)
        sizer.Add(edit_box, 1, wx.ALIGN_CENTER)
        sizer.AddSpacer(2)
        browse_bmp = wx.ArtProvider.GetBitmap(
            wx.ART_FOLDER_OPEN, wx.ART_CMN_DIALOG, (16, 16)
        )
        browse_button = wx.BitmapButton(panel, -1, bitmap=browse_bmp)
        browse_button.SetToolTip("Browse for %s folder" % text)
        sizer.Add(browse_button, 0, wx.ALIGN_CENTER)
        sizer.AddSpacer(2)

        new_bmp = wx.ArtProvider.GetBitmap(wx.ART_NEW_DIR, wx.ART_CMN_DIALOG, (16, 16))
        new_button = wx.BitmapButton(panel, -1, bitmap=new_bmp)
        new_button.SetToolTip("Make a new sub-folder")
        if os.path.isdir(value):
            new_button.Disable()
        sizer.Add(new_button, 0, wx.ALIGN_CENTER)
        if refresh_action is not None:
            refresh_bitmap = wx.ArtProvider.GetBitmap(
                wx.ART_REDO, wx.ART_CMN_DIALOG, (16, 16)
            )
            refresh_button = wx.BitmapButton(panel, -1, bitmap=refresh_bitmap)
            sizer.AddSpacer(2)
            sizer.Add(refresh_button, 0, wx.ALIGN_CENTER, 1)
            refresh_button.SetToolTip("Refresh the Default Input Folder list")

            def on_refresh(event):
                refresh_action()

            refresh_button.Bind(wx.EVT_BUTTON, on_refresh)
        panel.SetSizer(sizer)

        def on_new_folder(event):
            if os.path.exists(edit_box.GetValue()):
                return
            if (
                wx.MessageBox(
                    "Do you really want to create the %s folder?" % edit_box.GetValue(),
                    style=wx.YES_NO,
                )
                == wx.YES
            ):
                os.makedirs(edit_box.GetValue())
                self.__on_edit_box_change(event, edit_box, text, actions)

        def on_edit_box_change(event):
            if os.path.isdir(edit_box.GetValue()):
                new_button.Disable()
                new_button.SetToolTip("%s is a directory" % edit_box.GetValue())
            else:
                new_button.Enable()
                new_button.SetToolTip(
                    "Press button to create the %s folder" % edit_box.GetValue()
                )
            self.__on_edit_box_change(event, edit_box, text, actions)
            event.Skip()

        help_button.Bind(wx.EVT_BUTTON, lambda event: self.__on_help(event, help_text))
        panel.Bind(
            wx.EVT_BUTTON,
            lambda event: self.__on_browse(event, edit_box, text),
            browse_button,
        )
        panel.Bind(wx.EVT_TEXT, on_edit_box_change, edit_box)
        panel.Bind(wx.EVT_COMBOBOX, on_edit_box_change, edit_box)
        panel.Bind(wx.EVT_BUTTON, on_new_folder, new_button)
        return edit_box

    def __make_odds_and_ends_panel(self):
        panel = self.__odds_and_ends_panel
        add_image_directory_listener(self.__on_preferences_image_directory_event)
        add_output_directory_listener(self.__on_preferences_output_directory_event)
        self.__hold_a_reference_to_progress_callback = self.progress_callback
        add_progress_callback(self.__hold_a_reference_to_progress_callback)
        panel.Bind(wx.EVT_WINDOW_DESTROY, self.__on_destroy, panel)

    def update_worker_count_info(self, n_workers):
        """Update the # of running workers in the progress UI

        n_workers - # of workers running
        """
        if n_workers == 1:
            label = "Running 1 worker."
        else:
            label = "Running %d workers." % n_workers
        self.__worker_count_ctrl.SetLabel(label)

    def __make_progress_panel(self):
        panel = self.__progress_panel
        self.__progress_msg_ctrl = wx.StaticText(panel)
        self.__worker_count_ctrl = wx.StaticText(panel)
        self.__progress_bar = wx.Gauge(panel, -1, size=(100, -1))
        self.__progress_bar.SetValue(25)
        self.__timer = wx.StaticText(panel)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.AddMany(
            [
                ((1, 1), 1),
                (self.__progress_msg_ctrl, 0, wx.ALIGN_BOTTOM),
                ((10, 0), 0),
                (self.__worker_count_ctrl, 0, wx.ALIGN_BOTTOM),
                ((10, 0), 0),
                (self.__progress_bar, 0, wx.ALIGN_BOTTOM),
                ((10, 0), 0),
                (self.__timer, 0, wx.ALIGN_BOTTOM),
            ]
        )
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
            for operation_id in self.__progress_stack[(loc + 1) :]:
                del self.__progress_dictionary[operation_id]
            self.__progress_stack = self.__progress_stack[: (loc + 1)]
        self.__progress_dictionary[operation_id] = (progress, message)
        wx.SetCursor(wx.Cursor(wx.CURSOR_WAIT))
        message = ", ".join(
            [
                "%s (%d %%)" % (message, int(progress * 100))
                for progress, message in [
                    self.__progress_dictionary[o] for o in self.__progress_stack
                ]
            ]
        )
        self.set_message_text(message)
        wx.SafeYield(None, True)  # ouch, can't repaint without it.

    def check_preferences(self):
        """Return True if preferences are OK (e.g., directories exist)"""
        path = self.__image_edit_box.GetValue()
        if not os.path.isdir(path):
            if (
                wx.MessageBox(
                    (
                        'The Default Input Folder is "%s", but '
                        "the directory does not exist. Do you want to "
                        "create it?"
                    )
                    % path,
                    "Warning, cannot run pipeline",
                    style=wx.YES_NO,
                )
                == wx.NO
            ):
                return False, "Image directory does not exist"
            os.makedirs(path)
            set_default_image_directory(path)
        path = self.__output_edit_box.GetValue()
        if not os.path.isdir(path):
            if (
                wx.MessageBox(
                    (
                        'The Default Output Folder is "%s", but '
                        "the directory does not exist. Do you want to "
                        "create it?"
                    )
                    % path,
                    "Warning, cannot run pipeline",
                    style=wx.YES_NO,
                )
                == wx.NO
            ):
                return False, "Output directory does not exist"
            os.makedirs(path)
            set_default_output_directory(path)
        return True, "OK"

    def __on_destroy(self, event):
        remove_image_directory_listener(self.__on_preferences_image_directory_event)
        remove_output_directory_listener(self.__on_preferences_output_directory_event)

    def attach_to_pipeline_list_view(self, pipeline_list_view):
        self.__pipeline_list_view = pipeline_list_view

    def on_analyze_images(self):
        # begin tracking progress
        self.__progress_watcher = ProgressWatcher(
            self.__progress_panel, self.update_progress, multiprocessing=use_analysis,
        )
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
        self.__progress_msg_ctrl.SetLabel(message)
        if remaining_time is not None:
            self.__progress_bar.SetValue(
                (100 * elapsed_time) / (elapsed_time + remaining_time + 0.00001)
            )
            timestr = "Time %s/%s" % (
                secs_to_timestr(elapsed_time),
                secs_to_timestr(elapsed_time + remaining_time),
            )
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
        if self.__status_text.GetLabel() != text:
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
        dir_dialog = wx.DirDialog(
            self.__panel, str.capitalize(text), edit_box.GetValue()
        )
        if dir_dialog.ShowModal() == wx.ID_OK:
            edit_box.SetValue(dir_dialog.GetPath())
            fake_event = wx.CommandEvent(wx.wxEVT_COMMAND_TEXT_UPDATED)
            fake_event.SetEventObject(edit_box)
            fake_event.SetId(edit_box.Id)
            edit_box.GetEventHandler().ProcessEvent(fake_event)

    def __on_edit_box_change(self, event, edit_box, text, actions):
        path = edit_box.GetValue()
        error_text = "The %s is not a directory" % text
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

                # Remove items that are simply the abspath
                filtered_items = list(
                    filter(lambda item: os.path.abspath(item) != abspath, items)
                )
                # Clearing the whole edit box also wipes typed text. Specifically remove items which are invalid.
                for i in range(len(items), 0, -1):
                    if items[i - 1] not in filtered_items:
                        edit_box.Delete(i)
                    else:
                        # Prevent exact duplicate entries
                        filtered_items.remove(items[i - 1])
            self.pop_error_text(error_text)
        else:
            self.set_error_text(error_text)

    def __on_help(self, event, help_text):
        dlg = HTMLDialog(self.__panel, "Help", rst_to_html_fragment(help_text),)
        dlg.Show()

    def __on_pixel_size_changed(self, event):
        error_text = "Pixel size must be a number"
        text = self.__pixel_size_edit_box.GetValue()
        if text.isdigit():
            set_pixel_size(int(text))
            self.pop_error_text(error_text)
        else:
            self.set_error_text(error_text)

    def __on_preferences_output_directory_event(self, event):
        old_selection = self.__output_edit_box.GetSelection()
        if self.__output_edit_box.GetValue() != get_default_output_directory():
            self.__output_edit_box.SetValue(get_default_output_directory())

    def __on_preferences_image_directory_event(self, event):
        if self.__image_edit_box.GetValue() != get_default_image_directory():
            self.__image_edit_box.SetValue(get_default_image_directory())

    def __notify_pipeline_list_view_directory_change(self, path):
        # modules may need revalidation
        if self.__pipeline_list_view is not None:
            self.__pipeline_list_view.notify_directory_change()

    @staticmethod
    def refresh_input_directory():
        fire_image_directory_changed_event()
