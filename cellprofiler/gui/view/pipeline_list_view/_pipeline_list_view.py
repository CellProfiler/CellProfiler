import io
import logging
import math
import os
import sys
import time

import wx
from cellprofiler_core.constants.pipeline import DIRECTION_UP
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.pipeline import PipelineLoaded
from cellprofiler_core.pipeline import ModuleAdded
from cellprofiler_core.pipeline import ModuleMoved
from cellprofiler_core.pipeline import ModuleRemoved
from cellprofiler_core.pipeline import PipelineCleared
from cellprofiler_core.pipeline import ModuleEdited
from cellprofiler_core.pipeline import ModuleEnabled
from cellprofiler_core.pipeline import ModuleDisabled
from cellprofiler_core.pipeline import ModuleShowWindow
from cellprofiler_core.pipeline import dump
from cellprofiler_core.preferences import EXT_PROJECT_CHOICES
from cellprofiler_core.preferences import EXT_PIPELINE_CHOICES

from cellprofiler.gui.constants.frame import ID_EDIT_DELETE
from cellprofiler.gui.constants.frame import ID_EDIT_DUPLICATE
from cellprofiler.gui.constants.frame import ID_EDIT_ENABLE_MODULE
from cellprofiler.gui.constants.frame import ID_HELP_MODULE
from cellprofiler.gui.constants.frame import ID_DEBUG_RUN_FROM_THIS_MODULE
from cellprofiler.gui.constants.frame import ID_DEBUG_STEP_FROM_THIS_MODULE
from cellprofiler.gui.constants.pipeline_list_view import EVT_PLV_SLIDER_MOTION
from cellprofiler.gui.constants.pipeline_list_view import EVT_PLV_ERROR_COLUMN_CLICKED
from cellprofiler.gui.constants.pipeline_list_view import EVT_PLV_EYE_COLUMN_CLICKED
from cellprofiler.gui.constants.pipeline_list_view import EVT_PLV_STEP_COLUMN_CLICKED
from cellprofiler.gui.constants.pipeline_list_view import EVT_PLV_PAUSE_COLUMN_CLICKED
from cellprofiler.gui.constants.pipeline_list_view import PLV_STATE_UNAVAILABLE
from cellprofiler.gui.constants.pipeline_list_view import PLV_STATE_PROCEED
from cellprofiler.gui.constants.pipeline_list_view import MODULE_NAME_COLUMN
from cellprofiler.gui.constants.pipeline_list_view import (
    EVT_PLV_VALID_STEP_COLUMN_CLICKED,
)
from cellprofiler.gui.constants.pipeline_list_view import PLV_STATE_WARNING
from cellprofiler.gui.constants.pipeline_list_view import PLV_STATE_ERROR
from cellprofiler.gui.constants.pipeline_list_view import PLV_STATE_ERROR_MASK
from cellprofiler.gui.widget.frame._figure_frame import window_name
from cellprofiler.gui.widget.frame._figure_frame import find_fig
from cellprofiler.gui.view.module_view import ValidationRequestController
from ._pipeline_data_object import PipelineDataObject
from ._pipeline_drop_target import PipelineDropTarget
from ._pipeline_list_ctrl import PipelineListCtrl
from cellprofiler.gui.utilities.module_view import request_module_validation


class PipelineListView:
    """View on a set of modules

    Here is the window hierarchy within the panel:
    pipeline static box
    top_sizer
        input panel
             box sizer
                 "Input" static box
                      static box sizer
                          input module list control
        "Modules" static box
            self.__sizer
                self.__pipeline_slider
                self.list_ctrl
        "Outputs" static box
            static box sizer
                self.outputs_panel
                    box sizer
                        show preferences button
    """

    def __init__(self, panel, frame):
        self.__pipeline = None
        self.__panel = panel
        self.__frame = frame
        self.__module_controls_panel = None
        assert isinstance(panel, wx.Window)
        top_sizer = wx.BoxSizer(orient=wx.VERTICAL)
        panel.SetSizer(top_sizer)
        self.__input_controls = []
        self.__input_sizer = wx.BoxSizer(wx.VERTICAL)
        top_sizer.Add(self.__input_sizer, 0, wx.EXPAND)
        self.input_list_ctrl = None
        self.make_input_panel()
        self.__sizer = wx.BoxSizer(wx.HORIZONTAL)
        top_sizer.Add(self.__sizer, 1, wx.EXPAND)
        outputs_sizer = wx.BoxSizer(wx.VERTICAL)
        top_sizer.Add(outputs_sizer, 0, wx.EXPAND)
        self.outputs_panel = wx.Panel(panel)
        outputs_sizer.Add(self.outputs_panel, 1, wx.EXPAND)
        self.outputs_panel.SetSizer(wx.BoxSizer())
        self.outputs_panel.SetBackgroundStyle(wx.BG_STYLE_ERASE)
        self.outputs_button = wx.Button(
            self.outputs_panel, label="View output settings", style=wx.BU_EXACTFIT
        )
        self.outputs_panel.GetSizer().AddStretchSpacer(1)
        self.outputs_panel.GetSizer().Add(self.outputs_button, 0, wx.ALL, 2)
        self.outputs_panel.GetSizer().AddStretchSpacer(1)
        self.outputs_button.Bind(wx.EVT_BUTTON, self.on_outputs_button)
        self.outputs_panel.SetAutoLayout(True)
        self.__panel.Layout()
        self.outputs_panel.Layout()
        self.__panel.SetAutoLayout(True)
        self.make_list()
        self.set_debug_mode(False)
        self.__adjust_rows()
        self.__submission_time = 0
        self.drag_underway = False
        self.drag_start = None
        self.drag_time = None
        self.list_ctrl.SetDropTarget(PipelineDropTarget(self))
        self.validation_requests = []
        self.__allow_editing = True
        self.__has_file_list = False

    def allow_editing(self, allow):
        """Allow or disallow pipeline editing

        allow - true to allow, false to prevent
        """
        self.__allow_editing = allow

    def make_list(self):
        """Make the list control with the pipeline items in it"""
        self.list_ctrl = PipelineListCtrl(self.__panel)
        self.__sizer.Add(self.list_ctrl, 1, wx.EXPAND | wx.TOP, border=2)
        #
        # Bind events
        #
        self.list_ctrl.Bind(wx.EVT_LIST_ITEM_SELECTED, self.__on_item_selected)
        self.list_ctrl.Bind(wx.EVT_LIST_ITEM_DESELECTED, self.__on_item_deselected)
        self.list_ctrl.Bind(EVT_PLV_SLIDER_MOTION, self.__on_slider_motion)
        self.list_ctrl.Bind(wx.EVT_LEFT_DCLICK, self.__on_list_dclick)
        self.list_ctrl.Bind(wx.EVT_CONTEXT_MENU, self.__on_list_context_menu)
        self.list_ctrl.Bind(
            EVT_PLV_ERROR_COLUMN_CLICKED, self.__on_error_column_clicked
        )
        self.list_ctrl.Bind(EVT_PLV_EYE_COLUMN_CLICKED, self.__on_eye_column_clicked)
        self.list_ctrl.Bind(EVT_PLV_STEP_COLUMN_CLICKED, self.__on_step_column_clicked)
        self.list_ctrl.Bind(
            EVT_PLV_PAUSE_COLUMN_CLICKED, self.__on_pause_column_clicked
        )
        self.list_ctrl.Bind(wx.EVT_LIST_BEGIN_DRAG, self.start_drag_operation)
        self.list_ctrl.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self.__on_item_activated)
        self.input_list_ctrl.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self.__on_item_activated)
        #
        # Accelerators
        #
        accelerator_table = wx.AcceleratorTable(
            [(wx.ACCEL_NORMAL, wx.WXK_DELETE, ID_EDIT_DELETE)]
        )
        self.list_ctrl.SetAcceleratorTable(accelerator_table)

    def make_input_panel(self):
        self.input_list_ctrl = PipelineListCtrl(self.__panel)
        self.input_list_ctrl.set_show_step(False)
        self.input_list_ctrl.set_show_go_pause(False)
        self.input_list_ctrl.set_show_frame_column(False)
        self.input_list_ctrl.set_allow_disable(False)
        self.input_list_ctrl.always_draw_current_as_if_selected = True
        self.__input_controls.append(self.input_list_ctrl)
        self.__input_sizer.Add(self.input_list_ctrl, 1, wx.EXPAND)
        self.add_transparent_window_for_tooltip(self.input_list_ctrl)

    def add_transparent_window_for_tooltip(self, input_list_ctrl):
        #
        # You can't display a tooltip over a disabled window. But you can
        # display a tooltip over a transparent window in front of the disabled
        # window.
        #
        if sys.platform.startswith("linux"):
            self.transparent_window = None
            return  # Doesn't work right.
        transparent_window = wx.Panel(self.__panel)

        def on_background_paint(event):
            assert isinstance(event, wx.EraseEvent)
            dc = event.GetDC()
            assert isinstance(dc, wx.DC)
            # Mostly, this painting activity is for debugging, so you
            # can see how big the control is. But you need to handle
            # the event to keep the control from being painted.
            dc.SetBackgroundMode(wx.PENSTYLE_TRANSPARENT)
            dc.SetBrush(wx.TRANSPARENT_BRUSH)
            dc.SetPen(wx.TRANSPARENT_PEN)
            r = transparent_window.GetRect()
            dc.DrawRectangle(0, 0, r.Width, r.Height)
            return True

        transparent_window.Bind(wx.EVT_ERASE_BACKGROUND, on_background_paint)

        def on_fake_size(event):
            assert isinstance(event, wx.SizeEvent)
            transparent_window.SetSize(event.GetSize())
            event.Skip()

        input_list_ctrl.Bind(wx.EVT_SIZE, on_fake_size)

        def on_fake_move(event):
            assert isinstance(event, wx.MoveEvent)
            transparent_window.Move(event.GetPosition())
            event.Skip()

        input_list_ctrl.Bind(wx.EVT_MOVE, on_fake_move)
        transparent_window.SetToolTip(
            "The current pipeline is a legacy pipeline that does not use these modules"
        )
        self.transparent_window = transparent_window

    def show_input_panel(self, show):
        """Show or hide the controls for input modules

        show - True to show the controls, False to hide
        """
        self.input_list_ctrl.Enable(show)
        if not show:
            self.input_list_ctrl.DeleteAllItems()
            fake_pipeline = Pipeline()
            fake_pipeline.init_modules()
            for i, module in enumerate(fake_pipeline.modules(False)):
                item = PipelineListCtrl.PipelineListCtrlItem(module)
                item.set_state(PLV_STATE_UNAVAILABLE, PLV_STATE_UNAVAILABLE)
                self.input_list_ctrl.InsertItem(i, item)
        else:
            # remove the fake modules if present
            idx = 0
            while idx < self.input_list_ctrl.ItemCount:
                if self.input_list_ctrl.items[idx].is_unavailable():
                    self.input_list_ctrl.DeleteItem(idx)
                else:
                    idx += 1

        if self.transparent_window:
            self.transparent_window.Show(not show)

    def on_outputs_button(self, event):
        self.__frame.show_preferences(True)

    def request_validation(self, module=None):
        """Request validation of the pipeline, starting at the given module"""
        if module is None:
            if self.__pipeline is not None and len(self.__pipeline.modules()) > 0:
                module = self.__pipeline.modules()[0]
            else:
                return

        earliest_module = module.module_num
        for stale_validation_request in self.validation_requests:
            if stale_validation_request.module_num < earliest_module:
                earliest_module = stale_validation_request.module_num
            stale_validation_request.cancel()
        self.validation_requests = []
        settings_hash = self.__pipeline.settings_hash()
        for module in self.__pipeline.modules():
            if module.module_num >= earliest_module:

                def on_validate_module(
                    setting_idx,
                    message,
                    level,
                    module_num=module.module_num,
                    settings_hash=settings_hash,
                ):
                    self.on_validate_module(
                        setting_idx, message, level, module_num, settings_hash
                    )

                validation_request = ValidationRequestController(
                    self.__pipeline, module, on_validate_module
                )
                self.validation_requests.append(validation_request)
                request_module_validation(validation_request)

    def set_debug_mode(self, mode):
        if (mode is True) and (self.__pipeline is not None):
            modules = list(
                filter((lambda m: not m.is_input_module()), self.__pipeline.modules())
            )
            if len(modules) > 0:
                self.select_one_module(modules[0].module_num)
        self.list_ctrl.set_test_mode(mode)
        self.__debug_mode = mode
        self.__sizer.Layout()
        self.request_validation()

    def attach_to_pipeline(self, pipeline, controller):
        """Attach the viewer to the pipeline to allow it to listen for changes

        """
        self.__pipeline = pipeline
        self.__controller = controller
        pipeline.add_listener(self.notify)
        controller.attach_to_pipeline_list_view(self)

    def set_current_debug_module(self, module):
        assert not module.is_input_module()
        list_ctrl, index = self.get_ctrl_and_index(module)
        if list_ctrl is self.list_ctrl:
            self.list_ctrl.set_running_item(index)
        self.select_one_module(module.module_num)

    def reset_debug_module(self):
        """Set the pipeline slider to the first module to be debugged

        Skip the input modules. If there are no other modules, return None,
        otherwise return the first module
        """
        self.list_ctrl.last_running_item = 0
        for module in self.__pipeline.modules():
            if not module.is_input_module():
                self.set_current_debug_module(module)
                return module
        return None

    def get_current_debug_module(self, ignore_disabled=True):
        """Get the current debug module according to the slider"""
        index = self.list_ctrl.running_item
        if index is not None:
            return self.list_ctrl.items[index].module
        return None

    def advance_debug_module(self):
        """Move to the next debug module in the pipeline

        returns the module or None if we are at the end
        """
        index = self.list_ctrl.running_item
        while True:
            index += 1
            if index >= self.list_ctrl.GetItemCount():
                return None
            module = self.list_ctrl.items[index].module
            if module.enabled:
                break
        self.list_ctrl.set_running_item(index)
        self.set_current_debug_module(module)
        return module

    def attach_to_module_view(self, module_view):
        self.__module_view = module_view
        module_view.add_listener(self.__on_setting_changed_event)

    def notify(self, pipeline, event):
        """Pipeline event notifications come through here

        """
        if isinstance(event, PipelineLoaded):
            self.__on_pipeline_loaded(pipeline, event)
        elif isinstance(event, ModuleAdded):
            self.__on_module_added(pipeline, event)
        elif isinstance(event, ModuleMoved):
            self.__on_module_moved(pipeline, event)
        elif isinstance(event, ModuleRemoved):
            self.__on_module_removed(pipeline, event)
        elif isinstance(event, PipelineCleared):
            self.__on_pipeline_cleared(pipeline, event)
        elif isinstance(event, ModuleEdited):
            for list_ctrl in self.list_ctrl, self.input_list_ctrl:
                active_item = list_ctrl.get_active_item()
                if (
                    active_item is not None
                    and active_item.module.module_num == event.module_num
                ):
                    self.request_validation(active_item.module)
                    break
            else:
                self.select_one_module(event.module_num)
                for module in self.__pipeline.modules():
                    if (
                        module.module_num >= event.module_num
                        and not module.is_input_module()
                    ):
                        self.request_validation(module)
                        debug_module = self.get_current_debug_module()
                        if (
                            debug_module is not None
                            and debug_module.module_num > event.module_num
                        ):
                            self.set_current_debug_module(module)
                        break

        elif isinstance(event, ModuleEnabled):
            self.__on_module_enabled(event)
        elif isinstance(event, ModuleDisabled):
            self.__on_module_disabled(event)
        elif isinstance(event, ModuleShowWindow):
            self.__on_show_window(event)

    def notify_directory_change(self):
        # we can't know which modules use this information
        self.request_validation()

    def notify_has_file_list(self, has_files):
        """Tell the pipeline list view that the workspace has images

        has_files - True if there are files in the workspace file list

        We indicate that the pipeline can't proceed past "Images" if
        there are no files.
        """
        modules = self.__pipeline.modules()
        self.__has_file_list = has_files
        if len(modules) > 0 and modules[0].is_input_module():
            state = PLV_STATE_PROCEED if has_files else 0
            self.input_list_ctrl.SetItemState(0, state, PLV_STATE_PROCEED)

    def iter_list_items(self):
        """Iterate over the list items in all list controls

        yields a tuple of control, index for each item in
        the input and main list controls.
        """
        for ctrl in (self.input_list_ctrl, self.list_ctrl):
            for i in range(ctrl.GetItemCount()):
                yield ctrl, i

    def get_ctrl_and_index(self, module):
        """Given a module, return its list control and index

        module - module to find

        returns tuple of list control and index

        raises an exception if module is not in proper list control
        """
        ctrl = self.input_list_ctrl if module.is_input_module() else self.list_ctrl
        assert isinstance(ctrl, PipelineListCtrl)
        for i in range(ctrl.GetItemCount()):
            if ctrl.items[i].module is module:
                return ctrl, i
        raise IndexError(
            "The module, %s, was not found in the list control" % module.module_name
        )

    def select_one_module(self, module_num):
        """Select only the given module number in the list box"""
        for module in self.__pipeline.modules(False):
            if module.module_num == module_num:
                break
        else:
            logging.warn("Could not find module %d" % module_num)
            for ctrl, idx in self.iter_list_items():
                ctrl.Select(idx, False)
            self.__on_item_selected(None)
            return
        ctrl, idx = self.get_ctrl_and_index(module)
        ctrl.activate_item(idx, True, False)

    def select_module(self, module_num, selected=True):
        """Select the given one-based module number in the list
        This is mostly for testing
        """
        for module in self.__pipeline.modules(False):
            if module.module_num == module_num:
                break
        else:
            logging.warn("Could not find module %d" % module_num)
            return
        ctrl, idx = self.get_ctrl_and_index(module)
        ctrl.Select(idx, selected)
        self.__on_item_selected(None)

    def get_selected_modules(self):
        modules = []
        for ctrl, idx in self.iter_list_items():
            if ctrl.IsSelected(idx):
                modules.append(ctrl.items[idx].module)

        return modules

    @staticmethod
    def get_event_module(event):
        """Retrieve a module from an event's selection

        event - an event from a PipelineListCtrl with an associated selection
        """
        return event.EventObject.items[event.GetInt()].module

    def __on_slider_motion(self, event):
        """Handle EVT_PLV_SLIDER_MOTION

        event - a NotifyEvent with the selection indicating
                the new slider position. The event can be vetoed.

        """
        module = self.get_event_module(event)
        if not module.enabled:
            event.Veto()

    def __on_list_dclick(self, event):
        list_ctrl = event.GetEventObject()
        item, hit_code, subitem = list_ctrl.HitTestSubItem(event.Position)

        if (
            0 <= item < list_ctrl.ItemCount
            and (hit_code & wx.LIST_HITTEST_ONITEM)
            and subitem == MODULE_NAME_COLUMN
        ):
            module = list_ctrl.items[item].module
            w = self.find_module_figure_window(module)
            if w is not None:
                w.Raise()
                w.SetFocus()

    @staticmethod
    def find_module_figure_window(module):
        name = window_name(module)
        return find_fig(name=name)

    def __on_step_column_clicked(self, event):
        module = self.get_event_module(event)
        if (
            self.get_current_debug_module().module_num >= module.module_num
            and module.enabled
        ):
            mod_evt = self.list_ctrl.make_event(
                EVT_PLV_VALID_STEP_COLUMN_CLICKED, index=None, module=module
            )
            self.list_ctrl.GetEventHandler().ProcessEvent(mod_evt)
        self.list_ctrl.Refresh(eraseBackground=False)

    def __on_pause_column_clicked(self, event):
        module = self.get_event_module(event)
        module.wants_pause = not module.wants_pause
        self.list_ctrl.Refresh(eraseBackground=False)

    def __on_eye_column_clicked(self, event):
        module = self.get_event_module(event)
        self.__pipeline.show_module_window(module, not module.show_window)

    def __on_show_window(self, event):
        """Handle a ModuleShowWindow pipeline event"""
        self.list_ctrl.Refresh(eraseBackground=False)
        figure = self.find_module_figure_window(event.module)
        if figure is not None:
            figure.Close()

    def __on_error_column_clicked(self, event):
        module = self.get_event_module(event)
        if not module.is_input_module() and self.__allow_editing:
            if module.enabled:
                self.__pipeline.disable_module(module)
            else:
                self.__pipeline.enable_module(module)

    def __on_list_context_menu(self, event):
        if event.EventObject is not self.list_ctrl:
            return

        menu = wx.Menu()
        try:
            module = self.get_active_module()
            num_modules = len(self.get_selected_modules())
            if self.list_ctrl.active_item is not None:
                if num_modules == 1:
                    sub_menu = wx.Menu()
                    self.__controller.populate_edit_menu(sub_menu)
                    menu.AppendSubMenu(sub_menu, "&Add")
                    menu.Append(
                        ID_EDIT_DELETE,
                        "&Delete {} (#{})".format(
                            module.module_name, module.module_num
                        ),
                    )
                    menu.Append(
                        ID_EDIT_DUPLICATE,
                        "Duplicate {} (#{})".format(
                            module.module_name, module.module_num
                        ),
                    )
                    menu.Append(
                        ID_EDIT_ENABLE_MODULE,
                        "Enable {} (#{})".format(module.module_name, module.module_num),
                    )
                    menu.Append(
                        ID_HELP_MODULE,
                        "&Help for {} (#{})".format(
                            module.module_name, module.module_num
                        ),
                    )
                    if self.__debug_mode:
                        _, active_index = self.get_ctrl_and_index(module)
                        _, debug_index = self.get_ctrl_and_index(
                            self.get_current_debug_module()
                        )

                        if active_index <= debug_index and module.enabled:
                            menu.Append(
                                ID_DEBUG_RUN_FROM_THIS_MODULE,
                                "&Run from {} (#{})".format(
                                    module.module_name, module.module_num
                                ),
                            )
                            menu.Append(
                                ID_DEBUG_STEP_FROM_THIS_MODULE,
                                "&Step from {} (#{})".format(
                                    module.module_name, module.module_num
                                ),
                            )
                elif num_modules > 1:
                    # Multiple modules are selected
                    menu.Append(
                        ID_EDIT_DELETE,
                        "&Delete selected modules ({})".format(num_modules),
                    )
                    menu.Append(
                        ID_EDIT_DUPLICATE,
                        "Duplicate selected modules ({})".format(num_modules),
                    )
                    menu.Append(
                        ID_EDIT_ENABLE_MODULE,
                        "Enable selected modules ({})".format(num_modules),
                    )

            else:
                self.__controller.populate_edit_menu(menu)

            self.__frame.PopupMenu(menu)
        finally:
            menu.Destroy()

    def start_drag_operation(self, event):
        """Start dragging whatever is selected"""
        if event.EventObject is not self.list_ctrl:
            event.Veto()
            return
        modules_to_save = self.get_selected_modules()
        if len(modules_to_save) == 0:
            event.Veto()
            return
        fd = io.StringIO()
        temp_pipeline = Pipeline()
        for module in modules_to_save:
            temp_pipeline.add_module(module)
        dump(temp_pipeline, fd, save_image_plane_details=False, version=5)
        pipeline_data_object = PipelineDataObject()
        pipeline_data_object.SetData(fd.getvalue().encode())

        text_data_object = wx.TextDataObject()
        text_data_object.SetText(fd.getvalue())

        data_object = wx.DataObjectComposite()
        data_object.Add(pipeline_data_object)
        data_object.Add(text_data_object)
        drop_source = wx.DropSource(self.list_ctrl)
        drop_source.SetData(data_object)
        self.drag_underway = False
        self.drag_start = None
        self.drag_time = time.time()
        selected_module_ids = [m.id for m in self.get_selected_modules()]
        self.__pipeline.start_undoable_action()
        try:
            result = drop_source.DoDragDrop(wx.Drag_AllowMove)
            self.drag_underway = False
            if result in (wx.DragMove, wx.DragCopy):
                for identifier in selected_module_ids:
                    for module in self.__pipeline.modules(False):
                        if module.id == identifier:
                            self.__pipeline.remove_module(module.module_num)
                            break
        finally:
            self.list_ctrl.end_drop()
            self.__pipeline.stop_undoable_action("Drag and drop")

    def provide_drag_feedback(self, x, y, data):
        """Show the drop insert point and return True if allowed

        x, y - drag point in window coordinates
        data - drop object

        return True if allowed
        """
        index = self.where_to_drop(x, y)
        if self.allow_drag(x, y):
            self.list_ctrl.update_drop_insert_point(index)
            return True
        return False

    def allow_drag(self, x, y):
        """Return True if dragging is allowed

        If drag is within-window, three seconds must have
        elapsed since dragging or the user must move
        the cursor 10 pixels. Drop at drag site is not
        allowed.
        """
        if self.drag_start is None:
            self.drag_start = (x, y)
            return False
        if not self.__allow_editing:
            return False
        index = self.where_to_drop(x, y)
        if index is None:
            self.list_ctrl.end_drop()
            return False
        if not self.drag_underway:
            # Decide to commit to dragging
            now = time.time()
            while self.drag_time is not None and self.drag_start is not None:
                if now - self.drag_time > 3:
                    break
                distance = math.sqrt(
                    (x - self.drag_start[0]) ** 2 + (y - self.drag_start[1]) ** 2
                )
                if distance > 10:
                    break
                return False
            self.drag_underway = True
        if self.drag_start is not None:
            start_index = self.list_ctrl.HitTest(self.drag_start)
            if start_index[0] == index:
                return False
        return True

    def on_drop(self, x, y):
        if not self.allow_drag(x, y):
            return False
        return True

    def get_input_item_count(self):
        """Return the number of input items in the pipeline"""
        n_input_items = 0
        for module in self.__pipeline.modules(False):
            if module.is_input_module():
                n_input_items += 1
        return n_input_items

    def where_to_drop(self, x, y):
        pt = wx.Point(x, y)
        index, code = self.list_ctrl.HitTest(pt)
        if code & wx.LIST_HITTEST_ONITEM:
            return self.list_ctrl.get_insert_index(pt)
        elif code & wx.LIST_HITTEST_BELOW:
            return self.list_ctrl.GetItemCount()
        return None

    def on_data(self, x, y, action, data):
        index = self.where_to_drop(x, y)
        if index is not None:
            self.do_drop(index, action, data)

    def on_filelist_data(self, x, y, action, filenames):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if len(ext) > 1 and ext[1:] in EXT_PROJECT_CHOICES:
                self.__frame.Raise()
                if (
                    wx.MessageBox(
                        "Do you want to load the project, " "%s" "?" % filename,
                        caption="Load project",
                        style=wx.YES_NO | wx.ICON_QUESTION,
                        parent=self.__frame,
                    )
                    == wx.YES
                ):
                    self.__frame.pipeline_controller.do_open_workspace(filename)
                    break
            elif len(ext) > 1 and ext[1:] in EXT_PIPELINE_CHOICES:
                self.__frame.Raise()
                if (
                    wx.MessageBox(
                        "Do you want to import the pipeline, " "%s" "?" % filename,
                        caption="Load pipeline",
                        style=wx.YES_NO | wx.ICON_QUESTION,
                        parent=self.__frame,
                    )
                    == wx.YES
                ):
                    self.__frame.pipeline_controller.do_load_pipeline(filename)
                    break

    def do_drop(self, index, action, data):
        #
        # Insert the new modules
        #
        wx.BeginBusyCursor()
        try:
            pipeline = Pipeline()
            pipeline.load(io.StringIO(data))
            n_input_modules = self.get_input_item_count()
            for i, module in enumerate(pipeline.modules(False)):
                module.module_num = i + index + n_input_modules + 1
                self.__pipeline.add_module(module)
            for i in range(len(pipeline.modules(False))):
                self.list_ctrl.SetItemState(
                    i + index, wx.LIST_STATE_SELECTED, wx.LIST_STATE_SELECTED
                )
        finally:
            wx.EndBusyCursor()

    def __on_pipeline_loaded(self, pipeline, event):
        """Repopulate the list view after the pipeline loads

        """
        self.resetItems(pipeline)
        self.request_validation()
        if len(self.__pipeline.modules()) > 0:
            self.select_one_module(1)
            self.__frame.show_module_ui(True)

    def resetItems(self, pipeline):
        """Reset the list view and repopulate the list items"""
        self.list_ctrl.DeleteAllItems()
        self.input_list_ctrl.DeleteAllItems()
        assert isinstance(pipeline, Pipeline)

        for module in pipeline.modules(False):
            self.__populate_row(module)
        self.__adjust_rows()
        self.__controller.enable_module_controls_panel_buttons()

    def __adjust_rows(self):
        """Adjust slider and dimensions after adding or removing rows"""
        if self.__pipeline is not None:
            for module in self.__pipeline.modules(False):
                if module.is_input_module():
                    self.show_input_panel(True)
                    break
            else:
                self.show_input_panel(False)
        self.__panel.Layout()

    def __populate_row(self, module):
        """Populate a row in the grid with a module."""
        list_ctrl = self.input_list_ctrl if module.is_input_module() else self.list_ctrl
        for row in range(list_ctrl.GetItemCount()):
            other_module = list_ctrl.items[row].module
            if other_module.module_num > module.module_num:
                break
        else:
            row = list_ctrl.GetItemCount()
        item = PipelineListCtrl.PipelineListCtrlItem(module)
        if module.is_input_module():
            self.input_list_ctrl.InsertItem(row, item)
        else:
            self.list_ctrl.InsertItem(row, item)

    def __on_pipeline_cleared(self, pipeline, event):
        self.resetItems(pipeline)
        self.request_validation()
        if len(self.__pipeline.modules()) > 0:
            self.select_one_module(1)
            self.__frame.show_module_ui(True)
        self.notify_has_file_list(self.__has_file_list)

    def __on_module_added(self, pipeline, event):
        module = pipeline.modules(False)[event.module_num - 1]
        self.__populate_row(module)
        self.__adjust_rows()
        if len(self.get_selected_modules()) <= 1:
            self.select_one_module(event.module_num)
        self.request_validation(module)
        self.notify_has_file_list(self.__has_file_list)

    def __on_module_removed(self, pipeline, event):
        pipeline_module_ids = [module.id for module in pipeline.modules(False)]
        for list_ctrl in (self.list_ctrl, self.input_list_ctrl):
            idx = 0
            while idx < list_ctrl.GetItemCount():
                if list_ctrl.items[idx].module.id in pipeline_module_ids:
                    idx += 1
                else:
                    list_ctrl.DeleteItem(idx)
        self.__adjust_rows()
        self.__controller.enable_module_controls_panel_buttons()
        self.request_validation()

    def __on_module_moved(self, pipeline, event):
        module = pipeline.modules(False)[event.module_num - 1]
        list_ctrl, index = self.get_ctrl_and_index(module)
        if event.direction == DIRECTION_UP:
            # if this module was moved up, the one before it was moved down
            # and is now after
            other_module = pipeline.modules(False)[event.module_num]
            modules = [module, other_module]
        else:
            # if this module was moved down, the one after it was moved up
            # and is now before
            other_module = pipeline.modules(False)[event.module_num - 2]
            modules = [other_module, module]

        self.request_validation(modules[0])
        other_list_ctrl, other_index = self.get_ctrl_and_index(other_module)
        if other_list_ctrl != list_ctrl:
            # in different list controls, nothing changes
            return
        if list_ctrl.active_item == index:
            new_active_item = other_index
        elif list_ctrl.active_item == other_index:
            new_active_item = index
        else:
            new_active_item = None
        temp = list_ctrl.items[index]
        list_ctrl.items[index] = list_ctrl.items[other_index]
        list_ctrl.items[other_index] = temp
        self.__adjust_rows()
        if new_active_item is not None:
            list_ctrl.activate_item(new_active_item, False, True)
        self.__controller.enable_module_controls_panel_buttons()
        list_ctrl.SetFocus()

    def __on_module_enabled(self, event):
        self.refresh_module_display(event.module)
        self.request_validation(event.module)
        self.notify_has_file_list(self.__has_file_list)

    def refresh_module_display(self, module):
        """Refresh the display of a module"""
        list_ctrl, index = self.get_ctrl_and_index(module)
        list_ctrl.Refresh(eraseBackground=False)

    def get_active_module(self):
        """Return the module that's currently active"""
        for lc in (self.input_list_ctrl, self.list_ctrl):
            if lc.active_item is not None:
                return lc.items[lc.active_item].module
        return None

    def __on_module_disabled(self, event):
        if event.module == self.get_current_debug_module(False):
            # Must change the current debug module to something enabled
            for module in self.__pipeline.modules():
                if module.module_num > event.module.module_num:
                    self.set_current_debug_module(module)
                    break
            else:
                for module in reversed(self.__pipeline.modules()):
                    if (
                        module.module_num < event.module.module_num
                        and not module.is_input_module()
                    ):
                        self.set_current_debug_module(module)
                        break
                else:
                    self.__controller.stop_debugging()
        self.refresh_module_display(event.module)
        self.request_validation()
        self.notify_has_file_list(self.__has_file_list)

    def __on_item_selected(self, event):
        self.__controller.enable_module_controls_panel_buttons()

    def __on_item_activated(self, event):
        if self.__module_view:
            module = self.get_event_module(event)
            self.__module_view.set_selection(module.module_num)
            if event.EventObject is self.list_ctrl:
                self.input_list_ctrl.deactivate_active_item()
            else:
                self.list_ctrl.deactivate_active_item()
                for index in range(self.list_ctrl.GetItemCount()):
                    if self.list_ctrl.IsSelected(index):
                        self.list_ctrl.Select(index, False)
        self.__controller.enable_module_controls_panel_buttons()

    def __on_item_deselected(self, event):
        self.__controller.enable_module_controls_panel_buttons()

    def __on_setting_changed_event(self, caller, event):
        """Handle a setting change

        The debugging viewer needs to rewind to rerun a module after a change
        """
        setting = event.get_setting()
        module = event.get_module()
        list_ctrl, index = self.get_ctrl_and_index(module)
        if self.list_ctrl.running_item is not None:
            if not module.is_input_module() and self.list_ctrl.running_item > index:
                self.list_ctrl.set_running_item(index)

    def on_stop_debugging(self):
        self.list_ctrl.set_test_mode(False)

    def on_validate_module(
        self, setting_idx, message, level, module_num, settings_hash
    ):
        if settings_hash != self.__pipeline.settings_hash():
            return

        modules = [x for x in self.__pipeline.modules() if x.module_num == module_num]
        if len(modules) == 0:
            return
        module = modules[0]
        list_ctrl, index = self.get_ctrl_and_index(module)
        if level == logging.WARNING:
            flags = PLV_STATE_WARNING
        elif level == logging.ERROR:
            flags = PLV_STATE_ERROR
        else:
            flags = 0
        list_ctrl.SetItemErrorToolTipString(index, message)
        list_ctrl.SetItemState(index, flags, PLV_STATE_ERROR_MASK)
