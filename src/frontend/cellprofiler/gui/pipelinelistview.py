# coding=utf-8
"""PipelineListView.py
"""

import io
import logging
import math
import os
import sys
import time

import wx
from cellprofiler_core.constants.pipeline import DIRECTION_UP
from cellprofiler_core.pipeline import (
    ModuleShowWindow,
    ModuleDisabled,
    ModuleEnabled,
    PipelineLoaded,
    ModuleAdded,
    ModuleMoved,
    ModuleRemoved,
    PipelineCleared,
    ModuleEdited,
    dump,
    Pipeline,
)
from cellprofiler_core.preferences import EXT_PROJECT_CHOICES, EXT_PIPELINE_CHOICES
from cellprofiler_core.setting.subscriber import Subscriber
from cellprofiler_core.setting.text import Name
from cellprofiler_core.setting.subscriber import ImageListSubscriber
from cellprofiler_core.setting.subscriber import LabelListSubscriber
from cellprofiler_core.setting import Measurement

import cellprofiler.gui
import cellprofiler.gui.figure
import cellprofiler.gui.module_view._validation_request_controller
import cellprofiler.gui.moduleview
import cellprofiler.gui.pipeline
import cellprofiler.gui.utilities.module_view
import cellprofiler.icons


LOGGER = logging.getLogger(__name__)

IMG_OK = cellprofiler.icons.get_builtin_image("check")
IMG_ERROR = cellprofiler.icons.get_builtin_image("remove-sign")
IMG_EYE = cellprofiler.icons.get_builtin_image("eye-open")
IMG_CLOSED_EYE = cellprofiler.icons.get_builtin_image("eye-close")
IMG_STEP = cellprofiler.icons.get_builtin_image("IMG_ANALYZE_16")
IMG_STEPPED = cellprofiler.icons.get_builtin_image("IMG_ANALYZED")
IMG_PAUSE = cellprofiler.icons.get_builtin_image("IMG_PAUSE")
IMG_GO = cellprofiler.icons.get_builtin_image("IMG_GO_DIM")
IMG_INPUT = cellprofiler.icons.get_builtin_image("IMG_USE_INPUT")
IMG_OUTPUT = cellprofiler.icons.get_builtin_image("IMG_USE_OUTPUT")
IMG_SOURCE = cellprofiler.icons.get_builtin_image("IMG_USE_SOURCE")
IMG_DISABLED = cellprofiler.icons.get_builtin_image("unchecked")
IMG_UNAVAILABLE = cellprofiler.icons.get_builtin_image("IMG_UNAVAILABLE")
IMG_SLIDER = cellprofiler.icons.get_builtin_image("IMG_SLIDER")
IMG_SLIDER_ACTIVE = cellprofiler.icons.get_builtin_image("IMG_SLIDER_ACTIVE")
IMG_DOWNARROW = cellprofiler.icons.get_builtin_image("downarrow")
BMP_WARNING = wx.ArtProvider.GetBitmap(wx.ART_WARNING, size=(16, 16))

NO_PIPELINE_LOADED = "No pipeline loaded"
PADDING = 1


def plv_get_bitmap(data):
    return wx.Bitmap(data)


STEP_COLUMN = 0
PAUSE_COLUMN = 1
EYE_COLUMN = 2
ERROR_COLUMN = 3
MODULE_NAME_COLUMN = 4
NUM_COLUMNS = 5
INPUT_ERROR_COLUMN = 0
INPUT_MODULE_NAME_COLUMN = 1
NUM_INPUT_COLUMNS = 2

ERROR = "error"
WARNING = "warning"
OK = "ok"
DISABLED = "disabled"
EYE = "eye-open"
CLOSED_EYE = "eye-close"
PAUSE = "pause"
GO = "go"
NOTDEBUG = "notdebug"
UNAVAILABLE = "unavailable"

PLV_HITTEST_SLIDER = 4096
"""Module has an associated warning"""
PLV_STATE_WARNING = 1024
"""Module has an associated error"""
PLV_STATE_ERROR = 2048
"""Mask of warning / error bits"""
PLV_STATE_ERROR_MASK = PLV_STATE_ERROR + PLV_STATE_WARNING
"""Bit is set if the module is unavailable = not in pipeline"""
PLV_STATE_UNAVAILABLE = 4096
"""Bit is clear if the pipeline can't proceeed past this module"""
PLV_STATE_PROCEED = 8192
"""Report that the slider has moved"""
EVT_PLV_SLIDER_MOTION = wx.PyEventBinder(wx.NewEventType())
EVT_PLV_STEP_COLUMN_CLICKED = wx.PyEventBinder(wx.NewEventType())
EVT_PLV_PAUSE_COLUMN_CLICKED = wx.PyEventBinder(wx.NewEventType())
EVT_PLV_EYE_COLUMN_CLICKED = wx.PyEventBinder(wx.NewEventType())
EVT_PLV_ERROR_COLUMN_CLICKED = wx.PyEventBinder(wx.NewEventType())
EVT_PLV_VALID_STEP_COLUMN_CLICKED = wx.PyEventBinder(wx.NewEventType())

############################
#
# Image index dictionary - image names -> indexes
#
############################
image_index_dictionary = {}


def get_image_index(name):
    """Return the index of an image in the image list"""
    global image_index_dictionary
    if name not in image_index_dictionary:
        image_index_dictionary[name] = len(image_index_dictionary)
    return image_index_dictionary[name]


CHECK_TIMEOUT_SEC = 2
CHECK_FAIL_SEC = 20


class PipelineListView(object):
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
            self.outputs_panel, label="Output Settings", style=wx.BU_EXACTFIT
        )
        self.wsv_button = wx.Button(
            self.outputs_panel, label="View Workspace", style=wx.BU_EXACTFIT
        )
        self.outputs_panel.GetSizer().AddStretchSpacer(1)
        self.outputs_panel.GetSizer().Add(self.outputs_button, 0, wx.ALL, 2)
        self.outputs_panel.GetSizer().Add(self.wsv_button, 0, wx.ALL, 2)
        self.outputs_panel.GetSizer().AddStretchSpacer(1)
        self.outputs_button.Bind(wx.EVT_BUTTON, self.on_outputs_button)
        self.wsv_button.Bind(wx.EVT_BUTTON, self.on_wsv_button)
        self.wsv_button.Enable(False)
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
        from cellprofiler.gui.cpframe import ID_EDIT_DELETE

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
            fake_pipeline = cellprofiler.gui.pipeline.Pipeline()
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

    def on_wsv_button(self, event):
        self.__frame.pipeline_controller.on_view_workspace(event)

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

                validation_request = cellprofiler.gui.module_view._validation_request_controller.ValidationRequestController(
                    self.__pipeline, module, on_validate_module
                )
                self.validation_requests.append(validation_request)
                cellprofiler.gui.utilities.module_view.request_module_validation(
                    validation_request
                )

    def set_debug_mode(self, mode):
        if (mode is True) and (self.__pipeline is not None):
            modules = list(
                filter((lambda m: not m.is_input_module()), self.__pipeline.modules())
            )
            if len(modules) > 0:
                self.select_one_module(modules[0].module_num)
        self.list_ctrl.set_test_mode(mode)
        self.wsv_button.Enable(mode)
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
        self.list_ctrl.show_io_trace = False
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
            LOGGER.warning("Could not find module %d" % module_num)
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
            LOGGER.warning("Could not find module %d" % module_num)
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
        if item is None:
            # Open the add modules window
            self.__frame.pipeline_controller.open_add_modules()
            return
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
        from ..gui.utilities.figure import find_fig
        from ..gui.utilities.figure import window_name

        name = window_name(module)
        return find_fig(name=name)

    def __on_step_column_clicked(self, event):
        module = self.get_event_module(event)
        if not self.list_ctrl.test_mode:
            return
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
        from cellprofiler.gui.cpframe import (
            ID_EDIT_DELETE,
            ID_EDIT_DUPLICATE,
            ID_HELP_MODULE,
            ID_EDIT_ENABLE_MODULE,
            ID_EDIT_DISPLAY_MODULE,
            ID_DEBUG_RUN_FROM_THIS_MODULE,
            ID_DEBUG_STEP_FROM_THIS_MODULE,
            ID_FIND_USAGES,
        )

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
                        "Disable {} (#{})".format(module.module_name, module.module_num),
                    )
                    menu.Append(
                        ID_EDIT_DISPLAY_MODULE,
                        "Disable Display of {} (#{})".format(module.module_name, module.module_num),
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
                    elif module.enabled:
                        menu.Append(
                            ID_FIND_USAGES,
                            f"Trace inputs/outputs for {module.module_name} (#{module.module_num})",
                        )

                        if module.enabled:
                            menu.Bind(
                                wx.EVT_MENU,
                                self.on_io_trace,
                                id=cellprofiler.gui.cpframe.ID_FIND_USAGES,
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
                        "Disable selected modules ({})".format(num_modules),
                    )
                    menu.Append(
                        ID_EDIT_DISPLAY_MODULE,
                        "Disable Display of selected modules ({})".format(num_modules),
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
            result = drop_source.DoDragDrop(wx.Drag_DefaultMove)
            self.drag_underway = False
            if result == wx.DragMove:
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
            pipeline = cellprofiler.gui.pipeline.Pipeline()
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
        assert isinstance(pipeline, cellprofiler.gui.pipeline.Pipeline)

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

    def on_io_trace(self, event):
        self.list_ctrl.show_io_trace = False
        self.list_ctrl.Refresh(eraseBackground=False)

        # Get the selected module
        tgt_module = self.get_selected_modules()
        if len(tgt_module) != 1:
            return
        else:
            tgt_module = tgt_module[0]
        inputs, outputs, measures_in, measures_out = self._get_inputs_outputs(tgt_module)
        need_measures_out = len(measures_in) > 0
        after_tgt = False
        for module in self.__pipeline.modules(exclude_disabled=False):
            module.io_status = None
            if not module.enabled:
                continue
            if module == tgt_module:
                after_tgt = True
                module.io_status = "source"
                continue
            mod_inputs, mod_outputs, mod_m_in, mod_m_out = self._get_inputs_outputs(module, need_measures_out)
            if not after_tgt and "Export" in tgt_module.module_name and "Measure" in module.module_name:
                module.io_status = "output"
            elif after_tgt and "Export" in module.module_name and len(measures_out) > 0:
                module.io_status = "input"
            elif not inputs.isdisjoint(mod_outputs):
                module.io_status = "output"
            elif not outputs.isdisjoint(mod_inputs):
                module.io_status = "input"
            elif not after_tgt and not measures_in.isdisjoint(mod_m_out):
                module.io_status = "output"
            elif after_tgt and not measures_out.isdisjoint(mod_m_in):
                module.io_status = "input"
        self.list_ctrl.show_io_trace = True
        self.list_ctrl.Refresh(eraseBackground=False)

    def _get_inputs_outputs(self, module, need_measure_outputs=True):
        inputs = []
        outputs = []
        measures_in = []
        for setting in module.visible_settings():
            if isinstance(setting, (ImageListSubscriber, LabelListSubscriber)):
                inputs += setting.value
            elif isinstance(setting, Subscriber):
                inputs.append(setting.value)
            elif isinstance(setting, Name):
                outputs.append(setting.value)
            elif isinstance(setting, Measurement):
                measures_in.append(setting.value)
        if need_measure_outputs:
            measures_out = set([measure[1] for measure in module.get_measurement_columns(self.__pipeline)])
        else:
            measures_out = set()
        return set(inputs), set(outputs), set(measures_in), measures_out

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


PIPELINE_DATA_FORMAT = "application.cellprofiler.pipeline"


class PipelineDataObject(wx.CustomDataObject):
    def __init__(self):
        super(PipelineDataObject, self).__init__(wx.DataFormat(PIPELINE_DATA_FORMAT))


class PipelineDropTarget(wx.DropTarget):
    def __init__(self, window):
        super(PipelineDropTarget, self).__init__()
        self.window = window
        self.data_object = wx.DataObjectComposite()
        self.pipeline_data_object = PipelineDataObject()
        self.file_data_object = wx.FileDataObject()
        self.data_object.Add(self.pipeline_data_object)
        self.data_object.Add(self.file_data_object)
        self.SetDataObject(self.data_object)

    def OnDragOver(self, x, y, data):
        if not self.window.provide_drag_feedback(x, y, data):
            return wx.DragNone
        if wx.GetKeyState(wx.WXK_CONTROL) == 0:
            return wx.DragMove
        return wx.DragCopy

    def OnDrop(self, x, y):
        return self.window.on_drop(x, y)

    def OnData(self, x, y, action):
        if self.GetData():
            if (
                self.data_object.GetReceivedFormat().GetType()
                == self.pipeline_data_object.GetFormat().GetType()
            ):
                pipeline_data = self.pipeline_data_object.GetData().tobytes().decode()
                if pipeline_data is not None:
                    self.window.on_data(x, y, action, pipeline_data)
            elif self.data_object.GetReceivedFormat().GetType() == wx.DF_FILENAME:
                self.window.on_filelist_data(
                    x, y, action, self.file_data_object.GetFilenames()
                )
        if action == 1:
            # Bug in wx 4.1 returns the wrong action ID on Windows. Get the right one.
            action = self.OnDragOver(x, y, None)
        return action


class PipelineListCtrl(wx.ScrolledWindow):
    """A custom widget for the pipeline module list"""

    class PipelineListCtrlItem(object):
        """An item in a pipeline list control"""

        def __init__(self, module):
            self.module = module
            self.__state = PLV_STATE_PROCEED
            self.tooltip = ""

        @property
        def module_name(self):
            """The module name of the item's module"""
            return self.module.module_name

        def set_state(self, state, state_mask):
            """Set the item's state

            state - the state bit values to set

            state_mask - the mask indicating which state bits to set.
            """
            self.__state = (self.__state & ~state_mask) | (state & state_mask)

        def is_selected(self):
            """Return True if item is selected"""
            return bool(self.__state & wx.LIST_STATE_SELECTED)

        selected = property(is_selected)

        def select(self, value=True):
            self.set_state(
                wx.LIST_STATE_SELECTED if value else 0, wx.LIST_STATE_SELECTED
            )

        def get_error_state(self):
            """Return the error state: ERROR, WARNING or OK"""
            if self.__state & PLV_STATE_ERROR:
                return ERROR
            if self.__state & PLV_STATE_WARNING:
                return WARNING
            return OK

        def can_proceed(self):
            """Return True if the pipeline can proceed past this module

            This is the state of the PLV_STATE_PROCEED flag. The pipeline
            might not be able to proceed because of an error or warning as
            well.
            """
            return (self.__state & PLV_STATE_PROCEED) == PLV_STATE_PROCEED

        error_state = property(get_error_state)

        def is_shown(self):
            """The module's frame should be shown if True"""
            return self.module.show_window

        def is_enabled(self):
            """The module should not be executed"""
            return self.module.enabled

        enabled = property(is_enabled)

        def is_paused(self):
            """The module is breakpointed in test mode"""
            return self.module.wants_pause

        def get_io_status(self):
            """The module is breakpointed in test mode"""
            if hasattr(self.module, "io_status"):
                return self.module.io_status
            else:
                return None

        def is_unavailable(self):
            """The module is unavailable = not in pipeline"""
            return bool(self.__state & PLV_STATE_UNAVAILABLE)

    def __init__(self, *args, **kwargs):
        super(PipelineListCtrl, self).__init__(*args, **kwargs)
        self.bmp_ok = plv_get_bitmap(IMG_OK)
        self.bmp_error = plv_get_bitmap(IMG_ERROR)
        self.bmp_eye = plv_get_bitmap(IMG_EYE)
        self.bmp_closed_eye = plv_get_bitmap(IMG_CLOSED_EYE)
        self.bmp_step = plv_get_bitmap(IMG_STEP)
        self.bmp_stepped = plv_get_bitmap(IMG_STEPPED)
        self.bmp_go = plv_get_bitmap(IMG_GO)
        self.bmp_pause = plv_get_bitmap(IMG_PAUSE)
        self.bmp_input = plv_get_bitmap(IMG_INPUT)
        self.bmp_output = plv_get_bitmap(IMG_OUTPUT)
        self.bmp_source = plv_get_bitmap(IMG_SOURCE)
        self.bmp_unavailable = plv_get_bitmap(IMG_UNAVAILABLE)
        self.bmp_disabled = plv_get_bitmap(IMG_DISABLED)
        self.bmp_slider = plv_get_bitmap(IMG_SLIDER)
        self.bmp_slider_active = plv_get_bitmap(IMG_SLIDER_ACTIVE)
        self.bmp_downarrow = plv_get_bitmap(IMG_DOWNARROW)
        # The items to display
        self.items = []
        # The current or active item (has wx.CONTROL_CURRENT display flag)
        self.active_item = None
        # The anchor of an extended selection
        self.anchor = None
        # The item highlighted by the test mode slider
        self.running_item = 0
        # The last module in the pipeline that has been run on this image set
        self.last_running_item = 0
        # True if in test mode
        self.test_mode = False
        # True if allowed to disable a module
        self.allow_disable = True
        # True to show the step column, false to hide
        self.show_step = True
        # True to show the pause column, false to hide
        self.show_go_pause = True
        # True to show inheritence icons, false to hide
        self.show_io_trace = False
        # True to show the eyeball column, false to hide it
        self.show_show_frame_column = True
        # Space reserved for test mode slider
        self.slider_width = 10
        # The index at which a drop would take place, if dropping.
        self.drop_insert_point = None
        # Shading border around buttons
        self.border = 1
        # Gap between buttons
        self.gap = 1
        # Gap before and after text
        self.text_gap = 3
        # The height of one row in the display
        self.line_height = max(
            16 + self.border + self.gap, self.GetFullTextExtent("M")[1]
        )
        # The width of a icon column
        self.column_width = self.line_height
        # The row # of the currently pressed icon button
        self.pressed_row = None
        # The column of the currently pressed icon button
        self.pressed_column = None
        # True if the currently pressed button will fire when the mouse button
        # is released.
        self.button_is_active = False
        # True if the slider is currently being slid by a mouse capture action
        self.active_slider = False
        # True to draw the current item as if it were selected, even when
        # it's not.
        self.always_draw_current_as_if_selected = False
        # A pen to use to draw shadow edges on buttons
        self.shadow_pen = wx.Pen(wx.SystemSettings.GetColour(wx.SYS_COLOUR_3DSHADOW))
        # A pen to use to draw lighted edges on buttons
        self.light_pen = wx.Pen(wx.SystemSettings.GetColour(wx.SYS_COLOUR_3DLIGHT))
        self.SetScrollRate(self.line_height, self.line_height)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
        self.Bind(wx.EVT_LEFT_UP, self.on_left_up)
        self.Bind(wx.EVT_MOTION, self.on_mouse_move)
        self.Bind(wx.EVT_KEY_DOWN, self.on_key_down)
        self.Bind(wx.EVT_MOUSE_CAPTURE_LOST, self.on_capture_lost)
        self.Bind(wx.EVT_KILL_FOCUS, self.on_focus_change)
        self.Bind(wx.EVT_SET_FOCUS, self.on_focus_change)

    def AdjustScrollbars(self):
        self.SetScrollbars(1, self.line_height, self.BestSize[0], len(self.items))

    def on_focus_change(self, event):
        self.Refresh(eraseBackground=False)

    def HitTestSubItem(self, position):
        """Mimic ListCtrl's HitTestSubItem

        position - x, y position of mouse click

        returns item hit, hit code (same as ListCtrl's) and column #
        (None if no hit or hit the slider, or PAUSE_COLUMN, EYE_COLUMN,
         ERROR_COLUMN or MODULE_NAME_COLUMN)

         Hit codes:
         wx.LIST_HITTEST_NOWHERE - in the slider column, but not on the slider
         wx.LIST_HITTEST_BELOW - below any item (item #, column # not valid)
         wx.LIST_HITTEST_ONITEMLABEL - on the module name of the item
         wx.LIST_HITTEST_ONITEMICON - on one of the icons
         PLV_HITTEST_SLIDER - on the slider handle.
        """
        position = self.CalcUnscrolledPosition(position)
        x = position[0]
        y = position[1]
        x0 = self.slider_width
        if self.test_mode:
            if x < x0:
                r = self.get_slider_rect()
                if r.Contains(x, y):
                    return None, PLV_HITTEST_SLIDER, None
                return None, wx.LIST_HITTEST_NOWHERE, None
        elif x < x0:
            return None, wx.LIST_HITTEST_NOWHERE, None
        column = int((x - x0) / self.column_width)
        if (not (self.show_go_pause and self.test_mode) and not self.show_io_trace) and column == PAUSE_COLUMN:
            return None, wx.LIST_HITTEST_NOWHERE, None
        if not self.show_show_frame_column and column == EYE_COLUMN:
            return None, wx.LIST_HITTEST_NOWHERE, None
        row = int(y / self.line_height)
        if row >= len(self.items):
            return None, wx.LIST_HITTEST_BELOW, None
        if column < 4:
            return row, wx.LIST_HITTEST_ONITEMICON, column
        return row, wx.LIST_HITTEST_ONITEMLABEL, MODULE_NAME_COLUMN

    def HitTest(self, position):
        """Mimic ListCtrl's HitTest

        position - x, y position of mouse click

        returns hit code (see HitTestSubItem)
        """
        return self.HitTestSubItem(position)[:2]

    def GetSelectedItemCount(self):
        """Return the # of items selected"""
        return len([True for item in self.items if item.selected])

    SelectedItemCount = property(GetSelectedItemCount)

    def GetItemCount(self):
        """Return the total # of items

        """
        return len(self.items)

    ItemCount = property(GetItemCount)

    def SetItemState(self, item, state, state_mask):
        """Set the state for the given item

        item - index of item

        state - one of the list states

        """
        self.items[item].set_state(state, state_mask)
        self.Refresh(eraseBackground=False)

    def SetItemErrorToolTipString(self, item, text):
        """Set the string to display when hovering over the error indicator"""
        self.items[item].tooltip = text

    def Select(self, index, state):
        """Mark the item as selected or not

        index - index of the item to mark as selected

        state - True to be selected, False for not
        """
        self.items[index].select(state)
        if state:
            plv_event = self.make_event(wx.EVT_LIST_ITEM_SELECTED, index)
        else:
            plv_event = self.make_event(wx.EVT_LIST_ITEM_DESELECTED, index)
        self.GetEventHandler().ProcessEvent(plv_event)
        self.Refresh(eraseBackground=False)

    def CanSelect(self):
        """Return True if selection is allowed in this control"""
        return self.allow_disable

    def SelectAll(self):
        """Select all modules"""
        if self.CanSelect():
            for i in range(self.GetItemCount()):
                if not self.IsSelected(i):
                    self.Select(i, True)

    def IsSelected(self, index):
        """Return True if the item at the given index is selected"""
        return bool(self.items[index].is_selected())

    def DeleteAllItems(self):
        self.items = []
        self.active_item = None
        self.anchor = None
        self.running_item = None
        self.pressed_row = None
        self.pressed_column = None
        self.Refresh(eraseBackground=False)

    def DeleteItem(self, index):
        """Remove the item at the given index"""
        del self.items[index]

        if self.active_item is not None:
            if self.active_item == index:
                self.active_item = None
            elif self.active_item > index:
                self.active_item -= 1

        if self.anchor:
            if self.anchor == index:
                self.anchor = None
            elif self.anchor > index:
                self.anchor -= 1

        if self.running_item:
            if self.running_item > index:
                self.running_item -= 1

        self.AdjustScrollbars()

        self.Refresh(eraseBackground=False)

    def InsertItem(self, index, item):
        """Insert an item at the given index

        index - the item will appear at this slot on the list.

        item - a PipelineListCtrlItem
        """
        self.items.insert(index, item)

        if self.active_item:
            if self.active_item >= index:
                self.active_item += 1

        if self.anchor:
            if self.anchor >= index:
                self.anchor += 1

        if self.running_item:
            if self.running_item >= index:
                self.running_item += 1

        self.AdjustScrollbars()

        self.SetInitialSize(self.GetBestSize())
        self.Refresh(eraseBackground=False)

    def get_insert_index(self, position):
        """Return the index for an insert into the item list, based on position

        position - position of the mouse in the window

        returns the index that should be used when inserting a new item into
        the list.
        """
        position = self.CalcUnscrolledPosition(position)
        y = position[1]
        row = int((y + self.line_height / 2) / self.line_height)
        return max(0, min(row, self.ItemCount))

    def update_drop_insert_point(self, index):
        """Change the visual drop indication

        index - the first dropped module would end up here.
        """
        self.drop_insert_point = index
        self.Refresh(eraseBackground=False)

    def end_drop(self):
        """Remove the visual drop indication"""
        self.drop_insert_point = None
        self.Refresh(eraseBackground=False)

    def get_index(self, position):
        """Return the index of the item at the given position"""
        position = self.CalcUnscrolledPosition(position)
        y = position[1]
        row = int(y / self.line_height)
        return max(0, min(row, self.ItemCount - 1))

    def get_active_item(self):
        """Return the active PipelineListCtrlItem or None"""
        if self.active_item is not None:
            return self.items[self.active_item]
        return None

    def set_test_mode(self, mode):
        self.test_mode = mode
        self.AdjustScrollbars()
        self.Refresh(eraseBackground=False)

    def set_show_step(self, state):
        """Show or hide the go / pause test-mode icons

        state - true to show them, false to hide them
        """
        self.show_step = state
        self.AdjustScrollbars()
        self.Refresh(eraseBackground=False)

    def set_show_go_pause(self, state):
        """Show or hide the go / pause test-mode icons

        state - true to show them, false to hide them
        """
        self.show_go_pause = state
        self.AdjustScrollbars()
        self.Refresh(eraseBackground=False)

    def set_show_frame_column(self, state):
        """Show or hide the show / hide frame column

        state - True to show, False to hide
        """
        self.show_show_frame_column = state
        self.AdjustScrollbars()
        self.Refresh(eraseBackground=False)

    def set_allow_disable(self, state):
        """Allow disabling of modules / disallow disabling of modules"""
        self.allow_disable = state
        self.Refresh(eraseBackground=False)

    def set_running_item(self, idx):
        """The index of the next module to run in test mode"""
        self.running_item = idx
        if self.running_item > self.last_running_item:
            self.last_running_item = self.running_item
        self.Refresh(eraseBackground=False)

    def DoGetBestSize(self):
        x0 = self.column_width * 3 + self.slider_width + self.text_gap
        max_width = 0
        for i, item in enumerate(self.items):
            width, height, _, _ = self.GetFullTextExtent(item.module_name)
            max_width = max(width, max_width)
        total_width = x0 + max_width + self.border * 2 + self.gap + self.text_gap
        height = max((len(self.items) - 1) * self.line_height, 0)
        return total_width, height

    def DoGetVirtualSize(self):
        return self.DoGetBestSize()

    def AcceptsFocus(self):
        return True

    def get_button_rect(self, column, idx):
        x0 = column * self.column_width + self.slider_width
        if self.pressed_column == column and self.pressed_row == idx:
            x0 += 1
            y0 = 1
        else:
            y0 = 0
        return wx.Rect(x0 + self.gap, y0 + idx * self.line_height, 16, 16)

    def get_step_rect(self, idx):
        return self.get_button_rect(0, idx)

    def get_go_pause_rect(self, idx):
        return self.get_button_rect(1, idx)

    def get_eye_rect(self, idx):
        return self.get_button_rect(2, idx)

    def get_error_rect(self, idx):
        return self.get_button_rect(3, idx)

    def get_text_rect(self, index):
        x0 = self.slider_width + self.column_width * 4 + self.text_gap
        return wx.Rect(
            x0 + self.gap,
            index * self.line_height,
            self.GetSize()[0] - x0 - self.gap,
            self.line_height,
        )

    def get_slider_rect(self):
        """Return the rectangle encompassing the slider bitmap

        returns None if not displayed, otherwise the coordinates of the bitmap
        """
        if not self.test_mode:
            return None
        top = (
            self.line_height - self.bmp_slider.GetHeight()
        ) / 2 + self.line_height * self.running_item
        return wx.Rect(1, top, self.bmp_slider.GetWidth(), self.bmp_slider.GetHeight())

    def on_paint(self, event):
        assert isinstance(event, wx.PaintEvent)

        dc = wx.BufferedPaintDC(self)

        self.PrepareDC(dc)

        dc.SetBackground(wx.Brush(wx.SystemSettings.GetColour(wx.SYS_COLOUR_LISTBOX)))

        dc.Clear()

        text_color = wx.SystemSettings.GetColour(wx.SYS_COLOUR_LISTBOXTEXT)

        text_color_selected = wx.SystemSettings.GetColour(
            wx.SYS_COLOUR_LISTBOXHIGHLIGHTTEXT
        )

        if len(self.items) == 0:
            text = "Drop a pipeline file here (.cppipe or .cpproj)\n or double-click to add modules"
            dc.SetTextForeground(wx.SystemSettings.GetColour(wx.SYS_COLOUR_GRAYTEXT))
            dc.DrawLabel(
                text, wx.Bitmap(), wx.Rect(self.GetSize()), alignment=wx.ALIGN_CENTER,
            )

        for index, item in enumerate(self.items):
            item_text_color = text_color

            dc.SetFont(self.GetFont())

            if self.show_step and self.test_mode:
                rectangle = self.get_step_rect(index)
                bitmap = (
                    self.bmp_step if self.running_item == index else self.bmp_stepped
                )
                if self.running_item >= index and item.enabled:
                    dc.DrawBitmap(bitmap, rectangle.GetLeft(), rectangle.GetTop(), True)

            if self.show_go_pause and self.test_mode:
                rectangle = self.get_go_pause_rect(index)
                bitmap = self.bmp_pause if item.is_paused() else self.bmp_go

                dc.DrawBitmap(bitmap, rectangle.GetLeft(), rectangle.GetTop(), True)

            if self.show_io_trace and not self.test_mode:
                rectangle = self.get_go_pause_rect(index)
                status = item.get_io_status()
                bitmap = False
                if status == "input":
                    bitmap = self.bmp_input
                elif status == "output":
                    bitmap = self.bmp_output
                elif status == "source":
                    bitmap = self.bmp_source
                if bitmap:
                    dc.DrawBitmap(bitmap, rectangle.GetLeft(), rectangle.GetTop(), True)

            if self.show_show_frame_column:
                rectangle = self.get_eye_rect(index)

                bitmap = self.bmp_eye if item.is_shown() else self.bmp_closed_eye

                dc.DrawBitmap(bitmap, rectangle.GetLeft(), rectangle.GetTop(), True)

            rectangle = self.get_error_rect(index)

            bitmap = (
                self.bmp_unavailable
                if item.is_unavailable()
                else self.bmp_disabled
                if not item.enabled
                else self.bmp_error
                if item.error_state == ERROR
                else BMP_WARNING
                if item.error_state == WARNING
                else self.bmp_ok
            )

            dc.DrawBitmap(bitmap, rectangle.GetLeft(), rectangle.GetTop(), True)

            rectangle = self.get_text_rect(index)

            flags = 0 if self.Enabled else wx.CONTROL_DISABLED
            font = self.Font

            if item.selected:
                flags |= wx.CONTROL_SELECTED
                item_text_color = text_color_selected
                font = font.MakeBold()

            if self.active_item == index:
                flags |= wx.CONTROL_CURRENT
                font = font.MakeBold()

                if self.always_draw_current_as_if_selected:
                    flags |= wx.CONTROL_SELECTED
                    item_text_color = text_color_selected

            if self.FindFocus() is self:
                if item.selected or self.active_item == index:
                    flags |= wx.CONTROL_FOCUSED
            else:
                # On Windows, the highlight color is white. If focus is lost, the background color is light grey.
                # These colors together makes the font very difficult to read. The default text color is dark. Let's
                # use it instead.
                item_text_color = text_color

            dc.SetFont(font)
            cellprofiler.gui.draw_item_selection_rect(self, dc, rectangle, flags)

            if self.test_mode and self.running_item == index:
                dc.SetFont(font.MakeUnderlined())

            dc.SetBackgroundMode(wx.PENSTYLE_TRANSPARENT)

            dc.SetTextForeground(item_text_color)

            dc.DrawText(
                item.module_name,
                rectangle.GetLeft() + self.text_gap,
                rectangle.GetTop(),
            )

        if self.drop_insert_point is not None:
            y = self.line_height * self.drop_insert_point

            dc.SetPen(wx.BLACK_PEN)

            dc.DrawLine(0, y, self.GetSize()[0], y)

    def make_event(self, py_event_binder, index=None, module=None):
        event = wx.NotifyEvent(py_event_binder.evtType[0])
        event.SetEventObject(self)
        if index is not None:
            event.SetInt(index)
        if module is not None:
            event.module = module
        return event

    def on_left_down(self, event):
        assert isinstance(event, wx.MouseEvent)
        self.SetFocus()
        index, hit_test, column = self.HitTestSubItem(event.GetPosition())
        if hit_test == PLV_HITTEST_SLIDER:
            self.active_slider = True
            self.CaptureMouse()
            self.RefreshRect(self.get_slider_rect())
        elif hit_test & wx.LIST_HITTEST_ONITEMLABEL:
            item_id = None if index >= len(self.items) else id(self.items[index])
            if event.ShiftDown() and self.active_item is not None and self.CanSelect():
                # Extend the selection
                begin = min(self.active_item, index)
                end = max(self.active_item, index) + 1
                for i in range(begin, end):
                    self.Select(i, True)
                toggle_selection = False
                multiple_selection = True
                activate_before_drag = True
            else:
                activate_before_drag = not self.IsSelected(index)
                toggle_selection = True
                multiple_selection = event.ControlDown()
            if activate_before_drag:
                self.activate_item(index, toggle_selection, multiple_selection)
            plv_event = self.make_event(wx.EVT_LIST_BEGIN_DRAG, index)
            self.GetEventHandler().ProcessEvent(plv_event)
            if not activate_before_drag:
                new_item_id = (
                    None if index >= len(self.items) else id(self.items[index])
                )
                if new_item_id == item_id:
                    self.activate_item(index, toggle_selection, multiple_selection)
        elif hit_test & wx.LIST_HITTEST_ONITEMICON:
            if column == PAUSE_COLUMN and not self.test_mode:
                return
            elif column != ERROR_COLUMN or self.CanSelect():
                self.pressed_row = index
                self.pressed_column = column
                self.button_is_active = True
                self.Refresh(eraseBackground=False)

    def deactivate_active_item(self):
        """Remove the activation UI indication from the current active item"""
        self.active_item = None
        self.Refresh(eraseBackground=False)

    def activate_item(
        self, index, toggle_selection, multiple_selection, anchoring=True
    ):
        """Move the active item

        index - index of item to activate

        toggle_selection - true to toggle the selection state, false to
                           always select.

        multiple_selection - true to allow multiple selections, false
                           to deselect all but the activated.

        anchoring - true to place the anchor for extended selection
                    false if the anchor should be kept where it is

        returns the selection state
        """
        self.active_item = index
        plv_event = self.make_event(wx.EVT_LIST_ITEM_ACTIVATED, index)
        self.GetEventHandler().ProcessEvent(plv_event)
        if self.allow_disable:
            if self.IsSelected(index) and toggle_selection:
                self.Select(index, False)
                self.Refresh(eraseBackground=False)
                return False
            if not multiple_selection:
                for i, item in enumerate(self.items):
                    if self.IsSelected(i):
                        self.Select(i, False)
            self.Select(index, True)
            if anchoring:
                self.anchor = index
        window_height = int(self.GetSize()[1] / self.line_height)
        #
        # Always keep the active item in view
        #
        sx = self.GetScrollPos(wx.HORIZONTAL)
        sy = self.GetScrollPos(wx.VERTICAL)
        if index < sy:
            self.Scroll(sx, index)
        elif index >= sy + window_height:
            self.Scroll(sx, index + window_height - 1)

        self.Refresh(eraseBackground=False)
        return True

    def on_left_up(self, event):
        if self.GetCapture() == self:
            self.ReleaseMouse()
        if self.active_slider:
            self.active_slider = False
            self.RefreshRect(self.get_slider_rect())
        if self.button_is_active and self.pressed_column is not None:
            if self.pressed_column == ERROR_COLUMN:
                code = EVT_PLV_ERROR_COLUMN_CLICKED
            elif self.pressed_column == STEP_COLUMN:
                code = EVT_PLV_STEP_COLUMN_CLICKED
            elif self.pressed_column == PAUSE_COLUMN:
                code = EVT_PLV_PAUSE_COLUMN_CLICKED
            else:
                code = EVT_PLV_EYE_COLUMN_CLICKED
            r = self.get_button_rect(self.pressed_column, self.pressed_row or 0)
            r.Inflate(2, 2)
            self.RefreshRect(r)
            self.button_is_active = False
            self.pressed_column = None
            plv_event = self.make_event(code, self.pressed_row)
            self.GetEventHandler().ProcessEvent(plv_event)

    def on_mouse_move(self, event):
        index, hit_test, column = self.HitTestSubItem(event.Position)
        if self.active_slider:
            index = self.get_index(event.Position)
            if self.running_item != index:
                plv_event = self.make_event(EVT_PLV_SLIDER_MOTION, index)
                self.GetEventHandler().ProcessEvent(plv_event)
                if plv_event.IsAllowed() and index <= self.last_running_item:
                    self.running_item = index
                    self.Refresh(eraseBackground=False)
        elif self.HasCapture() and self.pressed_column is not None:
            button_is_active = (
                index == self.pressed_row and column == self.pressed_column
            )
            if button_is_active != self.button_is_active:
                self.button_is_active = button_is_active
                self.Refresh(eraseBackground=False)
        else:
            tooltip_text = None
            item = (
                None
                if (index is None or index >= self.GetItemCount())
                else self.items[index]
            )
            if hit_test & wx.LIST_HITTEST_ONITEM:
                if column == EYE_COLUMN:
                    if item.module.show_window:
                        tooltip_text = (
                            "%s will show its display. Click icon to hide display"
                            % item.module.module_name
                        )
                    else:
                        tooltip_text = (
                            "%s will not show its display. Click icon to show display"
                            % item.module.module_name
                        )
                elif column == PAUSE_COLUMN:
                    if self.show_io_trace:
                        status = item.get_io_status()
                        if status == "output":
                            tooltip_text = f"This module produces images, objects or measurements needed by the " \
                                           f"inspected module. "
                        elif status == "input":
                            tooltip_text = f"This module uses images, objects or measurements produced by the " \
                                           f"inspected module. "
                        elif status == "source":
                            tooltip_text = "This is the inspected module"
                    elif item.module.wants_pause:
                        tooltip_text = (
                            "Test mode will stop before executing %s. Click icon to change"
                            % item.module.module_name
                        )
                    else:
                        tooltip_text = (
                            "Test mode will not stop before executing %s. Click icon to change"
                            % item.module.module_name
                        )
                elif column == ERROR_COLUMN:
                    if item.get_error_state() in (WARNING, ERROR):
                        tooltip_text = item.tooltip
                    elif item.module.enabled and not item.module.is_input_module():
                        tooltip_text = (
                            "Click to disable the %s module" % item.module.module_name
                        )
                    elif not item.module.is_input_module():
                        tooltip_text = (
                            "Click to enable the %s module" % item.module.module_name
                        )
            if tooltip_text is not None:
                self.SetToolTip(tooltip_text)
            else:
                self.SetToolTip(None)

    def cancel_capture(self):
        if self.HasCapture() and self.pressed_column is not None:
            self.ReleaseMouse()
            self.pressed_column = None
            self.button_is_active = False

    def on_key_down(self, event):
        assert isinstance(event, wx.KeyEvent)
        if self.GetItemCount() > 0 and self.active_item is not None:
            multiple_selection = event.ControlDown() or event.ShiftDown()
            anchoring_selection = not event.ShiftDown()
            if (
                event.GetKeyCode() in (wx.WXK_DOWN, wx.WXK_NUMPAD_DOWN)
                and self.active_item < self.GetItemCount() - 1
            ):
                self.cancel_capture()
                # Retreating from a previous shift select
                if not anchoring_selection and self.anchor > self.active_item:
                    self.Select(self.active_item, False)
                self.activate_item(
                    self.active_item + 1,
                    False,
                    multiple_selection,
                    anchoring=anchoring_selection,
                )
            elif (
                event.GetKeyCode() in (wx.WXK_UP, wx.WXK_NUMPAD_UP)
                and self.active_item > 0
            ):
                self.cancel_capture()
                if not anchoring_selection and self.anchor < self.active_item:
                    self.Select(self.active_item, False)
                self.activate_item(
                    self.active_item - 1, False, multiple_selection, anchoring_selection
                )
            elif event.GetKeyCode() in (wx.WXK_HOME, wx.WXK_NUMPAD_HOME):
                self.cancel_capture()
                if not anchoring_selection:
                    # Extend selection to anchor / remove selection beyond anchor
                    for i in range(self.anchor + 1, self.active_item + 1):
                        self.Select(i, False)
                    for i in range(1, self.anchor):
                        self.Select(i, True)
                self.activate_item(
                    0, False, multiple_selection, anchoring=anchoring_selection
                )
            elif event.GetKeyCode() in (wx.WXK_END, wx.WXK_NUMPAD_END):
                self.cancel_capture()
                if not anchoring_selection:
                    # Extend selection from current to end / remove
                    # selection before anchor
                    for i in range(self.active_item, self.anchor):
                        self.Select(i, False)
                    for i in range(self.anchor + 1, self.GetItemCount()):
                        self.Select(i, True)
                self.activate_item(
                    self.GetItemCount() - 1,
                    False,
                    multiple_selection,
                    anchoring=anchoring_selection,
                )
            else:
                event.Skip()
        else:
            event.Skip()

    def on_capture_lost(self, event):
        self.active_slider = False
        self.button_is_active = False
        self.pressed_column = None
        self.Refresh(eraseBackground=False)
