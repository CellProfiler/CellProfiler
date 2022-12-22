import logging
import os
import stat
import sys

import matplotlib.cm
import numpy
import wx
import wx.grid
import wx.lib.colourselect
import wx.lib.resizewidget
import wx.lib.scrolledpanel
import wx.lib.mixins.gridlabelrenderer as wxglr
from cellprofiler_core.pipeline import ModuleEdited
from cellprofiler_core.pipeline import ModuleRemoved
from cellprofiler_core.pipeline import PipelineCleared
from cellprofiler_core.pipeline import PipelineLoaded
from cellprofiler_core.preferences import DEFAULT_INPUT_FOLDER_NAME, URL_FOLDER_NAME
from cellprofiler_core.preferences import DEFAULT_INPUT_SUBFOLDER_NAME
from cellprofiler_core.preferences import DEFAULT_OUTPUT_FOLDER_NAME
from cellprofiler_core.preferences import DEFAULT_OUTPUT_SUBFOLDER_NAME
from cellprofiler_core.preferences import get_background_color
from cellprofiler_core.preferences import get_default_colormap
from cellprofiler_core.preferences import get_default_image_directory
from cellprofiler_core.preferences import get_default_output_directory
from cellprofiler_core.preferences import get_error_color
from cellprofiler_core.setting import Binary, PathListDisplay, Setting
from cellprofiler_core.setting import BinaryMatrix
from cellprofiler_core.setting import Color
from cellprofiler_core.setting import Coordinates
from cellprofiler_core.setting import DataTypes
from cellprofiler_core.setting import Divider
from cellprofiler_core.setting import DoThings
from cellprofiler_core.setting import FigureSubscriber
from cellprofiler_core.setting import FileCollectionDisplay
from cellprofiler_core.setting import HTMLText
from cellprofiler_core.setting import ImagePlane
from cellprofiler_core.setting import Joiner
from cellprofiler_core.setting import Measurement
from cellprofiler_core.setting import RegexpText
from cellprofiler_core.setting import StructuringElement
from cellprofiler_core.setting import Table
from cellprofiler_core.setting import TreeChoice
from cellprofiler_core.setting import ValidationError
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.choice import Colormap
from cellprofiler_core.setting.choice import CustomChoice
from cellprofiler_core.setting.do_something import DoSomething, ImageSetDisplay
from cellprofiler_core.setting.do_something import PathListExtractButton
from cellprofiler_core.setting.do_something import PathListRefreshButton
from cellprofiler_core.setting.filter import Filter
from cellprofiler_core.setting.multichoice import MeasurementMultiChoice
from cellprofiler_core.setting.multichoice import MultiChoice
from cellprofiler_core.setting.multichoice import SubdirectoryFilter
from cellprofiler_core.setting.multichoice import SubscriberMultiChoice
from cellprofiler_core.setting.range import FloatRange
from cellprofiler_core.setting.range import IntegerOrUnboundedRange
from cellprofiler_core.setting.range import IntegerRange
from cellprofiler_core.setting.subscriber import (
    ImageListSubscriber,
    LabelSubscriber,
    GridSubscriber,
)
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.subscriber import LabelListSubscriber
from cellprofiler_core.setting.text import Directory
from cellprofiler_core.setting.text import Filename
from cellprofiler_core.setting.text import Pathname

from ._file_collection_display_controller import FileCollectionDisplayController
from ._module_sizer import ModuleSizer
from ._setting_edited_event import SettingEditedEvent
from ._table_controller import TableController
from ._validation_request_controller import ValidationRequestController
from .. import _tree_checkbox_dialog
from ..gridrenderers import RowLabelRenderer, ColLabelRenderer, CornerLabelRenderer
from .. import metadatactrl
from .. import namesubscriber
from .. import regexp_editor
from ..constants.module_view import ABSOLUTE
from ..constants.module_view import CHECK_TIMEOUT_SEC
from ..constants.module_view import EDIT_TIMEOUT_SEC
from ..constants.module_view import FROM_EDGE
from ..constants.module_view import WARNING_COLOR
from ..html.utils import rst_to_html_fragment
from ..htmldialog import HTMLDialog
from ..utilities.module_view import absrel_control_name
from ..utilities.module_view import button_control_name
from ..utilities.module_view import category_control_name
from ..utilities.module_view import category_text_control_name
from ..utilities.module_view import colorbar_ctrl_name
from ..utilities.module_view import combobox_ctrl_name
from ..utilities.module_view import custom_label_name
from ..utilities.module_view import edit_control_name
from ..utilities.module_view import encode_label
from ..utilities.module_view import feature_control_name
from ..utilities.module_view import feature_text_control_name
from ..utilities.module_view import folder_label_name
from ..utilities.module_view import grid_control_name
from ..utilities.module_view import help_ctrl_name
from ..utilities.module_view import max_control_name
from ..utilities.module_view import min_control_name
from ..utilities.module_view import object_control_name
from ..utilities.module_view import object_text_control_name
from ..utilities.module_view import request_module_validation
from ..utilities.module_view import scale_control_name
from ..utilities.module_view import scale_text_ctrl_name
from ..utilities.module_view import subedit_control_name
from ..utilities.module_view import text_control_name
from ..utilities.module_view import x_control_name
from ..utilities.module_view import y_control_name

LOGGER = logging.getLogger(__name__)


class ModuleView:
    """The module view implements a view on CellProfiler.Module

    The module view implements a view on CellProfiler.Module. The view consists
    of a table composed of one row per setting. The first column of the table
    has the explanatory text and the second has a control which
    gives the ui for editing the setting.
    """

    def __init__(self, top_panel, workspace, frame=None, notes_panel=None):
        """Constructor

        module_panel - the top-level panel used by the view
        workspace - the current workspace
        notes_panel - panel in which to construct the notes GUI
        """
        pipeline = workspace.pipeline
        self.__workspace = workspace
        self.__module = None
        self.refresh_pending = False
        self.notes_panel = notes_panel
        self.__frame = frame
        self.top_panel = top_panel
        background_color = get_background_color()
        #############################################
        #
        # Build the top-level GUI windows
        #
        #
        # Module panel structure:
        #
        # top_panel
        #   box sizer
        #     "Module settings" box
        #        static box sizer
        #          module_panel
        #            custom module sizer
        #              module setting controls
        #
        #############################################
        top_panel.Sizer = wx.BoxSizer()
        module_settings_box_sizer = wx.BoxSizer()
        top_panel.Sizer.Add(module_settings_box_sizer, 1, wx.EXPAND)
        self.__module_panel = wx.lib.scrolledpanel.ScrolledPanel(top_panel)
        self.__module_panel.SetupScrolling()
        module_settings_box_sizer.Add(self.__module_panel, 1, wx.EXPAND)
        self.__sizer = ModuleSizer(0, 3)
        self.module_panel.SetSizer(self.__sizer)
        self.module_panel.Bind(wx.EVT_CHILD_FOCUS, self.skip_event)
        if notes_panel is not None:
            self.make_notes_gui()

        self.__pipeline = pipeline
        self.__listeners = []
        self.__value_listeners = []
        self.__inside_notify = False
        self.__handle_change = True
        self.__notes_text = None
        self.__started = False
        self.__validation_request = None

    def start(self):
        """Start the module view

        Start the module view after the pipeline and workspace have been
        properly initialized.
        """
        self.__pipeline.add_listener(self.__on_pipeline_event)
        self.__workspace.add_notification_callback(self.__on_workspace_event)
        self.__started = True

    @staticmethod
    def skip_event(event):
        event.Skip(False)

    def get_module_panel(self):
        """The panel that hosts the module controls

        This is exposed for testing purposes.
        """
        return self.__module_panel

    module_panel = property(get_module_panel)

    # ~*~
    def get_current_module(self):
        return self.__module

    # ~^~

    def clear_selection(self):
        if self.__module:
            for listener in self.__value_listeners:
                listener["notifier"].remove_listener(listener["listener"])
            self.__value_listeners = []
            self.__module = None
        self.__sizer.Reset(0)
        if self.notes_panel is not None:
            self.notes_panel.Hide()

    def get_module_settings_label(self):
        if self.__module is None:
            return "Module settings"
        return "Module settings (%s #%02d)" % (
            self.__module.module_name,
            self.__module.module_num,
        )

    def hide_settings(self):
        for child in self.__module_panel.Children:
            child.Hide()

    def check_settings(self, module_name, settings):
        try:
            assert len(settings) > 0
        except:
            wx.MessageBox(
                "Module %s.visible_settings() did not return a list!\n  value: %s"
                % (module_name, settings),
                "Pipeline Error",
                wx.ICON_ERROR,
                self.__module_panel,
            )
            settings = []
        try:
            assert all([isinstance(s, Setting) for s in settings])
        except:
            wx.MessageBox(
                "Module %s.visible_settings() returned something other than a list of Settings!\n  value: %s"
                % (module_name, settings),
                "Pipeline Error",
                wx.ICON_ERROR,
                self.__module_panel,
            )
            settings = []
        return settings

    DO_FREEZE = wx.VERSION < (2, 9, 0, 0)

    def set_selection(self, module_num):
        """Initialize the controls in the view to the settings of the module"""
        if self.DO_FREEZE:
            self.module_panel.Freeze()
        self.__handle_change = False
        imageset_control = None
        path_control = None
        table_control = None
        try:
            new_module = self.__pipeline.module(module_num)
            reselecting = self.__module and self.__module.id == new_module.id
            if not reselecting:
                if self.__module is not None:
                    self.__module.on_deactivated()
                self.clear_selection()
                self.request_validation(new_module)
                try:
                    # Need to initialize some controls.
                    new_module.test_valid(self.__pipeline)
                except Exception as e:
                    LOGGER.debug(e.message)
            self.__module = new_module
            self.__controls = []
            self.__static_texts = []
            data = []
            if reselecting:
                self.hide_settings()
            else:
                self.__module.on_activated(self.__workspace)
                if self.notes_panel is not None:
                    self.notes_panel.Show(True)
            settings = self.check_settings(
                self.__module.module_name, self.__module.visible_settings()
            )
            self.__sizer.Reset(len(settings), 3, False)
            sizer = self.__sizer

            #################################
            #
            # Set the module's notes
            #
            #################################
            if self.notes_panel is not None:
                self.module_notes_control.SetValue(
                    "\n".join([str(note) for note in self.__module.notes])
                )

            #################################
            #
            # Populate the GUI elements for each of the settings
            #
            #################################
            for i, v in enumerate(settings):
                if isinstance(v, PathListDisplay):
                    path_control = v
                    self.__frame.pipeline_controller.set_path_list_filtering(
                        v.using_filter
                    )
                    continue
                if isinstance(v, ImageSetDisplay):
                    v.on_event_fired = self.__frame.reset_imageset_ctrl
                    imageset_control = v
                    continue
                flag = wx.EXPAND
                border = 0
                control_name = edit_control_name(v)
                text_name = text_control_name(v)
                static_text = self.__module_panel.FindWindowByName(text_name)
                control = self.__module_panel.FindWindowByName(control_name)
                if static_text:
                    static_text.Show()
                    static_text.Label = encode_label(v.text)
                else:
                    static_text = wx.StaticText(
                        self.__module_panel,
                        -1,
                        encode_label(v.text),
                        style=wx.ALIGN_RIGHT,
                        name=text_name,
                    )
                text_sizer_item = sizer.Add(
                    static_text,
                    3,
                    wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL | wx.ALL,
                    2,
                )
                if control:
                    control.Show()
                self.__static_texts.append(static_text)
                if isinstance(v, Binary):
                    control = self.make_binary_control(v, control_name, control)
                    flag = wx.ALIGN_LEFT
                    text_sizer_item.Flag = (
                        wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL | wx.ALL
                    )
                elif isinstance(v, MeasurementMultiChoice):
                    control = self.make_measurement_multichoice_control(
                        v, control_name, control
                    )
                elif isinstance(v, SubdirectoryFilter):
                    control = self.make_subdirectory_filter_control(
                        v, control_name, control
                    )
                elif isinstance(v, MultiChoice):
                    control = self.make_multichoice_control(v, control_name, control)
                elif isinstance(v, CustomChoice):
                    control = self.make_choice_control(
                        v, v.get_choices(), control_name, wx.CB_DROPDOWN, control
                    )
                elif isinstance(v, Colormap):
                    control = self.make_colormap_control(v, control_name, control)
                elif isinstance(v, Choice):
                    control = self.make_choice_control(
                        v, v.get_choices(), control_name, wx.CB_READONLY, control
                    )
                    flag = wx.ALIGN_LEFT
                elif isinstance(v, (ImageListSubscriber, LabelListSubscriber,),):
                    choices = v.get_choices(self.__pipeline)
                    control = self.make_list_name_subscriber_control(
                        v, choices, control_name, control
                    )
                    flag = wx.EXPAND
                elif isinstance(v, (ImageSubscriber, LabelSubscriber, GridSubscriber)):
                    choices = v.get_choices(self.__pipeline)
                    control = self.make_name_subscriber_control(
                        v, choices, control_name, control
                    )
                    flag = wx.ALIGN_LEFT
                elif isinstance(v, FigureSubscriber):
                    choices = v.get_choices(self.__pipeline)
                    control = self.make_choice_control(
                        v, choices, control_name, wx.CB_DROPDOWN, control
                    )
                    flag = wx.ALIGN_LEFT
                elif isinstance(v, DoSomething):
                    if isinstance(v, PathListRefreshButton) and v.callback is None:
                        v.callback = self.__frame.pipeline_controller.on_update_pathlist
                    if isinstance(v, PathListExtractButton) and v.callback is None:
                        v.callback = self.__frame.pipeline_controller.on_extract_metadata
                    control = self.make_callback_control(v, control_name, control)
                    flag = wx.ALIGN_LEFT
                elif isinstance(v, DoThings):
                    control = self.make_callback_controls(v, control_name, control)
                elif isinstance(v, IntegerOrUnboundedRange):
                    control = self.make_unbounded_range_control(v, control)
                elif isinstance(v, IntegerRange) or isinstance(v, FloatRange):
                    control = self.make_range_control(v, control)
                elif isinstance(v, Coordinates):
                    control = self.make_coordinates_control(v, control)
                elif isinstance(v, RegexpText):
                    control = self.make_regexp_control(v, control)
                elif isinstance(v, Measurement):
                    control = self.make_measurement_control(v, control)
                elif isinstance(v, Divider):
                    if control is None:
                        if v.line:
                            control = wx.StaticLine(
                                self.__module_panel, name=control_name
                            )
                        else:
                            control = wx.StaticText(
                                self.__module_panel, name=control_name
                            )
                    flag = wx.EXPAND | wx.ALL
                    border = 2
                elif isinstance(v, Filename):
                    control = self.make_filename_text_control(v, control)
                elif isinstance(v, Directory):
                    control = self.make_directory_path_control(v, control_name, control)
                elif isinstance(v, Pathname):
                    control = self.make_pathname_control(v, control)
                elif isinstance(v, ImagePlane):
                    control = self.make_image_plane_control(v, control)
                elif isinstance(v, Color):
                    control = self.make_color_control(v, control_name, control)
                elif isinstance(v, TreeChoice):
                    control = self.make_tree_choice_control(v, control_name, control)
                    flag = wx.ALIGN_LEFT
                elif isinstance(v, Filter):
                    from ._filter_panel_controller import FilterPanelController

                    if control is not None:
                        control.filter_panel_controller.update()
                    else:
                        fc = FilterPanelController(self, v, control)
                        control = fc.panel
                        control.filter_panel_controller = fc
                elif isinstance(v, FileCollectionDisplay):
                    if control is not None:
                        # control.file_collection_display.update()
                        pass
                    else:
                        fcd = FileCollectionDisplayController(self, v, self.__pipeline)
                        control = fcd.panel
                        fcd.panel.file_collection_display = fcd
                elif isinstance(v, Table):
                    if v.use_sash:
                        table_control = v
                        grid = self.__frame.get_grid_ctrl()
                        table = grid.GetTable()
                        if isinstance(table, TableController) and table.v is v:
                            table.update_grid()
                            self.__frame.show_grid_ctrl()
                            continue
                        else:
                            table = TableController(v)
                            table.bind_to_grid(grid)
                            self.__frame.show_grid_ctrl(table)
                        continue
                    control = self.make_table_control(v, control)
                    flag = wx.EXPAND
                elif isinstance(v, HTMLText):
                    control = self.make_html_control(v, control)
                    flag = wx.EXPAND | wx.ALL
                elif isinstance(v, Joiner):
                    from ._joiner_controller import JoinerController

                    control = JoinerController.update_control(self, v)
                    flag = wx.ALIGN_LEFT
                elif isinstance(v, BinaryMatrix):
                    from ._binary_matrix_controller import BinaryMatrixController

                    control = BinaryMatrixController.update_control(self, v)
                    flag = wx.ALIGN_LEFT
                elif isinstance(v, DataTypes):
                    from ._data_type_controller import DataTypeController

                    control = DataTypeController.update_control(self, v)
                    flag = wx.ALIGN_LEFT
                elif isinstance(v, StructuringElement):
                    control = self.make_structuring_element_control(
                        v, control_name, control
                    )
                    text_sizer_item.Flag = (
                        wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL | wx.ALL
                    )
                else:
                    control = self.make_text_control(v, control_name, control)
                sizer.Add(control, 0, flag, border)
                self.__controls.append(control)
                help_name = help_ctrl_name(v)
                help_control = self.module_panel.FindWindowByName(help_name)

                if help_control is None:
                    if v.doc is None:
                        help_control = wx.StaticText(
                            self.__module_panel, -1, "", name=help_name
                        )
                    else:
                        help_control = self.make_help_control(
                            v.doc, v.text, name=help_name
                        )
                else:
                    help_control.Show()
                sizer.Add(help_control, 0, wx.LEFT, 2)
        finally:
            self.__handle_change = True

            if self.__frame is not None:
                if self.__started:
                    self.__frame.show_module_ui(True)
                if imageset_control is not None:
                    self.__frame.show_imageset_ctrl()
                    self.__frame.reset_imageset_ctrl(refresh_image_set=False)
                elif table_control is None:
                    self.__frame.show_imageset_sash(False)
                self.__frame.show_path_list_ctrl(path_control is not None)
                #
                # Lay out the module panel, then tell the scroll window
                # to fit it in order to update the scrollbars
                # see http://stackoverflow.com/questions/5912761/wxpython-scrolled-panel-not-updating-scroll-bars
                #
                self.__frame.layout_pmi_panel()
                self.top_panel.Layout()
                self.module_panel.FitInside()
                if self.DO_FREEZE:
                    self.module_panel.Thaw()
            else:
                if self.DO_FREEZE:
                    self.module_panel.Thaw()

    def make_structuring_element_control(self, v, control_name, control):
        shape_control_name = combobox_ctrl_name(v)
        size_control_name = text_control_name(v)

        if control is None:
            control = wx.Panel(
                self.module_panel, style=wx.TAB_TRAVERSAL, name=control_name
            )

            shape_control_text = wx.StaticText(
                control, label="Shape", style=wx.ALIGN_LEFT
            )

            shape_control = wx.ComboBox(
                control,
                choices=v.get_choices(),
                style=wx.CB_READONLY,
                name=shape_control_name,
                value=v.shape,
            )

            def on_select_shape(event, setting=v, control=shape_control):
                setting.shape = event.GetString()
                new_value = setting.value_text
                self.on_value_change(setting, control, new_value, event)

            shape_control.Bind(wx.EVT_COMBOBOX, on_select_shape)

            size_control_text = wx.StaticText(
                control, label="Size", style=wx.ALIGN_LEFT
            )

            size_control = wx.TextCtrl(
                control, name=size_control_name, value=str(v.size)
            )

            def on_set_size(event):
                size = event.GetString()
                v.size = int(size) if size else ""
                new_value = v.value_text
                self.notify(SettingEditedEvent(v, self.__module, new_value, event))

            self.__module_panel.Bind(wx.EVT_TEXT, on_set_size, size_control)

            shape_sizer = wx.BoxSizer(wx.VERTICAL)

            shape_sizer.Add(shape_control_text)

            shape_sizer.Add(shape_control)

            size_sizer = wx.BoxSizer(wx.VERTICAL)

            size_sizer.Add(size_control_text)

            size_sizer.Add(size_control)

            sizer = wx.BoxSizer(wx.HORIZONTAL)

            sizer.Add(shape_sizer)

            sizer.Add(size_sizer)

            control.SetSizer(sizer)
        else:
            shape_control = self.module_panel.FindWindowByName(shape_control_name)

            size_control = self.module_panel.FindWindowByName(size_control_name)

            shape_control.SetStringSelection(v.shape)

            size_control.Value = str(v.size)

        return control

    def make_notes_gui(self):
        """Make the GUI elements that contain the module notes"""
        #
        # The notes sizer contains a static box that surrounds the notes
        # plus the notes text control.
        #
        notes_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.notes_panel.SetSizer(notes_sizer)
        self.module_notes_control = wx.TextCtrl(
            self.notes_panel, -1, style=wx.TE_MULTILINE | wx.TE_PROCESS_ENTER
        )
        #
        # If you call self.module_notes_control.GetFullTextExtent(),
        # you will find that the font has a descender of 4 pixels. But
        # it looks like GetFullTextExtent is not implemented everywhere
        # so I hardcode here
        #
        height = self.module_notes_control.GetFullTextExtent("M")[1] * 2 + 4
        height = self.module_notes_control.ClientToWindowSize(wx.Size(1, height))[1]
        self.module_notes_control.SetInitialSize(wx.Size(100, 100))
        notes_sizer.Add(self.module_notes_control, 1, wx.ALL, 10)

        def on_notes_changed(event):
            if not self.__handle_change:
                return
            if self.__module is not None:
                notes = self.module_notes_control.GetValue()
                self.__module.notes = notes.split("\n")
                self.notify(
                    SettingEditedEvent("module_notes", self.__module, notes, event)
                )

        self.notes_panel.Bind(wx.EVT_TEXT, on_notes_changed, self.module_notes_control)

    def make_binary_control(self, v, control_name, control):
        """Make a checkbox control for a Binary setting"""
        if not control:
            control = wx.RadioBox(
                self.__module_panel, choices=["Yes", "No"], name=control_name,
            )

            def callback(event, setting=v, control=control):
                self.__on_radiobox_change(event, setting, control)

            control.Bind(wx.EVT_RADIOBOX, callback)
        current_selection = control.GetStringSelection()
        if current_selection != v.value_text:
            control.SetStringSelection(v.value_text)
        return control

    def make_name_subscriber_control(self, v, choices, control_name, control):
        """Make a read-only combobox with extra feedback about source modules,
        and a context menu with choices navigable by module name.

        v            - the setting
        choices      - a list of (name, module_name, module_number)
        control_name - assign this name to the control
        """
        if v.value not in [c[0] for c in choices]:
            choices = choices + [(v.value, "", 0, False)]
        if not control:
            control = namesubscriber.NameSubscriberComboBox(
                self.__module_panel, value=v.value, choices=choices, name=control_name
            )

            def callback(event, setting=v, control=control):
                # the NameSubscriberComboBox behaves like a combobox
                self.__on_combobox_change(event, setting, control)

            control.add_callback(callback)
        else:
            if list(choices) != list(control.Items):
                control.Items = choices
        if (
            getattr(v, "has_tooltips", False)
            and v.has_tooltips
            and (control.Value in v.tooltips)
        ):
            control.SetToolTip(wx.ToolTip(v.tooltips[control.Value]))
        return control

    def make_list_name_subscriber_control(self, v, choices, control_name, control):
        """Make a read-only combobox with extra feedback about source modules,
        and a context menu with choices navigable by module name.

        v            - the setting
        choices      - a list of (name, module_name, module_number)
        control_name - assign this name to the control
        """
        if not control:
            namelabel = "Image" if isinstance(v, ImageListSubscriber) else "Object"
            control = namesubscriber.NameSubscriberListBox(
                self.__module_panel,
                checked=v.value,
                choices=choices,
                name=control_name,
                nametype=namelabel,
            )

            def callback(event, setting=v, control=control):
                self.__on_checklistbox_change(event, setting, control)

            control.add_callback(callback)
        else:
            if list(choices) != list(control.Items):
                control.Items = choices
        return control

    def make_choice_control(self, v, choices, control_name, style, control):
        """Make a combo-box that shows choices

        v            - the setting
        choices      - the possible values for the setting
        control_name - assign this name to the control
        style        - one of the CB_ styles
        """
        assert isinstance(v, (Choice, FigureSubscriber,),)
        try:
            v.test_valid(self.__pipeline)
        except:
            pass
        if v.value not in choices and style == wx.CB_READONLY:
            choices = choices + [v.value]
        if not control:
            control = wx.ComboBox(
                self.__module_panel,
                -1,
                v.value,
                choices=choices,
                style=style,
                name=control_name,
            )

            def callback(event, setting=v, control=control):
                self.__on_combobox_change(event, setting, control)

            def ignore_mousewheel(event):
                return

            control.Bind(wx.EVT_MOUSEWHEEL, ignore_mousewheel)
            self.__module_panel.Bind(wx.EVT_COMBOBOX, callback, control)
            if style == wx.CB_DROPDOWN:

                def on_cell_change(event, setting=v, control=control):
                    self.__on_cell_change(event, setting, control)

                self.__module_panel.Bind(wx.EVT_TEXT, on_cell_change, control)
        else:
            old_choices = control.Items
            if len(choices) != len(old_choices) or not all(
                [x == y for x, y in zip(choices, old_choices)]
            ):
                if v.value in old_choices:
                    # For Mac, if you change the choices and the current
                    # combo-box value isn't in the choices, it throws
                    # an exception. Windows is much more forgiving.
                    # But the Mac has those buttons that look like little
                    # jellies, so it is better.
                    control.Value = v.value
                control.Items = choices
            try:
                # more desperate MAC cruft
                i_am_different = control.Value != v.value
            except:
                i_am_different = True
            if len(choices) > 0 and i_am_different:
                control.Value = v.value

        if (
            getattr(v, "has_tooltips", False)
            and v.has_tooltips
            and control.Value in v.tooltips
        ):
            control.SetToolTip(wx.ToolTip(v.tooltips[control.Value]))
        return control

    def make_measurement_multichoice_control(self, v, control_name, control):
        """Make a button that, when pressed, launches the tree editor"""
        if control is None:
            control = wx.Button(
                self.module_panel, -1, "Press button to select measurements"
            )

            def on_press(event):
                d = {}
                assert isinstance(v, MeasurementMultiChoice)
                if len(v.choices) == 0:
                    v.populate_choices(self.__pipeline)
                #
                # Populate the tree
                #
                choices = set(v.choices)
                for choice in choices:
                    object_name, feature = v.split_choice(choice)
                    pieces = [object_name] + feature.split("_")
                    d1 = d
                    for piece in pieces:
                        if piece not in d1:
                            d1[piece] = {}
                            d1[None] = 0
                        d1 = d1[piece]
                    d1[None] = False
                #
                # Mark selected leaf states as true
                #
                for selection in v.selections:
                    if selection not in choices:
                        continue
                    object_name, feature = v.split_choice(selection)
                    pieces = [object_name] + feature.split("_")
                    d1 = d
                    for piece in pieces:
                        if piece not in d1:
                            break
                        d1 = d1[piece]
                    d1[None] = True

                #
                # Backtrack recursively through tree to get branch states
                #
                def get_state(d):
                    leaf_state = d[None]
                    for subtree_key in [x for x in list(d.keys()) if x is not None]:
                        subtree_state = get_state(d[subtree_key])
                        if leaf_state == 0:
                            leaf_state = subtree_state
                        elif leaf_state != subtree_state:
                            leaf_state = None
                    d[None] = leaf_state
                    return leaf_state

                get_state(d)
                dlg = _tree_checkbox_dialog.TreeCheckboxDialog(
                    self.module_panel, d, size=(480, 480)
                )
                dlg.Title = "Select measurements"
                if dlg.ShowModal() == wx.ID_OK:

                    def collect_state(object_name, prefix, d):
                        if d[None] is False:
                            return []
                        result = []
                        if d[None] is True and prefix is not None:
                            name = v.make_measurement_choice(object_name, prefix)
                            if name in choices:
                                result.append(name)
                        for key in [x for x in list(d.keys()) if x is not None]:
                            if prefix is None:
                                sub_prefix = key
                            else:
                                sub_prefix = "_".join((prefix, key))
                            result += collect_state(object_name, sub_prefix, d[key])
                        return result

                    selections = []
                    for object_name in [x for x in list(d.keys()) if x is not None]:
                        selections += collect_state(object_name, None, d[object_name])
                    proposed_value = v.get_value_string(selections)
                    setting_edited_event = SettingEditedEvent(
                        v, self.__module, proposed_value, event
                    )
                    self.notify(setting_edited_event)
                    self.reset_view()

            control.Bind(wx.EVT_BUTTON, on_press)
        else:
            control.Show()
        return control

    def make_subdirectory_filter_control(self, v, control_name, control):
        if control is None:
            control = wx.Button(self.module_panel, -1, "Press to select folders")

            def on_press(event):
                assert isinstance(v, SubdirectoryFilter)

                root = v.directory_path.get_absolute_path()
                self.module_panel.SetCursor(wx.Cursor(wx.CURSOR_WAIT))
                try:

                    def fn_populate(root):
                        d = {None: True}
                        try:
                            for dirname in os.listdir(root):
                                dirpath = os.path.join(root, dirname)
                                dirstat = os.stat(dirpath)
                                if not stat.S_ISDIR(dirstat.st_mode):
                                    continue
                                if stat.S_ISLNK(dirstat.st_mode):
                                    continue
                                if (stat.S_IREAD & dirstat.st_mode) == 0:
                                    continue
                                d[dirname] = lambda dirpath=dirpath: fn_populate(
                                    dirpath
                                )
                        except:
                            print("Warning: failed to list directory %s" % root)
                        return d

                    d = fn_populate(root)
                    selections = v.get_selections()

                    def populate_selection(d, selection, root):
                        s0 = selection[0]
                        if s0 not in d:
                            d[s0] = fn_populate(os.path.join(root, s0))
                        elif hasattr(d[s0], "__call__"):
                            d[s0] = d[s0]()
                        if len(selection) == 1:
                            d[s0][None] = False
                        else:
                            if d[s0][None] is not False:
                                populate_selection(
                                    d[s0], selection[1:], os.path.join(root, s0)
                                )
                        if d[s0][None] is False:
                            # At best, the root is not all true
                            d[None] = None

                    def split_all(x):
                        head, tail = os.path.split(x)
                        if (len(head) == 0) or (len(tail) == 0):
                            return [x]
                        else:
                            return split_all(head) + [tail]

                    for selection in selections:
                        selection_parts = split_all(selection)
                        populate_selection(d, selection_parts, root)
                finally:
                    self.module_panel.SetCursor(wx.NullCursor)

                dlg = _tree_checkbox_dialog.TreeCheckboxDialog(
                    self.module_panel, d, size=(320, 480)
                )
                dlg.set_parent_reflects_child(False)
                dlg.Title = "Select folders"
                if dlg.ShowModal() == wx.ID_OK:

                    def collect_state(prefix, d):
                        if d is None:
                            return []
                        if hasattr(d, "__call__") or d[None]:
                            return []
                        elif d[None] is False:
                            return [prefix]
                        result = []
                        for key in list(d.keys()):
                            if key is None:
                                continue
                            result += collect_state(os.path.join(prefix, key), d[key])
                        return result

                    selections = []
                    for object_name in [x for x in list(d.keys()) if x is not None]:
                        selections += collect_state(object_name, d[object_name])
                    proposed_value = v.get_value_string(selections)
                    setting_edited_event = SettingEditedEvent(
                        v, self.__module, proposed_value, event
                    )
                    self.notify(setting_edited_event)
                    self.reset_view()

            control.Bind(wx.EVT_BUTTON, on_press)
        else:
            control.Show()
        return control

    def make_multichoice_control(self, v, control_name, control):
        selections = v.selections
        assert isinstance(v, MultiChoice)
        if isinstance(v, SubscriberMultiChoice):
            # Get the choices from the providers
            v.load_choices(self.__pipeline)
        choices = v.choices + [
            selection for selection in selections if selection not in v.choices
        ]
        if not control:
            control = wx.ListBox(
                self.__module_panel,
                -1,
                choices=choices,
                style=wx.LB_EXTENDED,
                name=control_name,
            )
            for selection in selections:
                index = choices.index(selection)
                control.SetSelection(index)
                if selection not in v.choices:
                    control.SetItemForegroundColour(index, get_error_color())

            def callback(event, setting=v, control=control):
                self.__on_multichoice_change(event, setting, control)

            self.__module_panel.Bind(wx.EVT_LISTBOX, callback, control)
        else:
            old_choices = control.Items
            if len(choices) != len(old_choices) or not all(
                [x == y for x, y in zip(choices, old_choices)]
            ):
                control.Items = choices
            for i in range(len(choices)):
                if control.IsSelected(i):
                    if choices[i] not in selections:
                        control.Deselect(i)
                elif choices[i] in selections:
                    control.Select(i)
                    if choices[i] not in v.choices:
                        control.SetItemForegroundColour(i, get_error_color())
        return control

    def make_colormap_control(self, v, control_name, control):
        """Make a combo-box that shows colormap choices
        v            - the setting
        choices      - the possible values for the setting
        control_name - assign this name to the control
        style        - one of the CB_ styles
        """
        try:
            if v.value == "Default":
                cmap_name = get_default_colormap()
            else:
                cmap_name = v.value
            cm = matplotlib.cm.get_cmap(cmap_name)
            sm = matplotlib.cm.ScalarMappable(cmap=cm)
            i, j = numpy.mgrid[0:12, 0:128]
            if cm.N < 128:
                j *= int((cm.N + 128) / 128)
            image = (sm.to_rgba(j) * 255).astype(numpy.uint8)
            bitmap = wx.Bitmap.FromBufferRGBA(128, 12, image.tostring())
        except:
            LOGGER.warning("Failed to create the %s colorbar" % cmap_name)
            bitmap = None
        if not control:
            control = wx.Panel(self.__module_panel, -1, name=control_name)
            sizer = wx.BoxSizer(wx.VERTICAL)
            control.SetSizer(sizer)
            colorbar = wx.StaticBitmap(control, -1, name=colorbar_ctrl_name(v))
            if bitmap is not None:
                colorbar.SetBitmap(bitmap)
            sizer.Add(colorbar, 0, wx.EXPAND | wx.BOTTOM, 2)

            combo = wx.ComboBox(
                control,
                -1,
                v.value,
                choices=v.choices,
                style=wx.CB_READONLY,
                name=combobox_ctrl_name(v),
            )
            sizer.Add(combo, 1, wx.EXPAND)

            def callback(event, setting=v, control=combo):
                self.__on_combobox_change(event, setting, combo)

            def ignore_mousewheel(event):
                return

            combo.Bind(wx.EVT_MOUSEWHEEL, ignore_mousewheel)
            self.__module_panel.Bind(wx.EVT_COMBOBOX, callback, combo)
        else:
            combo = control.FindWindowByName(combobox_ctrl_name(v))
            colorbar = control.FindWindowByName(colorbar_ctrl_name(v))
            old_choices = combo.Items
            if len(v.choices) != len(old_choices) or not all(
                [x == y for x, y in zip(v.choices, old_choices)]
            ):
                combo.Items = v.choices
            if combo.Value != v.value:
                combo.Value = v.value
            if bitmap is not None:
                colorbar.SetBitmap(bitmap)
        return control

    def make_color_control(self, v, control_name, control):
        try:
            color = wx.Colour()
            color.Set(v.value)
        except:
            color = wx.BLACK
            if (
                not hasattr(control, "bad_color_name")
                or control.bad_color_name != v.value
            ):
                LOGGER.warning("Failed to set color to %s" % v.value)
                control.bad_color_name = v.value
        if control is None:
            control = wx.lib.colourselect.ColourSelect(
                self.__module_panel, colour=color
            )
            control.SetName(control_name)

            def on_press(event, v=v, control=control):
                proposed_value = control.GetColour().GetAsString(
                    wx.C2S_NAME | wx.C2S_HTML_SYNTAX
                )
                setting_edited_event = SettingEditedEvent(
                    v, self.__module, proposed_value, event
                )
                self.notify(setting_edited_event)
                self.reset_view()

            #
            # There's a little display bugginess that, when the window's
            # size changes, the colored bitmap does not.
            #
            def on_size(event, control=control):
                control.SetBitmap(control.MakeBitmap())

            control.Bind(wx.lib.colourselect.EVT_COLOURSELECT, on_press)
            control.Bind(wx.EVT_SIZE, on_size)
        else:
            control.SetColour(color)
        return control

    def make_tree_choice_control(self, v, control_name, control):
        new_label = ">".join(v.get_path_parts())

        def make_bitmap(control, flags):
            assert isinstance(control, wx.BitmapButton)
            text_width, text_height = control.GetFullTextExtent(new_label)
            gap = 4
            drop_width = wx.SystemSettings.GetMetric(wx.SYS_VSCROLL_ARROW_X)
            drop_height = wx.SystemSettings.GetMetric(wx.SYS_VSCROLL_ARROW_Y)
            width = text_width + 2 * gap + drop_width
            height = max(text_height, drop_height) + 4
            bitmap = wx.Bitmap(width, height)
            dc = wx.MemoryDC(bitmap)
            dc.SetFont(control.GetFont())
            brush = wx.Brush(control.GetBackgroundColour())
            dc.SetBackground(brush)
            dc.Clear()
            wx.RendererNative.Get().DrawComboBox(
                control, dc, wx.Rect(0, 0, width, height), flags
            )
            dc.DrawText(new_label, 2, 2)
            return bitmap

        if control is None:
            control = wx.BitmapButton(self.module_panel, style=wx.BU_EXACTFIT)
            control.label_text = None

            def on_press(event, v=v, control=control):
                id_dict = {}

                def on_event(event, v=v, control=control, id_dict=None):
                    if id_dict is None:
                        id_dict = id_dict
                    new_path = v.encode_path_parts(id_dict[event.GetId()])
                    self.on_value_change(v, control, new_path, event)

                def make_menu(tree, id_dict=None, path=None):
                    if id_dict is None:
                        id_dict = id_dict
                    if path is None:
                        path = []
                    menu = wx.Menu()
                    for node in tree:
                        text, subtree = node[:2]
                        subpath = path + [text]
                        if v.fn_is_leaf(node):
                            item = menu.Append(-1, text)
                            id_dict[item.GetId()] = subpath
                            if wx.VERSION >= (2, 9) and sys.platform != "win32":
                                wx.EVT_MENU(menu, item.GetId(), on_event)
                        if subtree is not None and len(subtree) > 0:
                            submenu = make_menu(subtree, path=subpath)
                            menu.Append(-1, text, submenu)
                    return menu

                menu = make_menu(v.get_tree())
                assert isinstance(control, wx.Window)
                if wx.VERSION < (2, 9) or sys.platform == "win32":
                    menu.Bind(wx.EVT_MENU, on_event)
                control.PopupMenu(menu, 0, control.GetSize()[1])
                menu.Destroy()

            control.Bind(wx.EVT_BUTTON, on_press)
        old_label = control.label_text
        if old_label != new_label:
            control.label_text = new_label
            for getter, setter, flags in (
                (control.GetBitmapLabel, control.SetBitmapLabel, 0),
                (control.GetBitmapFocus, control.SetBitmapFocus, wx.CONTROL_FOCUSED),
                (
                    control.GetBitmapSelected,
                    control.SetBitmapSelected,
                    wx.CONTROL_SELECTED,
                ),
            ):
                old_bitmap = getter()
                setter(make_bitmap(control, flags))
                if old_bitmap is not None:
                    old_bitmap.Destroy()
        return control

    def make_callback_control(self, v, control_name, control):
        """Make a control that calls back using the callback buried in the setting"""
        if not control:
            control = wx.Button(self.module_panel, -1, v.label, name=control_name)

            def callback(event, setting=v):
                self.__on_do_something(event, setting)

            self.module_panel.Bind(wx.EVT_BUTTON, callback, control)
        else:
            control.Label = v.label
        return control

    def make_callback_controls(self, v, control_name, control):
        """Make a panel of buttons for each of the setting's actions

        v - a DoThings setting

        control_name - the name that we apply to the panel

        control - either None or the panel containing the buttons
        """
        assert isinstance(v, DoThings)
        if not control:
            control = wx.Panel(self.module_panel, name=control_name)
            control.SetSizer(wx.BoxSizer(wx.HORIZONTAL))
            for i in range(v.count):
                if i != 0:
                    control.Sizer.AddSpacer(2)
                button = wx.Button(control, name=button_control_name(v, i))
                control.Sizer.Add(button, 0, wx.ALIGN_LEFT)

                def callback(event, index=i):
                    v.on_event_fired(index)
                    setting_edited_event = SettingEditedEvent(
                        v, self.__module, None, event
                    )
                    self.notify(setting_edited_event)
                    self.__module.on_setting_changed(v, self.__pipeline)
                    self.reset_view()

                button.Bind(wx.EVT_BUTTON, callback)
        for i in range(v.count):
            button = control.FindWindowByName(button_control_name(v, i))
            button.Label = v.get_label(i)
        return control

    def make_regexp_control(self, v, control):
        """Make a textbox control + regular expression button"""
        if not control:
            panel = wx.Panel(self.__module_panel, -1, name=edit_control_name(v))
            control = panel
            sizer = wx.BoxSizer(wx.HORIZONTAL)
            panel.SetSizer(sizer)
            text_ctrl = wx.TextCtrl(panel, -1, str(v.value), name=text_control_name(v))
            sizer.Add(text_ctrl, 1, wx.EXPAND | wx.RIGHT, 1)
            bitmap = wx.ArtProvider.GetBitmap(wx.ART_FIND, wx.ART_TOOLBAR, (16, 16))
            bitmap_button = wx.BitmapButton(
                panel, bitmap=bitmap, name=button_control_name(v)
            )
            sizer.Add(bitmap_button, 0, wx.EXPAND)

            def on_cell_change(event, setting=v, control=text_ctrl):
                self.__on_cell_change(event, setting, control, timeout=False)

            def on_button_pressed(event, setting=v, control=text_ctrl):
                #
                # Find a file in the image directory
                #
                filename = "plateA-2008-08-06_A12_s1_w1_[89A882DE-E675-4C12-9F8E-46C9976C4ABE].tif"
                try:
                    if setting.get_example_fn is None:
                        path = get_default_image_directory()
                        filenames = [
                            x
                            for x in os.listdir(path)
                            if x.find(".") != -1
                            and os.path.splitext(x)[1].upper()
                            in (".TIF", ".JPG", ".PNG", ".BMP")
                        ]
                        if len(filenames):
                            filename = filenames[0]
                    else:
                        filename = setting.get_example_fn()
                except:
                    pass

                if v.guess == RegexpText.GUESS_FOLDER:
                    guesses = regexp_editor.RE_FOLDER_GUESSES
                else:
                    guesses = regexp_editor.RE_FILENAME_GUESSES

                new_value = regexp_editor.edit_regexp(
                    panel, control.GetValue(), filename, guesses
                )
                if new_value:
                    control.SetValue(new_value)
                    self.__on_cell_change(event, setting, control)

            def on_kill_focus(event, setting=v, control=text_ctrl):
                # Make sure not to call set_selection again if a set_selection is already
                # in process. Doing so may have adverse effects (e.g. disappearing textboxes)
                if self.__module is not None and self.__handle_change:
                    self.set_selection(self.__module.module_num)
                event.Skip()

            self.__module_panel.Bind(wx.EVT_TEXT, on_cell_change, text_ctrl)
            self.__module_panel.Bind(wx.EVT_BUTTON, on_button_pressed, bitmap_button)
            #
            # http://www.velocityreviews.com/forums/t359823-textctrl-focus-events-in-wxwidgets.html
            # explains why bind is to control itself
            #
            text_ctrl.Bind(wx.EVT_KILL_FOCUS, on_kill_focus)
        else:
            text_control = control.FindWindow(text_control_name(v))
            if v.value != text_control.Value:
                text_control.Value = v.value
        return control

    def make_filename_text_control(self, v, control):
        """Make a filename text control"""
        edit_name = subedit_control_name(v)
        control_name = edit_control_name(v)
        button_name = button_control_name(v)
        if control is None:
            control = wx.Panel(self.module_panel, -1, name=control_name)
            sizer = wx.BoxSizer(wx.HORIZONTAL)
            control.SetSizer(sizer)
            if v.metadata_display:
                edit_control = metadatactrl.MetadataControl(
                    self.__pipeline,
                    self.__module,
                    control,
                    value=v.value,
                    name=edit_name,
                )
            else:
                edit_control = wx.TextCtrl(control, -1, str(v.value), name=edit_name)
            sizer.Add(edit_control, 1, wx.ALIGN_LEFT | wx.ALIGN_TOP)

            def on_cell_change(event, setting=v, control=edit_control):
                self.__on_cell_change(event, setting, control)

            self.__module_panel.Bind(wx.EVT_TEXT, on_cell_change, edit_control)

            bitmap = wx.ArtProvider.GetBitmap(wx.ART_FILE_OPEN, wx.ART_BUTTON, (16, 16))
            button_control = wx.BitmapButton(control, bitmap=bitmap, name=button_name)

            def on_press(event):
                """Open a file browser"""
                if v.mode == Filename.MODE_OPEN:
                    mode = wx.FD_OPEN
                elif v.mode == Filename.MODE_APPEND:
                    mode = wx.FD_SAVE
                else:
                    mode = wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
                dlg = wx.FileDialog(control, v.browse_msg, style=mode)
                if v.get_directory_fn is not None:
                    dlg.SetDirectory(v.get_directory_fn())
                if v.exts is not None:
                    dlg.SetWildcard("|".join(["|".join(tuple(x)) for x in v.exts]))
                if dlg.ShowModal() == wx.ID_OK:
                    if v.set_directory_fn is not None:
                        v.set_directory_fn(dlg.GetDirectory())
                    v.value = dlg.GetFilename()
                    setting_edited_event = SettingEditedEvent(
                        v, self.__module, v.value, event
                    )
                    self.notify(setting_edited_event)
                    self.reset_view()

            button_control.Bind(wx.EVT_BUTTON, on_press)
            sizer.Add(button_control, 0, wx.EXPAND | wx.LEFT, 2)
        else:
            edit_control = self.module_panel.FindWindowByName(edit_name)
            button_control = self.module_panel.FindWindowByName(button_name)
            if edit_control.Value != v.value:
                edit_control.Value = v.value
            button_control.Show(v.browsable)
        return control

    def make_directory_path_control(self, v, control_name, control):
        assert isinstance(v, Directory)
        dir_ctrl_name = combobox_ctrl_name(v)
        custom_ctrl_name = subedit_control_name(v)
        custom_ctrl_label_name = custom_label_name(v)
        browse_ctrl_name = button_control_name(v)
        folder_label_ctrl_name = folder_label_name(v)

        if control is None:
            control = wx.Panel(
                self.module_panel, style=wx.TAB_TRAVERSAL, name=control_name
            )
            sizer = wx.BoxSizer(wx.VERTICAL)
            control.SetSizer(sizer)
            choice_sizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(choice_sizer, 0, wx.ALIGN_TOP | wx.ALIGN_LEFT)
            dir_ctrl = wx.Choice(control, choices=v.dir_choices, name=dir_ctrl_name)
            choice_sizer.Add(dir_ctrl, 0, wx.ALIGN_LEFT | wx.BOTTOM, 2)
            choice_sizer.AddSpacer(3)
            folder_label = wx.StaticText(control, name=folder_label_ctrl_name)
            choice_sizer.Add(folder_label, 0, wx.ALIGN_CENTER_VERTICAL)

            custom_sizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(custom_sizer, 1, wx.EXPAND)
            custom_label = wx.StaticText(control, name=custom_ctrl_label_name)
            custom_sizer.Add(custom_label, 0, wx.ALIGN_CENTER_VERTICAL)
            if v.allow_metadata:
                custom_ctrl = metadatactrl.MetadataControl(
                    self.__pipeline,
                    self.__module,
                    control,
                    value=v.custom_path,
                    name=custom_ctrl_name,
                )
            else:
                custom_ctrl = wx.TextCtrl(
                    control, -1, v.custom_path, name=custom_ctrl_name
                )
            custom_sizer.Add(custom_ctrl, 1, wx.ALIGN_CENTER_VERTICAL)
            browse_bitmap = wx.ArtProvider.GetBitmap(
                wx.ART_FOLDER, wx.ART_CMN_DIALOG, (16, 16)
            )
            browse_ctrl = wx.BitmapButton(
                control, bitmap=browse_bitmap, name=browse_ctrl_name
            )
            custom_sizer.Add(browse_ctrl, 0, wx.ALIGN_CENTER | wx.LEFT, 2)

            def on_dir_choice_change(event, v=v, dir_ctrl=dir_ctrl):
                """Handle a change to the directory choice combobox"""
                if not self.__handle_change:
                    return
                proposed_value = v.join_string(dir_choice=dir_ctrl.GetStringSelection())
                setting_edited_event = SettingEditedEvent(
                    v, self.__module, proposed_value, event
                )
                self.notify(setting_edited_event)
                self.reset_view()

            def on_custom_path_change(event, v=v, custom_ctrl=custom_ctrl):
                """Handle a change to the custom path"""
                if not self.__handle_change:
                    return
                proposed_value = v.join_string(custom_path=custom_ctrl.Value)
                setting_edited_event = SettingEditedEvent(
                    v, self.__module, proposed_value, event
                )
                self.notify(setting_edited_event)
                self.reset_view(1000)

            def on_browse_pressed(
                event, v=v, dir_ctrl=dir_ctrl, custom_ctrl=custom_ctrl
            ):
                """Handle browse button pressed"""
                dlg = wx.DirDialog(self.module_panel, v.text, v.get_absolute_path())
                if dlg.ShowModal() == wx.ID_OK:
                    dir_choice, custom_path = v.get_parts_from_path(dlg.GetPath())
                    proposed_value = v.join_string(dir_choice, custom_path)
                    if v.allow_metadata:
                        # Do escapes on backslashes
                        proposed_value = proposed_value.replace("\\", "\\\\")
                    setting_edited_event = SettingEditedEvent(
                        v, self.__module, proposed_value, event
                    )
                    self.notify(setting_edited_event)
                    self.reset_view()

            dir_ctrl.Bind(wx.EVT_CHOICE, on_dir_choice_change)
            custom_ctrl.Bind(wx.EVT_TEXT, on_custom_path_change)
            browse_ctrl.Bind(wx.EVT_BUTTON, on_browse_pressed)
        else:
            dir_ctrl = self.module_panel.FindWindowByName(dir_ctrl_name)
            custom_ctrl = self.module_panel.FindWindowByName(custom_ctrl_name)
            custom_label = self.module_panel.FindWindowByName(custom_ctrl_label_name)
            browse_ctrl = self.module_panel.FindWindowByName(browse_ctrl_name)
            folder_label = self.module_panel.FindWindowByName(folder_label_ctrl_name)
        if dir_ctrl.StringSelection != v.dir_choice:
            dir_ctrl.StringSelection = v.dir_choice
        if v.is_custom_choice:
            if not custom_ctrl.IsShown():
                custom_ctrl.Show()
            if not custom_label.IsShown():
                custom_label.Show()
            if not browse_ctrl.IsShown():
                browse_ctrl.Show()
            if v.dir_choice in (
                DEFAULT_INPUT_SUBFOLDER_NAME,
                DEFAULT_OUTPUT_SUBFOLDER_NAME,
            ):
                custom_label.Label = "Sub-folder:"
            elif v.dir_choice == URL_FOLDER_NAME:
                custom_label.Hide()
                custom_ctrl.Hide()
                browse_ctrl.Hide()
            if custom_ctrl.Value != v.custom_path:
                custom_ctrl.Value = v.custom_path
        else:
            custom_label.Hide()
            custom_ctrl.Hide()
            browse_ctrl.Hide()
        if v.dir_choice in (DEFAULT_INPUT_FOLDER_NAME, DEFAULT_INPUT_SUBFOLDER_NAME,):
            folder_label.Label = "( %s )" % get_default_image_directory()
        elif v.dir_choice in (
            DEFAULT_OUTPUT_FOLDER_NAME,
            DEFAULT_OUTPUT_SUBFOLDER_NAME,
        ):
            folder_label.Label = "( %s )" % get_default_output_directory()
        else:
            folder_label.Label = wx.EmptyString
        dir_ctrl.SetToolTip(folder_label.Label)
        return control

    def make_pathname_control(self, v, control):
        if control is None:
            control = wx.Panel(self.module_panel, -1, name=edit_control_name(v))
            control.SetSizer(wx.BoxSizer(wx.HORIZONTAL))
            text_control = wx.TextCtrl(control, -1, name=subedit_control_name(v))
            text_control.Bind(
                wx.EVT_TEXT, lambda event: self.__on_cell_change(event, v, text_control)
            )
            browse_bitmap = wx.ArtProvider.GetBitmap(
                wx.ART_FOLDER, wx.ART_CMN_DIALOG, (16, 16)
            )
            browse_ctrl = wx.BitmapButton(
                control, bitmap=browse_bitmap, name=button_control_name(v)
            )
            control.GetSizer().Add(text_control, 1, wx.EXPAND)
            control.GetSizer().AddSpacer(3)
            control.GetSizer().Add(browse_ctrl, 0, wx.EXPAND)

            def on_browse(event):
                dlg = wx.FileDialog(self.module_panel)
                try:
                    dlg.SetTitle("Browse for metadata file")
                    dlg.SetWildcard(v.wildcard)
                    if dlg.ShowModal() == wx.ID_OK:
                        self.on_value_change(v, control, dlg.GetPath(), event)
                finally:
                    dlg.Destroy()

            browse_ctrl.Bind(wx.EVT_BUTTON, on_browse)
        else:
            text_control = control.FindWindowByName(subedit_control_name(v))
        if text_control.Value != v.value:
            text_control.Value = v.value
        return control

    def make_image_plane_control(self, v, control):
        """Make a control to pick an image plane from the file list"""
        from cellprofiler_core.utilities.pathname import url2pathname

        assert isinstance(v, ImagePlane)
        if not control:
            control = wx.Panel(self.module_panel, name=edit_control_name(v))
            control.SetSizer(wx.BoxSizer(wx.HORIZONTAL))
            url_control = wx.TextCtrl(
                control, style=wx.TE_READONLY, name=text_control_name(v)
            )
            control.GetSizer().Add(url_control, 1, wx.EXPAND)
            control.GetSizer().AddSpacer(2)
            browse_button = wx.Button(
                control, label="Browse", name=button_control_name(v)
            )
            control.GetSizer().Add(browse_button, 0, wx.EXPAND)

            def on_button(event):
                selected_plane = self.__frame.pipeline_controller.pick_from_pathlist(
                    v.get_plane(), instructions="Select an image plane from the list below"
                )
                if selected_plane is not None:
                    value = v.build(selected_plane)
                    self.on_value_change(v, control, value, event)
                    url_control.Value = url2pathname(v.value)

            browse_button.Bind(wx.EVT_BUTTON, on_button)
        else:
            url_control = control.FindWindowByName(text_control_name(v))
        label = v.url or ""
        if label.startswith("file:"):
            label = url2pathname(label)
        url_control.Value = label
        return control

    def make_text_control(self, v, control_name, control):
        """Make a textbox control"""
        text = None
        if not control:
            if v.metadata_display:
                control = metadatactrl.MetadataControl(
                    self.__pipeline,
                    self.__module,
                    self.__module_panel,
                    value=v.value,
                    name=control_name,
                )
            else:
                style = 0
                text = v.get_value_text()
                if not isinstance(text, str):
                    text = str(text)
                if getattr(v, "multiline_display", False):
                    style = wx.TE_MULTILINE | wx.TE_PROCESS_ENTER
                    lines = text.split("\n")
                else:
                    lines = [text]

                control = wx.TextCtrl(
                    self.__module_panel, -1, text, name=control_name, style=style
                )

            def on_cell_change(event, setting=v, control=control):
                self.__on_cell_change(event, setting, control)

            self.__module_panel.Bind(wx.EVT_TEXT, on_cell_change, control)
        elif not (v.get_value_text() == control.Value):
            text = v.get_value_text()
            if not isinstance(text, str):
                text = str(text)
            control.Value = text
        if text is not None:
            lines = text.split("\n")
            if len(lines) > 0:
                width = max([control.GetTextExtent(line)[0] for line in lines])
                height = sum([control.GetTextExtent(line)[1] for line in lines])
                min_width, min_height = control.GetTextExtent("M")
                if width < min_width:
                    width = min_width
                if height < min_height:
                    height = min_height
                bw, bh = control.GetWindowBorderSize()
                control.SetMinSize((bw * 2 + width, bh * 2 + height))
        return control

    def make_range_control(self, v, panel):
        """Make a "control" composed of a panel and two edit boxes representing a range"""
        if not panel:
            panel = wx.Panel(self.__module_panel, -1, name=edit_control_name(v))
            sizer = wx.BoxSizer(wx.HORIZONTAL)
            panel.SetSizer(sizer)
            min_ctrl = wx.TextCtrl(panel, -1, v.min_text, name=min_control_name(v))
            sizer.Add(min_ctrl, 0, wx.EXPAND | wx.RIGHT, 1)
            max_ctrl = wx.TextCtrl(panel, -1, v.max_text, name=max_control_name(v))
            # max_ctrl.SetInitialSize(wx.Size(best_width,-1))
            sizer.Add(max_ctrl, 0, wx.EXPAND)

            def on_min_change(event, setting=v, control=min_ctrl):
                self.__on_min_change(event, setting, control)

            self.__module_panel.Bind(wx.EVT_TEXT, on_min_change, min_ctrl)

            def on_max_change(event, setting=v, control=max_ctrl):
                self.__on_max_change(event, setting, control)

            self.__module_panel.Bind(wx.EVT_TEXT, on_max_change, max_ctrl)
        else:
            min_ctrl = panel.FindWindowByName(min_control_name(v))
            if min_ctrl.Value != v.min_text:
                min_ctrl.Value = v.min_text
            max_ctrl = panel.FindWindowByName(max_control_name(v))
            if max_ctrl.Value != v.max_text:
                max_ctrl.Value = v.max_text

        for ctrl in (min_ctrl, max_ctrl):
            self.fit_ctrl(ctrl)
        return panel

    def make_unbounded_range_control(self, v, panel):
        """Make a "control" composed of a panel and two combo-boxes representing a range

        v - an IntegerOrUnboundedRange setting
        panel - put it in this panel

        The combo box has the word to use to indicate that the range is unbounded
        and the text portion is the value
        """
        if not panel:
            panel = wx.Panel(self.__module_panel, -1, name=edit_control_name(v))
            sizer = wx.BoxSizer(wx.HORIZONTAL)
            panel.SetSizer(sizer)
            min_ctrl = wx.TextCtrl(
                panel, -1, value=str(v.min), name=min_control_name(v)
            )
            best_width = min_ctrl.GetCharWidth() * 5
            min_ctrl.SetInitialSize(wx.Size(best_width, -1))
            sizer.Add(min_ctrl, 0, wx.EXPAND | wx.RIGHT, 1)
            max_ctrl = wx.TextCtrl(
                panel, -1, value=v.display_max, name=max_control_name(v)
            )
            max_ctrl.SetInitialSize(wx.Size(best_width, -1))
            sizer.Add(max_ctrl, 0, wx.EXPAND)
            value = ABSOLUTE if v.is_abs() else FROM_EDGE
            absrel_ctrl = wx.ComboBox(
                panel,
                -1,
                value,
                choices=[ABSOLUTE, FROM_EDGE],
                name=absrel_control_name(v),
                style=wx.CB_DROPDOWN | wx.CB_READONLY,
            )
            sizer.Add(absrel_ctrl, 0, wx.EXPAND | wx.RIGHT, 1)

            def ignore_mousewheel(event):
                return

            def on_min_change(event, setting=v, control=min_ctrl):
                if not self.__handle_change:
                    return
                proposed_value = setting.compose_min_text(control.GetValue())
                setting_edited_event = SettingEditedEvent(
                    setting, self.__module, proposed_value, event
                )
                self.notify(setting_edited_event)
                self.fit_ctrl(control)

            self.__module_panel.Bind(wx.EVT_TEXT, on_min_change, min_ctrl)
            absrel_ctrl.Bind(wx.EVT_MOUSEWHEEL, ignore_mousewheel)

            def on_max_change(
                event, setting=v, control=max_ctrl, absrel_ctrl=absrel_ctrl
            ):
                if not self.__handle_change:
                    return
                proposed_value = setting.compose_display_max_text(control.GetValue())
                setting_edited_event = SettingEditedEvent(
                    setting, self.__module, proposed_value, event
                )
                self.notify(setting_edited_event)
                self.fit_ctrl(control)

            self.__module_panel.Bind(wx.EVT_TEXT, on_max_change, max_ctrl)

            def on_absrel_change(event, setting=v, control=absrel_ctrl):
                if not self.__handle_change:
                    return

                if control.GetValue() == ABSOLUTE:
                    proposed_value = setting.compose_abs()
                else:
                    proposed_value = setting.compose_rel()
                if proposed_value is not None:
                    setting_edited_event = SettingEditedEvent(
                        setting, self.__module, proposed_value, event
                    )
                    self.notify(setting_edited_event)

            self.__module_panel.Bind(wx.EVT_COMBOBOX, on_absrel_change, absrel_ctrl)
        else:
            min_ctrl = panel.FindWindowByName(min_control_name(v))
            if min_ctrl.Value != v.display_min:
                min_ctrl.Value = v.display_min
            max_ctrl = panel.FindWindowByName(max_control_name(v))
            if max_ctrl.Value != v.display_max:
                min_ctrl.Value = v.display_max
            absrel_ctrl = panel.FindWindowByName(absrel_control_name(v))
            absrel_value = ABSOLUTE if v.is_abs() else FROM_EDGE
            if absrel_ctrl.Value != absrel_value:
                absrel_ctrl.Value = absrel_value

        return panel

    def make_coordinates_control(self, v, panel):
        """Make a "control" composed of a panel and two edit boxes representing X and Y"""
        if not panel:
            panel = wx.Panel(self.__module_panel, -1, name=edit_control_name(v))
            sizer = wx.BoxSizer(wx.HORIZONTAL)
            panel.SetSizer(sizer)
            sizer.Add(wx.StaticText(panel, -1, "X:"), 0, wx.EXPAND | wx.RIGHT, 1)
            x_ctrl = wx.TextCtrl(panel, -1, v.get_x_text(), name=x_control_name(v))
            best_width = x_ctrl.GetCharWidth() * 5
            x_ctrl.SetInitialSize(wx.Size(best_width, -1))
            sizer.Add(x_ctrl, 0, wx.EXPAND | wx.RIGHT, 1)
            sizer.Add(wx.StaticText(panel, -1, "Y:"), 0, wx.EXPAND | wx.RIGHT, 1)
            y_ctrl = wx.TextCtrl(panel, -1, v.get_y_text(), name=y_control_name(v))
            y_ctrl.SetInitialSize(wx.Size(best_width, -1))
            sizer.Add(y_ctrl, 0, wx.EXPAND)

            def on_x_change(event, setting=v, control=x_ctrl):
                if not self.__handle_change:
                    return
                old_value = str(setting)
                proposed_value = "%s,%s" % (str(control.GetValue()), str(setting.y))
                setting_edited_event = SettingEditedEvent(
                    setting, self.__module, proposed_value, event
                )
                self.notify(setting_edited_event)

            self.__module_panel.Bind(wx.EVT_TEXT, on_x_change, x_ctrl)

            def on_y_change(event, setting=v, control=y_ctrl):
                if not self.__handle_change:
                    return
                old_value = str(setting)
                proposed_value = "%s,%s" % (str(setting.x), str(control.GetValue()))
                setting_edited_event = SettingEditedEvent(
                    setting, self.__module, proposed_value, event
                )
                self.notify(setting_edited_event)

            self.__module_panel.Bind(wx.EVT_TEXT, on_y_change, y_ctrl)
        else:
            x_ctrl = panel.FindWindowByName(x_control_name(v))
            if x_ctrl.Value != v.get_x_text():
                x_ctrl.Value = v.get_x_text()
            y_ctrl = panel.FindWindowByName(y_control_name(v))
            if y_ctrl.Value != v.get_y_text():
                y_ctrl.Value = v.get_y_text()

        return panel

    def make_measurement_control(self, v, panel):
        """Make a composite measurement control

        The measurement control has the following parts:
        Category - a measurement category like AreaShape or Intensity
        Feature name - the feature being measured or algorithm applied
        Image name - an optional image that was used to compute the measurement
        Object name - an optional set of objects used to compute the measurement
        Scale - an optional scale, generally in pixels, that controls the size
                of the measured features.
        """
        #
        # We either come in here with:
        # * panel = None - create the controls
        # * panel != None - find the controls
        #
        category = v.get_category(self.__pipeline)
        categories = v.get_category_choices(self.__pipeline)
        feature_name = v.get_feature_name(self.__pipeline)
        feature_names = v.get_feature_name_choices(self.__pipeline)
        image_name = v.get_image_name(self.__pipeline)
        image_names = v.get_image_name_choices(self.__pipeline)
        object_name = v.get_object_name(self.__pipeline)
        object_names = v.get_object_name_choices(self.__pipeline)
        scale = v.get_scale(self.__pipeline)
        scales = v.get_scale_choices(self.__pipeline)

        if not panel:
            panel = wx.Panel(self.__module_panel, -1, name=edit_control_name(v))
            sizer = wx.BoxSizer(wx.VERTICAL)
            panel.SetSizer(sizer)
            #
            # The category combo-box
            #
            sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(sub_sizer, 0, wx.ALIGN_LEFT)
            category_text_ctrl = wx.StaticText(
                panel, label="Category:", name=category_text_control_name(v)
            )
            sub_sizer.Add(category_text_ctrl, 0, wx.EXPAND | wx.ALL, 2)
            category_ctrl = wx.ComboBox(
                panel, style=wx.CB_READONLY, name=category_control_name(v), choices=categories
            )
            sub_sizer.Add(category_ctrl, 0, wx.EXPAND | wx.ALL, 2)
            #
            # The measurement / feature combo-box
            #
            sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(sub_sizer, 0, wx.ALIGN_LEFT)
            feature_text_ctrl = wx.StaticText(
                panel, label="Measurement:", name=feature_text_control_name(v)
            )
            sub_sizer.Add(feature_text_ctrl, 0, wx.EXPAND | wx.ALL, 2)
            feature_ctrl = wx.ComboBox(
                panel, style=wx.CB_READONLY, name=feature_control_name(v), choices=feature_names
            )
            sub_sizer.Add(feature_ctrl, 0, wx.EXPAND | wx.ALL, 2)
            #
            # The object combo-box which sometimes doubles as an image combo-box
            #
            sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(sub_sizer, 0, wx.ALIGN_LEFT)
            object_text_ctrl = wx.StaticText(
                panel, label="Object:", name=object_text_control_name(v)
            )
            sub_sizer.Add(object_text_ctrl, 0, wx.EXPAND | wx.ALL, 2)
            object_ctrl = wx.ComboBox(
                panel, style=wx.CB_READONLY, name=object_control_name(v), choices=image_names+object_names
            )
            sub_sizer.Add(object_ctrl, 0, wx.EXPAND | wx.ALL, 2)
            #
            # The scale combo-box
            sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(sub_sizer, 0, wx.ALIGN_LEFT)
            scale_text_ctrl = wx.StaticText(
                panel, label="Scale:", name=scale_text_ctrl_name(v)
            )
            sub_sizer.Add(scale_text_ctrl, 0, wx.EXPAND | wx.ALL, 2)
            scale_ctrl = wx.ComboBox(
                panel, style=wx.CB_READONLY, name=scale_control_name(v), choices=scales
            )
            sub_sizer.Add(scale_ctrl, 0, wx.EXPAND | wx.ALL, 2)
            max_width = 0
            for sub_sizer_item in sizer.GetChildren():
                static = sub_sizer_item.Sizer.GetChildren()[1].Window
                max_width = max(max_width, static.Size.GetWidth())
            for sub_sizer_item in sizer.GetChildren():
                static = sub_sizer_item.Sizer.GetChildren()[1].Window
                static.Size = wx.Size(max_width, static.Size.GetHeight())
                static.SetSizeHints(max_width, -1, max_width)

            #
            # Bind all controls to the function that constructs a value
            # out of the parts
            #
            def on_change(
                event,
                v=v,
                category_ctrl=category_ctrl,
                feature_ctrl=feature_ctrl,
                object_ctrl=object_ctrl,
                scale_ctrl=scale_ctrl,
            ):
                """Reconstruct the measurement value if anything changes"""
                if not self.__handle_change:
                    return

                def value_of(ctrl):
                    return ctrl.Value if ctrl.Selection != -1 else None

                value = v.construct_value(
                    value_of(category_ctrl),
                    value_of(feature_ctrl),
                    value_of(object_ctrl),
                    value_of(scale_ctrl),
                )
                setting_edited_event = SettingEditedEvent(
                    v, self.__module, value, event
                )
                self.notify(setting_edited_event)
                self.reset_view()

            def ignore_mousewheel(evt):
                return

            for ctrl in (category_ctrl, feature_ctrl, object_ctrl, scale_ctrl):
                panel.Bind(wx.EVT_MOUSEWHEEL, ignore_mousewheel)
                panel.Bind(wx.EVT_COMBOBOX, on_change, ctrl)
        else:
            #
            # Find the controls from inside the panel
            #
            category_ctrl = panel.FindWindowByName(category_control_name(v))
            category_text_ctrl = panel.FindWindowByName(category_text_control_name(v))
            feature_ctrl = panel.FindWindowByName(feature_control_name(v))
            feature_text_ctrl = panel.FindWindowByName(feature_text_control_name(v))
            object_ctrl = panel.FindWindowByName(object_control_name(v))
            object_text_ctrl = panel.FindWindowByName(object_text_control_name(v))
            scale_ctrl = panel.FindWindowByName(scale_control_name(v))
            scale_text_ctrl = panel.FindWindowByName(scale_text_ctrl_name(v))

        def set_up_combobox(ctrl, text_ctrl, choices, value, always_show=False):
            if len(choices):
                if value is None:
                    choices = ["[None]"] + choices
                if not (
                    len(ctrl.Strings) == len(choices)
                    and all([x == y for x, y in zip(ctrl.Strings, choices)])
                ):
                    ctrl.Clear()
                    ctrl.AppendItems(choices)
                    ctrl.SetSelection(0)
                if value is not None:
                    try:
                        if ctrl.Value != value:
                            ctrl.Value = value
                    except:
                        # Crashes on the Mac sometimes
                        ctrl.Value = value
                else:
                    if ctrl.Value != "[None]":
                        ctrl.SetSelection(0)
                ctrl.Show()
                text_ctrl.Show()
            elif always_show:
                ctrl.Clear()
                ctrl.Value = "No measurements available"
            else:
                ctrl.Hide()
                ctrl.Clear()
                text_ctrl.Hide()

        set_up_combobox(category_ctrl, category_text_ctrl, categories, category, True)
        set_up_combobox(feature_ctrl, feature_text_ctrl, feature_names, feature_name)
        #
        # The object combo-box might have image choices
        #
        if len(object_names) > 0:
            if len(image_names) > 0:
                object_text_ctrl.Label = "Image or Objects:"
                object_names += image_names
            else:
                object_text_ctrl.Label = "Objects:"
        else:
            object_text_ctrl.Label = "Image:"
            object_names = image_names
        if object_name is None:
            object_name = image_name
        set_up_combobox(object_ctrl, object_text_ctrl, object_names, object_name)
        set_up_combobox(scale_ctrl, scale_text_ctrl, scales, scale)
        return panel

    def make_html_control(self, v, control):
        from cellprofiler.gui.html.htmlwindow import HtmlClickableWindow

        if control is None:
            control = HtmlClickableWindow(
                self.module_panel, -1, name=edit_control_name(v)
            )
            if v.size is not None:
                unit = float(wx.SystemSettings.GetMetric(wx.SYS_CAPTION_Y))
                if unit == -1:
                    unit = 32.0
                control.SetMinSize((v.size[0] * unit, v.size[1] * unit))
        control.SetPage(v.content)
        return control

    def make_help_control(self, content, title="Help", name=wx.ButtonNameStr):
        control = wx.Button(self.__module_panel, -1, "?", (0, 0), (30, -1), name=name)

        def callback(event):
            dialog = HTMLDialog(
                self.__module_panel, title, rst_to_html_fragment(content),
            )
            dialog.CentreOnParent()
            dialog.Show()

        control.Bind(wx.EVT_BUTTON, callback, control)
        return control
    class CornerButtonGrid(wx.grid.Grid, wxglr.GridWithLabelRenderersMixin):
        def __init__(self, *args, **kwargs):
            kwargs = kwargs.copy()
            if "fn_clicked" in kwargs:
                fn_clicked = kwargs.pop("fn_clicked")
            else:
                fn_clicked = None
            label = kwargs.pop("label", "Update")
            tooltip = kwargs.pop("tooltip", "Update this table")
            wx.grid.Grid.__init__(self, *args, **kwargs)
            wxglr.GridWithLabelRenderersMixin.__init__(self)
            self._corner_label_renderer = CornerLabelRenderer(self, fn_clicked, tooltip=tooltip, label=label)
            self.SetCornerLabelRenderer(self._corner_label_renderer)
            self.SetDefaultRowLabelRenderer(RowLabelRenderer())
            self.SetDefaultColLabelRenderer(ColLabelRenderer())
            self.sort_reverse = False
            self.Bind(wx.grid.EVT_GRID_COL_SORT, self.sort_cols)

        @property
        def fn_clicked(self):
            return self._corner_label_renderer.fn_clicked

        @fn_clicked.setter
        def fn_clicked(self, value):
            self._corner_label_renderer.fn_clicked = value

        @property
        def tooltip(self):
            return self._corner_label_renderer.tooltip

        @tooltip.setter
        def tooltip(self, value):
            self._corner_label_renderer.tooltip = value

        @property
        def label(self):
            return self._corner_label_renderer.label

        @label.setter
        def label(self, value):
            self._corner_label_renderer.label = value

        def sort_cols(self, event):
            if len(self.GetSelectedCols()) != 1:
                return
            tgtcolumn = event.GetCol()
            if self.GetSortingColumn() != tgtcolumn:
                self.sort_reverse = False
            else:
                self.sort_reverse = not self.sort_reverse
            tab = self.GetTable()
            tab.v.data.sort(
                key=lambda thedata: thedata[tgtcolumn] if thedata[tgtcolumn] is not None else "",
                reverse=self.sort_reverse
            )
            self.SetSortingColumn(tgtcolumn)
            self.ClearSelection()

    def make_table_control(self, v, control):
        if control is None:
            control = wx.lib.resizewidget.ResizeWidget(
                self.module_panel, name=edit_control_name(v)
            )

            if v.corner_button is None:
                grid = wx.grid.Grid(control, name=grid_control_name(v))
            else:
                grid = self.CornerButtonGrid(
                    control, name=grid_control_name(v), **v.corner_button
                )
            data_table = TableController(v)
            grid.SetTable(data_table)
            grid.Table.bind_to_grid(grid)
        else:
            grid = control.FindWindowByName(grid_control_name(v))
            grid.Table.update_grid()
        grid.ForceRefresh()
        grid.SetInitialSize(v.min_size)
        control.AdjustToSize(
            (
                v.min_size[0] + wx.lib.resizewidget.RW_THICKNESS,
                v.min_size[1] + wx.lib.resizewidget.RW_THICKNESS,
            )
        )
        return control

    def add_listener(self, listener):
        self.__listeners.append(listener)

    def remove_listener(self, listener):
        self.__listeners.remove(listener)

    def notify(self, event):
        self.__inside_notify = True
        try:
            for listener in self.__listeners:
                listener(self, event)
        finally:
            self.__inside_notify = False

    def __on_column_sized(self, event):
        self.__module_panel.GetTopLevelParent().Layout()

    def __on_radiobox_change(self, event, setting, control):
        if not self.__handle_change:
            return

        setting.on_event_fired(control.GetStringSelection() == "Yes")

        self.on_value_change(setting, control, control.GetStringSelection(), event)

        self.__module.on_setting_changed(setting, self.__pipeline)

    def __on_combobox_change(self, event, setting, control):
        if not self.__handle_change:
            return
        self.on_value_change(setting, control, control.GetValue(), event)

    def __on_checklistbox_change(self, event, setting, control):
        if not self.__handle_change:
            return
        if hasattr(event, "refresh_now"):
            timeout = None
        else:
            timeout = CHECK_TIMEOUT_SEC * 1000
        self.on_value_change(
            setting,
            control,
            control.GetChecked(),
            event,
            timeout=timeout,
        )

    def __on_multichoice_change(self, event, setting, control):
        if not self.__handle_change:
            return

        proposed_value = ",".join([control.Items[i] for i in control.Selections])
        self.on_value_change(setting, control, proposed_value, event)

    def __on_cell_change(self, event, setting, control, timeout=True):
        if not self.__handle_change:
            return
        proposed_value = str(control.GetValue())
        timeout_sec = EDIT_TIMEOUT_SEC * 1000 if timeout else False
        self.on_value_change(setting, control, proposed_value, event, timeout_sec)

    def on_value_change(self, setting, control, proposed_value, event, timeout=None):
        """Handle a change in value to a setting

        setting - the setting that changed
        control - the WX control whose UI signalled the change
        proposed_value - the proposed new value for the setting
        event - the UI event signalling the change
        timeout - None = reset view immediately, False = don't reset view
                  otherwise the # of milliseconds to wait before
                  refresh.
        """
        setting_edited_event = SettingEditedEvent(
            setting, self.__module, proposed_value, event
        )
        self.notify(setting_edited_event)
        if timeout is None:
            self.reset_view()  # use the default timeout
        elif timeout is not False:
            self.reset_view(timeout)

    @staticmethod
    def fit_ctrl(ctrl):
        """Fit the control to its text size"""
        width, height = ctrl.GetTextExtent(ctrl.Value + "MM")
        ctrl.SetSizeHints(wx.Size(width, -1))
        ctrl.Parent.Fit()

    def __on_min_change(self, event, setting, control):
        if not self.__handle_change:
            return
        old_value = str(setting)
        proposed_value = setting.compose_min_text(control.Value)
        setting_edited_event = SettingEditedEvent(
            setting, self.__module, proposed_value, event
        )
        self.notify(setting_edited_event)
        self.fit_ctrl(control)

    def __on_max_change(self, event, setting, control):
        if not self.__handle_change:
            return
        old_value = str(setting)
        proposed_value = setting.compose_max_text(control.Value)
        setting_edited_event = SettingEditedEvent(
            setting, self.__module, proposed_value, event
        )
        self.notify(setting_edited_event)
        self.fit_ctrl(control)

    def request_validation(self, module=None):
        """Request validation of the current module in its current state"""
        if module is None:
            module = self.__module
        if self.__validation_request is not None:
            self.__validation_request.cancel()
        self.__validation_request = ValidationRequestController(
            self.__pipeline, module, self.on_validation
        )
        request_module_validation(self.__validation_request)

    def __on_pipeline_event(self, pipeline, event):
        if isinstance(event, PipelineCleared) or isinstance(event, PipelineLoaded):
            if self.__module not in self.__pipeline.modules(False):
                self.clear_selection()
        elif isinstance(event, ModuleEdited):
            if (
                not self.__inside_notify
                and self.__module is not None
                and self.__module.module_num == event.module_num
            ):
                self.reset_view()
            if (
                self.__module is not None
                and self.__module.module_num == event.module_num
            ):
                self.request_validation()
        elif isinstance(event, ModuleRemoved):
            if (
                self.__module is not None
                and event.module_num == self.__module.module_num
            ):
                self.clear_selection()

    def __on_workspace_event(self, event):
        import cellprofiler.gui._workspace_model as cpw

        if isinstance(
            event,
            (cpw.Workspace.WorkspaceLoadedEvent, cpw.Workspace.WorkspaceCreatedEvent,),
        ):
            # Detach and reattach the current module to get it reacclimated
            # to the current workspace and reselect
            if self.__module is not None:
                self.__module.on_deactivated()
                self.__module.on_activated(self.__workspace)
                self.do_reset()

    def __on_do_something(self, event, setting):
        setting.on_event_fired()
        setting_edited_event = SettingEditedEvent(setting, self.__module, None, event)
        self.notify(setting_edited_event)
        self.__module.on_setting_changed(setting, self.__pipeline)
        self.reset_view()

    def on_validation(self, setting_idx, message, level):
        default_fg_color = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT)
        default_bg_color = get_background_color()
        if not self.__module:  # defensive coding, in case the module was deleted
            return

        visible_settings = self.__module.visible_settings()
        bad_setting = None
        if setting_idx is not None:
            # an error was detected by the validation thread.  The pipeline may
            # have changed in the meantime, so we revalidate here to make sure
            # what we display is up to date.
            if setting_idx >= len(visible_settings):
                return  # obviously changed, don't update display
            try:
                # fast-path: check the reported setting first
                level = logging.ERROR
                visible_settings[setting_idx].test_valid(self.__pipeline)
                self.__module.test_valid(self.__pipeline)
                level = logging.WARNING
                self.__module.test_module_warnings(self.__pipeline)
            except ValidationError as instance:
                message = instance.message
                bad_setting = instance.get_setting()
                LOGGER.debug(f'Bad setting in Module "{self.__module.module_name}", setting "{bad_setting.text}": {message}')
        # update settings' foreground/background
        try:
            for setting in visible_settings:
                self.set_tool_tip(
                    setting, message if (setting is bad_setting) else None
                )
                static_text_name = text_control_name(setting)
                static_text = self.__module_panel.FindWindowByName(static_text_name)
                if static_text is not None:
                    desired_fg, desired_bg = default_fg_color, default_bg_color
                    if setting is bad_setting:
                        if level == logging.ERROR:
                            desired_fg = get_error_color()
                        elif level == logging.WARNING:
                            desired_bg = WARNING_COLOR
        except Exception:
            LOGGER.debug(
                "Caught bare exception in ModuleView.on_validate()", exc_info=True
            )
            pass

    def set_tool_tip(self, setting, message):
        """Set the tool tip for a setting to display a message

        setting - set the tooltip for this setting

        message - message to display or None for no tool tip
        """
        control_name = edit_control_name(setting)
        control = self.__module_panel.FindWindowByName(control_name)
        if message is None:

            def set_tool_tip(ctrl):
                ctrl.SetToolTip(None)

        else:

            def set_tool_tip(ctrl, message=message):
                ctrl.SetToolTip(message)

        if control is not None:
            set_tool_tip(control)
            for child in control.GetChildren():
                set_tool_tip(child)
        static_text_name = text_control_name(setting)
        static_text = self.__module_panel.FindWindowByName(static_text_name)
        if static_text is not None:
            set_tool_tip(static_text)

    def reset_view(self, refresh_delay=250):
        """Redo all of the controls after something has changed

        refresh_delay - wait this many ms before refreshing the display
        """
        if self.__module is None:
            return
        if self.refresh_pending:
            return
        self.refresh_pending = True
        wx.CallLater(refresh_delay, self.do_reset)

    def do_reset(self):
        self.refresh_pending = False
        focus_control = wx.Window.FindFocus()
        if focus_control is not None:
            focus_name = focus_control.GetName()
        else:
            focus_name = None
        if self.__module is None:
            return
        self.set_selection(self.__module.module_num)
        if focus_name:
            focus_control = self.module_panel.FindWindowByName(focus_name)
            if focus_control:
                focus_control.SetFocus()
                if isinstance(focus_control, wx.TextCtrl):
                    focus_control.SetSelection(
                        focus_control.GetLastPosition(), focus_control.GetLastPosition()
                    )

    def disable(self):
        self.__module_panel.Disable()

    def enable(self):
        self.__module_panel.Enable()

    def get_max_width(self):
        sizer = self.__sizer
        return (
            sizer.calc_max_text_width()
            + sizer.calc_edit_size()[0]
            + sizer.calc_help_size()[0]
        )
