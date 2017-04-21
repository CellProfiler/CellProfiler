# coding=utf-8
"""ModuleView.py - implements a view on a module
"""

import cellprofiler.gui.pipeline
import cellprofiler.icons
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.setting
import cornerbuttonmixin
import htmldialog
import logging
import matplotlib.cm
import metadatactrl
import namesubscriber
import numpy
import os
import Queue
import regexp_editor
import stat
import sys
import threading
import time
import treecheckboxdialog
import uuid
import weakref
import wx
import wx.grid
import wx.lib.colourselect
import wx.lib.rcsizer
import wx.lib.resizewidget
import wx.lib.scrolledpanel

logger = logging.getLogger(__name__)

WARNING_COLOR = wx.Colour(224, 224, 0, 255)
RANGE_TEXT_WIDTH = 40  # number of pixels in a range text box TO_DO - calculate it
ABSOLUTE = "Absolute"
FROM_EDGE = "From edge"

CHECK_TIMEOUT_SEC = 2
EDIT_TIMEOUT_SEC = 5

# validation queue priorities, to allow faster updates for the displayed module
PRI_VALIDATE_DISPLAY = 0
PRI_VALIDATE_BACKGROUND = 1


class SettingEditedEvent(object):
    """Represents an attempt by the user to edit a setting

    """

    def __init__(self, setting, module, proposed_value, event):
        self.__module = module
        self.__setting = setting
        self.__proposed_value = proposed_value
        self.__event = event
        self.__accept_change = True

    def get_setting(self):
        """Return the setting being edited

        """
        return self.__setting

    def get_proposed_value(self):
        """Return the value proposed by the user

        """
        return self.__proposed_value

    def get_module(self):
        """Get the module holding the setting"""
        return self.__module

    def cancel(self):
        self.__accept_change = False

    def accept_change(self):
        return self.__accept_change

    def ui_event(self):
        """The event from the UI that triggered the edit

        """
        return self.__event


def text_control_name(v):
    """Return the name of a setting's text control
    v - the setting
    The text control name is built using the setting's key
    """
    return "%s_text" % (str(v.key()))


def button_control_name(v, idx=None):
    """Return the name of a setting's button

    v - the setting

    idx - if present, the index of one of several buttons for the setting
    """
    if idx is None:
        return "%s_button" % (str(v.key()))
    else:
        return "%s_button_%d" % (str(v.key()), idx)


def edit_control_name(v):
    """Return the name of a setting's edit control
    v - the setting
    The edit control name is built using the setting's key
    """
    return str(v.key())


def min_control_name(v):
    """For a range, return the control that sets the minimum value
    v - the setting
    """
    return "%s_min" % (str(v.key()))


def max_control_name(v):
    """For a range, return the control that sets the maximum value
    v - the setting
    """
    return "%s_max" % (str(v.key()))


def absrel_control_name(v):
    """For a range, return the control that chooses between absolute and relative

    v - the setting
    Absolute - far coordinate is an absolute value
    From edge - far coordinate is a distance from the far edge
    """
    return "%s_absrel" % (str(v.key()))


def x_control_name(v):
    """For coordinates, return the control that sets the x value
    v - the setting
    """
    return "%s_x" % (str(v.key()))


def y_control_name(v):
    """For coordinates, return the control that sets the y value
    v - the setting
    """
    return "%s_y" % (str(v.key()))


def category_control_name(v):
    """For measurements, return the control that sets the measurement category

    v - the setting
    """
    return "%s_category" % (str(v.key()))


def category_text_control_name(v):
    return "%s_category_text" % (str(v.key()))


def feature_control_name(v):
    """For measurements, return the control that sets the feature name

    v - the setting
    """
    return "%s_feature" % (str(v.key()))


def feature_text_control_name(v):
    return "%s_feature_text" % (str(v.key()))


def image_control_name(v):
    """For measurements, return the control that sets the image name

    v - the setting
    """
    return "%s_image" % (str(v.key()))


def image_text_control_name(v):
    return "%s_image_text" % (str(v.key()))


def object_control_name(v):
    """For measurements, return the control that sets the object name

    v - the setting
    """
    return "%s_object" % (str(v.key()))


def object_text_control_name(v):
    return "%s_object_text" % (str(v.key()))


def scale_control_name(v):
    """For measurements, return the control that sets the measurement scale

    v - the setting
    """
    return "%s_scale" % (str(v.key()))


def scale_text_ctrl_name(v):
    return "%s_scale_text" % (str(v.key()))


def combobox_ctrl_name(v):
    return "%s_combobox" % (str(v.key()))


def colorbar_ctrl_name(v):
    return "%s_colorbar" % (str(v.key()))


def help_ctrl_name(v):
    return "%s_help" % str(v.key())


def subedit_control_name(v):
    return "%s_subedit" % str(v.key())


def grid_control_name(v):
    return "%s_grid" % str(v.key())


def custom_label_name(v):
    return "%s_customlabel" % str(v.key())


def folder_label_name(v):
    return "%s_folderlabel" % str(v.key())


def encode_label(text):
    """Encode text escapes for the static control and button labels

    The ampersand (&) needs to be encoded as && for wx.StaticText
    and wx.Button in order to keep it from signifying an accelerator.
    """
    return text.replace('&', '&&')


class ModuleView(object):
    """The module view implements a view on CellProfiler.Module

    The module view implements a view on CellProfiler.Module. The view consists
    of a table composed of one row per setting. The first column of the table
    has the explanatory text and the second has a control which
    gives the ui for editing the setting.
    """

    def __init__(self, top_panel,
                 workspace,
                 as_datatool=False,
                 frame=None,
                 notes_panel=None):
        """Constructor

        module_panel - the top-level panel used by the view
        workspace - the current workspace
        as_datatool - True if module is being run as a data tool
        notes_panel - panel in which to construct the notes GUI
        """
        pipeline = workspace.pipeline
        self.__workspace = workspace
        self.__module = None
        self.refresh_pending = False
        self.notes_panel = notes_panel
        self.__frame = frame
        self.top_panel = top_panel
        background_color = cellprofiler.preferences.get_background_color()
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
        self.module_panel.Sizer = self.__sizer
        self.module_panel.Bind(wx.EVT_CHILD_FOCUS, self.skip_event)
        if notes_panel is not None:
            self.make_notes_gui()

        self.__pipeline = pipeline
        self.__as_datatool = as_datatool
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
        self.__workspace.add_notification_callback(
            self.__on_workspace_event)
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
                listener['notifier'].remove_listener(listener['listener'])
            self.__value_listeners = []
            self.__module = None
        self.__sizer.Reset(0)
        if self.notes_panel is not None:
            self.notes_panel.Hide()

    def get_module_settings_label(self):
        if self.__module is None:
            return "Module settings"
        return "Module settings (%s #%02d)" % (
            self.__module.module_name, self.__module.module_num)

    def hide_settings(self):
        for child in self.__module_panel.Children:
            child.Hide()

    def check_settings(self, module_name, settings):
        try:
            assert len(settings) > 0
        except:
            wx.MessageBox("Module %s.visible_settings() did not return a list!\n  value: %s" % (module_name, settings),
                          "Pipeline Error", wx.ICON_ERROR, self.__module_panel)
            settings = []
        try:
            assert all([isinstance(s, cellprofiler.setting.Setting) for s in settings])
        except:
            wx.MessageBox(
                "Module %s.visible_settings() returned something other than a list of Settings!\n  value: %s" % (
                    module_name, settings),
                "Pipeline Error", wx.ICON_ERROR, self.__module_panel)
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
            reselecting = (self.__module and
                           self.__module.id == new_module.id)
            if not reselecting:
                if self.__module is not None:
                    self.__module.on_deactivated()
                self.clear_selection()
                self.request_validation(new_module)
                try:
                    # Need to initialize some controls.
                    new_module.test_valid(self.__pipeline)
                except:
                    pass
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
            settings = self.check_settings(self.__module.module_name,
                                           self.__module.visible_settings())
            self.__sizer.Reset(len(settings), 3, False)
            sizer = self.__sizer

            #################################
            #
            # Set the module's notes
            #
            #################################
            if self.notes_panel is not None:
                self.module_notes_control.Value = u"\n".join(
                    [_.decode("utf-8") for _ in self.__module.notes])

            #################################
            #
            # Populate the GUI elements for each of the settings
            #
            #################################
            for i, v in enumerate(settings):
                if isinstance(v, cellprofiler.setting.PathListDisplay):
                    path_control = v
                    self.__frame.pipeline_controller. \
                        set_path_list_filtering(v.using_filter)
                    continue
                if isinstance(v, cellprofiler.setting.ImageSetDisplay):
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
                    static_text = wx.StaticText(self.__module_panel,
                                                -1,
                                                encode_label(v.text),
                                                style=wx.ALIGN_RIGHT,
                                                name=text_name)
                text_sizer_item = sizer.Add(static_text, 3, wx.EXPAND | wx.ALL, 2)
                if control:
                    control.Show()
                self.__static_texts.append(static_text)
                if isinstance(v, cellprofiler.setting.Binary):
                    control = self.make_binary_control(v, control_name, control)
                    flag = wx.ALIGN_LEFT
                    text_sizer_item.Flag = \
                        wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL | wx.ALL
                elif isinstance(v, cellprofiler.setting.MeasurementMultiChoice):
                    control = self.make_measurement_multichoice_control(
                        v, control_name, control)
                elif isinstance(v, cellprofiler.setting.SubdirectoryFilter):
                    control = self.make_subdirectory_filter_control(
                        v, control_name, control)
                elif isinstance(v, cellprofiler.setting.MultiChoice):
                    control = self.make_multichoice_control(v, control_name,
                                                            control)
                elif isinstance(v, cellprofiler.setting.CustomChoice):
                    control = self.make_choice_control(v, v.get_choices(),
                                                       control_name,
                                                       wx.CB_DROPDOWN,
                                                       control)
                elif isinstance(v, cellprofiler.setting.Colormap):
                    control = self.make_colormap_control(v, control_name,
                                                         control)
                elif isinstance(v, cellprofiler.setting.Choice):
                    control = self.make_choice_control(v, v.get_choices(),
                                                       control_name,
                                                       wx.CB_READONLY,
                                                       control)
                    flag = wx.ALIGN_LEFT
                elif isinstance(v, cellprofiler.setting.NameSubscriber):
                    choices = v.get_choices(self.__pipeline)
                    control = self.make_name_subscriber_control(v, choices,
                                                                control_name,
                                                                control)
                    flag = wx.ALIGN_LEFT
                elif isinstance(v, cellprofiler.setting.FigureSubscriber):
                    choices = v.get_choices(self.__pipeline)
                    control = self.make_choice_control(v, choices,
                                                       control_name,
                                                       wx.CB_DROPDOWN,
                                                       control)
                    flag = wx.ALIGN_LEFT
                elif isinstance(v, cellprofiler.setting.DoSomething):
                    if (isinstance(v, cellprofiler.setting.PathListRefreshButton) and
                                v.callback is None):
                        v.callback = \
                            self.__frame.pipeline_controller.on_update_pathlist
                    control = self.make_callback_control(v, control_name,
                                                         control)
                    flag = wx.ALIGN_LEFT
                elif isinstance(v, cellprofiler.setting.DoThings):
                    control = self.make_callback_controls(
                        v, control_name, control)
                elif isinstance(v,
                                cellprofiler.setting.IntegerOrUnboundedRange):
                    control = self.make_unbounded_range_control(v, control)
                elif isinstance(v, cellprofiler.setting.IntegerRange) or \
                        isinstance(v, cellprofiler.setting.FloatRange):
                    control = self.make_range_control(v, control)
                elif isinstance(v, cellprofiler.setting.Coordinates):
                    control = self.make_coordinates_control(v, control)
                elif isinstance(v, cellprofiler.setting.RegexpText):
                    control = self.make_regexp_control(v, control)
                elif isinstance(v, cellprofiler.setting.Measurement):
                    control = self.make_measurement_control(v, control)
                elif isinstance(v, cellprofiler.setting.Divider):
                    if control is None:
                        if v.line:
                            control = wx.StaticLine(self.__module_panel,
                                                    name=control_name)
                        else:
                            control = wx.StaticText(self.__module_panel,
                                                    name=control_name)
                    flag = wx.EXPAND | wx.ALL
                    border = 2
                elif isinstance(v, cellprofiler.setting.FilenameText):
                    control = self.make_filename_text_control(v, control)
                elif isinstance(v, cellprofiler.setting.DirectoryPath):
                    control = self.make_directory_path_control(v, control_name,
                                                               control)
                elif isinstance(v, cellprofiler.setting.Pathname):
                    control = self.make_pathname_control(v, control)
                elif isinstance(v, cellprofiler.setting.ImagePlane):
                    control = self.make_image_plane_control(v, control)
                elif isinstance(v, cellprofiler.setting.Color):
                    control = self.make_color_control(v, control_name, control)
                elif isinstance(v, cellprofiler.setting.TreeChoice):
                    control = self.make_tree_choice_control(v, control_name, control)
                    flag = wx.ALIGN_LEFT
                elif isinstance(v, cellprofiler.setting.Filter):
                    if control is not None:
                        control.filter_panel_controller.update()
                    else:
                        fc = FilterPanelController(self, v, control)
                        control = fc.panel
                        control.filter_panel_controller = fc
                elif isinstance(v, cellprofiler.setting.FileCollectionDisplay):
                    if control is not None:
                        # control.file_collection_display.update()
                        pass
                    else:
                        fcd = FileCollectionDisplayController(
                            self, v, self.__pipeline)
                        control = fcd.panel
                        fcd.panel.file_collection_display = fcd
                elif isinstance(v, cellprofiler.setting.Table):
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
                elif isinstance(v, cellprofiler.setting.HTMLText):
                    control = self.make_html_control(v, control)
                    flag = wx.EXPAND | wx.ALL
                elif isinstance(v, cellprofiler.setting.Joiner):
                    control = JoinerController.update_control(self, v)
                    flag = wx.ALIGN_LEFT
                elif isinstance(v, cellprofiler.setting.BinaryMatrix):
                    control = BinaryMatrixController.update_control(self, v)
                    flag = wx.ALIGN_LEFT
                elif isinstance(v, cellprofiler.setting.DataTypes):
                    control = DataTypeController.update_control(self, v)
                    flag = wx.ALIGN_LEFT
                elif isinstance(v, cellprofiler.setting.StructuringElement):
                    control = self.make_structuring_element_control(v, control_name, control)
                    text_sizer_item.Flag = wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL | wx.ALL
                else:
                    control = self.make_text_control(v, control_name, control)
                sizer.Add(control, 0, flag, border)
                self.__controls.append(control)
                help_name = help_ctrl_name(v)
                help_control = self.module_panel.FindWindowByName(help_name)

                if help_control is None:
                    if v.doc is None:
                        help_control = wx.StaticText(self.__module_panel,
                                                     -1, "",
                                                     name=help_name)
                    else:
                        help_control = self.make_help_control(v.doc, v.text,
                                                              name=help_name)
                else:
                    help_control.Show()
                sizer.Add(help_control, 0, wx.LEFT, 2)
        finally:
            self.__handle_change = True
            if self.__as_datatool:
                self.module_panel.Layout()
                if self.DO_FREEZE:
                    self.module_panel.Thaw()
            elif self.__frame is not None:
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
                self.module_panel,
                style=wx.TAB_TRAVERSAL,
                name=control_name
            )

            shape_control_text = wx.StaticText(
                control,
                label="Shape",
                style=wx.ALIGN_LEFT
            )

            shape_control = wx.ComboBox(
                control,
                choices=v.get_choices(),
                style=wx.CB_READONLY,
                name=shape_control_name,
                value=v.shape
            )

            def on_select_shape(event, setting=v, control=shape_control):
                setting.shape = event.GetString()
                new_value = setting.value_text
                self.on_value_change(setting, control, new_value, event)

            shape_control.Bind(wx.EVT_COMBOBOX, on_select_shape)

            size_control_text = wx.StaticText(
                control,
                label="Size",
                style=wx.ALIGN_LEFT
            )

            size_control = wx.TextCtrl(
                control,
                name=size_control_name,
                value=str(v.size)
            )

            def on_set_size(event):
                v.size = int(event.GetString())
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
            self.notes_panel, -1, style=wx.TE_MULTILINE | wx.TE_PROCESS_ENTER)
        #
        # If you call self.module_notes_control.GetFullTextExtent(),
        # you will find that the font has a descender of 4 pixels. But
        # it looks like GetFullTextExtent is not implemented everywhere
        # so I hardcode here
        #
        height = self.module_notes_control.GetTextExtent("M")[1] * 2 + 4
        height = self.module_notes_control.ClientToWindowSize(
            wx.Size(1, height))[1]
        self.module_notes_control.SetInitialSize(wx.Size(100, 100))
        notes_sizer.Add(self.module_notes_control, 1, wx.ALL, 10)

        def on_notes_changed(event):
            if not self.__handle_change:
                return
            if self.__module is not None:
                notes = self.module_notes_control.Value.encode('utf-8')
                self.__module.notes = notes.split('\n')

        self.notes_panel.Bind(wx.EVT_TEXT, on_notes_changed,
                              self.module_notes_control)

    def make_binary_control(self, v, control_name, control):
        """Make a checkbox control for a Binary setting"""
        if not control:
            control = wx.RadioBox(
                self.__module_panel,
                choices=[cellprofiler.setting.YES, cellprofiler.setting.NO],
                name=control_name)

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
            control = namesubscriber.NameSubscriberComboBox(self.__module_panel,
                                                            value=v.value,
                                                            choices=choices,
                                                            name=control_name)

            def callback(event, setting=v, control=control):
                # the NameSubscriberComboBox behaves like a combobox
                self.__on_combobox_change(event, setting, control)

            control.add_callback(callback)
        else:
            if list(choices) != list(control.Items):
                control.Items = choices
        if (getattr(v, 'has_tooltips', False) and
                v.has_tooltips and (control.Value in v.tooltips)):
            control.SetToolTip(wx.ToolTip(v.tooltips[control.Value]))
        return control

    def make_choice_control(self, v, choices, control_name, style, control):
        """Make a combo-box that shows choices

        v            - the setting
        choices      - the possible values for the setting
        control_name - assign this name to the control
        style        - one of the CB_ styles
        """
        assert isinstance(v, (cellprofiler.setting.Choice, cellprofiler.setting.FigureSubscriber))
        try:
            v.test_valid(self.__pipeline)
        except:
            pass
        if v.value not in choices and style == wx.CB_READONLY:
            choices = choices + [v.value]
        if not control:
            control = wx.ComboBox(self.__module_panel, -1, v.value,
                                  choices=choices,
                                  style=style,
                                  name=control_name)

            def callback(event, setting=v, control=control):
                self.__on_combobox_change(event, setting, control)

            self.__module_panel.Bind(wx.EVT_COMBOBOX, callback, control)
            if style == wx.CB_DROPDOWN:
                def on_cell_change(event, setting=v, control=control):
                    self.__on_cell_change(event, setting, control)

                self.__module_panel.Bind(wx.EVT_TEXT, on_cell_change, control)
        else:
            old_choices = control.Items
            if len(choices) != len(old_choices) or \
                    not all([x == y for x, y in zip(choices, old_choices)]):
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
                i_am_different = (control.Value != v.value)
            except:
                i_am_different = True
            if len(choices) > 0 and i_am_different:
                control.Value = v.value

        if (getattr(v, 'has_tooltips', False) and
                v.has_tooltips and v.tooltips.has_key(control.Value)):
            control.SetToolTip(wx.ToolTip(v.tooltips[control.Value]))
        return control

    def make_measurement_multichoice_control(self, v, control_name, control):
        """Make a button that, when pressed, launches the tree editor"""
        if control is None:
            control = wx.Button(self.module_panel, -1,
                                "Press button to select measurements")

            def on_press(event):
                d = {}
                assert isinstance(v, cellprofiler.setting.MeasurementMultiChoice)
                if len(v.choices) == 0:
                    v.populate_choices(self.__pipeline)
                #
                # Populate the tree
                #
                choices = set(v.choices)
                for choice in choices:
                    object_name, feature = v.split_choice(choice)
                    pieces = [object_name] + feature.split('_')
                    d1 = d
                    for piece in pieces:
                        if not d1.has_key(piece):
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
                    pieces = [object_name] + feature.split('_')
                    d1 = d
                    for piece in pieces:
                        if not d1.has_key(piece):
                            break
                        d1 = d1[piece]
                    d1[None] = True

                #
                # Backtrack recursively through tree to get branch states
                #
                def get_state(d):
                    leaf_state = d[None]
                    for subtree_key in [x for x in d.keys() if x is not None]:
                        subtree_state = get_state(d[subtree_key])
                        if leaf_state == 0:
                            leaf_state = subtree_state
                        elif leaf_state != subtree_state:
                            leaf_state = None
                    d[None] = leaf_state
                    return leaf_state

                get_state(d)
                dlg = treecheckboxdialog.TreeCheckboxDialog(self.module_panel, d, size=(480, 480))
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
                        for key in [x for x in d.keys() if x is not None]:
                            if prefix is None:
                                sub_prefix = key
                            else:
                                sub_prefix = '_'.join((prefix, key))
                            result += collect_state(object_name, sub_prefix,
                                                    d[key])
                        return result

                    selections = []
                    for object_name in [x for x in d.keys() if x is not None]:
                        selections += collect_state(object_name, None,
                                                    d[object_name])
                    proposed_value = v.get_value_string(selections)
                    setting_edited_event = SettingEditedEvent(v, self.__module,
                                                              proposed_value,
                                                              event)
                    self.notify(setting_edited_event)
                    self.reset_view()

            control.Bind(wx.EVT_BUTTON, on_press)
        else:
            control.Show()
        return control

    def make_subdirectory_filter_control(self, v, control_name, control):
        if control is None:
            control = wx.Button(self.module_panel, -1,
                                "Press to select folders")

            def on_press(event):
                assert isinstance(v, cellprofiler.setting.SubdirectoryFilter)

                root = v.directory_path.get_absolute_path()
                self.module_panel.SetCursor(wx.StockCursor(wx.CURSOR_WAIT))
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
                                d[dirname] = lambda dirpath=dirpath: fn_populate(dirpath)
                        except:
                            print "Warning: failed to list directory %s" % root
                        return d

                    d = fn_populate(root)
                    selections = v.get_selections()

                    def populate_selection(d, selection, root):
                        s0 = selection[0]
                        if not d.has_key(s0):
                            d[s0] = fn_populate(os.path.join(root, s0))
                        elif hasattr(d[s0], "__call__"):
                            d[s0] = d[s0]()
                        if len(selection) == 1:
                            d[s0][None] = False
                        else:
                            if d[s0][None] is not False:
                                populate_selection(d[s0], selection[1:],
                                                   os.path.join(root, s0))
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

                dlg = treecheckboxdialog.TreeCheckboxDialog(self.module_panel, d, size=(320, 480))
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
                        for key in d.keys():
                            if key is None:
                                continue
                            result += collect_state(os.path.join(prefix, key),
                                                    d[key])
                        return result

                    selections = []
                    for object_name in [x for x in d.keys() if x is not None]:
                        selections += collect_state(object_name, d[object_name])
                    proposed_value = v.get_value_string(selections)
                    setting_edited_event = SettingEditedEvent(v, self.__module,
                                                              proposed_value,
                                                              event)
                    self.notify(setting_edited_event)
                    self.reset_view()

            control.Bind(wx.EVT_BUTTON, on_press)
        else:
            control.Show()
        return control

    def make_multichoice_control(self, v, control_name, control):
        selections = v.selections
        assert isinstance(v, cellprofiler.setting.MultiChoice)
        if isinstance(v, cellprofiler.setting.SubscriberMultiChoice):
            # Get the choices from the providers
            v.load_choices(self.__pipeline)
        choices = v.choices + [selection for selection in selections
                               if selection not in v.choices]
        if not control:
            control = wx.ListBox(self.__module_panel, -1, choices=choices,
                                 style=wx.LB_EXTENDED,
                                 name=control_name)
            for selection in selections:
                index = choices.index(selection)
                control.SetSelection(index)
                if selection not in v.choices:
                    control.SetItemForegroundColour(
                        index, cellprofiler.preferences.get_error_color())

            def callback(event, setting=v, control=control):
                self.__on_multichoice_change(event, setting, control)

            self.__module_panel.Bind(wx.EVT_LISTBOX, callback, control)
        else:
            old_choices = control.Items
            if (len(choices) != len(old_choices) or
                    not all([x == y for x, y in zip(choices, old_choices)])):
                control.Items = choices
            for i in range(len(choices)):
                if control.IsSelected(i):
                    if choices[i] not in selections:
                        control.Deselect(i)
                elif choices[i] in selections:
                    control.Select(i)
                    if choices[i] not in v.choices:
                        control.SetItemForegroundColour(
                            i, cellprofiler.preferences.get_error_color())
        return control

    def make_colormap_control(self, v, control_name, control):
        """Make a combo-box that shows colormap choices
        v            - the setting
        choices      - the possible values for the setting
        control_name - assign this name to the control
        style        - one of the CB_ styles
        """
        try:
            if v.value == cellprofiler.setting.DEFAULT:
                cmap_name = cellprofiler.preferences.get_default_colormap()
            else:
                cmap_name = v.value
            cm = matplotlib.cm.get_cmap(cmap_name)
            sm = matplotlib.cm.ScalarMappable(cmap=cm)
            i, j = numpy.mgrid[0:12, 0:128]
            if cm.N < 128:
                j *= int((cm.N + 128) / 128)
            image = (sm.to_rgba(j) * 255).astype(numpy.uint8)
            bitmap = wx.BitmapFromBufferRGBA(128, 12, image.tostring())
        except:
            logger.warning("Failed to create the %s colorbar" % cmap_name)
            bitmap = None
        if not control:
            control = wx.Panel(self.__module_panel, -1,
                               name=control_name)
            sizer = wx.BoxSizer(wx.VERTICAL)
            control.SetSizer(sizer)
            colorbar = wx.StaticBitmap(control, -1,
                                       name=colorbar_ctrl_name(v))
            if not bitmap is None:
                colorbar.SetBitmap(bitmap)
            sizer.Add(colorbar, 0, wx.EXPAND | wx.BOTTOM, 2)

            combo = wx.ComboBox(control, -1, v.value,
                                choices=v.choices,
                                style=wx.CB_READONLY,
                                name=combobox_ctrl_name(v))
            sizer.Add(combo, 1, wx.EXPAND)

            def callback(event, setting=v, control=combo):
                self.__on_combobox_change(event, setting, combo)

            self.__module_panel.Bind(wx.EVT_COMBOBOX, callback, combo)
        else:
            combo = control.FindWindowByName(combobox_ctrl_name(v))
            colorbar = control.FindWindowByName(colorbar_ctrl_name(v))
            old_choices = combo.Items
            if len(v.choices) != len(old_choices) or \
                    not all([x == y for x, y in zip(v.choices, old_choices)]):
                combo.Items = v.choices
            if combo.Value != v.value:
                combo.Value = v.value
            if not bitmap is None:
                colorbar.SetBitmap(bitmap)
        return control

    def make_color_control(self, v, control_name, control):
        try:
            color = wx.Colour()
            color.SetFromName(v.value)
        except:
            color = wx.BLACK
            if (not hasattr(control, "bad_color_name") or
                        control.bad_color_name != v.value):
                logger.warn("Failed to set color to %s" % v.value)
                control.bad_color_name = v.value
        if control is None:
            control = wx.lib.colourselect.ColourSelect(
                self.__module_panel,
                colour=color)
            control.SetName(control_name)

            def on_press(event, v=v, control=control):
                proposed_value = control.GetColour().GetAsString(
                    wx.C2S_NAME | wx.C2S_HTML_SYNTAX)
                setting_edited_event = SettingEditedEvent(
                    v, self.__module, proposed_value, event)
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
            text_width, text_height = control.GetTextExtent(new_label)
            gap = 4
            drop_width = wx.SystemSettings.GetMetric(wx.SYS_VSCROLL_ARROW_X)
            drop_height = wx.SystemSettings.GetMetric(wx.SYS_VSCROLL_ARROW_Y)
            width = text_width + 2 * gap + drop_width
            height = max(text_height, drop_height) + 4
            bitmap = wx.EmptyBitmap(width, height)
            dc = wx.MemoryDC(bitmap)
            dc.Font = control.Font
            brush = wx.Brush(control.BackgroundColour)
            dc.Background = brush
            dc.Clear()
            wx.RendererNative.Get().DrawComboBox(
                control, dc, wx.Rect(0, 0, width, height), flags)
            dc.DrawText(new_label, 2, 2)
            return bitmap

        if control is None:
            control = wx.BitmapButton(self.module_panel,
                                      style=wx.BU_EXACTFIT)
            control.label_text = None

            def on_press(event, v=v, control=control):
                id_dict = {}

                def on_event(event, v=v, control=control, id_dict=id_dict):
                    new_path = v.encode_path_parts(id_dict[event.GetId()])
                    self.on_value_change(v, control, new_path, event)

                def make_menu(tree, id_dict=id_dict, path=None):
                    if path is None:
                        path = []
                    menu = wx.Menu()
                    for node in tree:
                        text, subtree = node[:2]
                        subpath = path + [text]
                        if v.fn_is_leaf(node):
                            item = menu.Append(-1, text)
                            id_dict[item.GetId()] = subpath
                            if wx.VERSION >= (2, 9) and \
                                            sys.platform != 'win32':
                                wx.EVT_MENU(menu, item.GetId(), on_event)
                        if subtree is not None and len(subtree) > 0:
                            submenu = make_menu(subtree, path=subpath)
                            menu.AppendMenu(-1, text, submenu)
                    return menu

                menu = make_menu(v.get_tree())
                assert isinstance(control, wx.Window)
                if wx.VERSION < (2, 9) or sys.platform == 'win32':
                    menu.Bind(wx.EVT_MENU, on_event)
                control.PopupMenuXY(menu, 0, control.GetSize()[1])
                menu.Destroy()

            control.Bind(wx.EVT_BUTTON, on_press)
        old_label = control.label_text
        if old_label != new_label:
            control.label_text = new_label
            for getter, setter, flags in (
                    (control.GetBitmapLabel, control.SetBitmapLabel, 0),
                    (control.GetBitmapFocus, control.SetBitmapFocus,
                     wx.CONTROL_FOCUSED),
                    (control.GetBitmapSelected, control.SetBitmapSelected,
                     wx.CONTROL_SELECTED)):
                old_bitmap = getter()
                setter(make_bitmap(control, flags))
                if old_bitmap is not None:
                    old_bitmap.Destroy()
        return control

    def make_callback_control(self, v, control_name, control):
        """Make a control that calls back using the callback buried in the setting"""
        if not control:
            control = wx.Button(self.module_panel, -1,
                                v.label, name=control_name)

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
        assert isinstance(v, cellprofiler.setting.DoThings)
        if not control:
            control = wx.Panel(self.module_panel, name=control_name)
            control.Sizer = wx.BoxSizer(wx.HORIZONTAL)
            for i in range(v.count):
                if i != 0:
                    control.Sizer.AddSpacer(2)
                button = wx.Button(control,
                                   name=button_control_name(v, i))
                control.Sizer.Add(button, 0, wx.ALIGN_LEFT)

                def callback(event, index=i):
                    v.on_event_fired(index)
                    setting_edited_event = SettingEditedEvent(
                        v, self.__module, None, event)
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
            panel = wx.Panel(self.__module_panel,
                             -1,
                             name=edit_control_name(v))
            control = panel
            sizer = wx.BoxSizer(wx.HORIZONTAL)
            panel.SetSizer(sizer)
            text_ctrl = wx.TextCtrl(panel, -1, str(v.value),
                                    name=text_control_name(v))
            sizer.Add(text_ctrl, 1, wx.EXPAND | wx.RIGHT, 1)
            bitmap = wx.ArtProvider.GetBitmap(wx.ART_FIND, wx.ART_TOOLBAR, (16, 16))
            bitmap_button = wx.BitmapButton(panel, bitmap=bitmap,
                                            name=button_control_name(v))
            sizer.Add(bitmap_button, 0, wx.EXPAND)

            def on_cell_change(event, setting=v, control=text_ctrl):
                self.__on_cell_change(event, setting, control)

            def on_button_pressed(event, setting=v, control=text_ctrl):
                #
                # Find a file in the image directory
                #
                filename = "plateA-2008-08-06_A12_s1_w1_[89A882DE-E675-4C12-9F8E-46C9976C4ABE].tif"
                try:
                    if setting.get_example_fn is None:
                        path = cellprofiler.preferences.get_default_image_directory()
                        filenames = [x for x in os.listdir(path)
                                     if x.find('.') != -1 and
                                     os.path.splitext(x)[1].upper() in
                                     ('.TIF', '.JPG', '.PNG', '.BMP')]
                        if len(filenames):
                            filename = filenames[0]
                    else:
                        filename = setting.get_example_fn()
                except:
                    pass

                if v.guess == cellprofiler.setting.RegexpText.GUESS_FOLDER:
                    guesses = regexp_editor.RE_FOLDER_GUESSES
                else:
                    guesses = regexp_editor.RE_FILENAME_GUESSES

                new_value = regexp_editor.edit_regexp(panel, control.Value, filename, guesses)
                if new_value:
                    control.Value = new_value
                    self.__on_cell_change(event, setting, control)

            def on_kill_focus(event, setting=v, control=text_ctrl):
                if self.__module is not None:
                    self.set_selection(self.__module.module_num)

            self.__module_panel.Bind(wx.EVT_TEXT, on_cell_change, text_ctrl)
            self.__module_panel.Bind(wx.EVT_BUTTON, on_button_pressed, bitmap_button)
            #
            # http://www.velocityreviews.com/forums/t359823-textctrl-focus-events-in-wxwidgets.html
            # explains why bind is to control itself
            #
            text_ctrl.Bind(wx.EVT_KILL_FOCUS, on_kill_focus)
        else:
            text_control = control.FindWindowByName(text_control_name(v))
            if v.value != text_control.Value:
                text_control.Value = v.value
        return control

    def make_filename_text_control(self, v, control):
        """Make a filename text control"""
        edit_name = subedit_control_name(v)
        control_name = edit_control_name(v)
        button_name = button_control_name(v)
        if control is None:
            control = wx.Panel(self.module_panel, -1,
                               name=control_name)
            sizer = wx.BoxSizer(wx.HORIZONTAL)
            control.SetSizer(sizer)
            if v.metadata_display:
                edit_control = metadatactrl.MetadataControl(
                    self.__pipeline,
                    self.__module,
                    control,
                    value=v.value,
                    name=edit_name)
            else:
                edit_control = wx.TextCtrl(control, -1, str(v),
                                           name=edit_name)
            sizer.Add(edit_control, 1, wx.ALIGN_LEFT | wx.ALIGN_TOP)

            def on_cell_change(event, setting=v, control=edit_control):
                self.__on_cell_change(event, setting, control)

            self.__module_panel.Bind(wx.EVT_TEXT, on_cell_change, edit_control)

            bitmap = wx.ArtProvider.GetBitmap(wx.ART_FILE_OPEN,
                                              wx.ART_BUTTON, (16, 16))
            button_control = wx.BitmapButton(control, bitmap=bitmap,
                                             name=button_name)

            def on_press(event):
                """Open a file browser"""
                if v.mode == cellprofiler.setting.FilenameText.MODE_OPEN:
                    mode = wx.FD_OPEN
                elif v.mode == cellprofiler.setting.FilenameText.MODE_APPEND:
                    mode = wx.FD_SAVE
                else:
                    mode = wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
                dlg = wx.FileDialog(control, v.browse_msg, style=mode)
                if v.get_directory_fn is not None:
                    dlg.Directory = v.get_directory_fn()
                if v.exts is not None:
                    dlg.Wildcard = "|".join(["|".join(tuple(x)) for x in v.exts])
                if dlg.ShowModal() == wx.ID_OK:
                    if v.set_directory_fn is not None:
                        v.set_directory_fn(dlg.Directory)
                    v.value = dlg.Filename
                    setting_edited_event = SettingEditedEvent(
                        v, self.__module, v.value, event)
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
        assert isinstance(v, cellprofiler.setting.DirectoryPath)
        dir_ctrl_name = combobox_ctrl_name(v)
        custom_ctrl_name = subedit_control_name(v)
        custom_ctrl_label_name = custom_label_name(v)
        browse_ctrl_name = button_control_name(v)
        folder_label_ctrl_name = folder_label_name(v)

        if control is None:
            control = wx.Panel(self.module_panel,
                               style=wx.TAB_TRAVERSAL,
                               name=control_name)
            sizer = wx.BoxSizer(wx.VERTICAL)
            control.SetSizer(sizer)
            choice_sizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(choice_sizer, 0, wx.ALIGN_TOP | wx.ALIGN_LEFT)
            dir_ctrl = wx.Choice(control, choices=v.dir_choices,
                                 name=dir_ctrl_name)
            choice_sizer.Add(dir_ctrl, 0, wx.ALIGN_LEFT | wx.BOTTOM, 2)
            choice_sizer.AddSpacer(3)
            folder_label = wx.StaticText(
                control, name=folder_label_ctrl_name)
            choice_sizer.Add(folder_label, 0, wx.ALIGN_CENTER_VERTICAL)

            custom_sizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(custom_sizer, 1, wx.EXPAND)
            custom_label = wx.StaticText(control, name=custom_ctrl_label_name)
            custom_sizer.Add(custom_label, 0, wx.ALIGN_CENTER_VERTICAL)
            if v.allow_metadata:
                custom_ctrl = metadatactrl.MetadataControl(self.__pipeline,
                                                           self.__module,
                                                           control, value=v.custom_path,
                                                           name=custom_ctrl_name)
            else:
                custom_ctrl = wx.TextCtrl(control, -1, v.custom_path,
                                          name=custom_ctrl_name)
            custom_sizer.Add(custom_ctrl, 1, wx.ALIGN_CENTER_VERTICAL)
            browse_bitmap = wx.ArtProvider.GetBitmap(wx.ART_FOLDER,
                                                     wx.ART_CMN_DIALOG,
                                                     (16, 16))
            browse_ctrl = wx.BitmapButton(control, bitmap=browse_bitmap,
                                          name=browse_ctrl_name)
            custom_sizer.Add(browse_ctrl, 0, wx.ALIGN_CENTER | wx.LEFT, 2)

            def on_dir_choice_change(event, v=v, dir_ctrl=dir_ctrl):
                """Handle a change to the directory choice combobox"""
                if not self.__handle_change:
                    return
                proposed_value = v.join_string(dir_choice=dir_ctrl.StringSelection)
                setting_edited_event = SettingEditedEvent(
                    v, self.__module, proposed_value, event)
                self.notify(setting_edited_event)
                self.reset_view()

            def on_custom_path_change(event, v=v, custom_ctrl=custom_ctrl):
                """Handle a change to the custom path"""
                if not self.__handle_change:
                    return
                proposed_value = v.join_string(custom_path=custom_ctrl.Value)
                setting_edited_event = SettingEditedEvent(
                    v, self.__module, proposed_value, event)
                self.notify(setting_edited_event)
                self.reset_view(1000)

            def on_browse_pressed(event, v=v, dir_ctrl=dir_ctrl,
                                  custom_ctrl=custom_ctrl):
                """Handle browse button pressed"""
                dlg = wx.DirDialog(self.module_panel,
                                   v.text,
                                   v.get_absolute_path())
                if dlg.ShowModal() == wx.ID_OK:
                    dir_choice, custom_path = v.get_parts_from_path(dlg.Path)
                    proposed_value = v.join_string(dir_choice, custom_path)
                    if v.allow_metadata:
                        # Do escapes on backslashes
                        proposed_value = proposed_value.replace('\\', '\\\\')
                    setting_edited_event = SettingEditedEvent(
                        v, self.__module, proposed_value, event)
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
            if v.dir_choice in (cellprofiler.setting.DEFAULT_INPUT_SUBFOLDER_NAME,
                                cellprofiler.setting.DEFAULT_OUTPUT_SUBFOLDER_NAME):
                custom_label.Label = "Sub-folder:"
            elif v.dir_choice == cellprofiler.setting.URL_FOLDER_NAME:
                if v.support_urls == cellprofiler.setting.SUPPORT_URLS_SHOW_DIR:
                    custom_label.Label = "URL:"
                    custom_label.Show()
                    custom_ctrl.Show()
                else:
                    custom_label.Hide()
                    custom_ctrl.Hide()
                browse_ctrl.Hide()
            if custom_ctrl.Value != v.custom_path:
                custom_ctrl.Value = v.custom_path
        else:
            custom_label.Hide()
            custom_ctrl.Hide()
            browse_ctrl.Hide()
        if v.dir_choice in (cellprofiler.setting.DEFAULT_INPUT_FOLDER_NAME,
                            cellprofiler.setting.DEFAULT_INPUT_SUBFOLDER_NAME):
            folder_label.Label = \
                "( %s )" % cellprofiler.preferences.get_default_image_directory()
        elif v.dir_choice in (cellprofiler.setting.DEFAULT_OUTPUT_FOLDER_NAME,
                              cellprofiler.setting.DEFAULT_OUTPUT_SUBFOLDER_NAME):
            folder_label.Label = \
                "( %s )" % cellprofiler.preferences.get_default_output_directory()
        else:
            folder_label.Label = wx.EmptyString
        dir_ctrl.SetToolTipString(folder_label.Label)
        return control

    def make_pathname_control(self, v, control):
        if control is None:
            control = wx.Panel(self.module_panel, -1,
                               name=edit_control_name(v))
            control.Sizer = wx.BoxSizer(wx.HORIZONTAL)
            text_control = wx.TextCtrl(control, -1,
                                       name=subedit_control_name(v))
            text_control.Bind(
                wx.EVT_TEXT,
                lambda event: self.__on_cell_change(event, v, text_control))
            browse_bitmap = wx.ArtProvider.GetBitmap(wx.ART_FOLDER,
                                                     wx.ART_CMN_DIALOG,
                                                     (16, 16))
            browse_ctrl = wx.BitmapButton(control, bitmap=browse_bitmap,
                                          name=button_control_name(v))
            control.Sizer.Add(text_control, 1, wx.EXPAND)
            control.Sizer.AddSpacer((3, 0))
            control.Sizer.Add(browse_ctrl, 0, wx.EXPAND)

            def on_browse(event):
                dlg = wx.FileDialog(self.module_panel)
                try:
                    dlg.Title = "Browse for metadata file"
                    dlg.Wildcard = v.wildcard
                    if dlg.ShowModal() == wx.ID_OK:
                        self.on_value_change(v, control, dlg.Path, event)
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
        from cellprofiler.modules.loadimages import url2pathname

        assert isinstance(v, cellprofiler.setting.ImagePlane)
        if not control:
            control = wx.Panel(self.module_panel,
                               name=edit_control_name(v))
            control.Sizer = wx.BoxSizer(wx.HORIZONTAL)
            url_control = wx.TextCtrl(
                control,
                style=wx.TE_READONLY,
                name=text_control_name(v))
            control.Sizer.Add(url_control, 1, wx.EXPAND)
            control.Sizer.AddSpacer(2)
            browse_button = wx.Button(control, label="Browse",
                                      name=button_control_name(v))
            control.Sizer.Add(browse_button, 0, wx.EXPAND)

            def on_button(event):
                url = self.__frame.pipeline_controller.pick_from_pathlist(
                    v.url, instructions="Select an image file from the list below")
                if url is not None:
                    value = v.build(url)
                    self.on_value_change(v, control, value, event)

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
                    name=control_name
                )
            else:
                style = 0
                text = v.get_value_text()
                if not isinstance(text, (unicode, str)):
                    text = str(text)
                if getattr(v, "multiline_display", False):
                    style = wx.TE_MULTILINE | wx.TE_PROCESS_ENTER
                    lines = text.split("\n")
                else:
                    lines = [text]

                control = wx.TextCtrl(self.__module_panel,
                                      -1,
                                      text,
                                      name=control_name,
                                      style=style)

            def on_cell_change(event, setting=v, control=control):
                self.__on_cell_change(event, setting, control)

            self.__module_panel.Bind(wx.EVT_TEXT, on_cell_change, control)
        elif not (v.get_value_text() == control.Value):
            text = v.get_value_text()
            if not isinstance(text, (unicode, str)):
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
            min_ctrl = wx.TextCtrl(
                panel, -1, v.min_text, name=min_control_name(v))
            sizer.Add(min_ctrl, 0, wx.EXPAND | wx.RIGHT, 1)
            max_ctrl = wx.TextCtrl(
                panel, -1, v.max_text, name=max_control_name(v))
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
            min_ctrl = wx.TextCtrl(panel, -1, value=str(v.min),
                                   name=min_control_name(v))
            best_width = min_ctrl.GetCharWidth() * 5
            min_ctrl.SetInitialSize(wx.Size(best_width, -1))
            sizer.Add(min_ctrl, 0, wx.EXPAND | wx.RIGHT, 1)
            max_ctrl = wx.TextCtrl(panel, -1, value=v.display_max,
                                   name=max_control_name(v))
            max_ctrl.SetInitialSize(wx.Size(best_width, -1))
            sizer.Add(max_ctrl, 0, wx.EXPAND)
            value = ABSOLUTE if v.is_abs() else FROM_EDGE
            absrel_ctrl = wx.ComboBox(panel, -1, value,
                                      choices=[ABSOLUTE, FROM_EDGE],
                                      name=absrel_control_name(v),
                                      style=wx.CB_DROPDOWN | wx.CB_READONLY)
            sizer.Add(absrel_ctrl, 0, wx.EXPAND | wx.RIGHT, 1)

            def on_min_change(event, setting=v, control=min_ctrl):
                if not self.__handle_change:
                    return
                proposed_value = setting.compose_min_text(control.Value)
                setting_edited_event = SettingEditedEvent(setting, self.__module,
                                                          proposed_value, event)
                self.notify(setting_edited_event)
                self.fit_ctrl(control)

            self.__module_panel.Bind(wx.EVT_TEXT, on_min_change, min_ctrl)

            def on_max_change(event, setting=v, control=max_ctrl,
                              absrel_ctrl=absrel_ctrl):
                if not self.__handle_change:
                    return
                proposed_value = setting.compose_display_max_text(control.Value)
                setting_edited_event = SettingEditedEvent(setting, self.__module,
                                                          proposed_value, event)
                self.notify(setting_edited_event)
                self.fit_ctrl(control)

            self.__module_panel.Bind(wx.EVT_TEXT, on_max_change, max_ctrl)

            def on_absrel_change(event, setting=v, control=absrel_ctrl):
                if not self.__handle_change:
                    return

                if control.Value == ABSOLUTE:
                    proposed_value = setting.compose_abs()
                else:
                    proposed_value = setting.compose_rel()
                if proposed_value is not None:
                    setting_edited_event = SettingEditedEvent(setting, self.__module,
                                                              proposed_value, event)
                    self.notify(setting_edited_event)

            self.__module_panel.Bind(wx.EVT_COMBOBOX,
                                     on_absrel_change, absrel_ctrl)
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
            x_ctrl = wx.TextCtrl(
                panel, -1, v.get_x_text(), name=x_control_name(v))
            best_width = x_ctrl.GetCharWidth() * 5
            x_ctrl.SetInitialSize(wx.Size(best_width, -1))
            sizer.Add(x_ctrl, 0, wx.EXPAND | wx.RIGHT, 1)
            sizer.Add(wx.StaticText(panel, -1, "Y:"), 0, wx.EXPAND | wx.RIGHT, 1)
            y_ctrl = wx.TextCtrl(
                panel, -1, v.get_y_text(), name=y_control_name(v))
            y_ctrl.SetInitialSize(wx.Size(best_width, -1))
            sizer.Add(y_ctrl, 0, wx.EXPAND)

            def on_x_change(event, setting=v, control=x_ctrl):
                if not self.__handle_change:
                    return
                old_value = str(setting)
                proposed_value = "%s,%s" % (str(control.Value), str(setting.y))
                setting_edited_event = SettingEditedEvent(setting, self.__module,
                                                          proposed_value, event)
                self.notify(setting_edited_event)

            self.__module_panel.Bind(wx.EVT_TEXT, on_x_change, x_ctrl)

            def on_y_change(event, setting=v, control=y_ctrl):
                if not self.__handle_change:
                    return
                old_value = str(setting)
                proposed_value = "%s,%s" % (str(setting.x), str(control.Value))
                setting_edited_event = SettingEditedEvent(setting, self.__module,
                                                          proposed_value, event)
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
        if not panel:
            panel = wx.Panel(self.__module_panel, -1, name=edit_control_name(v))
            sizer = wx.BoxSizer(wx.VERTICAL)
            panel.SetSizer(sizer)
            #
            # The category combo-box
            #
            sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(sub_sizer, 0, wx.ALIGN_LEFT)
            category_text_ctrl = wx.StaticText(panel, label='Category:',
                                               name=category_text_control_name(v))
            sub_sizer.Add(category_text_ctrl, 0, wx.EXPAND | wx.ALL, 2)
            category_ctrl = wx.ComboBox(panel, style=wx.CB_READONLY,
                                        name=category_control_name(v))
            sub_sizer.Add(category_ctrl, 0, wx.EXPAND | wx.ALL, 2)
            #
            # The measurement / feature combo-box
            #
            sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(sub_sizer, 0, wx.ALIGN_LEFT)
            feature_text_ctrl = wx.StaticText(panel, label='Measurement:',
                                              name=feature_text_control_name(v))
            sub_sizer.Add(feature_text_ctrl, 0, wx.EXPAND | wx.ALL, 2)
            feature_ctrl = wx.ComboBox(panel, style=wx.CB_READONLY,
                                       name=feature_control_name(v))
            sub_sizer.Add(feature_ctrl, 0, wx.EXPAND | wx.ALL, 2)
            #
            # The object combo-box which sometimes doubles as an image combo-box
            #
            sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(sub_sizer, 0, wx.ALIGN_LEFT)
            object_text_ctrl = wx.StaticText(panel, label='Object:',
                                             name=object_text_control_name(v))
            sub_sizer.Add(object_text_ctrl, 0, wx.EXPAND | wx.ALL, 2)
            object_ctrl = wx.ComboBox(panel, style=wx.CB_READONLY,
                                      name=object_control_name(v))
            sub_sizer.Add(object_ctrl, 0, wx.EXPAND | wx.ALL, 2)
            #
            # The scale combo-box
            sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(sub_sizer, 0, wx.ALIGN_LEFT)
            scale_text_ctrl = wx.StaticText(panel, label='Scale:',
                                            name=scale_text_ctrl_name(v))
            sub_sizer.Add(scale_text_ctrl, 0, wx.EXPAND | wx.ALL, 2)
            scale_ctrl = wx.ComboBox(panel, style=wx.CB_READONLY,
                                     name=scale_control_name(v))
            sub_sizer.Add(scale_ctrl, 0, wx.EXPAND | wx.ALL, 2)
            max_width = 0
            for sub_sizer_item in sizer.GetChildren():
                static = sub_sizer_item.Sizer.GetChildren()[0].Window
                max_width = max(max_width, static.Size.width)
            for sub_sizer_item in sizer.GetChildren():
                static = sub_sizer_item.Sizer.GetChildren()[0].Window
                static.Size = wx.Size(max_width, static.Size.height)
                static.SetSizeHints(max_width, -1, max_width)

            #
            # Bind all controls to the function that constructs a value
            # out of the parts
            #
            def on_change(event, v=v, category_ctrl=category_ctrl,
                          feature_ctrl=feature_ctrl,
                          object_ctrl=object_ctrl,
                          scale_ctrl=scale_ctrl):
                """Reconstruct the measurement value if anything changes"""
                if not self.__handle_change:
                    return

                def value_of(ctrl):
                    return ctrl.Value if ctrl.Selection != -1 else None

                value = v.construct_value(value_of(category_ctrl),
                                          value_of(feature_ctrl),
                                          value_of(object_ctrl),
                                          value_of(scale_ctrl))
                setting_edited_event = SettingEditedEvent(v,
                                                          self.__module,
                                                          value,
                                                          event)
                self.notify(setting_edited_event)
                self.reset_view()

            for ctrl in (category_ctrl, feature_ctrl, object_ctrl, scale_ctrl):
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

        def set_up_combobox(ctrl, text_ctrl, choices, value, always_show=False):
            if len(choices):
                if value is None:
                    choices = ["[None]"] + choices
                if not (len(ctrl.Strings) == len(choices) and
                            all([x == y for x, y in zip(ctrl.Strings, choices)])):
                    ctrl.Clear()
                    ctrl.AppendItems(choices)
                    ctrl.SetSelection(0)
                if not value is None:
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

        set_up_combobox(category_ctrl, category_text_ctrl, categories,
                        category, True)
        set_up_combobox(feature_ctrl, feature_text_ctrl,
                        feature_names, feature_name)
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
        from cellprofiler.gui.html import HtmlClickableWindow
        if control is None:
            control = HtmlClickableWindow(self.module_panel, -1,
                                          name=edit_control_name(v))
            if v.size is not None:
                unit = float(wx.SystemSettings.GetMetric(wx.SYS_CAPTION_Y))
                if unit == -1:
                    unit = 32.0
                control.SetMinSize((v.size[0] * unit, v.size[1] * unit))
        control.SetPage(v.content)
        return control

    def make_help_control(self, content, title="Help",
                          name=wx.ButtonNameStr):
        control = wx.Button(self.__module_panel, -1, '?', (0, 0), (30, -1),
                            name=name)

        def callback(event):
            dialog = htmldialog.HTMLDialog(self.__module_panel, title, content)
            dialog.CentreOnParent()
            dialog.Show()

        control.Bind(wx.EVT_BUTTON, callback, control)
        return control

    class CornerButtonGrid(wx.grid.Grid, cornerbuttonmixin.CornerButtonMixin):
        def __init__(self, *args, **kwargs):
            kwargs = kwargs.copy()
            if "fn_clicked" in kwargs:
                fn_clicked = kwargs.pop("fn_clicked")
            else:
                fn_clicked = None
            label = kwargs.pop("label", "Update")
            tooltip = kwargs.pop("tooltip", "Update this table")
            wx.grid.Grid.__init__(self, *args, **kwargs)
            cornerbuttonmixin.CornerButtonMixin.__init__(self, fn_clicked, label, tooltip)

    def make_table_control(self, v, control):
        if control is None:
            control = wx.lib.resizewidget.ResizeWidget(
                self.module_panel,
                name=edit_control_name(v))

            if v.corner_button is None:
                grid = wx.grid.Grid(control, name=grid_control_name(v))
            else:
                grid = self.CornerButtonGrid(
                    control, name=grid_control_name(v), **v.corner_button)
            grid.SetTable(TableController(v))
            grid.Table.bind_to_grid(grid)
        else:
            grid = control.FindWindowByName(grid_control_name(v))
            grid.Table.update_grid()
        grid.ForceRefresh()
        grid.SetBestFittingSize(v.min_size)
        control.AdjustToSize((v.min_size[0] + wx.lib.resizewidget.RW_THICKNESS,
                              v.min_size[1] + wx.lib.resizewidget.RW_THICKNESS))
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
        self.on_value_change(
            setting, control, control.GetStringSelection(), event)

    def __on_combobox_change(self, event, setting, control):
        if not self.__handle_change:
            return
        self.on_value_change(setting, control, control.GetValue(), event)

    def __on_multichoice_change(self, event, setting, control):
        if not self.__handle_change:
            return

        proposed_value = u','.join([control.Items[i]
                                    for i in control.Selections])
        self.on_value_change(setting, control, proposed_value, event)

    def __on_cell_change(self, event, setting, control):
        if not self.__handle_change:
            return
        proposed_value = unicode(control.GetValue())
        self.on_value_change(setting, control, proposed_value, event,
                             EDIT_TIMEOUT_SEC * 1000)

    def on_value_change(self, setting, control, proposed_value, event,
                        timeout=None):
        """Handle a change in value to a setting

        setting - the setting that changed
        control - the WX control whose UI signalled the change
        proposed_value - the proposed new value for the setting
        event - the UI event signalling the change
        timeout - None = reset view immediately, False = don't reset view
                  otherwise the # of milliseconds to wait before
                  refresh.
        """
        setting_edited_event = SettingEditedEvent(setting,
                                                  self.__module,
                                                  proposed_value,
                                                  event)
        self.notify(setting_edited_event)
        if timeout is None:
            self.reset_view()  # use the default timeout
        elif timeout is not False:
            self.reset_view(timeout)

    @staticmethod
    def fit_ctrl(ctrl):
        """Fit the control to its text size"""
        width, height = ctrl.GetTextExtent(ctrl.Value + "MM")
        ctrl.SetSizeHintsSz(wx.Size(width, -1))
        ctrl.Parent.Fit()

    def __on_min_change(self, event, setting, control):
        if not self.__handle_change:
            return
        old_value = str(setting)
        proposed_value = setting.compose_min_text(control.Value)
        setting_edited_event = SettingEditedEvent(
            setting, self.__module, proposed_value, event)
        self.notify(setting_edited_event)
        self.fit_ctrl(control)

    def __on_max_change(self, event, setting, control):
        if not self.__handle_change:
            return
        old_value = str(setting)
        proposed_value = setting.compose_max_text(control.Value)
        setting_edited_event = SettingEditedEvent(
            setting, self.__module, proposed_value, event)
        self.notify(setting_edited_event)
        self.fit_ctrl(control)

    def request_validation(self, module=None):
        """Request validation of the current module in its current state"""
        if module is None:
            module = self.__module
        if self.__validation_request is not None:
            self.__validation_request.cancel()
        self.__validation_request = ValidationRequest(
            self.__pipeline, module, self.on_validation)
        request_module_validation(self.__validation_request)

    def __on_pipeline_event(self, pipeline, event):
        if (isinstance(event, cellprofiler.pipeline.PipelineClearedEvent) or
                isinstance(event, cellprofiler.pipeline.PipelineLoadedEvent)):
            if self.__module not in self.__pipeline.modules(False):
                self.clear_selection()
        elif isinstance(event, cellprofiler.pipeline.ModuleEditedPipelineEvent):
            if (not self.__inside_notify and self.__module is not None
                and self.__module.module_num == event.module_num):
                self.reset_view()
            if (self.__module is not None and
                        self.__module.module_num == event.module_num):
                self.request_validation()
        elif isinstance(event, cellprofiler.pipeline.ModuleRemovedPipelineEvent):
            if (self.__module is not None and
                        event.module_num == self.__module.module_num):
                self.clear_selection()

    def __on_workspace_event(self, event):
        import cellprofiler.gui.workspace as cpw
        if isinstance(event, (cpw.Workspace.WorkspaceLoadedEvent,
                              cpw.Workspace.WorkspaceCreatedEvent)):
            # Detach and reattach the current module to get it reacclimated
            # to the current workspace and reselect
            if self.__module is not None:
                self.__module.on_deactivated()
                self.__module.on_activated(self.__workspace)
                self.do_reset()

    def __on_do_something(self, event, setting):
        setting.on_event_fired()
        setting_edited_event = SettingEditedEvent(setting, self.__module,
                                                  None, event)
        self.notify(setting_edited_event)
        self.__module.on_setting_changed(setting, self.__pipeline)
        self.reset_view()

    def on_validation(self, setting_idx, message, level):
        default_fg_color = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT)
        default_bg_color = cellprofiler.preferences.get_background_color()
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
            except cellprofiler.setting.ValidationError, instance:
                message = instance.message
                bad_setting = instance.get_setting()
        # update settings' foreground/background
        try:
            for setting in visible_settings:
                self.set_tool_tip(setting, message if (setting is bad_setting) else None)
                static_text_name = text_control_name(setting)
                static_text = self.__module_panel.FindWindowByName(static_text_name)
                if static_text is not None:
                    desired_fg, desired_bg = default_fg_color, default_bg_color
                    if setting is bad_setting:
                        if level == logging.ERROR:
                            desired_fg = cellprofiler.preferences.get_error_color()
                        elif level == logging.WARNING:
                            desired_bg = WARNING_COLOR
        except Exception:
            logger.debug("Caught bare exception in ModuleView.on_validate()", exc_info=True)
            pass

    def set_tool_tip(self, setting, message):
        """Set the tool tip for a setting to display a message

        setting - set the tooltip for this setting

        message - message to display or None for no tool tip
        """
        control_name = edit_control_name(setting)
        control = self.__module_panel.FindWindowByName(
            control_name)
        if message is None:
            def set_tool_tip(ctrl):
                ctrl.SetToolTip(None)
        else:
            def set_tool_tip(ctrl, message=message):
                ctrl.SetToolTipString(message)
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
        if not focus_control is None:
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
                    focus_control.SetSelection(focus_control.GetLastPosition(), focus_control.GetLastPosition())

    def disable(self):
        self.__module_panel.Disable()

    def enable(self):
        self.__module_panel.Enable()

    def get_max_width(self):
        sizer = self.__sizer
        return sizer.calc_max_text_width() + sizer.calc_edit_size()[0] + sizer.calc_help_size()[0]


class FilterPanelController(object):
    """Handle representation of the filter panel

    The code for handling the filter UI is moderately massive, so it gets
    its own class, if for no other reason than to organize the code.
    """

    def __init__(self, module_view, v, panel):
        assert isinstance(module_view, ModuleView)
        assert isinstance(v, cellprofiler.setting.Filter)
        self.module_view = module_view
        self.v = v
        self.panel = wx.Panel(self.module_view.module_panel,
                              style=wx.TAB_TRAVERSAL,
                              name=edit_control_name(self.v))
        self.panel.Sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer_dict = {}
        self.sizer_item_dict = {}
        self.stretch_spacer_dict = {}
        self.hide_show_dict = {}
        self.update()

    def get_sizer(self, address):
        """Find or create the sizer that's associated with a particular address"""
        key = tuple(address)
        line_name = self.line_name(address)
        self.hide_show_dict[line_name] = True
        if self.sizer_dict.has_key(key):
            if len(address) > 0:
                self.hide_show_dict[self.remove_button_name(address)] = True
                self.hide_show_dict[self.add_button_name(address)] = True
                self.hide_show_dict[self.add_group_button_name(address)] = True
            return self.sizer_dict[key]
        #
        # Four possibilities:
        #
        # * The sizer is the top level one
        # * There is a sizer at the same level whose last address is one more.
        # * There are sizers at the same level whose next to last to address is
        #   one more than the next to last address of the address and whose
        #   last address is zero.
        # * None of the above which means the sizer can be added at the end.
        #
        line_style = wx.LI_HORIZONTAL | wx.BORDER_SUNKEN
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.indent(sizer, address)
        self.stretch_spacer_dict[key] = sizer.AddStretchSpacer()
        line = wx.StaticLine(self.panel, -1, style=line_style,
                             name=self.line_name(address))

        if len(address) == 0:
            key = None
        else:
            sizer.Add(self.make_delete_button(address), 0,
                      wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL)
            sizer.Add(self.make_add_rule_button(address), 0,
                      wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL)
            sizer.Add(self.make_add_rules_button(address), 0,
                      wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL)
            key = tuple(address[:-1] + [address[-1] + 1])
            if not self.sizer_dict.has_key(key):
                if len(address) == 1:
                    key = None
                else:
                    key = tuple(address[:-2] + [address[-2] + 1])
                    if not self.sizer_dict.has_key(key):
                        key = None
        if key is not None:
            next_sizer = self.sizer_dict[key]
            idx = self.get_sizer_index(self.panel.Sizer, next_sizer)
            self.panel.Sizer.Insert(idx, sizer, 0, wx.EXPAND)
            self.panel.Sizer.Insert(idx + 1, line, 0, wx.EXPAND)
        else:
            self.panel.Sizer.Add(sizer, 0, wx.EXPAND)
            self.panel.Sizer.Add(line, 0, wx.EXPAND)
        self.sizer_dict[tuple(address)] = sizer
        return sizer

    def get_tokens(self):
        try:
            tokens = self.v.parse()
        except Exception, e:
            logger.debug("Failed to parse filter (value=%s): %s",
                         self.v.value_text, str(e))
            tokens = self.v.default()
        #
        # Always require an "and" or "or" clause
        #
        if (len(tokens) == 0 or
                (tokens[0] not in
                     (cellprofiler.setting.Filter.AND_PREDICATE, cellprofiler.setting.Filter.OR_PREDICATE))):
            tokens = [cellprofiler.setting.Filter.AND_PREDICATE, tokens]
        return tokens

    def update(self):
        self.inside_update = True
        try:
            structure = self.get_tokens()
            for key in self.hide_show_dict:
                self.hide_show_dict[key] = False
            self.populate_subpanel(structure, [])
            for key, value in self.hide_show_dict.iteritems():
                self.panel.FindWindowByName(key).Show(value)
            self.panel.Layout()
        except:
            logger.exception("Threw exception while updating filter")
        finally:
            self.inside_update = False

    ANY_ALL_PREDICATES = [cellprofiler.setting.Filter.AND_PREDICATE,
                          cellprofiler.setting.Filter.OR_PREDICATE]

    def any_all_choices(self):
        return [x.display_name for x in self.ANY_ALL_PREDICATES]

    @staticmethod
    def indent(sizer, address):
        assert isinstance(sizer, wx.Sizer)
        if len(address) == 0:
            return
        sizer.AddSpacer((len(address) * 20, 0))

    def find_and_mark(self, name):
        """Find a control and mark it to be shown"""
        ctrl = self.panel.FindWindowByName(name)
        self.hide_show_dict[name] = True
        return ctrl

    @staticmethod
    def get_sizer_index(sizer, item):
        if isinstance(item, wx.Sizer):
            indexes = [i for i, s in enumerate(sizer.GetChildren())
                       if s.IsSizer() and s.GetSizer() is item]
        elif isinstance(item, wx.Window):
            indexes = [i for i, s in enumerate(sizer.GetChildren())
                       if s.IsWindow() and s.GetWindow() is item]
        elif isinstance(item, wx.SizerItem):
            return sizer.GetChildren().index(item)
        if len(indexes) > 0:
            return indexes[0]
        return None

    def on_value_change(self, event, new_text, timeout=None):
        if not self.inside_update:
            self.module_view.on_value_change(
                    self.v, self.panel, new_text, event, timeout)

    def make_delete_button(self, address):
        name = self.remove_button_name(address)
        button = self.find_and_mark(name)
        if button is not None:
            return button

        button = wx.Button(self.panel, -1, "-",
                           name=name,
                           style=wx.BU_EXACTFIT)
        button.Bind(wx.EVT_BUTTON,
                    lambda event: self.on_delete_rule(event, address))
        return button

    def on_delete_rule(self, event, address):
        logger.debug("Delete row at " + str(address))
        structure = self.v.parse()
        sequence = self.find_address(structure, address[:-1])
        del sequence[address[-1] + 1]
        new_text = self.v.build_string(structure)
        self.on_value_change(event, new_text)

    def make_add_rule_button(self, address):
        name = self.add_button_name(address)
        button = self.find_and_mark(name)
        if button is not None:
            return button

        button = wx.Button(self.panel, -1, "+",
                           name=name,
                           style=wx.BU_EXACTFIT)
        button.Bind(wx.EVT_BUTTON,
                    lambda event: self.on_add_rule(event, address))
        return button

    def on_add_rule(self, event, address):
        logger.debug("Add rule after " + str(address))
        structure = self.v.parse()
        sequence = self.find_address(structure, address[:-1])
        new_rule = self.v.default()
        sequence.insert(address[-1] + 2, new_rule)
        new_text = self.v.build_string(structure)
        self.on_value_change(event, new_text)

    def make_add_rules_button(self, address):
        name = self.add_group_button_name(address)
        button = self.find_and_mark(name)
        if button is not None:
            return button
        button = wx.Button(self.panel, -1, "...",
                           name=name,
                           style=wx.BU_EXACTFIT)
        button.Bind(wx.EVT_BUTTON,
                    lambda event: self.on_add_rules(event, address))
        return button

    def on_add_rules(self, event, address):
        logger.debug("Add rules after " + str(address))
        structure = self.v.parse()
        sequence = self.find_address(structure, address[:-1])
        new_rule = [cellprofiler.setting.Filter.OR_PREDICATE, self.v.default()]
        sequence.insert(address[-1] + 2, new_rule)
        new_text = self.v.build_string(structure)
        self.on_value_change(event, new_text)

    def make_predicate_choice(self, predicates, index, address, sizer):
        name = self.choice_name(index, address)
        choice_ctrl = self.find_and_mark(name)
        choices = [x.display_name for x in predicates]
        if choice_ctrl is not None:
            items = choice_ctrl.GetItems()
            if (len(items) != len(choices) or
                    any([choice not in items for choice in choices])):
                choice_ctrl.SetItems(choices)
            return choice_ctrl
        choice_ctrl = wx.Choice(self.panel, -1, choices=choices,
                                name=name)
        choice_ctrl.Bind(wx.EVT_CHOICE,
                         lambda event: self.on_predicate_changed(event, index, address))
        self.add_to_sizer(sizer, choice_ctrl, index, address)
        return choice_ctrl

    def on_predicate_changed(self, event, index, address):
        logger.debug("Predicate choice at %d / %s changed" %
                     (index, self.saddress(address)))
        structure = self.v.parse()
        sequence = self.find_address(structure, address)

        while len(sequence) <= index:
            # The sequence is bad (e.g. bad pipeline or metadata collection)
            # Fill in enough to deal
            #
            sequence.append(self.v.predicates[0]
                            if len(sequence) == 0
                            else sequence[-1].subpredicates[0])
        if index == 0:
            predicates = self.v.predicates
        else:
            predicates = sequence[index - 1].subpredicates
        new_predicate = predicates[event.GetSelection()]

        sequence[index] = new_predicate
        predicates = new_predicate.subpredicates
        #
        # Make sure following predicates are legal
        #
        for index in range(index + 1, len(sequence)):
            if isinstance(sequence[index], basestring):
                is_good = cellprofiler.setting.Filter.LITERAL_PREDICATE in predicates
            else:
                matches = [p for p in predicates
                           if sequence[index].symbol == p.symbol]
                is_good = len(matches) == 1
                if is_good:
                    sequence[index] = matches[0]
            if not is_good:
                del sequence[index:]
                sequence += self.v.default(predicates)
                break
            if not isinstance(sequence[index], cellprofiler.setting.Filter.FilterPredicate):
                break
            predicates = sequence[index].subpredicates
        new_text = self.v.build_string(structure)
        self.on_value_change(event, new_text)

    def add_to_sizer(self, sizer, item, index, address):
        """Insert the item in the sizer at the right location

        sizer - sizer for the line

        item - the control to be added

        index - index of the control within the sizer

        address - address of the sizer
        """
        key = tuple(address + [index])
        next_key = tuple(address + [index + 1])
        if self.sizer_item_dict.has_key(next_key):
            next_ctrl = self.sizer_item_dict[next_key]
        else:
            next_ctrl = self.stretch_spacer_dict[tuple(address)]
        index = self.get_sizer_index(sizer, next_ctrl)
        sizer.Insert(index, item, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_HORIZONTAL)
        if not self.sizer_item_dict.has_key(key):
            self.sizer_item_dict[key] = item

    def make_literal(self, token, index, address, sizer):
        name = self.literal_name(index, address)
        literal_ctrl = self.find_and_mark(name)
        if literal_ctrl is not None:
            if literal_ctrl.GetValue() != token:
                literal_ctrl.SetValue(token)
            return literal_ctrl
        literal_ctrl = wx.TextCtrl(self.panel, -1, token, name=name)
        literal_ctrl.Bind(wx.EVT_TEXT,
                          lambda event: self.on_literal_changed(event, index, address))
        self.add_to_sizer(sizer, literal_ctrl, index, address)
        return literal_ctrl

    def on_literal_changed(self, event, index, address):
        logger.debug("Literal at %d / %s changed" % (index, self.saddress(address)))
        try:
            structure = self.v.parse()
            sequence = self.find_address(structure, address)
            while len(sequence) <= index:
                # The sequence is bad (e.g. bad pipeline or metadata collection)
                # Fill in enough to deal
                #
                sequence.append(self.v.predicates[0]
                                if len(sequence) == 0
                                else sequence[-1].subpredicates[0])
            sequence[index] = event.GetString()
        except:
            structure = self.v.default()

        new_text = self.v.build_string(structure)
        self.on_value_change(event, new_text,
                             timeout=None if self.v.reset_view else False)

    def make_anyall_ctrl(self, address):
        anyall = wx.Choice(self.panel, -1, choices=self.any_all_choices(),
                           name=self.anyall_choice_name(address))
        anyall.Bind(wx.EVT_CHOICE,
                    lambda event: self.on_anyall_changed(event, address))
        return anyall

    def on_anyall_changed(self, event, address):
        logger.debug("Any / all choice at %s changed" % self.saddress(address))
        structure = self.v.parse()
        sequence = self.find_address(structure, address)
        predicate = self.ANY_ALL_PREDICATES[event.GetSelection()]
        sequence[0] = predicate
        new_text = self.v.build_string(structure)
        self.on_value_change(event, new_text)

    def find_address(self, sequence, address):
        """Find the sequence with the given address"""
        if len(address) == 0:
            return sequence
        subsequence = sequence[address[0] + 1]
        return self.find_address(subsequence, address[1:])

    def populate_subpanel(self, structure, address):
        parent_sizer = self.panel.Sizer
        any_all_name = self.anyall_choice_name(address)
        anyall = self.find_and_mark(any_all_name)
        self.hide_show_dict[self.static_text_name(0, address)] = True
        if len(address) == 0:
            self.hide_show_dict[self.static_text_name(1, address)] = True
        if anyall is None:
            anyall = self.make_anyall_ctrl(address)
            sizer = self.get_sizer(address)
            idx = self.get_sizer_index(sizer,
                                       self.stretch_spacer_dict[tuple(address)])
            if len(address) == 0:
                text = wx.StaticText(self.panel, -1, "Match",
                                     name=self.static_text_name(0, address))
                sizer.Insert(idx, text, 0,
                             wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
                sizer.InsertSpacer(idx + 1, (3, 0))
                sizer.Insert(idx + 2, anyall, 0,
                             wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
                sizer.InsertSpacer(idx + 3, (3, 0))
                text = wx.StaticText(self.panel, -1, "of the following rules",
                                     name=self.static_text_name(1, address))
                sizer.Insert(idx + 4, text,
                             0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
            else:
                sizer.Insert(idx, anyall, 0,
                             wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL | wx.RIGHT)
                sizer.InsertSpacer(idx + 1, (3, 0))
                text = wx.StaticText(self.panel, -1, "of the following are true",
                                     name=self.static_text_name(0, address))
                sizer.Insert(idx + 2, text,
                             0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        else:
            self.hide_show_dict[self.line_name(address)] = True
            if len(address) > 0:
                #
                # Show the buttons for the anyall if not top level
                #
                self.hide_show_dict[self.remove_button_name(address)] = True
                self.hide_show_dict[self.add_button_name(address)] = True
                self.hide_show_dict[self.add_group_button_name(address)] = True

        if anyall.GetStringSelection() != structure[0].display_name:
            anyall.SetStringSelection(structure[0].display_name)
            anyall.SetToolTipString(structure[0].doc)
        #
        # Now each subelement should be a list.
        #
        for subindex, substructure in enumerate(structure[1:]):
            subaddress = address + [subindex]
            if substructure[0].subpredicates is list:
                # A sublist
                self.populate_subpanel(substructure, subaddress)
            else:
                # A list of predicates
                sizer = self.get_sizer(subaddress)
                predicates = self.v.predicates
                for i, token in enumerate(substructure):
                    if isinstance(token, basestring):
                        literal_ctrl = self.make_literal(
                                token, i, subaddress, sizer)
                        predicates = []
                    else:
                        choice_ctrl = self.make_predicate_choice(
                                predicates, i, subaddress, sizer)
                        if choice_ctrl.GetStringSelection() != token.display_name:
                            choice_ctrl.SetStringSelection(token.display_name)
                        if token.doc is not None:
                            choice_ctrl.SetToolTipString(token.doc)
                        predicates = token.subpredicates
                i = len(substructure)
                while len(predicates) > 0:
                    #
                    # We can get here if there's a badly constructed token
                    # list - for instance if an invalid subpredicate was
                    # chosen or none existed because of some error, but now
                    # they do.
                    #
                    if (len(predicates) == 1 and
                                predicates[0] is cellprofiler.setting.Filter.LITERAL_PREDICATE):
                        self.make_literal("", i, subaddress, sizer)
                    else:
                        self.make_predicate_choice(predicates, i, subaddress,
                                                   sizer)
                    i += 1
                    predicates = predicates[0].subpredicates
        #
        # Don't allow delete of only rule
        #
        name = self.remove_button_name(address + [0])
        delete_button = self.panel.FindWindowByName(name)
        delete_button.Enable(len(structure) > 2)

    @property
    def key(self):
        return str(self.v.key())

    @staticmethod
    def saddress(address):
        return "_".join([str(x) for x in address])

    def anyall_choice_name(self, address):
        return "%s_filter_anyall_%s" % (self.key, self.saddress(address))

    def choice_name(self, index, address):
        return "%s_choice_%d_%s" % (self.key, index, self.saddress(address))

    def literal_name(self, index, address):
        return "%s_literal_%d_%s" % (self.key, index, self.saddress(address))

    def remove_button_name(self, address):
        return "%s_remove_%s" % (self.key, self.saddress(address))

    def add_button_name(self, address):
        return "%s_add_%s" % (self.key, self.saddress(address))

    def add_group_button_name(self, address):
        return "%s_group_%s" % (self.key, self.saddress(address))

    def line_name(self, address):
        return "%s_line_%s" % (self.key, self.saddress(address))

    def static_text_name(self, index, address):
        return "%s_static_text_%d_%s" % (self.key, index, self.saddress(address))


class FileCollectionDisplayController(object):
    """This class provides the UI for the file collection display

    The UI has a browse button, a hide checkbox and a tree control.

    Critical attributes:

    self.walks_in_progress - this is a dictionary of keys to directory walks
                             and metadata fetches that are happening in the
                             background. The value of the dictionary entry
                             is the function to call to stop the search.

                             There's a completion callback that's called to
                             remove an entry from the dictionary. When the
                             dictionary size reaches zero, the stop and pause
                             buttons are disabled.

    self.modpath_to_item - a modpath is a collection of path parts to some file
                             handled by the controller. There's a tree item
                             for every modpath in this dictionary and the
                             dictionary can be used for fast lookup of the
                             item without traversing the entire tree.
    """
    IMAGE_LIST = wx.ImageList(16, 16, 3)
    FOLDER_IMAGE_INDEX = IMAGE_LIST.Add(
            wx.ArtProvider.GetBitmap(wx.ART_FOLDER,
                                     wx.ART_OTHER, size=(16, 16)))
    FOLDER_OPEN_IMAGE_INDEX = IMAGE_LIST.Add(
            wx.ArtProvider.GetBitmap(wx.ART_FOLDER_OPEN,
                                     wx.ART_OTHER, size=(16, 16)))
    FILE_IMAGE_INDEX = IMAGE_LIST.Add(
            wx.ArtProvider.GetBitmap(wx.ART_NORMAL_FILE,
                                     wx.ART_OTHER, size=(16, 16)))
    IMAGE_PLANE_IMAGE_INDEX = IMAGE_LIST.Add(
            cellprofiler.icons.get_builtin_image("microscope-icon_16").ConvertToBitmap())
    IMAGE_PLANES_IMAGE_INDEX = IMAGE_LIST.Add(
            cellprofiler.icons.get_builtin_image("microscopes_16").ConvertToBitmap())
    COLOR_IMAGE_INDEX = IMAGE_LIST.Add(
            cellprofiler.icons.get_builtin_image("microscope-color_16").ConvertToBitmap())
    MOVIE_IMAGE_INDEX = IMAGE_LIST.Add(
            cellprofiler.icons.get_builtin_image("movie_16").ConvertToBitmap())

    ACTIVE_COLOR = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT)
    FILTERED_COLOR = wx.SystemSettings.GetColour(wx.SYS_COLOUR_GRAYTEXT)

    class FCDCDropTarget(wx.PyDropTarget):
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

        @staticmethod
        def OnEnter(x, y, d):
            return wx.DragCopy

        @staticmethod
        def OnDragOver(x, y, d):
            return wx.DragCopy

        def OnData(self, x, y, d):
            if self.GetData():
                df = self.composite_data_object.GetReceivedFormat().GetType()
                if df in (wx.DF_TEXT, wx.DF_UNICODETEXT):
                    self.OnDropText(x, y, self.text_data_object.GetText())
                elif df == wx.DF_FILENAME:
                    self.OnDropFiles(x, y,
                                     self.file_data_object.GetFilenames())
            return wx.DragCopy

        @staticmethod
        def OnDrop(x, y):
            return True

    def __init__(self, module_view, v, pipeline):
        assert isinstance(v, cellprofiler.setting.FileCollectionDisplay)
        self.module_view = module_view
        self.v = v
        assert isinstance(pipeline, cellprofiler.gui.pipeline.Pipeline)
        self.pipeline = pipeline
        self.panel = wx.Panel(self.module_view.module_panel, -1,
                              name=edit_control_name(v))
        self.panel.controller = self
        self.panel.Sizer = wx.BoxSizer(wx.VERTICAL)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.panel.Sizer.Add(sizer, 0, wx.EXPAND)
        self.status_text = wx.StaticText(self.panel, -1)
        sizer.Add(self.status_text, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER)
        sizer.AddStretchSpacer()
        sizer.Add(wx.StaticText(self.panel, -1,
                                "Drag folders and/or files here or"),
                  0, wx.ALIGN_LEFT | wx.ALIGN_CENTER)
        sizer.AddSpacer((3, 0))
        browse_button = wx.Button(self.panel, -1, "Browse...")
        sizer.Add(browse_button, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER)
        browse_button.Bind(wx.EVT_BUTTON, self.on_browse)
        tree_style = wx.TR_HIDE_ROOT | wx.TR_HAS_BUTTONS | wx.TR_MULTIPLE
        self.tree_ctrl = wx.TreeCtrl(self.panel, -1,
                                     style=tree_style)
        self.panel.Sizer.Add(self.tree_ctrl, 1, wx.EXPAND)
        self.tree_ctrl.SetImageList(self.IMAGE_LIST)
        self.tree_ctrl.Bind(wx.EVT_TREE_ITEM_MENU, self.on_tree_item_menu)
        self.tree_ctrl.Bind(wx.EVT_TREE_KEY_DOWN, self.on_tree_key_down)
        #
        # Don't auto-expand after the user collapses a node.
        #
        self.user_collapsed_a_node = False

        def on_item_collapsed(event):
            logger.debug("On item collapsed")
            self.user_collapsed_a_node = True

        self.tree_ctrl.Bind(wx.EVT_TREE_ITEM_COLLAPSED, on_item_collapsed)
        self.tree_ctrl.Bind(wx.EVT_TREE_ITEM_ACTIVATED, self.on_tree_doubleclick)
        self.tree_ctrl.Bind(wx.EVT_ERASE_BACKGROUND, self.on_erase_background)

        self.panel.Bind(wx.EVT_WINDOW_DESTROY, self.on_destroy)
        self.root_item = self.tree_ctrl.AddRoot("I am the invisible root")
        self.tree_ctrl.SetPyData(self.root_item, None)
        self.tree_ctrl.SetItemImage(self.root_item, self.FOLDER_IMAGE_INDEX)
        self.tree_ctrl.SetItemImage(self.root_item,
                                    self.FOLDER_OPEN_IMAGE_INDEX,
                                    wx.TreeItemIcon_Expanded)
        self.tree_ctrl.SetMinSize((100, 300))
        self.tree_ctrl.SetMaxSize((sys.maxint, 300))
        self.file_drop_target = self.FCDCDropTarget(self.on_drop_files,
                                                    self.on_drop_text)
        self.tree_ctrl.SetDropTarget(self.file_drop_target)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.panel.Sizer.Add(sizer, 0, wx.EXPAND)
        self.hide_show_ctrl = wx.CheckBox(self.panel, -1,
                                          self.v.hide_text)
        sizer.Add(self.hide_show_ctrl, 0,
                  wx.ALIGN_LEFT | wx.ALIGN_BOTTOM)
        self.hide_show_ctrl.Bind(wx.EVT_CHECKBOX, self.on_hide_show_checked)
        self.hide_show_ctrl.Value = not self.v.show_filtered
        sizer.AddStretchSpacer()
        self.stop_button = wx.Button(self.panel, -1, "Stop")
        self.stop_button.Enable(False)
        self.stop_button.Bind(wx.EVT_BUTTON, self.on_stop)
        sizer.Add(self.stop_button, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        self.pause_button = wx.Button(self.panel, -1, "Pause")
        self.pause_button.Enable(False)
        self.pause_button.Bind(wx.EVT_BUTTON, self.on_pause_resume)
        sizer.Add(self.pause_button, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        v.set_update_function(self.request_update)
        self.needs_update = False
        self.modpath_to_item = {}
        self.request_update()

    def __del__(self):
        self.on_destroy(None)

    def on_destroy(self, event):
        self.v.set_update_function()

    def on_erase_background(self, event):
        assert isinstance(event, wx.EraseEvent)
        dc = event.DC
        assert isinstance(dc, wx.DC)
        brush = wx.Brush(self.tree_ctrl.GetBackgroundColour())
        dc.SetBrush(brush)
        dc.SetPen(wx.TRANSPARENT_PEN)
        width, height = self.tree_ctrl.GetSize()
        dc.DrawRectangle(0, 0, width, height)
        if len(self.modpath_to_item) == 0:
            text = "Drop files and folders here"
            font = wx.Font(36, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL,
                           wx.FONTWEIGHT_BOLD)
            dc.SetTextForeground(wx.SystemSettings.GetColour(wx.SYS_COLOUR_GRAYTEXT))
            dc.SetFont(font)
            text_width, text_height = dc.GetTextExtent(text)
            dc.DrawText(text,
                        (width - text_width) / 2,
                        (height - text_height) / 2)

    def on_browse(self, event):
        logger.debug("Browsing for file collection directory")
        dlg = wx.DirDialog(self.panel, "Select a directory to add")
        try:
            if dlg.ShowModal() == wx.ID_OK:
                self.v.fn_on_drop([dlg.Path], True)
        finally:
            dlg.Destroy()

    def on_start_received(self):
        self.pause_button.Label = "Pause"
        self.pause_button.Enable(True)
        self.stop_button.Enable(True)

    def on_stop_received(self):
        self.pause_button.Enable(False)
        self.stop_button.Enable(False)

    def on_stop(self, event):
        """Stop button pressed"""
        self.v.fn_on_bkgnd_control(self.v.BKGND_STOP)
        self.pause_button.Label = "Pause"
        self.pause_button.Enable(False)
        self.stop_button.Enable(False)

    def on_pause_resume(self, event):
        """Pause / resume pressed"""
        if self.pause_button.Label == "Pause":
            action = self.v.BKGND_PAUSE
            self.pause_button.Label = "Resume"
        else:
            action = self.v.BKGND_RESUME
            self.pause_button.Label = "Pause"
        self.v.fn_on_bkgnd_control(action)

    def add_item(self, modpath, text=None, sort=True):
        """Add an item to the tree

        modpath - a collection of path parts to the item in the tree
        text - the text to appear in the item
        """
        parent_key = tuple(modpath[:-1])
        modpath = tuple(modpath)
        if self.modpath_to_item.has_key(modpath):
            item = self.modpath_to_item[modpath]
            if text is not None:
                self.tree_ctrl.SetItemText(item, text)
            return item

        if text is None:
            text = modpath[-1]
        if len(modpath) == 1:
            parent_item = self.root_item
        elif self.modpath_to_item.has_key(parent_key):
            parent_item = self.modpath_to_item[parent_key]
        else:
            parent_item = self.add_item(parent_key, sort=sort)
            self.tree_ctrl.SetItemImage(parent_item, self.FOLDER_IMAGE_INDEX)
            self.tree_ctrl.SetItemImage(parent_item,
                                        self.FOLDER_OPEN_IMAGE_INDEX,
                                        wx.TreeItemIcon_Expanded)

        want_erase = len(self.modpath_to_item) == 0
        #
        # Put in alpha order
        #
        n_children = self.tree_ctrl.GetChildrenCount(parent_item)
        if n_children == 0 or not sort:
            item = self.tree_ctrl.AppendItem(parent_item, text)
        else:
            child, cookie = self.tree_ctrl.GetFirstChild(parent_item)
            for i in range(n_children):
                ctext = self.tree_ctrl.GetItemText(child)
                if ctext > text:
                    item = self.tree_ctrl.InsertItemBefore(parent_item, i, text)
                    break
                child = self.tree_ctrl.GetNextSibling(child)
            else:
                item = self.tree_ctrl.AppendItem(parent_item, text)

        self.tree_ctrl.SetPyData(item, modpath[-1])
        self.modpath_to_item[modpath] = item
        if want_erase:
            self.tree_ctrl.Refresh(True)
        return item

    def remove_item(self, modpath):
        modpath = tuple(modpath)
        if self.modpath_to_item.has_key(modpath):
            item = self.modpath_to_item[modpath]
            n_children = self.tree_ctrl.GetChildrenCount(item, False)
            if n_children > 0:
                child, cookie = self.tree_ctrl.GetFirstChild(item)
                child_tokens = []
                for i in range(n_children):
                    child_tokens.append(self.tree_ctrl.GetItemPyData(child))
                    child = self.tree_ctrl.GetNextSibling(child)
                for child_token in child_tokens:
                    sub_modpath = list(modpath) + [child_token]
                    self.remove_item(sub_modpath)
            self.tree_ctrl.Delete(self.modpath_to_item[modpath])
            del self.modpath_to_item[modpath]

    @classmethod
    def get_modpath(cls, path):
        """Break a path into its components"""
        result = []
        while True:
            new_path, part = os.path.split(path)
            if len(new_path) == 0 or len(part) == 0:
                result.insert(0, path)
                return result
            result.insert(0, part)
            path = new_path

    def on_drop_files(self, x, y, filenames):
        self.v.fn_on_drop(filenames, True)

    def on_drop_text(self, x, y, text):
        """Text is assumed to be one file name per line"""
        filenames = [line.strip() for line in text.split("\n")
                     if len(line.strip()) > 0]
        self.v.fn_on_drop(filenames, False)

    def get_path_from_event(self, event):
        """Given a tree control event, find the path from the root

        event - event from tree control (e.g. EVT_TREE_ITEM_ACTIVATED)

        returns a sequence of path items from the root
        """
        item = event.GetItem()
        path = []
        while True:
            item_data = self.tree_ctrl.GetItemPyData(item)
            if item_data is None:
                break
            path.insert(0, item_data)
            item = self.tree_ctrl.GetItemParent(item)
        return path

    def on_tree_item_menu(self, event):
        logger.debug("On tree item menu")
        path = self.get_path_from_event(event)
        if len(path) == 0:
            logger.warn("Could not find item associated with tree event")
            return
        context_menu = self.v.get_context_menu(path)
        if len(context_menu) > 0:
            menu = wx.Menu()
            try:
                delete_menu_items = []
                for context_item in context_menu:
                    if isinstance(context_item,
                                  cellprofiler.setting.FileCollectionDisplay.DeleteMenuItem):
                        delete_menu_items.append(
                                menu.Append(-1, context_item.text).Id)
                    else:
                        menu.Append(-1, context_item)

                def on_menu(event):
                    logger.debug("On menu")

                    self.pipeline.start_undoable_action()
                    try:
                        for menu_item in menu.GetMenuItems():
                            if menu_item.Id == event.Id:
                                logger.debug("    Command = %s" % menu_item.Text)
                                if menu_item.Id in delete_menu_items:
                                    self.on_delete_selected(event)
                                else:
                                    self.v.fn_on_menu_command(path, menu_item.Text)
                                break
                    finally:
                        self.pipeline.stop_undoable_action()

                self.tree_ctrl.Bind(wx.EVT_MENU, on_menu)
                self.tree_ctrl.PopupMenu(menu, event.GetPoint())
                self.tree_ctrl.Unbind(wx.EVT_MENU, handler=on_menu)
            finally:
                menu.Destroy()

    def on_tree_doubleclick(self, event):
        path = self.get_path_from_event(event)
        if self.v.fn_on_menu_command(path, None):
            return True

    def on_tree_key_down(self, event):
        logger.debug("On tree key down")
        key = event.GetKeyCode()
        if key == wx.WXK_DELETE:
            self.on_delete_selected(event)

    def on_delete_selected(self, event):
        mods = [self.get_item_address(item)
                for item in self.tree_ctrl.GetSelections()]
        mods = filter(lambda x: x is not None, mods)
        self.v.on_remove([self.v.get_tree_modpaths(mod) for mod in mods])

    def get_item_address(self, item):
        """Get an item's address as a collection of names"""
        result = []
        while True:
            name = self.tree_ctrl.GetItemPyData(item)
            if name is None:
                break
            else:
                result.insert(0, name)
                item = self.tree_ctrl.GetItemParent(item)
        return result

    def get_item_from_modpath(self, modpath):
        """Get an item from its modpath

        returns the tree item id or None if not found.
        """
        return self.modpath_to_item.get(tuple(modpath))

    def request_update(self, hint=None, modpath=None):
        if hint == cellprofiler.setting.FileCollectionDisplay.BKGND_RESUME:
            self.on_start_received()
            return
        if hint == cellprofiler.setting.FileCollectionDisplay.BKGND_STOP:
            self.on_stop_received()
            self.status_text.Label = "Idle..."
            return
        if modpath is not None and len(modpath) > 0:
            #
            # Descend down the leftmost side of all of the tuples
            # to get something we can display
            #
            path = []
            mp = modpath[0]
            any_others = len(modpath) > 1
            if hint != cellprofiler.setting.FileCollectionDisplay.REMOVE:
                # It's likely that the leaf was removed and it doesn't
                # make sense to descend
                file_tree = self.v.file_tree
            is_filtered = False
            while True:
                if isinstance(mp, basestring) or isinstance(mp, tuple) and len(mp) == 3:
                    path.append(mp)
                    if hint != cellprofiler.setting.FileCollectionDisplay.REMOVE:
                        is_filtered = not file_tree[mp]
                    break
                part, mp_list = mp
                path.append(part)
                if hint != cellprofiler.setting.FileCollectionDisplay.REMOVE:
                    file_tree = file_tree[part]
                if len(mp_list) == 0:
                    is_filtered = not file_tree[None]
                    break
                any_others = any_others or len(mp_list) > 1
                mp = mp_list[0]
            if hint != cellprofiler.setting.FileCollectionDisplay.REMOVE:
                self.status_text.Label = \
                    ("Processing " + path[-1] if isinstance(path[-1], basestring)
                     else path[-2])
            self.status_text.Update()
            if not any_others:
                #
                # It's just a modification to a single node. Try and handle
                # here.
                #
                if hint == cellprofiler.setting.FileCollectionDisplay.METADATA:
                    if (not self.v.show_filtered) and is_filtered:
                        return
                    item_id = self.get_item_from_modpath(path)
                    if item_id is not None:
                        text, node_type, tooltip = self.v.get_node_info(path)
                        image_id = self.get_image_id_from_nodetype(node_type)
                        self.tree_ctrl.SetItemText(item_id, text)
                        self.tree_ctrl.SetItemImage(item_id, image_id)
                        return
                elif hint == cellprofiler.setting.FileCollectionDisplay.ADD:
                    if self.get_item_from_modpath(path) is None:
                        text, node_type, tooltip = self.v.get_node_info(path)
                        item_id = self.add_item(path, text)
                        image_id = self.get_image_id_from_nodetype(node_type)
                        self.tree_ctrl.SetItemImage(item_id, image_id)
                        self.manage_expansion()
                        return
                elif hint == cellprofiler.setting.FileCollectionDisplay.REMOVE:
                    if is_filtered:
                        return
                    self.remove_item(path)
                    if len(path) > 1:
                        super_modpath = tuple(path[:-1])
                        if super_modpath in self.modpath_to_item:
                            item = self.modpath_to_item[super_modpath]
                            n_children = self.tree_ctrl.GetChildrenCount(
                                    item, False)
                            if n_children == 0:
                                self.remove_item(super_modpath)

                    return
        self.update()

    def update(self):
        operation_id = uuid.uuid4()
        total = self.v.node_count()
        if total == 0:
            return
        self.update_subtree(self.v.file_tree, self.root_item, False, [],
                            operation_id, 0, total)
        self.manage_expansion()
        cellprofiler.preferences.report_progress(operation_id, 1, None)

    def manage_expansion(self):
        """Handle UI expansion issues

        Make sure that the tree is auto-expanded if appropriate and that
        the root nodes are expanded.
        """
        if not self.user_collapsed_a_node:
            #
            # Expand all until we reach a node that has more than
            # one child = ambiguous choice of which to expand
            #
            item = self.root_item
            while self.tree_ctrl.GetChildrenCount(item, False) == 1:
                # Can't expand the invisible root for Mac
                if sys.platform != "darwin" or item != self.root_item:
                    self.tree_ctrl.Expand(item)
                item, cookie = self.tree_ctrl.GetFirstChild(item)
            if self.tree_ctrl.GetChildrenCount(item, False) > 0:
                self.tree_ctrl.Expand(item)
        #
        # The bottom-most nodes don't have expand buttons (why?). If you
        # have two bottom-most nodes, neither will be expanded and there
        # is no way to expand them using the UI. So, we need to make sure
        # all bottom-most nodes are expanded, no matter what.
        #
        for i in range(self.tree_ctrl.GetChildrenCount(self.root_item, False)):
            if i == 0:
                bottom_item, thing = \
                    self.tree_ctrl.GetFirstChild(self.root_item)
            else:
                bottom_item, thing = \
                    self.tree_ctrl.GetNextChild(self.root_item, thing)
            if not self.tree_ctrl.IsExpanded(bottom_item):
                self.tree_ctrl.Expand(bottom_item)

    def update_subtree(self, file_tree, parent_item,
                       is_filtered, modpath, operation_id, count, total):
        existing_items = {}
        show_filtered = self.v.show_filtered
        needs_sort = False
        child_count = self.tree_ctrl.GetChildrenCount(parent_item, False)
        if child_count > 0:
            child_item_id, cookie = self.tree_ctrl.GetFirstChild(parent_item)
            for i in range(child_count):
                existing_items[self.tree_ctrl.GetItemPyData(child_item_id)] = \
                    [child_item_id, False]
                if i < child_count - 1:
                    child_item_id = \
                        self.tree_ctrl.GetNextSibling(child_item_id)

        for x in sorted(file_tree.keys()):
            sub_modpath = modpath + [x]
            if x is None:
                continue
            text, node_type, tooltip = self.v.get_node_info(sub_modpath)
            cellprofiler.preferences.report_progress(
                    operation_id,
                    float(count) / float(total),
                    "Processing %s" % text)
            count += 1
            image_id = self.get_image_id_from_nodetype(node_type)
            if isinstance(file_tree[x], bool) or isinstance(x, tuple):
                node_is_filtered = (not file_tree[x]) or is_filtered
                if node_is_filtered and not show_filtered:
                    continue
                if existing_items.has_key(x):
                    existing_items[x][1] = True
                    item_id = existing_items[x][0]
                    self.tree_ctrl.SetItemText(item_id, text)
                else:
                    item_id = self.add_item(sub_modpath, text, sort=False)
                    existing_items[x] = (item_id, True)
                    needs_sort = True

                self.tree_ctrl.SetItemImage(item_id, image_id)
            elif isinstance(file_tree[x], dict):
                subtree = file_tree[x]
                node_is_filtered = (not subtree[None]) or is_filtered
                unfiltered_subfolders, filtered_subfolders, \
                unfiltered_files, filtered_files = \
                    self.get_file_and_folder_counts(subtree)
                n_subfolders = unfiltered_subfolders + filtered_subfolders
                n_files = unfiltered_files + filtered_files
                if node_is_filtered and not show_filtered:
                    continue
                if node_type in (cellprofiler.setting.FileCollectionDisplay.NODE_COMPOSITE_IMAGE,
                                 cellprofiler.setting.FileCollectionDisplay.NODE_MOVIE):
                    expanded_image_id = image_id
                else:
                    image_id = self.FOLDER_IMAGE_INDEX
                    expanded_image_id = self.FOLDER_OPEN_IMAGE_INDEX
                    text = "" + x
                    if n_subfolders > 0 or n_files > 0:
                        text += " ("
                        if n_subfolders > 0:
                            if node_is_filtered:
                                text += "\t%d folders" % n_subfolders
                            else:
                                text += "\t%d of %d folders" % (
                                    unfiltered_subfolders, n_subfolders)
                            if n_files > 0:
                                text += ", "
                        if n_files > 0:
                            if node_is_filtered:
                                text += "\t%d files" % n_files
                            else:
                                text += "\t%d of %d files" % (
                                    unfiltered_files, n_files)
                        text += ")"
                if existing_items.has_key(x):
                    existing_items[x][1] = True
                    item_id = existing_items[x][0]
                    self.tree_ctrl.SetItemText(item_id, text)
                else:
                    item_id = self.add_item(sub_modpath, text, sort=False)
                    existing_items[x] = (item_id, True)
                    needs_sort = True
                self.tree_ctrl.SetItemImage(item_id, image_id)
                self.tree_ctrl.SetItemImage(item_id, expanded_image_id,
                                            wx.TreeItemIcon_Expanded)
                has_children = n_subfolders + n_files > 0
                self.tree_ctrl.SetItemHasChildren(item_id, has_children)
                count = self.update_subtree(
                        subtree, item_id, node_is_filtered,
                        sub_modpath, operation_id, count, total)

            color = self.FILTERED_COLOR if node_is_filtered else self.ACTIVE_COLOR
            self.tree_ctrl.SetItemTextColour(item_id, color)
        for last_part, (item_id, keep) in existing_items.iteritems():
            if not keep:
                self.remove_item(modpath + [last_part])
        if needs_sort:
            self.tree_ctrl.SortChildren(parent_item)
        return count

    def get_image_id_from_nodetype(self, node_type):
        if node_type == cellprofiler.setting.FileCollectionDisplay.NODE_COLOR_IMAGE:
            image_id = self.COLOR_IMAGE_INDEX
        elif node_type == cellprofiler.setting.FileCollectionDisplay.NODE_COMPOSITE_IMAGE:
            image_id = self.IMAGE_PLANES_IMAGE_INDEX
        elif node_type in (cellprofiler.setting.FileCollectionDisplay.NODE_MONOCHROME_IMAGE,
                           cellprofiler.setting.FileCollectionDisplay.NODE_IMAGE_PLANE):
            image_id = self.IMAGE_PLANE_IMAGE_INDEX
        elif node_type == cellprofiler.setting.FileCollectionDisplay.NODE_MOVIE:
            image_id = self.MOVIE_IMAGE_INDEX
        else:
            image_id = self.FILE_IMAGE_INDEX
        return image_id

    @classmethod
    def get_file_and_folder_counts(cls, tree):
        """Count the number of files and folders in the tree

        returns the number of immediate unfiltered and filtered subfolders
        and number of unfiltered and filtered files in the hierarchy
        """
        unfiltered_subfolders = filtered_subfolders = 0
        unfiltered_files = filtered_files = 0
        for key in tree:
            if key is None:
                continue
            if isinstance(tree[key], bool):
                if tree[key]:
                    unfiltered_files += 1
                else:
                    filtered_files += 1
            else:
                is_filtered = not tree[key][None]
                if is_filtered:
                    unfiltered_subfolders += 1
                else:
                    filtered_subfolders += 1
                ufolders, ffolders, ufiles, ffiles = \
                    cls.get_file_and_folder_counts(tree[key])
                filtered_files += ffiles
                if is_filtered:
                    filtered_files += ufiles
                else:
                    unfiltered_files += ufiles
        return unfiltered_subfolders, filtered_subfolders, unfiltered_files, \
               filtered_files

    def on_hide_show_checked(self, event):
        self.v.show_filtered = not self.hide_show_ctrl.Value
        self.request_update()


class JoinerController(object):
    """The JoinerController managers a joiner setting"""
    #
    # It's important that DISPLAY_NONE be an illegal name for metadata
    # so that it can be recognized by its string. If this isn't acceptable,
    # code must be added to keep track of its position in each dropdown.
    #
    DISPLAY_NONE = "(None)"

    def __init__(self, module_view, v):
        super(self.__class__, self).__init__()
        assert isinstance(module_view, ModuleView)
        self.module_view = module_view
        self.v = v
        self.panel = wx.Panel(module_view.module_panel, -1,
                              name=edit_control_name(v))
        self.panel.Sizer = wx.lib.rcsizer.RowColSizer()
        self.panel.joiner_controller = self
        self.update()

    def get_header_control_name(self, colidx):
        return "header_%d_%s" % (colidx, str(self.v.key()))

    def get_add_button_control_name(self, rowidx):
        return "add_button_%d_%s" % (rowidx, str(self.v.key()))

    def get_delete_button_control_name(self, rowidx):
        return "delete_button_%d_%s" % (rowidx, str(self.v.key()))

    def get_up_button_control_name(self, rowidx):
        return "up_button_%d_%s" % (rowidx, str(self.v.key()))

    def get_down_button_control_name(self, rowidx):
        return "down_button_%d_%s" % (rowidx, str(self.v.key()))

    def get_choice_control_name(self, rowidx, colidx):
        return "choice_%d_%d_%s" % (rowidx, colidx, str(self.v.key()))

    @classmethod
    def update_control(cls, module_view, v):
        """Update the Joiner setting's control

        returns the control
        """
        assert isinstance(module_view, ModuleView)
        control = module_view.module_panel.FindWindowByName(edit_control_name(v))
        if control is None:
            jc = JoinerController(module_view, v)
            return jc.panel
        else:
            control.joiner_controller.update()
            return control

    @property
    def column_names(self):
        """Names of the entities in alphabetical order"""
        return sorted(self.v.entities.keys())

    @property
    def joins(self):
        """The join rows of the controlled setting

        Each row is a dictionary of key / value where key is the entity name
        and value is the column or metadata value for the join row.
        """
        return self.v.parse()

    def update(self):
        """Update the control to match the setting"""
        column_names = self.column_names
        joins = self.joins
        if len(joins) == 0:
            joins = [dict([(cn, "") for cn in column_names])]

        all_subcontrols = {}
        self.panel.Sizer.Clear()
        for ctrl in self.panel.GetChildren():
            assert isinstance(ctrl, wx.Window)
            all_subcontrols[ctrl.GetName()] = False

        for i, column_name in enumerate(column_names):
            header_control_name = self.get_header_control_name(i)
            ctrl = self.panel.FindWindowByName(header_control_name)
            if ctrl is None:
                ctrl = wx.StaticText(self.panel, -1, column_name,
                                     name=header_control_name)
            else:
                ctrl.Label = column_name
            self.panel.Sizer.Add(
                    ctrl, row=0, col=i,
                    flag=wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_BOTTOM)
            all_subcontrols[header_control_name] = True

        for i, join in enumerate(joins):
            for j, column_name in enumerate(column_names):
                choice_ctrl_name = self.get_choice_control_name(i, j)
                ctrl = self.panel.FindWindowByName(choice_ctrl_name)
                selection = join.get(column_name, self.DISPLAY_NONE)
                if selection is None:
                    selection = self.DISPLAY_NONE
                choices = sorted(self.v.entities.get(column_name, []))
                if self.v.allow_none:
                    choices += [self.DISPLAY_NONE]
                if selection not in choices:
                    choices += [selection]
                if ctrl is None:
                    ctrl = wx.Choice(self.panel, -1,
                                     choices=choices,
                                     name=choice_ctrl_name)
                    ctrl.Bind(wx.EVT_CHOICE,
                              lambda event, row=i, col=j:
                              self.on_choice_changed(event, row, col))
                else:
                    ctrl.SetItems(choices)
                ctrl.SetStringSelection(selection)
                self.panel.Sizer.Add(ctrl, row=i + 1, col=j,
                                     flag=wx.ALIGN_BOTTOM)
                all_subcontrols[choice_ctrl_name] = True

            add_button_name = self.get_add_button_control_name(i)
            ctrl = self.panel.FindWindowByName(add_button_name)
            if ctrl is None:
                ctrl = wx.Button(self.panel, -1, "+",
                                 name=add_button_name,
                                 style=wx.BU_EXACTFIT)
                ctrl.Bind(wx.EVT_BUTTON,
                          lambda event, position=i + 1:
                          self.on_insert_row(event, position))
            self.panel.Sizer.Add(ctrl, row=i + 1, col=len(column_names),
                                 flag=wx.ALIGN_BOTTOM)
            all_subcontrols[add_button_name] = True

            if len(joins) > 1:
                delete_button_name = self.get_delete_button_control_name(i)
                ctrl = self.panel.FindWindowByName(delete_button_name)
                if ctrl is None:
                    ctrl = wx.Button(self.panel, -1, "-",
                                     name=delete_button_name,
                                     style=wx.BU_EXACTFIT)
                    ctrl.Bind(wx.EVT_BUTTON,
                              lambda event, position=i:
                              self.on_delete_row(event, position))
                self.panel.Sizer.Add(ctrl, row=i + 1, col=len(column_names) + 1,
                                     flag=wx.ALIGN_BOTTOM)
                all_subcontrols[delete_button_name] = True

            if i > 0:
                move_up_button_name = self.get_up_button_control_name(i)
                ctrl = self.panel.FindWindowByName(move_up_button_name)
                if ctrl is None:
                    img = wx.ArtProvider.GetBitmap(wx.ART_GO_UP,
                                                   wx.ART_BUTTON,
                                                   (16, 16))
                    ctrl = wx.BitmapButton(self.panel, -1, img,
                                           name=move_up_button_name)
                    ctrl.Bind(wx.EVT_BUTTON,
                              lambda event, position=i:
                              self.on_move_row_up(event, position))
                self.panel.Sizer.Add(ctrl, row=i + 1, col=len(column_names) + 2,
                                     flag=wx.ALIGN_BOTTOM)
                all_subcontrols[move_up_button_name] = True

            if i < len(joins) - 1:
                move_down_button_name = self.get_down_button_control_name(i)
                ctrl = self.panel.FindWindowByName(move_down_button_name)
                if ctrl is None:
                    img = wx.ArtProvider.GetBitmap(wx.ART_GO_DOWN,
                                                   wx.ART_BUTTON,
                                                   (16, 16))
                    ctrl = wx.BitmapButton(self.panel, -1, img,
                                           name=move_down_button_name)
                    ctrl.Bind(wx.EVT_BUTTON,
                              lambda event, position=i:
                              self.on_move_row_down(event, position))
                self.panel.Sizer.Add(ctrl, row=i + 1, col=len(column_names) + 3,
                                     flag=wx.ALIGN_BOTTOM)
                all_subcontrols[move_down_button_name] = True

        for key, value in all_subcontrols.iteritems():
            ctrl = self.panel.FindWindowByName(key)
            ctrl.Show(value)

    def on_choice_changed(self, event, row, column):
        new_value = event.EventObject.GetItems()[event.GetSelection()]
        if new_value == self.DISPLAY_NONE:
            new_value = None
        joins = list(self.joins)
        while len(joins) <= row:
            joins.append(dict([(cn, "") for cn in self.column_names]))
        join = joins[row].copy()
        join[self.column_names[column]] = new_value
        joins[row] = join
        self.module_view.on_value_change(self.v, self.panel,
                                         self.v.build_string(joins), event)

    def on_insert_row(self, event, position):
        joins = list(self.joins)
        new_join = dict([(column_name, None) for column_name in self.column_names])
        joins.insert(position, new_join)
        self.module_view.on_value_change(self.v, self.panel,
                                         self.v.build_string(joins), event)

    def on_delete_row(self, event, position):
        joins = list(self.joins)
        del joins[position]
        self.module_view.on_value_change(self.v, self.panel,
                                         self.v.build_string(joins), event)

    def on_move_row_up(self, event, position):
        joins = list(self.joins)
        joins = joins[0:(position - 1)] + [joins[position], joins[position - 1]] + \
                joins[(position + 1):]
        self.module_view.on_value_change(self.v, self.panel,
                                         self.v.build_string(joins), event)

    def on_move_row_down(self, event, position):
        joins = list(self.joins)
        joins = joins[0:position] + [joins[position + 1], joins[position]] + \
                joins[(position + 2):]
        self.module_view.on_value_change(self.v, self.panel,
                                         self.v.build_string(joins), event)


class BinaryMatrixController(object):
    """A controller for the BinaryMatrix setting
    """

    def __init__(self, module_view, v):
        """Initialize the controller

        module_view - the module_view that contains the controller's panel

        v - the setting
        """
        self.module_view = module_view
        self.setting = v
        self.panel = wx.Panel(module_view.module_panel,
                              name=edit_control_name(v))
        self.panel.controller = self
        self.panel.Sizer = wx.BoxSizer(wx.VERTICAL)
        self.matrix_ctrl = wx.Panel(self.panel)
        self.panel.Sizer.Add(self.matrix_ctrl, 0, wx.EXPAND | wx.ALL, 10)
        self.matrix_ctrl.SetMinSize(wx.Size(50, 50))
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.panel.Sizer.Add(sizer, 0, wx.EXPAND | wx.ALL, 2)
        sizer.Add(wx.StaticText(self.panel, label="Width:"),
                  0, wx.ALIGN_RIGHT | wx.ALIGN_CENTER)
        sizer.AddSpacer(1)
        self.width_ctrl = wx.SpinCtrl(self.panel)
        self.width_ctrl.SetRange(1, 100)
        sizer.Add(self.width_ctrl, 1, wx.EXPAND)
        sizer.AddSpacer(4)
        sizer.Add(wx.StaticText(self.panel, label="Height:"),
                  0, wx.ALIGN_RIGHT | wx.ALIGN_CENTER)
        sizer.AddSpacer(1)
        self.height_ctrl = wx.SpinCtrl(self.panel)
        self.height_ctrl.SetRange(1, 100)
        sizer.Add(self.height_ctrl, 1, wx.EXPAND)

        self.matrix_ctrl.Bind(wx.EVT_PAINT, self.on_matrix_ctrl_paint)
        self.matrix_ctrl.Bind(wx.EVT_LEFT_UP, self.on_matrix_ctrl_clicked)
        self.width_ctrl.Bind(wx.EVT_SPINCTRL, self.on_width_changed)
        self.height_ctrl.Bind(wx.EVT_SPINCTRL, self.on_height_changed)
        self.update()

    def update(self):
        h, w = self.setting.get_size()
        hh, ww = [(x - 1) / 2 for x in h, w]
        if self.height_ctrl.Value != hh:
            self.height_ctrl.Value = hh
        if self.width_ctrl.Value != ww:
            self.width_ctrl.Value = ww
        r = self.get_matrix_element_rect(h - 1, w - 1)
        b = wx.SystemSettings.GetMetric(wx.SYS_EDGE_X)
        self.matrix_ctrl.SetMinSize(wx.Size(r.GetRight() + b, r.GetBottom() + b))
        self.matrix_ctrl.Refresh(eraseBackground=False)

    def on_matrix_ctrl_clicked(self, event):
        assert isinstance(event, wx.MouseEvent)
        i, j = self.hit_test(event.X, event.Y)
        if i is not None:
            matrix = self.setting.get_matrix()
            matrix[i][j] = not matrix[i][j]
            value = cellprofiler.setting.BinaryMatrix.to_value(matrix)
            self.module_view.on_value_change(self.setting, self.panel,
                                             value, event)

    def on_matrix_ctrl_paint(self, event):
        paint_dc = wx.BufferedPaintDC(self.matrix_ctrl)
        paint_dc.BeginDrawing()
        matrix = self.setting.get_matrix()
        h = len(matrix)
        w = len(matrix[0])
        bx, ex, dx, by, ey, dy = [
            wx.SystemSettings.GetMetric(m) for m in (
                wx.SYS_BORDER_X, wx.SYS_EDGE_X, wx.SYS_SMALLICON_X,
                wx.SYS_BORDER_Y, wx.SYS_EDGE_Y, wx.SYS_SMALLICON_Y)]
        paint_dc.Background = wx.Brush(cellprofiler.preferences.get_background_color())
        paint_dc.Clear()
        pShadow = wx.Pen(
                wx.SystemSettings.GetColour(wx.SYS_COLOUR_BTNSHADOW), 1, wx.SOLID)
        pHighlight = wx.Pen(
                wx.SystemSettings.GetColour(wx.SYS_COLOUR_BTNHIGHLIGHT), 1, wx.SOLID)
        bBackground, bForeground = [
            wx.Brush(color) for color in
            wx.Colour(80, 80, 80, 255), wx.WHITE]
        rw = 2 * ex + dx
        rh = 2 * ey + dy
        for x in range(w):
            for y in range(h):
                rx = x * rw + x * bx
                ry = y * rh + y * by
                value = matrix[y][x]
                paint_dc.Pen = pHighlight if value else pShadow
                for k in range(ex):
                    paint_dc.DrawLine(rx + k, ry + k, rx + rw - k - 1, ry + k)
                    paint_dc.DrawLine(rx + k, ry + k, rx + k, ry + rh - k - 1)
                paint_dc.Pen = pShadow if value else pHighlight
                for k in range(ex):
                    paint_dc.DrawLine(rx + k, ry + rh - k - 1, rx + rw - k - 1, ry + rh - k - 1)
                    paint_dc.DrawLine(rx + rw - k - 1, ry + k, rx + rw - k - 1, ry + rh - k - 1)
                paint_dc.Pen = wx.TRANSPARENT_PEN
                paint_dc.Brush = bForeground if value else bBackground
                paint_dc.DrawRectangle(rx + ex, ry + ey, dx, dy)
        paint_dc.EndDrawing()
        event.Skip()

    @staticmethod
    def get_matrix_element_rect(i, j):
        bx, ex, dx, by, ey, dy = [
            wx.SystemSettings.GetMetric(m) for m in (
                wx.SYS_BORDER_X, wx.SYS_EDGE_X, wx.SYS_SMALLICON_X,
                wx.SYS_BORDER_Y, wx.SYS_EDGE_Y, wx.SYS_SMALLICON_Y)]
        return wx.Rect(ex * (2 * j + 1) + dx * j + bx * j,
                       ey * (2 * i + 1) + dy * i + by * i, dx, dy)

    def hit_test(self, x, y):
        """Return the i, j coordinates at the mouse

        returns i, j or None, None if misses the hit test
        """
        bx, ex, dx, by, ey, dy = [
            wx.SystemSettings.GetMetric(m) for m in (
                wx.SYS_BORDER_X, wx.SYS_EDGE_X, wx.SYS_SMALLICON_X,
                wx.SYS_BORDER_Y, wx.SYS_EDGE_Y, wx.SYS_SMALLICON_Y)]
        i = int((y - ey) / (2 * ey + dy + by))
        j = int((x - ex) / (2 * ex + dx + bx))
        h, w = self.setting.get_size()
        if (i < 0 or j < 0 or i >= h or j >= w or
                not self.get_matrix_element_rect(i, j).ContainsXY(x, y)):
            return None, None
        return i, j

    def on_width_changed(self, event):
        matrix = self.setting.get_matrix()
        h, w = self.setting.get_size()
        d = self.width_ctrl.Value * 2 + 1 - w
        n = abs(int(d / 2))
        if d < 0:
            matrix = [row[n:-n] for row in matrix]
        elif d > 0:
            matrix = [[False] * n + row + [False] * n for row in matrix]
        else:
            return
        value = cellprofiler.setting.BinaryMatrix.to_value(matrix)
        self.module_view.on_value_change(self.setting, self.panel,
                                         value, event)

    def on_height_changed(self, event):
        matrix = self.setting.get_matrix()
        h, w = self.setting.get_size()
        d = self.height_ctrl.Value * 2 + 1 - h
        n = abs(int(d / 2))
        if d < 0:
            matrix = matrix[n:-n]
        elif d > 0:
            matrix = [[False] * w for _ in range(n)] + matrix + \
                     [[False] * w for _ in range(n)]
        else:
            return
        value = cellprofiler.setting.BinaryMatrix.to_value(matrix)
        self.module_view.on_value_change(self.setting, self.panel,
                                         value, event)

    @classmethod
    def update_control(cls, module_view, v):
        """Update the Joiner setting's control

        returns the control
        """
        assert isinstance(module_view, ModuleView)
        control = module_view.module_panel.FindWindowByName(edit_control_name(v))
        if control is None:
            controller = BinaryMatrixController(module_view, v)
            return controller.panel
        else:
            control.controller.update()
            return control


class DataTypeController(object):
    """The DataTypeController manages a DataType setting"""
    DTC_NONE = "None"
    DTC_TEXT = "Text"
    DTC_INTEGER = "Integer"
    DTC_FLOAT = "Float"
    DTC_TO_DT = {
        DTC_NONE: cellprofiler.setting.DataTypes.DT_NONE,
        DTC_TEXT: cellprofiler.setting.DataTypes.DT_TEXT,
        DTC_INTEGER: cellprofiler.setting.DataTypes.DT_INTEGER,
        DTC_FLOAT: cellprofiler.setting.DataTypes.DT_FLOAT,
        None: cellprofiler.setting.DataTypes.DT_TEXT}
    DT_TO_DTC = {
        cellprofiler.setting.DataTypes.DT_NONE: DTC_NONE,
        cellprofiler.setting.DataTypes.DT_TEXT: DTC_TEXT,
        cellprofiler.setting.DataTypes.DT_INTEGER: DTC_INTEGER,
        cellprofiler.setting.DataTypes.DT_FLOAT: DTC_FLOAT}

    def __init__(self, module_view, v):
        assert isinstance(v, cellprofiler.setting.DataTypes)
        self.module_view = module_view
        self.v = v
        self.panel = module_view.module_panel.FindWindowByName(
                edit_control_name(v))
        if self.panel is None:
            class DoesntInheritBackgroundColor(wx.Panel):
                def InheritsBackgroundColour(self):
                    return False

            self.panel = DoesntInheritBackgroundColor(
                    module_view.module_panel, -1,
                    name=edit_control_name(v))
            self.panel.Sizer = wx.lib.rcsizer.RowColSizer()
            self.panel.Bind(wx.EVT_PAINT, self.on_paint)
        self.panel.controller = self
        self.n_items = 0
        self.update()

    def on_paint(self, event):
        dc = wx.BufferedPaintDC(self.panel)
        dc.BeginDrawing()
        dc.Background = wx.Brush(wx.SystemSettings.GetColour(
                wx.SYS_COLOUR_WINDOW))
        dc.Clear()
        dc.Pen = wx.Pen(wx.SystemSettings.GetColour(
                wx.SYS_COLOUR_GRAYTEXT))
        sizer = self.panel.Sizer
        _, panel_width = self.panel.GetClientSize()
        assert isinstance(sizer, wx.lib.rcsizer.RowColSizer)
        bottom_choice_name = self.get_choice_control_name(self.n_items)
        bottom_choice = self.panel.FindWindowByName(bottom_choice_name)
        if bottom_choice is not None:
            r = bottom_choice.GetRect()
            dc.DrawLine(r.Left - 2, 1, r.Left - 2, r.Bottom)
        for i in range(1, self.n_items + 1):
            choice_name = self.get_choice_control_name(i)
            choice = self.panel.FindWindowByName(choice_name)
            if choice is not None:
                r = choice.GetRect()
                dc.DrawLine(1, r.Top - 2, panel_width - 1, r.Top - 2)
        dc.EndDrawing()
        event.Skip()

    def get_label_control_name(self, rowidx):
        """The name of the label control that holds the feature name"""
        return "label_control_%d_%s" % (rowidx, str(self.v.key()))

    def get_choice_control_name(self, rowidx):
        """The name of the choice control holding the data type choices"""
        return "choice_control_%d_%s" % (rowidx, str(self.v.key()))

    def update(self):
        """Change the UI state to match that of the DataTypes setting"""
        d = self.v.get_data_types()
        needs_bind = []
        sizer = self.panel.Sizer
        assert isinstance(sizer, wx.lib.rcsizer.RowColSizer)
        for child in self.panel.GetChildren():
            sizer.Hide(child)

        label_header_name = self.get_label_control_name(0)
        choice_header_name = self.get_choice_control_name(0)
        for header_name, column, text in (
                (label_header_name, 0, "Key"),
                (choice_header_name, 1, "Data type")):
            ctrl = self.panel.FindWindowByName(header_name)
            if ctrl is None:
                ctrl = wx.StaticText(self.panel,
                                     label=text,
                                     name=header_name)
                ctrl.Font = wx.Font(
                        ctrl.Font.GetPointSize(),
                        ctrl.Font.GetFamily(),
                        ctrl.Font.GetStyle(),
                        wx.FONTWEIGHT_BOLD, False,
                        ctrl.Font.GetFaceName())
                sizer.Add(ctrl,
                          flag=wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_CENTER_VERTICAL | wx.BOTTOM,
                          border=1,
                          row=0, col=column)
            else:
                sizer.Show(ctrl)
        for i, feature in enumerate(sorted(d.keys())):
            label_name = self.get_label_control_name(i + 1)
            choice_name = self.get_choice_control_name(i + 1)
            label = self.panel.FindWindowByName(label_name)
            if label is None:
                label = wx.StaticText(
                        self.panel,
                        label=feature,
                        name=label_name)
                sizer.Add(label,
                          flag=wx.ALIGN_LEFT | wx.BOTTOM | wx.LEFT | wx.RIGHT,
                          border=3,
                          row=i + 1, col=0)
            else:
                sizer.Show(label)
                if label.Label != feature:
                    label.Label = feature
            choice = self.panel.FindWindowByName(choice_name)
            if choice is None:
                choice = wx.Choice(
                        self.panel,
                        choices=[self.DTC_TEXT, self.DTC_INTEGER, self.DTC_FLOAT,
                                 self.DTC_NONE],
                        name=choice_name)
                sizer.Add(choice,
                          flag=wx.EXPAND | wx.BOTTOM | wx.RIGHT,
                          border=3,
                          row=i + 1, col=1)
                needs_bind.append(choice)
            else:
                sizer.Show(choice)
            value = self.DT_TO_DTC.get(d[feature], self.DTC_TEXT)
            if choice.GetStringSelection() != value:
                choice.SetStringSelection(value)
        self.n_items = len(d.keys())
        for choice in needs_bind:
            choice.Bind(wx.EVT_CHOICE, self.on_choice_changed)

    def on_choice_changed(self, event):
        result = {}
        for i in range(1, self.n_items + 1):
            label = self.panel.FindWindowByName(self.get_label_control_name(i))
            choice = self.panel.FindWindowByName(
                    self.get_choice_control_name(i))
            result[label.Label] = self.DTC_TO_DT[choice.GetStringSelection()]
        result = cellprofiler.setting.DataTypes.encode_data_types(result)
        if self.v.value != result:
            self.module_view.on_value_change(self.v, self.panel, result, event)

    @classmethod
    def update_control(cls, module_view, v):
        """Update the Joiner setting's control

        returns the control
        """
        assert isinstance(module_view, ModuleView)
        control = module_view.module_panel.FindWindowByName(edit_control_name(v))
        if control is None:
            controller = DataTypeController(module_view, v)
            return controller.panel
        else:
            control.controller.update()
            return control


class TableController(wx.grid.PyGridTableBase):
    DEFAULT_ATTR = wx.grid.GridCellAttr()
    ERROR_ATTR = wx.grid.GridCellAttr()
    ERROR_ATTR.TextColour = cellprofiler.preferences.get_error_color()

    def __init__(self, v):
        super(self.__class__, self).__init__()
        assert isinstance(v, cellprofiler.setting.Table)
        self.v = v
        self.column_size = [v.max_field_size] * len(v.column_names)

    def bind_to_grid(self, grid):
        """Bind to intercept events on the grid

        Binds on_mouse_motion and on_column_resize in order to do tooltips.
        Sets up editing / auto size and other to customize for table type.
        """
        self.grid = grid
        grid.AutoSize()
        grid.EnableEditing(False)
        grid.SetDefaultCellOverflow(False)
        if self.v.corner_button is None:
            grid.fn_clicked = None
        else:
            fn_clicked = self.v.corner_button["fn_clicked"]

            def on_corner_button_clicked():
                fn_clicked()
                self.update_grid()
                grid.ForceRefresh()
                grid.Parent.Layout()

            grid.fn_clicked = on_corner_button_clicked
            grid.label = self.v.corner_button.get("label", "Update")
            grid.tooltip = self.v.corner_button.get("tooltip", "")
        #
        # Below largely taken from
        # http://wiki.wxpython.org/wxGrid%20ToolTips
        #
        self.last_pos = (None, None)
        grid.GetGridWindow().Bind(wx.EVT_MOTION, self.on_mouse_motion)
        grid.Bind(wx.grid.EVT_GRID_COL_SIZE, self.on_column_resize)

    def update_grid(self):
        """Update the grid after the table data has changed"""
        need_column_layout = False
        grid = self.grid
        v = self.v
        if len(v.column_names) < grid.GetNumberCols():
            tm = wx.grid.GridTableMessage(
                    grid.Table,
                    wx.grid.GRIDTABLE_NOTIFY_COLS_DELETED,
                    0, grid.GetNumberCols() - len(v.column_names))
            grid.ProcessTableMessage(tm)
            need_column_layout = True
        elif grid.GetNumberCols() < len(v.column_names):
            tm = wx.grid.GridTableMessage(
                    grid.Table,
                    wx.grid.GRIDTABLE_NOTIFY_COLS_INSERTED,
                    0, len(v.column_names) - grid.GetNumberCols())
            grid.ProcessTableMessage(tm)
            need_column_layout = True
        if len(v.data) < grid.GetNumberRows():
            tm = wx.grid.GridTableMessage(
                    grid.Table,
                    wx.grid.GRIDTABLE_NOTIFY_ROWS_DELETED,
                    0, grid.GetNumberRows() - len(v.data))
            grid.ProcessTableMessage(tm)
        elif grid.GetNumberRows() < len(v.data):
            tm = wx.grid.GridTableMessage(
                    grid.Table,
                    wx.grid.GRIDTABLE_NOTIFY_ROWS_INSERTED,
                    0, len(v.data) - grid.GetNumberRows())
            grid.ProcessTableMessage(tm)
        if need_column_layout:
            grid.AutoSizeColumns()

    def on_mouse_motion(self, event):
        x, y = self.grid.CalcUnscrolledPosition(event.GetPosition())
        row = self.grid.YToRow(y)
        col = self.grid.XToCol(x)
        this_pos = (row, col)
        if (this_pos != self.last_pos and row >= 0 and col >= 0 and
                    row < len(self.v.data) and col < len(self.v.data[row])):
            self.last_pos = this_pos
            s = self.v.data[row][col]
            if s is None:
                s = ''
            elif not isinstance(s, basestring):
                s = str(s)
            self.grid.GetGridWindow().SetToolTipString(s)
        event.Skip()

    def on_column_resize(self, event):
        grid = self.grid
        col = event.GetRowOrCol()
        width = grid.GetColSize(col)
        table = grid.GetTable()
        self.column_size[col] = int(width * 1.1) / grid.CharWidth
        tm = wx.grid.GridTableMessage(
                self,
                wx.grid.GRIDTABLE_REQUEST_VIEW_GET_VALUES)
        grid.ProcessTableMessage(tm)
        grid.ForceRefresh()

    def GetAttr(self, row, col, kind):
        attrs = self.v.get_cell_attributes(
                row, self.v.column_names[col])
        attr = self.DEFAULT_ATTR
        if attrs is not None and self.v.ATTR_ERROR in attrs:
            attr = self.ERROR_ATTR
        attr.IncRef()  # OH so bogus, don't refcount = bus error
        return attr

    def CanHaveAttributes(self):
        return True

    def GetNumberRows(self):
        return len(self.v.data)

    def GetNumberCols(self):
        return len(self.v.column_names)

    def IsEmptyCell(self, row, col):
        return (len(self.v.data) <= row or
                len(self.v.data[row]) <= col or
                self.v.data[row][col] is None)

    def GetValue(self, row, col):
        if self.IsEmptyCell(row, col):
            return None
        s = unicode(self.v.data[row][col])
        if len(self.column_size) <= col:
            self.column_size += [self.v.max_field_size] * (col - len(self.column_size) + 1)
        field_size = self.column_size[col]
        if len(s) > field_size:
            half = int(field_size - 3) / 2
            s = s[:half] + "..." + s[-half:]
        return s

    def GetRowLabelValue(self, row):
        attrs = self.v.get_row_attributes(row)
        if attrs is not None and self.v.ATTR_ERROR in attrs:
            return "%d: Error" % (row + 1)
        return str(row + 1)

    def GetColLabelValue(self, col):
        return self.v.column_names[col]

    def AppendCols(self, numCols):
        return True

    def AppendRows(self, numRows):
        return True

    def InsertCols(self, index, numCols):
        return True

    def InsertRows(self, index, numRows):
        return True

    def DeleteCols(self, index, numCols):
        return True

    def DeleteRows(self, index, numRows):
        return True


class ModuleSizer(wx.PySizer):
    """The module sizer uses the maximum best width of the setting
    edit controls to compute the column widths, then it sets the text
    controls to wrap within the remaining space, then it uses the best
    height of each text control to lay out the rows.
    """

    def __init__(self, rows, cols=2):
        wx.PySizer.__init__(self)
        self.__rows = rows
        self.__cols = cols
        self.__min_text_width = 150
        self.__height_padding = 5
        self.__printed_exception = False
        self.__items = []

    def get_item(self, i, j):
        if len(self.__items) <= j or len(self.__items[j]) <= i:
            return None
        return self.__items[j][i]

    def Reset(self, rows, cols=3, destroy_windows=True):
        if destroy_windows:
            windows = []
            for j in range(self.__rows):
                for i in range(self.__cols):
                    item = self.get_item(i, j)
                    if item is None:
                        continue
                    if item.IsWindow():
                        window = item.GetWindow()
                        if isinstance(window, wx.Window):
                            windows.append(window)
            for window in windows:
                window.Hide()
                window.Destroy()
        self.Clear(False)
        self.__rows = rows
        self.__cols = cols
        self.__items = []

    def Add(self, control, *args, **kwargs):
        if len(self.__items) == 0 or len(self.__items[-1]) == self.__cols:
            self.__items.append([])
        item = super(ModuleSizer, self).Add(control, *args, **kwargs)
        self.__items[-1].append(item)
        return item

    def CalcMin(self):
        """Calculate the minimum from the edit controls.  Returns a
        wx.Size where the height is the total height of the grid and
        the width is self.__min_text_width plus the widths of the edit
        controls and help controls.
        """
        try:
            if (self.__rows * self.__cols == 0 or
                        self.Children is None or
                        len(self.Children) == 0):
                return wx.Size(0, 0)
            height = self.__height_padding
            for j in range(0, self.__rows):
                borders = [self.get_item(col, j).GetBorder()
                           for col in range(2)
                           if self.get_item(col, j) is not None]
                if len(borders) == 0:
                    height += 10
                else:
                    height_border = max(borders)
                    height += self.get_row_height(j) + 2 * height_border
            height += self.__height_padding
            self.__printed_exception = False
            return wx.Size(self.calc_edit_size()[0] + self.__min_text_width +
                           self.calc_help_size()[0],
                           height)
        except:
            # This happens, hopefully transiently, on the Mac
            if not self.__printed_exception:
                logger.error("WX internal error detected", exc_info=True)
                self.__printed_exception = True
            return wx.Size(0, 0)

    def get_row_height(self, j):
        height = 0
        for i in range(self.__cols):
            item = self.get_item(i, j)
            if item is None:
                continue
            if item.IsWindow() and isinstance(item.GetWindow(), wx.StaticLine):
                height = max(height, item.CalcMin()[1] * 1.25)
            else:
                height = max(height, item.CalcMin()[1])
        return height

    def calc_column_size(self, j):
        """Return a wx.Size with the total height of the controls in
        column j and the maximum of their widths.
        """
        height = 0
        width = 0
        for i in range(self.__rows):
            item = self.get_item(j, i)
            if item is None:
                continue
            size = item.CalcMin()
            height += size[1]
            width = max(width, size[0])
        return wx.Size(width, height)

    def calc_help_size(self):
        return self.calc_column_size(2)

    def calc_edit_size(self):
        return self.calc_column_size(1)

    def calc_max_text_width(self):
        width = self.__min_text_width
        for i in range(self.__rows):
            item = self.get_item(0, i)
            if item is None:
                continue
            control = item.GetWindow()
            assert isinstance(control, wx.StaticText), 'Control at column 0, ' \
                                                       '%d of grid is not StaticText: %s' % (i, str(control))
            text = control.GetLabel().replace('\n', ' ')
            ctrl_width = control.GetTextExtent(text)[0] + 2 * item.GetBorder()
            width = max(width, ctrl_width)
        return width

    def RecalcSizes(self):
        """Recalculate the sizes of our items, resizing the text boxes
        as we go.
        """
        if self.__rows * self.__cols == 0:
            return
        try:
            size = self.GetSize()
            width = size[0] - 20
            edit_width = self.calc_edit_size()[0]
            help_width = self.calc_help_size()[0]
            max_text_width = self.calc_max_text_width()
            if edit_width + help_width + max_text_width < width:
                edit_width = width - max_text_width - help_width
            elif edit_width * 4 < width:
                edit_width = width / 4
            text_width = max([width - edit_width - help_width,
                              self.__min_text_width])
            widths = [text_width, edit_width, help_width]
            #
            # Change all static text controls to wrap at the text width. Then
            # ask the items how high they are and do the layout of the line.
            #
            height = self.__height_padding
            panel = self.GetContainingWindow()
            for i in range(self.__rows):
                text_item = self.get_item(0, i)
                edit_item = self.get_item(1, i)
                if edit_item is None:
                    continue
                inner_text_width = text_width - 2 * text_item.GetBorder()
                control = text_item.GetWindow()
                assert isinstance(control, wx.StaticText), 'Control at column 0, %d of grid is not StaticText: %s' % (
                    i, str(control))
                text = control.GetLabel()
                edit_control = edit_item.GetWindow()
                height_border = max([x.GetBorder() for x in (edit_item, text_item)])
                if (isinstance(edit_control, wx.StaticLine) and
                            len(text) == 0):
                    #
                    # A line spans both columns
                    #
                    text_item.Show(False)
                    # make the divider height the same as a text row plus some
                    item_height = self.get_row_height(i)
                    assert isinstance(edit_item, wx.SizerItem)
                    border = edit_item.GetBorder()
                    third_width = (text_width + edit_width - 2 * border) / 3
                    item_location = wx.Point(text_width - third_width / 2,
                                             height + border + item_height / 2)
                    item_size = wx.Size(third_width, edit_item.Size[1])
                    item_location = panel.CalcScrolledPosition(item_location)
                    edit_item.SetDimension(item_location, item_size)
                else:
                    text_item.Show(True)
                    if (text_width > self.__min_text_width and
                            (text.find('\n') != -1 or
                                     control.GetTextExtent(text)[0] > inner_text_width)):
                        text = text.replace('\n', ' ')
                        control.SetLabel(text)
                        control.Wrap(inner_text_width)
                    row_height = self.get_row_height(i)
                    for j in range(self.__cols):
                        item = self.get_item(j, i)
                        item_x = sum(widths[0:j])
                        item_y = height
                        if (item.Flag & wx.EXPAND) == 0:
                            item_size = item.CalcMin()
                            if item.Flag & wx.ALIGN_CENTER_VERTICAL:
                                item_y = height + (row_height - item_size[1]) / 2
                            if item.Flag & wx.ALIGN_CENTER_HORIZONTAL:
                                item_x += (widths[j] - item_size[0]) / 2
                            elif item.Flag & wx.ALIGN_RIGHT:
                                item_x += widths[j] - item_size[0]
                        else:
                            item_size = wx.Size(widths[j], item.CalcMin()[1])
                        item_location = wx.Point(item_x, item_y)
                        item_location = panel.CalcScrolledPosition(item_location)
                        item.SetDimension(item_location, item_size)
                height += self.get_row_height(i) + 2 * height_border
        except:
            # This happens, hopefully transiently, on the Mac
            if not self.__printed_exception:
                logger.warning("Detected WX error", exc_info=True)
                self.__printed_exception = True


validation_queue = Queue.Queue()
pipeline_queue_thread = None  # global, protected by above lock
request_pipeline_cache = threading.local()  # used to cache the last requested pipeline
validation_queue_keep_running = True


class ValidationRequest(object):
    """A request for module validation"""

    def __init__(self, pipeline, module, callback):
        """Initialize the validation request

        pipeline - pipeline in question
        module - module in question
        callback - call this callback if there is an error. Do it on the GUI thread
        """
        self.pipeline = cache_pipeline(pipeline)
        self.module_num = module.module_num
        self.test_mode = pipeline.test_mode
        self.callback = callback
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


def cache_pipeline(pipeline):
    """Return a single cached copy of a pipeline to limit the # of copies"""
    d = getattr(request_pipeline_cache, "d", None)
    if d is None:
        d = weakref.WeakValueDictionary()
        setattr(request_pipeline_cache, "d", d)
    settings_hash = pipeline.settings_hash()
    result = d.get(settings_hash)
    if result is None:
        result = d[settings_hash] = pipeline.copy(False)
    return result


def validate_module(pipeline, module_num, test_mode, callback):
    """Validate a module and execute the callback on error on the main thread

    pipeline - a pipeline to be validated
    module_num - the module number of the module to be validated
    test_mode - whether pipeline is in test mode
    callback - a callback with the signature, "fn(setting, message, pipeline_data)"
    where setting is the setting that is in error and message is the message to
    display.
    """
    pipeline.test_mode = test_mode
    modules = [m for m in pipeline.modules() if m.module_num == module_num]
    if len(modules) != 1:
        return
    module = modules[0]
    level = logging.INFO
    setting_idx = None
    message = None
    try:
        level = logging.ERROR
        module.test_valid(pipeline)  # this method validates each visible
        # setting first, then the module itself.
        level = logging.WARNING
        module.test_module_warnings(pipeline)
        level = logging.INFO
    except cellprofiler.setting.ValidationError, instance:
        message = instance.message
        setting_idx = [m.key() for m in module.visible_settings()].index(instance.get_setting().key())
    wx.CallAfter(callback, setting_idx, message, level)


def validation_queue_handler():
    from javabridge import attach, detach
    attach()
    try:
        while validation_queue_keep_running:
            request = validation_queue.get()
            if not isinstance(request, ValidationRequest) or request.cancelled:
                continue
            start = time.clock()
            try:
                validate_module(request.pipeline, request.module_num,
                                request.test_mode, request.callback)
            except:
                pass
            # Make sure this thread utilizes less than 1/2 of GIL clock
            wait_for = max(.25, time.clock() - start)
            time.sleep(wait_for)
    finally:
        detach()
    logger.info("Exiting the pipeline validation thread")


def request_module_validation(validation_request):
    """Request that a module be validated

    """
    global pipeline_queue_thread, validation_queue

    if pipeline_queue_thread is None:
        pipeline_queue_thread = threading.Thread(target=validation_queue_handler)
        pipeline_queue_thread.setName("Pipeline vaidation thread")
        pipeline_queue_thread.setDaemon(True)
        pipeline_queue_thread.start()
    validation_queue.put(validation_request)


def stop_validation_queue_thread():
    """Stop the thread that handles module validation"""
    global validation_queue_keep_running
    if pipeline_queue_thread is not None:
        validation_queue_keep_running = False
        validation_queue.put(None)
        pipeline_queue_thread.join()
