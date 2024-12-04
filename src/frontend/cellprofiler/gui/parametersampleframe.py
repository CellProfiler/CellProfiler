# coding=utf-8
"""ParameterSampleFrame.py - a window for specifying sampling options.

Author: AJ Pretorius
        University of Leeds
        a.j.pretorius@leeds.ac.uk
"""


import os
import traceback

import numpy
import skimage.io
import wx
import wx.lib.agw.floatspin
from cellprofiler_core.image import ImageSetList
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import ObjectSet
from cellprofiler_core.pipeline import RunException
from cellprofiler_core.setting import (
    Binary,
    Divider,
    FigureSubscriber,
    Measurement,
    ValidationError,
)
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.do_something import DoSomething, RemoveSettingButton
from cellprofiler_core.setting.range import (
    FloatRange,
    IntegerRange,
    IntegerOrUnboundedRange,
)
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.text import Float, Integer, Name, Text, ImageName

import cellprofiler.gui._workspace_model

PARAM_CLASS_TEXT_LABEL = "Input text label"
PARAM_CLASS_BOUNDED_DISCRETE = "Bounded, discrete"
PARAM_CLASS_UNBOUNDED = "Unbounded"
PARAM_CLASS_DECORATION = "Decoration"

PARAM_CLASS_UNBOUNDED_INT = "Unbounded int"
PARAM_CLASS_UNBOUNDED_FLOAT = "Unbounded float"

INT_LOWER_BOUND = 1
INT_UPPER_BOUND = 10000
FLOAT_LOWER_BOUND = 0.01
FLOAT_UPPER_BOUND = 10000.0
FLOAT_DIGITS_TO_ROUND_TO = 2
DEFAULT_NUMBER_SAMPLES = 10

ID_PARAM_SELECT_CHECK_LIST_BOX = wx.NewId()
ID_SAMPLE_BUTTON = wx.NewId()


# TODO: we have decided to tear out parameter sampling
# it is currently broken, and even if we fix it, it writes out
# data to files that are then meant to be browsed in another APP that no longer is maintained (but does still work).
# the client app can be found here (Panorama2):
# https://github.com/CellProfiler/cellprofiler.github.com/pull/180/files#diff-acbd757001916c40aaf34d2fad8e124318b460844302b7df34cc18f2d31d6d83
class ParameterSampleFrame(wx.Frame):
    """A frame to get sampling settings and to calculate samples for a
    CellProfiler module.

    A few important data members and data structures are maintained:
    'self.__module': the module selected by the user.
    'self.__pipeline': the pipeline containing the module.
    'self.__parameters_list': a list of all unbounded parameters that will
        be displayed to the user. Eg,
            [(label_1, m_1), (label_2, m_2), ...], where
            parameter i maps to the tuple (label_i, m_i) with 'label_i' its
            text label and 'm_i' its 1-based index in 'self.__module'.
    'self.__parameters_to_widgets_list': a list of lists representing the
        mapping from parameters (see above), to 'groups of widgets'. Ie,
        some parameters will have more than two values and, hence, will
        have two groups of widgets. Eg,
            [[0, 1], [[2]], [3, 4], represents a mapping
            param 0 -> widget group 0 and 1, etc.
        In practice, each such mapping will correspond exactly to a
        check box and one or more rows of spin controls, depending on how
        many values a particular parameter takes.
    'self.__sample_list': a list of samples of the selected parameters. Eg,
        suppose that parameters p_1, p_2 and p_3 are selected, then a list
        of the form
            [[s_11, s_21, s_31], [s_12, s_22, s_32], ...]
        will be returned.
        NB: for those parameters displayed, but not selected by the user, this
        list will contain the value 'None'. Eg, suppose p2 is deselected, then
        the list will have the form:
            [[s_11, None, s_31], [s_12, None, s_32], ...].
    """

    def __init__(
        self,
        parent,
        module,
        pipeline,
        identifier=-1,
        pos=wx.DefaultPosition,
        size=wx.DefaultSize,
        style=wx.DEFAULT_FRAME_STYLE,
        name=wx.FrameNameStr,
    ):
        # Flag to check during event handling: widgets generate events as they
        # are created leading to referencing errors before all data structs
        # have been initialized.
        self.__initialized = False

        # Init frame
        self.__module = module
        self.__pipeline = pipeline
        self.__frame = parent

        # Data members for running pipeline
        self.__measurements = None
        self.__object_set = None
        self.__image_set_list = None
        self.__keys = None
        self.__groupings = None
        self.__grouping_index = None
        self.__within_group_index = None
        self.__outlines = None
        self.__grids = None

        frame_title = "Sampling settings for module, " + self.__module.module_name
        wx.Frame.__init__(
            self, self.__frame, identifier, frame_title, pos, size, style, name
        )
        self.__frame_sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSize(wx.Size(700, 350))
        self.SetSizer(self.__frame_sizer)

        # Get parameters
        self.__parameters_list = self.get_parameters_list()
        if len(self.__parameters_list) == 0:
            dialog = wx.MessageDialog(
                self,
                self.__module.module_name + " has no unbounded parameters.",
                caption="No unbounded parameters",
                style=wx.OK,
            )
            dialog.ShowModal()
            self.Close(True)

        # Init settings panel
        self.__settings_panel = wx.Panel(self, -1)
        self.__settings_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        self.__settings_panel.SetSizer(self.__settings_panel_sizer)
        self.__frame_sizer.Add(self.__settings_panel, 1, wx.EXPAND | wx.ALL, 5)

        # Init settings scrolled window
        self.__settings_scrolled_window = wx.ScrolledWindow(self.__settings_panel)
        self.__settings_scrolled_window.SetScrollRate(20, 20)
        self.__settings_scrolled_window.EnableScrolling(True, True)
        self.__settings_scrolled_window_sizer = wx.FlexGridSizer(rows=0, cols=5)
        self.__settings_scrolled_window.SetSizer(self.__settings_scrolled_window_sizer)
        self.__settings_scrolled_window_sizer.AddGrowableCol(0)
        self.__settings_panel_sizer.Add(
            self.__settings_scrolled_window, 1, wx.EXPAND | wx.ALL, 5
        )

        # Headings
        self.__settings_scrolled_window_sizer.Add(
            wx.StaticText(self.__settings_scrolled_window, -1, "Parameter"),
            0,
            wx.ALIGN_CENTER | wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.TOP,
            5,
        )
        self.__settings_scrolled_window_sizer.Add(
            wx.StaticText(self.__settings_scrolled_window, -1, "Current value"),
            0,
            wx.ALIGN_CENTER | wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.TOP,
            5,
        )
        self.__settings_scrolled_window_sizer.Add(
            wx.StaticText(self.__settings_scrolled_window, -1, "Lower bound"),
            0,
            wx.ALIGN_CENTER | wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.TOP,
            5,
        )
        self.__settings_scrolled_window_sizer.Add(
            wx.StaticText(self.__settings_scrolled_window, -1, "Upper bound"),
            0,
            wx.ALIGN_CENTER | wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.TOP,
            5,
        )
        self.__settings_scrolled_window_sizer.Add(
            wx.StaticText(self.__settings_scrolled_window, -1, "Number samples"),
            0,
            wx.ALIGN_CENTER | wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.TOP | wx.RIGHT,
            5,
        )

        # Init dynamic widgets based on parameters
        self.__parameters_to_widgets_list = self.get_parameters_to_widgets_list(
            self.__parameters_list
        )

        self.__check_box_list = []
        self.__current_static_text_list = []
        self.__lower_bound_spin_ctrl_list = []
        self.__upper_bound_spin_ctrl_list = []
        self.__number_spin_ctrl_list = []
        for i in range(len(self.__parameters_to_widgets_list)):
            label = self.__parameters_list[i][0]
            setting = self.__module.setting(self.__parameters_list[i][1])
            value = setting.get_value()

            for j in range(len(self.__parameters_to_widgets_list[i])):
                # Checkbox & label
                if j == 0:
                    check_box = wx.CheckBox(self.__settings_scrolled_window, -1, label)
                    check_box.Bind(wx.EVT_CHECKBOX, self.on_check_box)
                    self.__settings_scrolled_window_sizer.Add(
                        check_box,
                        0,
                        wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.TOP,
                        5,
                    )
                    self.__check_box_list.append(check_box)
                else:
                    self.__settings_scrolled_window_sizer.Add(
                        wx.Panel(self.__settings_scrolled_window, -1)
                    )

                # Current value
                if self.is_parameter_int(setting):
                    if self.get_parameter_input_size(setting) == 1:
                        cur_value = str(value)
                    elif self.get_parameter_input_size(setting) == 2:
                        cur_value = str(value[j])
                elif self.is_parameter_float(setting):
                    if self.get_parameter_input_size(setting) == 1:
                        cur_value = str(round(value, FLOAT_DIGITS_TO_ROUND_TO))
                    elif self.get_parameter_input_size(setting) == 2:
                        cur_value = str(round(value[j], FLOAT_DIGITS_TO_ROUND_TO))
                current_static_text = wx.StaticText(
                    self.__settings_scrolled_window, -1, cur_value
                )
                self.__settings_scrolled_window_sizer.Add(
                    current_static_text,
                    0,
                    wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.TOP | wx.RIGHT,
                    5,
                )
                self.__current_static_text_list.append(current_static_text)

                # Lower and upper bounds
                if self.is_parameter_int(setting):
                    # Integer
                    lower_spin_ctrl = wx.SpinCtrl(
                        self.__settings_scrolled_window, wx.NewId()
                    )
                    lower_spin_ctrl.SetRange(INT_LOWER_BOUND, INT_UPPER_BOUND)
                    lower_spin_ctrl.Bind(wx.EVT_SPINCTRL, self.on_lower_spin_ctrl)

                    upper_spin_ctrl = wx.SpinCtrl(
                        self.__settings_scrolled_window, wx.NewId()
                    )
                    upper_spin_ctrl.SetRange(INT_LOWER_BOUND, INT_UPPER_BOUND)
                    upper_spin_ctrl.Bind(wx.EVT_SPINCTRL, self.on_upper_spin_ctrl)

                    interval = (
                        DEFAULT_NUMBER_SAMPLES - 1
                    )  # Used later to set upper bound
                elif self.is_parameter_float(setting):
                    # Float
                    lower_spin_ctrl = wx.lib.agw.floatspin.FloatSpin(
                        self.__settings_scrolled_window, wx.NewId(), size=wx.DefaultSize
                    )  # NB: if not set, spin buttons are hidden
                    lower_spin_ctrl.SetDigits(FLOAT_DIGITS_TO_ROUND_TO)
                    lower_spin_ctrl.SetIncrement(10 ** -FLOAT_DIGITS_TO_ROUND_TO)
                    lower_spin_ctrl.SetRange(FLOAT_LOWER_BOUND, FLOAT_UPPER_BOUND)
                    lower_spin_ctrl.Bind(
                        wx.lib.agw.floatspin.EVT_FLOATSPIN, self.on_lower_float_spin
                    )

                    upper_spin_ctrl = wx.lib.agw.floatspin.FloatSpin(
                        self.__settings_scrolled_window, wx.NewId(), size=wx.DefaultSize
                    )  # NB: if not set, spin buttons are hidden
                    upper_spin_ctrl.SetDigits(FLOAT_DIGITS_TO_ROUND_TO)
                    upper_spin_ctrl.SetIncrement(10 ** -FLOAT_DIGITS_TO_ROUND_TO)
                    upper_spin_ctrl.SetRange(FLOAT_LOWER_BOUND, FLOAT_UPPER_BOUND)
                    upper_spin_ctrl.Bind(
                        wx.lib.agw.floatspin.EVT_FLOATSPIN, self.on_upper_float_spin
                    )

                    interval = (
                        upper_spin_ctrl.GetIncrement()
                    )  # Used later to set upper bound

                if self.get_parameter_input_size(setting) == 1:
                    lower_spin_ctrl.SetValue(
                        value
                    )  # Set value after range to avoid rounding
                    upper_spin_ctrl.SetValue(value + interval)
                elif self.get_parameter_input_size(setting) == 2:
                    lower_spin_ctrl.SetValue(value[j])
                    upper_spin_ctrl.SetValue(value[j] + interval)

                lower_spin_ctrl.Enable(False)
                self.__settings_scrolled_window_sizer.Add(
                    lower_spin_ctrl, 0, wx.EXPAND | wx.ALIGN_LEFT | wx.LEFT | wx.TOP, 5
                )
                self.__lower_bound_spin_ctrl_list.append(lower_spin_ctrl)

                upper_spin_ctrl.Enable(False)
                self.__settings_scrolled_window_sizer.Add(
                    upper_spin_ctrl, 0, wx.LEFT | wx.TOP, 5
                )
                self.__upper_bound_spin_ctrl_list.append(upper_spin_ctrl)

                # Number samples
                num_spin_ctrl = wx.SpinCtrl(self.__settings_scrolled_window, wx.NewId())
                num_spin_ctrl.Enable(False)
                num_spin_ctrl.SetRange(1, 100)
                num_spin_ctrl.SetValue(10)
                num_spin_ctrl.Bind(wx.EVT_SPINCTRL, self.on_number_spin_ctrl)
                self.__settings_scrolled_window_sizer.Add(
                    num_spin_ctrl, 0, wx.LEFT | wx.TOP | wx.RIGHT, 5
                )
                self.__number_spin_ctrl_list.append(num_spin_ctrl)

        self.__settings_scrolled_window.Layout()
        self.__settings_panel.Layout()

        # Add button
        self.sample_button = wx.Button(self, ID_SAMPLE_BUTTON, "Sample")
        self.sample_button.Bind(wx.EVT_BUTTON, self.on_button)
        self.sample_button.Enable(False)
        self.__frame_sizer.Add(self.sample_button, 0, wx.ALIGN_RIGHT | wx.ALL, 5)

        self.__initialized = True

    def get_parameters_list(self):
        """Get and return the list of unbounded parameters.

        This method first identifies all unbounded parameters in 'module'.
        Next, it updates a list of mappings to tuples:
            a parameter's 0-based index -> (its label, its 1-based module index),
            eg, 0 -> (Crop, 2); 1 -> (Crop, 3); 2 -> (IdentifyPrimaryObjects, 4).
        """
        parameters_list = []
        # Get unique keys of visible parameters
        visible_settings_keys = []
        for setting in self.__module.visible_settings():
            visible_settings_keys.append(setting.key())
        # Update dictionary of visible, unbounded parameters:
        #   0-based index in 'parameters_list' -> (label, 1-based module number)
        for i, setting in enumerate(self.__module.settings()):
            if (
                setting.key() in visible_settings_keys
                and self.get_parameter_type(setting) == PARAM_CLASS_UNBOUNDED
            ):
                parameters_list.append((setting.get_text(), i + 1))
        return parameters_list

    def get_parameters_to_widgets_list(self, parameters_list):
        """Create mapping from parameters to widgets.

        Some parameters have two values, requiring two sets of widgets. This
        method updates 'parameters_to_widgets_list', a mapping from:
            a parameter's 0-based index -> its widget groups' 0-based indices,
            eg, 0 -> [0, 1]; 1 -> [2]; 2 -> [3, 4].
        The indices on the left will correspond to check boxes and the indices
        on the right will correspond to widgets to the right of the check
        boxes. This method assumes that 'parameters_list' is correctly
        initialized (see 'get_parameters_list()').
        """
        parameters_to_widgets_list = []
        index_count = 0
        for i in range(len(parameters_list)):
            setting = self.__module.setting(parameters_list[i][1])
            input_size = self.get_parameter_input_size(setting)
            widget_indices = []
            for j in range(input_size):
                widget_indices.append(index_count)
                index_count += 1
            parameters_to_widgets_list.append(widget_indices)
        return parameters_to_widgets_list

    @staticmethod
    def get_parameter_type(setting):
        """Get parameter type of 'setting' by considering its class."""
        if isinstance(setting, Binary):
            return PARAM_CLASS_BOUNDED_DISCRETE
        elif isinstance(setting, Choice):
            return PARAM_CLASS_BOUNDED_DISCRETE
        elif isinstance(setting, Divider):
            return PARAM_CLASS_DECORATION
        elif isinstance(setting, DoSomething):
            return PARAM_CLASS_DECORATION
        elif isinstance(setting, FigureSubscriber):
            return PARAM_CLASS_BOUNDED_DISCRETE
        elif isinstance(setting, Float):
            return PARAM_CLASS_UNBOUNDED
        elif isinstance(setting, FloatRange):
            return PARAM_CLASS_UNBOUNDED
        elif isinstance(setting, Integer):
            return PARAM_CLASS_UNBOUNDED
        elif isinstance(setting, IntegerRange):
            return PARAM_CLASS_UNBOUNDED
        elif isinstance(setting, IntegerOrUnboundedRange):
            return PARAM_CLASS_UNBOUNDED
        elif isinstance(setting, Measurement):
            # NB: Not sure what to do with 'Measurement' yet
            return "not sure"
        elif isinstance(setting, Name):
            return PARAM_CLASS_TEXT_LABEL
        elif isinstance(setting, ImageSubscriber):
            return PARAM_CLASS_BOUNDED_DISCRETE
        elif isinstance(setting, RemoveSettingButton):
            return PARAM_CLASS_DECORATION
        elif isinstance(setting, Text):
            return PARAM_CLASS_TEXT_LABEL
        else:
            return "whatever is left"

    @staticmethod
    def get_parameter_input_size(setting):
        """Get input size of 'setting'."""
        size = 1
        try:
            size = len(setting.get_value())
        except:
            pass
        return size

    @staticmethod
    def is_parameter_float(setting):
        if isinstance(setting, Float):
            return True
        elif isinstance(setting, FloatRange):
            return True
        else:
            return False

    @staticmethod
    def is_parameter_int(setting):
        if isinstance(setting, Integer):
            return True
        elif isinstance(setting, IntegerRange):
            return True
        elif isinstance(setting, IntegerOrUnboundedRange):
            return True
        else:
            return False

    def validate_input_ranges(self):
        """Validates input ranges specified by the input widgets.

        This method uses CellProfiler's validate methods on settings to test
        whether the ranges specified on the UI are acceptable. When these do
        not validate, the user is shown an error message that lists all
        violations as well as the error message received back from CellProfiler.
        """
        # First, test whether selected parameters are valid
        message = ""
        for i, checkbox in enumerate(self.__check_box_list):
            if checkbox.IsChecked():
                setting = self.__module.setting(self.__parameters_list[i][1])
                input_size = self.get_parameter_input_size(setting)
                if input_size == 1:
                    widget_idx = self.__parameters_to_widgets_list[i][0]
                    lower_value = self.__lower_bound_spin_ctrl_list[
                        widget_idx
                    ].GetValue()
                    upper_value = self.__upper_bound_spin_ctrl_list[
                        widget_idx
                    ].GetValue()
                elif input_size == 2:
                    widget_idx = self.__parameters_to_widgets_list[i][0]
                    lower_value = (
                        self.__lower_bound_spin_ctrl_list[widget_idx].GetValue(),
                        self.__lower_bound_spin_ctrl_list[widget_idx + 1].GetValue(),
                    )
                    upper_value = (
                        self.__upper_bound_spin_ctrl_list[widget_idx].GetValue(),
                        self.__upper_bound_spin_ctrl_list[widget_idx + 1].GetValue(),
                    )

                old_value = setting.get_value()
                try:
                    setting.set_value(lower_value)
                    setting.test_valid(self.__pipeline)
                except ValidationError as instance:
                    message += (
                        "'"
                        + str(setting.get_text())
                        + "': lower bound invalid, "
                        + "\n\t"
                        + str(instance.message)
                        + "\n"
                    )
                try:
                    setting.set_value(upper_value)
                    setting.test_valid(self.__pipeline)
                except ValidationError as instance:
                    message += (
                        "'"
                        + str(setting.get_text())
                        + "': upper bound invalid, "
                        + "\n\t"
                        + str(instance.message)
                        + "\n"
                    )
                setting.set_value(old_value)

        # Second, if there are invalid parameters, tell the user
        if len(message) > 0:
            message = "Invalid sample settings:\n\n" + message
            dialog = wx.MessageDialog(
                self,
                message,
                caption="Sample settings error",
                style=wx.ICON_ERROR | wx.OK,
            )
            dialog.ShowModal()
            raise Exception(message)

    def generate_parameter_samples(self):
        """Compute samples values for the selected parameters.

        This method returns a list of samples for the selected parameters by
        computing all possible combinations of sample values that lie within
        the ranges specified by the user. Eg, suppose p1, p2, p3 and p4 are
        selected, a list of the form
            [[s11, s21, s31, s41], [s12, s22, s32, s42], ...]
        will be returned.

        NB: for those parameters displayed, but not selected by the user, the
        value 'None' will be returned. Eg,
        suppose p2 is deselected, then a list of the form:
            [[s11, None, s31, s41], [s12, None, s32, s42], ...]
        will be returned.
        """
        sample_list = []

        # First, compute a list of samples for every selected parameter
        for i, checkbox in enumerate(self.__check_box_list):
            samples = []
            if checkbox.IsChecked():
                number = self.__parameters_list[i][1]
                setting = self.__module.setting(number)

                if self.get_parameter_input_size(setting) == 1:
                    # Parameters with single value
                    index = self.__parameters_to_widgets_list[i][0]
                    lower_bound = self.__lower_bound_spin_ctrl_list[index].GetValue()
                    upper_bound = self.__upper_bound_spin_ctrl_list[index].GetValue()
                    number_samples = self.__number_spin_ctrl_list[index].GetValue()
                    samples = self.compute_samples(
                        lower_bound,
                        upper_bound,
                        number_samples,
                        self.is_parameter_int(setting),
                    )
                else:
                    # Parameters with tuple values
                    samples_temp = []
                    for j in self.__parameters_to_widgets_list[i]:
                        lower_bound = self.__lower_bound_spin_ctrl_list[j].GetValue()
                        upper_bound = self.__upper_bound_spin_ctrl_list[j].GetValue()
                        number_samples = self.__number_spin_ctrl_list[j].GetValue()
                        samples_temp.append(
                            self.compute_samples(
                                lower_bound,
                                upper_bound,
                                number_samples,
                                self.is_parameter_int(setting),
                            )
                        )

                    samples_temp = list(self.cartesian_product(samples_temp))
                    tuples = []
                    for l in samples_temp:
                        tuples.append(tuple(l))
                    samples = tuples
            else:
                samples.append(None)

            sample_list.append(samples)

        # Second, compute all possible combinations of the lists of samples
        sample_list = list(self.cartesian_product(sample_list))

        return sample_list

    def calc_number_samples(self):
        """Computes number of samples.
        This method computes the number of samples that will be generated
        with the current UI settings without computing the samples themselves.
        """
        number = 0
        for i, check_box in enumerate(self.__check_box_list):
            if check_box.IsChecked():
                for j in self.__parameters_to_widgets_list[i]:
                    if number == 0:
                        number = self.__number_spin_ctrl_list[j].GetValue()
                    else:
                        number *= self.__number_spin_ctrl_list[j].GetValue()
        return number

    @staticmethod
    def compute_samples(lower_bound, upper_bound, number_samples, is_int):
        """Computes samples in the range [lower_bound, upper_bound].

        This method computes an returns a list of uniformly distributed samples
        in the range [lower_bound, upper_bound]. The returned list will be of
        size 'number_samples'. If 'is_int' is true, samples will be rounded to
        the nearest integer, except for the last, which will be rounded to
        'upper_bound'.
        """
        samples = []
        if number_samples > 1:
            delta = (upper_bound - lower_bound) / float(number_samples - 1)
            for i in range(number_samples):
                sample = lower_bound + i * delta
                if is_int:
                    if i == number_samples:
                        sample = upper_bound
                    else:
                        sample = int(sample)
                samples.append(sample)
        else:
            samples.append(lower_bound)
        return samples

    def cartesian_product(self, list_of_lists):
        """Computes all combinations of lists in 'list_of_lists'.

        Eg, [[1, 2],[3, 4]] -> [[1, 3], [1, 4], [2, 3], [2, 4]]
        As this functionality is provided by 'itertools.product()' in
        Python 2.6 and above, the implementation below was just grabbed
        off the web as a temporary solution:
            <http://stackoverflow.com/questions/2419370/
                how-can-i-compute-a-cartesian-product-iteratively>
        If one of the lists in 'list_of_lists' is empty - eg,
        [[1, 2], [], [3, 4]] - the method will not work.
        """
        if not list_of_lists:
            yield []
        else:
            for item in list_of_lists[0]:
                for product in self.cartesian_product(list_of_lists[1:]):
                    yield [item] + product

    def prepare_for_run(self):
        """Prepare to run pipeline.

        This code was largely copied from 'PipelineController.start_debugging()'
        with updates to the 'PipelineController' UI removed. I wish I had a
        better understanding of what exactly this does, but I'm pretty much
        using it as a black box for the time being.
        """
        self.__measurements = Measurements()
        self.__object_set = ObjectSet(can_overwrite=True)
        self.__image_set_list = ImageSetList()
        workspace = cellprofiler.gui._workspace_model.Workspace(
            self.__pipeline,
            None,
            None,
            None,
            self.__measurements,
            self.__image_set_list,
            self.__frame,
        )
        try:
            if not self.__pipeline.prepare_run(workspace):
                print("Error: failed to get image sets")
            self.__keys, self.__groupings = self.__pipeline.get_groupings(workspace)
        except ValueError as v:
            message = "Error while preparing for run:\n%s" % v
            wx.MessageBox(
                message, "Pipeline error", wx.OK | wx.ICON_ERROR, self.__frame
            )
        self.__grouping_index = 0
        self.__within_group_index = 0
        self.__pipeline.prepare_group(
            workspace, self.__groupings[0][0], self.__groupings[0][1]
        )
        self.__outlines = {}

    def run_module(self, module):
        """Run pipeline with 'module'.

        This code was largely copied from 'PipelineController.do_step()' with
        updates to the 'PipelineController' UI removed. As with the above method,
        I wish I had a better understanding of what exactly this does, but I'm
        pretty much using it as a black box.

        Assumption: 'self.prepare_for_run()' was called first.
        """
        failure = 1
        try:
            # ~*~
            # workspace = cellprofiler.gui.workspace.Workspace(
            #    self.__pipeline, module, image_set, self.__object_set,
            #    self.__measurements, self.__image_set_list,
            #    # Uncomment next line to display results in UI
            #    #self.__frame if module.show_window else None,
            #    None,
            #    outlines = self.__outlines)
            self.__workspace = cellprofiler.gui._workspace_model.Workspace(
                self.__pipeline,
                module,
                self.__measurements,
                self.__object_set,
                self.__measurements,
                self.__image_set_list,
                # Uncomment next line to display results in UI
                # self.__frame if module.show_window else None,
                None,
                outlines=self.__outlines,
            )
            # self.__grids = workspace.set_grids(self.__grids)
            self.__grids = self.__workspace.set_grids(self.__grids)
            # ~^~
            # print 'Running ' + str(module.get_module_num()) + ': ' + str(module.module_name)
            # ~*~
            # module.run(workspace)
            module.run(self.__workspace)
            # ~^~
            # ~*~
            # workspace.refresh()
            self.__workspace.refresh()
            # ~^~
            failure = 0
        except Exception as instance:
            traceback.print_exc()
            event = RunException(instance, module)
            self.__pipeline.notify_listeners(event)
            failure = 1
        if (
            module.module_name != "Restart" or failure == -1
        ) and self.__measurements is not None:
            module_error_measurement = "ModuleError_%02d%s" % (
                module.module_num,
                module.module_name,
            )
            self.__measurements.add_measurement(
                "Image", module_error_measurement, failure
            )
        return failure == 0

    def save_run_output(self, sample_num, directory_path, output_file):
        """Save the parameter settings and images for the current run of
        'self.__module'."""
        for i, setting in enumerate(self.__module.visible_settings()):
            value_to_write = ""

            # Do not write settings without values, ie, buttons etc
            if setting.get_text() != "":
                value_to_write = str(setting.get_value())
                if isinstance(setting, ImageName):
                    # Save image
                    image = self.__measurements.get_image(value_to_write)
                    path = os.path.join(
                        directory_path, value_to_write + "_" + str(sample_num)
                    )
                    self.save_image(image, path)
                    value_to_write += "_" + str(sample_num) + ".jpg"
                    # ~*~
                    # elif isinstance(setting, settings.ObjectNameProvider):
                    #    print 'Bingo! ' + str(value_to_write)
                    #    objects =\
                    #        self.__workspace.get_object_set().get_objects(setting.get_value())
                    #    value_to_write = str(len(objects.indices))
                    # ~^~
            if i < len(self.__module.visible_settings()) - 1 and value_to_write != "":
                value_to_write += "\t"
            if i == len(self.__module.visible_settings()) - 1:
                value_to_write += "\n"

            # Write line to 'open_file'
            if value_to_write != "":
                output_file.write(value_to_write)

                # ~*~
                # Fiddle around with cpimage.ImageSet
                #   -> This might be a neater way of getting at images
                # print 'ImageSet:'
                # for name in self.__workspace.get_image_set().get_names():
                #    print '\t' + str(name)
                #    image = self.__workspace.get_image_set().get_image(name)
                #    path =\
                #        os.path.join(directory_path, name)
                #    self.save_image(image, path)

                # Fiddle around with objects.ObjectSet
                #   -> This might be the way to get at the number of objects detected
                # print 'ObjectSet:'
                # for name in self.__workspace.get_object_set().get_object_names():
                #    print '\t' + str(name)
                #    objects = self.__workspace.get_object_set().get_objects(name)
                #    print '\t\t' + str(objects.indices)
                #    #parent_image = objects.get_parent_image()
                #    #path = \
                #    #    os.path.join(directory_path, 'tootie')
                #    #self.save_image(image, path)

                # for object_name in self.__measurements.get_object_names():
                #    print object_name
                #    for feature_name in self.__measurements.get_feature_names(object_name):
                #        print '\t' + str(feature_name)
                #        current_measurement = self.__measurements.get_current_measurement(
                #            object_name, feature_name)
                #        print '\t\t' + str(current_measurement)
                # ~^~

    @staticmethod
    def save_image(image, path):
        """TODO: add comments"""

        pixels = image.pixel_data
        if numpy.max(pixels) > 1 or numpy.min(pixels) < 0:
            pixels = pixels.copy()
            pixels[pixels < 0] = 0
            pixels[pixels > 1] = 1

        pixels = (pixels * 255).astype(numpy.uint8)

        pathname = "{}.jpg".format(path)

        skimage.io.imsave(pathname, pixels)

    def on_check_box(self, event):
        button_flag = False
        for i, checkbox in enumerate(self.__check_box_list):
            if checkbox.IsChecked():
                for j in self.__parameters_to_widgets_list[i]:
                    self.__lower_bound_spin_ctrl_list[j].Enable(True)
                    self.__upper_bound_spin_ctrl_list[j].Enable(True)
                    self.__number_spin_ctrl_list[j].Enable(True)
                button_flag = True
            else:
                for j in self.__parameters_to_widgets_list[i]:
                    self.__lower_bound_spin_ctrl_list[j].Enable(False)
                    self.__upper_bound_spin_ctrl_list[j].Enable(False)
                    self.__number_spin_ctrl_list[j].Enable(False)
        self.sample_button.Enable(button_flag)

    def on_lower_spin_ctrl(self, event):
        if self.__initialized:
            for i in range(len(self.__parameters_to_widgets_list)):
                for j in self.__parameters_to_widgets_list[i]:
                    if event.GetId() == self.__lower_bound_spin_ctrl_list[j].GetId():
                        lower_bound = self.__lower_bound_spin_ctrl_list[j].GetValue()
                        upper_bound = self.__upper_bound_spin_ctrl_list[j].GetValue()
                        increment = self.__number_spin_ctrl_list[j].GetValue() - 1
                        if upper_bound < lower_bound + increment:
                            self.__upper_bound_spin_ctrl_list[j].SetValue(
                                lower_bound + increment
                            )

    def on_upper_spin_ctrl(self, event):
        if self.__initialized:
            for i in range(len(self.__parameters_to_widgets_list)):
                for j in self.__parameters_to_widgets_list[i]:
                    if event.GetId() == self.__upper_bound_spin_ctrl_list[j].GetId():
                        lower_bound = self.__lower_bound_spin_ctrl_list[j].GetValue()
                        upper_bound = self.__upper_bound_spin_ctrl_list[j].GetValue()
                        increment = self.__number_spin_ctrl_list[j].GetValue() - 1
                        if upper_bound < lower_bound + increment:
                            self.__lower_bound_spin_ctrl_list[j].SetValue(
                                upper_bound - increment
                            )

    def on_lower_float_spin(self, event):
        if self.__initialized:
            for i in range(len(self.__parameters_to_widgets_list)):
                for j in self.__parameters_to_widgets_list[i]:
                    if event.GetId() == self.__lower_bound_spin_ctrl_list[j].GetId():
                        lower_bound = self.__lower_bound_spin_ctrl_list[j].GetValue()
                        upper_bound = self.__upper_bound_spin_ctrl_list[j].GetValue()
                        increment = self.__upper_bound_spin_ctrl_list[j].GetIncrement()
                        if upper_bound < lower_bound + increment:
                            self.__upper_bound_spin_ctrl_list[j].SetValue(
                                lower_bound + increment
                            )

    def on_upper_float_spin(self, event):
        if self.__initialized:
            for i in range(len(self.__parameters_to_widgets_list)):
                for j in self.__parameters_to_widgets_list[i]:
                    if event.GetId() == self.__upper_bound_spin_ctrl_list[j].GetId():
                        lower_bound = self.__lower_bound_spin_ctrl_list[j].GetValue()
                        upper_bound = self.__upper_bound_spin_ctrl_list[j].GetValue()
                        increment = self.__upper_bound_spin_ctrl_list[j].GetIncrement()
                        if upper_bound < lower_bound + increment:
                            self.__lower_bound_spin_ctrl_list[j].SetValue(
                                upper_bound - increment
                            )

    def on_number_spin_ctrl(self, event):
        if self.__initialized:
            for i in range(len(self.__parameters_to_widgets_list)):
                setting = self.__module.setting(i)
                if self.is_parameter_int(setting):
                    for j in self.__parameters_to_widgets_list[i]:
                        if event.GetId() == self.__number_spin_ctrl_list[j].GetId():
                            lower_bound = self.__lower_bound_spin_ctrl_list[
                                j
                            ].GetValue()
                            upper_bound = self.__upper_bound_spin_ctrl_list[
                                j
                            ].GetValue()
                            number = self.__number_spin_ctrl_list[j].GetValue()
                            diff = upper_bound - lower_bound
                            if diff < number:
                                self.__upper_bound_spin_ctrl_list[j].SetValue(
                                    lower_bound + (number - 1)
                                )

    def on_button(self, event):
        if event.GetId() == ID_SAMPLE_BUTTON:
            # try:
            self.validate_input_ranges()
            number = self.calc_number_samples()

            sample_dialog = wx.MessageDialog(
                self,
                "Proceed with calculating " + str(number) + " samples?",
                caption="Confirm sample size",
                style=wx.ICON_QUESTION | wx.OK | wx.CANCEL,
            )
            if sample_dialog.ShowModal() == wx.ID_OK:

                save_dialog = wx.FileDialog(
                    event.GetEventObject(),
                    message="Save sampled output",
                    wildcard="Tab separated values (*.tsv)|*.tsv",
                    style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
                )
                if save_dialog.ShowModal() == wx.ID_OK:
                    # 1. Copy original parameter values
                    original_values = []
                    for setting in self.__module.visible_settings():
                        original_values.append(setting.get_value())

                    # 2. Sample input parameters
                    self.__sample_list = self.generate_parameter_samples()

                    # 3. Open output file and write headers
                    output_file = open(save_dialog.GetPath(), "w")
                    headers = ""
                    for i, setting in enumerate(self.__module.visible_settings()):
                        # Do not write settings without values, ie, buttons etc
                        if setting.get_text() != "":
                            headers += setting.get_text()
                            if i < len(self.__module.visible_settings()) - 1:
                                headers += "\t"
                    headers += "\n"
                    output_file.write(headers)

                    # 4. Run pipeline once for each sample

                    # ~*~
                    self.Show(False)
                    progressDialog = wx.ProgressDialog(
                        parent=self,
                        title="Sampling parameters",
                        message="Run 1",
                        maximum=len(self.__sample_list),
                    )
                    size = progressDialog.GetSize()
                    size.SetWidth(2 * size.GetWidth())
                    progressDialog.SetSize(size)
                    # ~^~

                    for i, l in enumerate(self.__sample_list):
                        # print '\nStarting run ' + str(i+1) + '...'

                        for j, value in enumerate(l):
                            if value is not None:
                                setting_nr = self.__parameters_list[j][1]
                                setting = self.__module.setting(setting_nr)
                                setting.set_value(value)
                                # print str(setting.get_text()) + ' -> ' + str(setting.get_value())

                        # ~*~
                        progressDialog.Update(
                            i + 1,
                            newmsg="Executing run "
                            + str(i + 1)
                            + " of "
                            + str(len(self.__sample_list)),
                        )
                        # ~^~

                        # It's not very efficient to run the complete pipeline
                        # when only the last module's parameter values are
                        # different. However, this is the only way I can get
                        # the images to update correctly for the last module. Ie,
                        # when I don't prepare and run the pipeline from scratch
                        # for every different configuration of the last module,
                        # I get the images generated by the first run for every
                        # run.

                        # 4.1 Prepare to run pipeline
                        self.prepare_for_run()
                        # 4.2 Run modules
                        for module in self.__pipeline.modules():
                            if (
                                module.get_module_num()
                                <= self.__module.get_module_num()
                            ):
                                self.run_module(module)
                        # 4.3 Save output
                        self.save_run_output(i, save_dialog.GetDirectory(), output_file)

                        # This is the way to run headless, if only I could get at the images...
                        # self.stop_now = False
                        # running_pipeline = self.__pipeline.run_with_yield(
                        #    run_in_background=False,
                        #    status_callback=self.status_callback)
                        # while not self.stop_now:
                        #    measurements = running_pipeline.next()

                        # print '...run completed.'
                    # 5. Close output file
                    output_file.close()

                    # ~*~
                    progressDialog.Destroy()
                    # ~^~

                    # 6. Set parameters back to original values and close window
                    for i, setting in enumerate(self.__module.visible_settings()):
                        setting.set_value(original_values[i])
                    self.Close(True)

                    # def status_callback(self, module, image_set, *args):
                    #    if (module.get_module_num() == self.__module.get_module_num()+1):
                    #        self.stop_now = True
