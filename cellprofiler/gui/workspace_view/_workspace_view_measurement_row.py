import wx
import wx.lib
import wx.lib.colourselect

from ..constants.workspace_view import C_CHOOSER
from ..constants.workspace_view import C_COLOR
from ..constants.workspace_view import C_SHOW


class WorkspaceViewMeasurementRow:
    """Container for measurement controls"""

    def __init__(self, panel, grid_sizer, row_idx, on_change):
        """MeasurementRow contstructor

        panel - the panel that's going to be the host for the controls
        grid_sizer - put the controls in this grid sizer
        row_idx - the row # in the grid sizer
        on_change - a function (with no args) that's called whenever any control
                    is changed. This handler should call MeasurementRow.update
        """
        #
        # Create three-tiered measurement choice:
        #    object name
        #    category
        #    measurement name
        #
        assert isinstance(grid_sizer, wx.GridBagSizer)
        self.process_events = True
        self.change_fn = on_change
        self.choice_panel = wx.Panel(panel)
        grid_sizer.Add(
            self.choice_panel, (row_idx, C_CHOOSER,), flag=wx.EXPAND,
        )
        self.choice_sizer = wx.BoxSizer(wx.VERTICAL)
        self.choice_panel.SetSizer(self.choice_sizer)
        self.object_choice = wx.Choice(self.choice_panel)
        self.choice_sizer.Add(self.object_choice, 0, wx.EXPAND)
        self.category_choice = wx.Choice(self.choice_panel)
        self.category_choice.Hide()
        self.choice_sizer.Add(self.category_choice, 0, wx.EXPAND | wx.TOP, 2)
        self.measurement_choice = wx.Choice(self.choice_panel)
        self.measurement_choice.Hide()
        self.choice_sizer.Add(self.measurement_choice, 0, wx.EXPAND | wx.TOP, 2)
        #
        # Font button
        #
        self.font_button = wx.Button(panel, label="Font")
        grid_sizer.Add(
            self.font_button,
            (row_idx, C_COLOR,),
            flag=wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_TOP,
        )
        self.font_button.Bind(wx.EVT_BUTTON, self.on_choose_font)
        self.show_ctrl = wx.CheckBox(panel)
        grid_sizer.Add(
            self.show_ctrl,
            (row_idx, C_SHOW,),
            flag=wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_TOP,
        )
        #
        # The drawing characteristics
        #
        self.font = self.font_button.Font
        self.foreground_color = wx.SystemSettings.GetColour(wx.SYS_COLOUR_BTNTEXT)
        self.background_color = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW)
        self.background_alpha = 0.5
        self.box_style = "round"
        self.precision = 4  # of decimal places

        for control, event in (
            (self.object_choice, wx.EVT_CHOICE),
            (self.category_choice, wx.EVT_CHOICE),
            (self.measurement_choice, wx.EVT_CHOICE),
            (self.show_ctrl, wx.EVT_CHECKBOX),
        ):
            control.Bind(event, self.on_change)

    def on_choose_font(self, event):
        with wx.Dialog(self.choice_panel.Parent, title="Measurement appearance") as dlg:
            labels = []

            def add_label(sizer, label):
                ctrl = wx.StaticText(dlg, label=label)
                sizer.Add(ctrl, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
                labels.append(ctrl)
                sizer.AddSpacer(2)

            dlg.Sizer = wx.BoxSizer(wx.VERTICAL)
            sizer = wx.BoxSizer(wx.VERTICAL)
            dlg.Sizer.Add(sizer, 0, wx.EXPAND | wx.ALL, 5)
            subsizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(subsizer, 0, wx.EXPAND)
            sizer.AddSpacer(2)
            add_label(subsizer, "Font")
            font_picker = wx.FontPickerCtrl(dlg)
            font_picker.SetSelectedFont(self.font)
            font_picker.SetPickerCtrlGrowable(True)
            font_picker.Bind(wx.EVT_FONTPICKER_CHANGED, lambda event: dlg.Layout())
            subsizer.Add(font_picker, 0, wx.ALIGN_LEFT)
            #
            # Foreground color
            #
            subsizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(subsizer, 0, wx.EXPAND)
            sizer.AddSpacer(2)
            add_label(subsizer, "Text color")
            foreground_color = wx.lib.colourselect.ColourSelect(
                dlg, colour=self.foreground_color
            )
            subsizer.Add(foreground_color, 0, wx.ALIGN_LEFT)
            #
            # Background color and alpha
            #
            subsizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(subsizer, 0, wx.EXPAND)
            sizer.AddSpacer(2)
            add_label(subsizer, "Background color")
            background_color = wx.lib.colourselect.ColourSelect(
                dlg, colour=self.background_color
            )
            subsizer.Add(background_color, 0, wx.ALIGN_LEFT)

            subsizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(subsizer, 0, wx.EXPAND)
            sizer.AddSpacer(2)
            add_label(subsizer, "Alpha")
            alpha = wx.Slider(
                dlg,
                value=self.background_alpha * 100,
                minValue=0,
                maxValue=100,
                style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS,
            )
            alpha.SetMinSize(wx.Size(200, alpha.GetMinSize()[0]))
            subsizer.Add(alpha, 0, wx.EXPAND)

            subsizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(subsizer, 0, wx.EXPAND)
            sizer.AddSpacer(2)
            add_label(subsizer, "Box shape")
            box_style = wx.Choice(
                dlg, choices=["circle", "round", "roundtooth", "sawtooth", "square"]
            )
            box_style.SetStringSelection(self.box_style)
            subsizer.Add(box_style, 0, wx.ALIGN_LEFT)

            subsizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(subsizer, 0, wx.EXPAND)
            sizer.AddSpacer(2)
            add_label(subsizer, "Precision")
            precision = wx.SpinCtrl(dlg, value=str(self.precision), min=0)
            subsizer.AddSpacer(2)
            subsizer.Add(precision, 0, wx.ALIGN_LEFT)

            width = 0
            for label in labels:
                width = max(width, label.GetBestSize()[0])
            for label in labels:
                label.SetMinSize(wx.Size(width, label.GetBestSize()[1]))

            button_sizer = wx.StdDialogButtonSizer()
            dlg.Sizer.Add(button_sizer, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALL, 5)
            button_sizer.AddButton(wx.Button(dlg, wx.ID_OK))
            button_sizer.AddButton(wx.Button(dlg, wx.ID_CANCEL))
            button_sizer.Realize()
            dlg.Fit()
            if dlg.ShowModal() == wx.ID_OK:
                self.font = font_picker.GetSelectedFont()
                self.foreground_color = foreground_color.GetColour()
                self.background_color = background_color.GetColour()
                self.background_alpha = float(alpha.GetValue()) / 100
                self.box_style = box_style.GetStringSelection()
                self.precision = precision.GetValue()
                self.on_change(event)

    def on_change(self, event):
        if self.process_events:
            self.change_fn()

    @staticmethod
    def __get_selected_choice(control):
        assert isinstance(control, wx.Choice)
        if not control.IsShown():
            return None
        selection = control.GetSelection()
        if selection == wx.NOT_FOUND:
            return None
        return control.GetItems()[selection]

    def get_object_name(self):
        return self.__get_selected_choice(self.object_choice)

    def get_measurement_name(self):
        if self.get_object_name() is None:
            return None
        category = self.__get_selected_choice(self.category_choice)
        if category is None:
            return None
        feature = self.__get_selected_choice(self.measurement_choice)
        if feature is None:
            return None
        return "_".join((category, feature))

    def should_show(self):
        return self.show_ctrl.IsChecked()

    def update_choices(self, control, choices):
        assert isinstance(control, wx.Choice)
        old_names = control.GetItems()
        if tuple(sorted(old_names)) == tuple(sorted(choices)):
            return
        self.process_events = False
        try:
            old_choice = self.__get_selected_choice(control)
            if old_choice is not None and old_choice not in choices:
                choices = choices + [old_choice]
            control.SetItems(choices)
            if old_choice is not None:
                control.SetStringSelection(old_choice)
        finally:
            self.process_events = True

    def update(self, workspace):
        m = workspace.measurements
        self.update_choices(self.object_choice, m.get_object_names())
        object_name = self.get_object_name()
        if object_name is None or object_name not in m.get_object_names():
            self.category_choice.Hide()
            self.measurement_choice.Hide()
            return
        self.category_choice.Show()
        self.measurement_choice.Show()
        categories = set()
        measurements = set()
        current_category = self.__get_selected_choice(self.category_choice)
        for feature in m.get_feature_names(object_name):
            category, measurement = feature.split("_", 1)
            categories.add(category)
            if category == current_category:
                measurements.add(measurement)
        self.update_choices(self.category_choice, sorted(categories))
        self.update_choices(self.measurement_choice, sorted(measurements))

    def destroy(self):
        self.object_choice.Destroy()
        self.category_choice.Destroy()
        self.measurement_choice.Destroy()
        self.choice_panel.Destroy()
        self.font_button.Destroy()
        self.show_ctrl.Destroy()
