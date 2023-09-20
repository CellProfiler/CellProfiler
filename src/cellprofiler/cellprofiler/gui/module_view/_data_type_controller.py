import wx
import wx.lib.rcsizer
from cellprofiler_core.setting import DataTypes

from ._module_view import ModuleView
from ..utilities.module_view import edit_control_name


class DataTypeController:
    """The DataTypeController manages a DataType setting"""

    DTC_NONE = "None"
    DTC_TEXT = "Text"
    DTC_INTEGER = "Integer"
    DTC_FLOAT = "Float"
    DTC_TO_DT = {
        DTC_NONE: DataTypes.DT_NONE,
        DTC_TEXT: DataTypes.DT_TEXT,
        DTC_INTEGER: DataTypes.DT_INTEGER,
        DTC_FLOAT: DataTypes.DT_FLOAT,
        None: DataTypes.DT_TEXT,
    }
    DT_TO_DTC = {
        DataTypes.DT_NONE: DTC_NONE,
        DataTypes.DT_TEXT: DTC_TEXT,
        DataTypes.DT_INTEGER: DTC_INTEGER,
        DataTypes.DT_FLOAT: DTC_FLOAT,
    }

    def __init__(self, module_view, v):
        assert isinstance(v, DataTypes)
        self.module_view = module_view
        self.v = v
        self.panel = module_view.module_panel.FindWindowByName(edit_control_name(v))
        if self.panel is None:

            class DoesntInheritBackgroundColor(wx.Panel):
                def InheritsBackgroundColour(self):
                    return False

            self.panel = DoesntInheritBackgroundColor(
                module_view.module_panel, -1, name=edit_control_name(v)
            )
            self.panel.Sizer = wx.lib.rcsizer.RowColSizer()
            self.panel.Bind(wx.EVT_PAINT, self.on_paint)
        self.panel.controller = self
        self.n_items = 0
        self.update()

    def on_paint(self, event):
        dc = wx.BufferedPaintDC(self.panel)
        dc.SetBackground(wx.Brush(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW)))
        dc.Clear()
        dc.SetPen(wx.Pen(wx.SystemSettings.GetColour(wx.SYS_COLOUR_GRAYTEXT)))
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
            (choice_header_name, 1, "Data type"),
        ):
            ctrl = self.panel.FindWindowByName(header_name)
            if ctrl is None:
                ctrl = wx.StaticText(self.panel, label=text, name=header_name)
                ctrl.Font = wx.Font(
                    ctrl.Font.GetPointSize(),
                    ctrl.Font.GetFamily(),
                    ctrl.Font.GetStyle(),
                    wx.FONTWEIGHT_BOLD,
                    False,
                    ctrl.Font.GetFaceName(),
                )
                sizer.Add(
                    ctrl,
                    flag=wx.ALIGN_CENTER_HORIZONTAL
                    | wx.ALIGN_CENTER_VERTICAL
                    | wx.BOTTOM,
                    border=1,
                    row=0,
                    col=column,
                )
            else:
                sizer.Show(ctrl)
        for i, feature in enumerate(sorted(d.keys())):
            label_name = self.get_label_control_name(i + 1)
            choice_name = self.get_choice_control_name(i + 1)
            label = self.panel.FindWindowByName(label_name)
            if label is None:
                label = wx.StaticText(self.panel, label=feature, name=label_name)
                sizer.Add(
                    label,
                    flag=wx.ALIGN_LEFT | wx.BOTTOM | wx.LEFT | wx.RIGHT,
                    border=3,
                    row=i + 1,
                    col=0,
                )
            else:
                sizer.Show(label)
                if label.Label != feature:
                    label.Label = feature
            choice = self.panel.FindWindowByName(choice_name)
            if choice is None:
                choice = wx.Choice(
                    self.panel,
                    choices=[
                        self.DTC_TEXT,
                        self.DTC_INTEGER,
                        self.DTC_FLOAT,
                        self.DTC_NONE,
                    ],
                    name=choice_name,
                )
                sizer.Add(
                    choice,
                    flag=wx.EXPAND | wx.BOTTOM | wx.RIGHT,
                    border=3,
                    row=i + 1,
                    col=1,
                )
                needs_bind.append(choice)
            else:
                sizer.Show(choice)
            value = self.DT_TO_DTC.get(d[feature], self.DTC_TEXT)
            if choice.GetStringSelection() != value:
                choice.SetStringSelection(value)
        self.n_items = len(list(d.keys()))
        for choice in needs_bind:
            choice.Bind(wx.EVT_CHOICE, self.on_choice_changed)

    def on_choice_changed(self, event):
        result = {}
        for i in range(1, self.n_items + 1):
            label = self.panel.FindWindowByName(self.get_label_control_name(i))
            choice = self.panel.FindWindowByName(self.get_choice_control_name(i))
            result[label.Label] = self.DTC_TO_DT[choice.GetStringSelection()]
        result = DataTypes.encode_data_types(result)
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
