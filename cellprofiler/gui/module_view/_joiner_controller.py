import wx
import wx.lib.rcsizer

from ._module_view import ModuleView
from ..utilities.module_view import edit_control_name


class JoinerController:
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
        self.panel = wx.Panel(module_view.module_panel, -1, name=edit_control_name(v))
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
                ctrl = wx.StaticText(
                    self.panel, -1, column_name, name=header_control_name
                )
            else:
                ctrl.Label = column_name
            self.panel.Sizer.Add(
                ctrl, row=0, col=i, flag=wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_BOTTOM
            )
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
                    ctrl = wx.Choice(
                        self.panel, -1, choices=choices, name=choice_ctrl_name
                    )
                    ctrl.Bind(
                        wx.EVT_CHOICE,
                        lambda event, row=i, col=j: self.on_choice_changed(
                            event, row, col
                        ),
                    )
                else:
                    ctrl.SetItems(choices)
                ctrl.SetStringSelection(selection)
                self.panel.Sizer.Add(ctrl, row=i + 1, col=j, flag=wx.ALIGN_BOTTOM)
                all_subcontrols[choice_ctrl_name] = True

            add_button_name = self.get_add_button_control_name(i)
            ctrl = self.panel.FindWindowByName(add_button_name)
            if ctrl is None:
                ctrl = wx.Button(
                    self.panel, -1, "+", name=add_button_name, style=wx.BU_EXACTFIT
                )
                ctrl.Bind(
                    wx.EVT_BUTTON,
                    lambda event, position=i + 1: self.on_insert_row(event, position),
                )
            self.panel.Sizer.Add(
                ctrl, row=i + 1, col=len(column_names), flag=wx.ALIGN_BOTTOM
            )
            all_subcontrols[add_button_name] = True

            if len(joins) > 1:
                delete_button_name = self.get_delete_button_control_name(i)
                ctrl = self.panel.FindWindowByName(delete_button_name)
                if ctrl is None:
                    ctrl = wx.Button(
                        self.panel,
                        -1,
                        "-",
                        name=delete_button_name,
                        style=wx.BU_EXACTFIT,
                    )
                    ctrl.Bind(
                        wx.EVT_BUTTON,
                        lambda event, position=i: self.on_delete_row(event, position),
                    )
                self.panel.Sizer.Add(
                    ctrl, row=i + 1, col=len(column_names) + 1, flag=wx.ALIGN_BOTTOM
                )
                all_subcontrols[delete_button_name] = True

            if i > 0:
                move_up_button_name = self.get_up_button_control_name(i)
                ctrl = self.panel.FindWindowByName(move_up_button_name)
                if ctrl is None:
                    img = wx.ArtProvider.GetBitmap(
                        wx.ART_GO_UP, wx.ART_BUTTON, (16, 16)
                    )
                    ctrl = wx.BitmapButton(
                        self.panel, -1, img, name=move_up_button_name
                    )
                    ctrl.Bind(
                        wx.EVT_BUTTON,
                        lambda event, position=i: self.on_move_row_up(event, position),
                    )
                self.panel.Sizer.Add(
                    ctrl, row=i + 1, col=len(column_names) + 2, flag=wx.ALIGN_BOTTOM
                )
                all_subcontrols[move_up_button_name] = True

            if i < len(joins) - 1:
                move_down_button_name = self.get_down_button_control_name(i)
                ctrl = self.panel.FindWindowByName(move_down_button_name)
                if ctrl is None:
                    img = wx.ArtProvider.GetBitmap(
                        wx.ART_GO_DOWN, wx.ART_BUTTON, (16, 16)
                    )
                    ctrl = wx.BitmapButton(
                        self.panel, -1, img, name=move_down_button_name
                    )
                    ctrl.Bind(
                        wx.EVT_BUTTON,
                        lambda event, position=i: self.on_move_row_down(
                            event, position
                        ),
                    )
                self.panel.Sizer.Add(
                    ctrl, row=i + 1, col=len(column_names) + 3, flag=wx.ALIGN_BOTTOM
                )
                all_subcontrols[move_down_button_name] = True

        for key, value in list(all_subcontrols.items()):
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
        if wx.GetKeyState(wx.WXK_SHIFT):
            # Fill other empty fields if present.
            for col in self.column_names:
                if join[col] in ("", None) and new_value in self.v.entities[col]:
                    join[col] = new_value
        joins[row] = join
        self.module_view.on_value_change(
            self.v, self.panel, self.v.build_string(joins), event
        )

    def on_insert_row(self, event, position):
        joins = list(self.joins)
        new_join = dict([(column_name, None) for column_name in self.column_names])
        joins.insert(position, new_join)
        self.module_view.on_value_change(
            self.v, self.panel, self.v.build_string(joins), event
        )

    def on_delete_row(self, event, position):
        joins = list(self.joins)
        del joins[position]
        self.module_view.on_value_change(
            self.v, self.panel, self.v.build_string(joins), event
        )

    def on_move_row_up(self, event, position):
        joins = list(self.joins)
        joins = (
            joins[0 : (position - 1)]
            + [joins[position], joins[position - 1]]
            + joins[(position + 1) :]
        )
        self.module_view.on_value_change(
            self.v, self.panel, self.v.build_string(joins), event
        )

    def on_move_row_down(self, event, position):
        joins = list(self.joins)
        joins = (
            joins[0:position]
            + [joins[position + 1], joins[position]]
            + joins[(position + 2) :]
        )
        self.module_view.on_value_change(
            self.v, self.panel, self.v.build_string(joins), event
        )
