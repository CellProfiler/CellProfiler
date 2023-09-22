import logging

import wx
from cellprofiler_core.setting.filter import Filter, FilterPredicate
from cellprofiler_core.setting.filter._filter import (
    AND_PREDICATE,
    OR_PREDICATE,
    LITERAL_PREDICATE,
)

from ._module_view import ModuleView
from ..utilities.module_view import edit_control_name

LOGGER = logging.getLogger(__name__)


class FilterPanelController(object):
    """Handle representation of the filter panel

    The code for handling the filter UI is moderately massive, so it gets
    its own class, if for no other reason than to organize the code.
    """

    def __init__(self, module_view, v, panel):
        assert isinstance(module_view, ModuleView)
        assert isinstance(v, Filter)
        self.module_view = module_view
        self.v = v
        self.panel = wx.Panel(
            self.module_view.module_panel,
            style=wx.TAB_TRAVERSAL,
            name=edit_control_name(self.v),
        )
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
        if key in self.sizer_dict:
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
        line = wx.StaticLine(
            self.panel, -1, style=line_style, name=self.line_name(address)
        )

        if len(address) == 0:
            key = None
        else:
            sizer.Add(
                self.make_delete_button(address), 0, wx.ALIGN_CENTER_VERTICAL,
            )
            sizer.Add(
                self.make_add_rule_button(address), 0, wx.ALIGN_CENTER_VERTICAL,
            )
            sizer.Add(
                self.make_add_rules_button(address), 0, wx.ALIGN_CENTER_VERTICAL,
            )
            key = tuple(address[:-1] + [address[-1] + 1])
            if key not in self.sizer_dict:
                if len(address) == 1:
                    key = None
                else:
                    key = tuple(address[:-2] + [address[-2] + 1])
                    if key not in self.sizer_dict:
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
        except Exception as e:
            LOGGER.debug(
                "Failed to parse filter (value=%s): %s", self.v.value_text, str(e)
            )
            tokens = self.v.default()
        #
        # Always require an "and" or "or" clause
        #
        if len(tokens) == 0 or (tokens[0] not in (AND_PREDICATE, OR_PREDICATE,)):
            tokens = [AND_PREDICATE, tokens]
        return tokens

    def update(self):
        self.inside_update = True
        try:
            structure = self.get_tokens()
            for key in self.hide_show_dict:
                self.hide_show_dict[key] = False
            self.populate_subpanel(structure, [])
            for key, value in list(self.hide_show_dict.items()):
                self.panel.FindWindowByName(key).Show(value)
            self.panel.Layout()
        except:
            LOGGER.exception("Threw exception while updating filter")
        finally:
            self.inside_update = False

    ANY_ALL_PREDICATES = [
        AND_PREDICATE,
        OR_PREDICATE,
    ]

    def any_all_choices(self):
        return [x.display_name for x in self.ANY_ALL_PREDICATES]

    @staticmethod
    def indent(sizer, address):
        assert isinstance(sizer, wx.Sizer)
        if len(address) == 0:
            return
        sizer.AddSpacer(len(address) * 20)

    def find_and_mark(self, name):
        """Find a control and mark it to be shown"""
        ctrl = self.panel.FindWindowByName(name)
        self.hide_show_dict[name] = True
        return ctrl

    @staticmethod
    def get_sizer_index(sizer, item):
        if isinstance(item, wx.Sizer):
            indexes = [
                i
                for i, s in enumerate(sizer.GetChildren())
                if s.IsSizer() and s.GetSizer() is item
            ]
        elif isinstance(item, wx.Window):
            indexes = [
                i
                for i, s in enumerate(sizer.GetChildren())
                if s.IsWindow() and s.GetWindow() is item
            ]
        elif isinstance(item, wx.SizerItem):
            return sizer.GetChildren().index(item)
        if len(indexes) > 0:
            return indexes[0]
        return None

    def on_value_change(self, event, new_text, timeout=None):
        if not self.inside_update:
            self.module_view.on_value_change(
                self.v, self.panel, new_text, event, timeout
            )

    def make_delete_button(self, address):
        name = self.remove_button_name(address)
        button = self.find_and_mark(name)
        if button is not None:
            return button

        button = wx.Button(self.panel, -1, "-", name=name, style=wx.BU_EXACTFIT)
        button.Bind(wx.EVT_BUTTON, lambda event: self.on_delete_rule(event, address))
        return button

    def on_delete_rule(self, event, address):
        LOGGER.debug("Delete row at " + str(address))
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

        button = wx.Button(self.panel, -1, "+", name=name, style=wx.BU_EXACTFIT)
        button.Bind(wx.EVT_BUTTON, lambda event: self.on_add_rule(event, address))
        return button

    def on_add_rule(self, event, address):
        LOGGER.debug("Add rule after " + str(address))
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
        button = wx.Button(self.panel, -1, "...", name=name, style=wx.BU_EXACTFIT)
        button.Bind(wx.EVT_BUTTON, lambda event: self.on_add_rules(event, address))
        return button

    def on_add_rules(self, event, address):
        LOGGER.debug("Add rules after " + str(address))
        structure = self.v.parse()
        sequence = self.find_address(structure, address[:-1])
        new_rule = [OR_PREDICATE, self.v.default()]
        sequence.insert(address[-1] + 2, new_rule)
        new_text = self.v.build_string(structure)
        self.on_value_change(event, new_text)

    def make_predicate_choice(self, predicates, index, address, sizer):
        name = self.choice_name(index, address)
        choice_ctrl = self.find_and_mark(name)
        choices = [x.display_name for x in predicates]
        if choice_ctrl is not None:
            items = choice_ctrl.GetItems()
            if len(items) != len(choices) or any(
                [choice not in items for choice in choices]
            ):
                choice_ctrl.SetItems(choices)
            return choice_ctrl
        choice_ctrl = wx.Choice(self.panel, -1, choices=choices, name=name)
        choice_ctrl.Bind(
            wx.EVT_CHOICE,
            lambda event: self.on_predicate_changed(event, index, address),
        )
        choice_ctrl.Bind(wx.EVT_MOUSEWHEEL, self.ignore_mousewheel)
        self.add_to_sizer(sizer, choice_ctrl, index, address)
        return choice_ctrl

    def on_predicate_changed(self, event, index, address):
        LOGGER.debug(
            "Predicate choice at %d / %s changed" % (index, self.saddress(address))
        )
        structure = self.v.parse()
        sequence = self.find_address(structure, address)

        while len(sequence) <= index:
            # The sequence is bad (e.g., bad pipeline or metadata collection)
            # Fill in enough to deal
            #
            sequence.append(
                self.v.predicates[0]
                if len(sequence) == 0
                else sequence[-1].subpredicates[0]
            )
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
            if isinstance(sequence[index], str):
                is_good = LITERAL_PREDICATE in predicates
            else:
                matches = [p for p in predicates if sequence[index].symbol == p.symbol]
                is_good = len(matches) == 1
                if is_good:
                    sequence[index] = matches[0]
            if not is_good:
                del sequence[index:]
                sequence += self.v.default(predicates)
                break
            if not isinstance(sequence[index], FilterPredicate):
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
        if next_key in self.sizer_item_dict:
            next_ctrl = self.sizer_item_dict[next_key]
        else:
            next_ctrl = self.stretch_spacer_dict[tuple(address)]
        index = self.get_sizer_index(sizer, next_ctrl)
        sizer.Insert(index, item, 0, wx.ALIGN_LEFT)
        if key not in self.sizer_item_dict:
            self.sizer_item_dict[key] = item

    def make_literal(self, token, index, address, sizer):
        name = self.literal_name(index, address)
        literal_ctrl = self.find_and_mark(name)
        if literal_ctrl is not None:
            if literal_ctrl.GetValue() != token:
                literal_ctrl.SetValue(token)
            return literal_ctrl
        literal_ctrl = wx.TextCtrl(self.panel, -1, token, name=name)
        literal_ctrl.Bind(
            wx.EVT_TEXT, lambda event: self.on_literal_changed(event, index, address)
        )
        self.add_to_sizer(sizer, literal_ctrl, index, address)
        return literal_ctrl

    def on_literal_changed(self, event, index, address):
        LOGGER.debug("Literal at %d / %s changed" % (index, self.saddress(address)))
        try:
            structure = self.v.parse()
            sequence = self.find_address(structure, address)
            while len(sequence) <= index:
                # The sequence is bad (e.g., bad pipeline or metadata collection)
                # Fill in enough to deal
                #
                sequence.append(
                    self.v.predicates[0]
                    if len(sequence) == 0
                    else sequence[-1].subpredicates[0]
                )
            sequence[index] = event.GetString()
        except:
            structure = self.v.default()

        new_text = self.v.build_string(structure)
        self.on_value_change(
            event, new_text, timeout=None if self.v.reset_view else False
        )

    def make_anyall_ctrl(self, address):
        anyall = wx.Choice(
            self.panel,
            -1,
            choices=self.any_all_choices(),
            name=self.anyall_choice_name(address),
        )
        anyall.Bind(wx.EVT_CHOICE, lambda event: self.on_anyall_changed(event, address))
        anyall.Bind(wx.EVT_MOUSEWHEEL, self.ignore_mousewheel)
        return anyall

    def on_anyall_changed(self, event, address):
        LOGGER.debug("Any / all choice at %s changed" % self.saddress(address))
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
            idx = self.get_sizer_index(sizer, self.stretch_spacer_dict[tuple(address)])
            if len(address) == 0:
                text = wx.StaticText(
                    self.panel, -1, "Match", name=self.static_text_name(0, address)
                )
                sizer.Insert(idx, text, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
                sizer.Insert(idx + 1, 3, 0)
                sizer.Insert(
                    idx + 2, anyall, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL
                )
                sizer.Insert(idx + 3, 3, 0)
                text = wx.StaticText(
                    self.panel,
                    -1,
                    "of the following rules",
                    name=self.static_text_name(1, address),
                )
                sizer.Insert(idx + 4, text, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
            else:
                sizer.Insert(
                    idx, anyall, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL | wx.RIGHT
                )
                sizer.Insert(idx + 1, 3, 0)
                text = wx.StaticText(
                    self.panel,
                    -1,
                    "of the following are true",
                    name=self.static_text_name(0, address),
                )
                sizer.Insert(idx + 2, text, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
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
            anyall.SetToolTip(structure[0].doc)
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
                    if isinstance(token, str):
                        literal_ctrl = self.make_literal(token, i, subaddress, sizer)
                        predicates = []
                    else:
                        choice_ctrl = self.make_predicate_choice(
                            predicates, i, subaddress, sizer
                        )
                        if choice_ctrl.GetStringSelection() != token.display_name:
                            choice_ctrl.SetStringSelection(token.display_name)
                        if token.doc is not None:
                            choice_ctrl.SetToolTip(token.doc)
                        predicates = token.subpredicates
                i = len(substructure)
                while len(predicates) > 0:
                    #
                    # We can get here if there's a badly constructed token
                    # list - for instance if an invalid subpredicate was
                    # chosen or none existed because of some error, but now
                    # they do.
                    #
                    if len(predicates) == 1 and predicates[0] is LITERAL_PREDICATE:
                        self.make_literal("", i, subaddress, sizer)
                    else:
                        self.make_predicate_choice(predicates, i, subaddress, sizer)
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

    def ignore_mousewheel(self, event):
        return
