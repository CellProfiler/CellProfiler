# coding=utf-8
"""metadatadlg.py - dialog for editing an expression that might contain metadata
"""

import wx
import wx.lib.masked
from cellprofiler_core.constants.measurement import C_FRAME
from cellprofiler_core.constants.measurement import C_METADATA
from cellprofiler_core.constants.measurement import C_SERIES
from cellprofiler_core.constants.measurement import IMAGE
from cellprofiler_core.preferences import get_primary_outline_color

__choice_ids = []


def get_choice_id(index):
    global __choice_ids
    while len(__choice_ids) <= index:
        __choice_ids.append(wx.NewId())
    return __choice_ids[index]


class MetadataControl(wx.Control):
    class MetadataToken(object):
        def __init__(self):
            self.value = ""

    def __init__(self, pipeline, module, *args, **kwargs):
        """Initialize the field

        pipeline - the pipeline being run
        module - the module containing the setting
        value (optional) - initial value for control
        padding (optional) - padding around text in pixels
        """
        kwargs = kwargs.copy()
        style = kwargs.get("style", wx.BORDER_DEFAULT)
        value = kwargs.pop("value", "")
        self.padding = kwargs.pop("padding", 1)
        self.offset = 0
        if (style & wx.BORDER_MASK) == wx.BORDER_DEFAULT:
            self.native_border = True
            self.padding += 2
            style = (style & ~wx.BORDER_MASK) | wx.BORDER_NONE
        else:
            self.native_border = False
        kwargs["style"] = style | wx.WANTS_CHARS

        super(MetadataControl, self).__init__(*args, **kwargs)
        columns = pipeline.get_measurement_columns(module)
        choices = [
            C_SERIES,
            C_FRAME,
        ]
        for column in columns:
            object_name, feature, coltype = column[:3]
            choice = feature[(len(C_METADATA) + 1) :]
            if object_name == IMAGE and feature.startswith(C_METADATA):
                choices.append(choice)
        self.__metadata_choices = choices
        self.SetValue(value)
        self.__cursor_pos = len(self.__tokens)
        self.__caret = None
        self.__metadata_choice_ids = {}
        self.__metadata_choice_dict = {}
        for i, choice in enumerate(sorted(choices)):
            choice_id = get_choice_id(i)
            self.__metadata_choice_ids[choice_id] = choice
            self.__metadata_choice_dict[choice] = choice_id
            self.Bind(wx.EVT_MENU, self.select_value, id=choice_id)

        self.selection = [0, 0]

        def on_focus(event):
            if self.__caret is None:
                self.make_caret()
            self.show_caret()

        def on_lose_focus(event):
            if self.__caret is not None:
                self.__caret.Hide()
                del self.__caret
                self.__caret = None

        def on_show(event):
            if event:
                if isinstance(event.EventObject, MetadataControl):
                    self.make_caret()
            else:
                del self.__caret
                self.__caret = None

        self.Bind(wx.EVT_SET_FOCUS, on_focus)
        self.Bind(wx.EVT_KILL_FOCUS, on_lose_focus)
        self.Bind(wx.EVT_KEY_DOWN, self.on_key_down)
        self.Bind(wx.EVT_CHAR, self.on_char)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.Bind(wx.EVT_MOTION, self.OnMouseMotion)
        self.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)
        self.Bind(wx.EVT_RIGHT_DOWN, self.on_right_down)
        self.Bind(wx.EVT_CONTEXT_MENU, self.on_context_menu)
        self.Bind(wx.EVT_SHOW, on_show)
        self.Bind(wx.EVT_SIZE, self.OnSize)

    @property
    def xoffset(self):
        return self.offset + self.padding

    def make_caret(self):
        self.__caret = wx.Caret(self, wx.Size(1, self.Size[1] - 2 * self.padding))

    def SetValue(self, value):
        #
        # Scan through the value, switching state
        #
        STATE_INITIAL = 0  # Not in a tag
        STATE_BACKSLASH = 1  # At first backslash
        STATE_PRE = 2  # Found either \? or \g
        STATE_INSIDE = 3  # Inside a metadata tag

        self.__tokens = []
        state = STATE_INITIAL
        index = 0
        while index < len(value):
            if state == STATE_INITIAL:
                if value[index] == "\\":
                    state = STATE_BACKSLASH
                else:
                    self.__tokens.append(value[index])
            elif state == STATE_BACKSLASH:
                if value[index] in ("g", "?"):
                    state = STATE_PRE
                else:
                    self.__tokens.append(str(value[index]))
                    state = STATE_INITIAL
            elif state == STATE_PRE:
                if value[index] != "<":
                    # WTF? bad input, output last 3 tokens
                    self.__tokens += value[(index - 2) : (index + 1)]
                    state = STATE_INITIAL
                else:
                    self.__tokens += [self.MetadataToken()]
                    state = STATE_INSIDE
            else:
                assert state == STATE_INSIDE
                if value[index] != ">":
                    self.__tokens[-1].value += value[index]
                else:
                    state = STATE_INITIAL
            index += 1
        self.SetMinSize(self.DoGetBestSize())
        self.__cursor_pos = len(self.__tokens)
        self.Cursor = wx.Cursor(wx.CURSOR_IBEAM)

    def GetValue(self):
        """The setting value underlying the text representation"""
        value = ""
        for token in self.__tokens:
            if isinstance(token, self.MetadataToken):
                value += "\\g<" + token.value + ">"
            elif token == "\\":
                value += "\\\\"
            else:
                value += token
        return value

    value = property(GetValue, SetValue)
    Value = property(GetValue, SetValue)

    def adjust_scroll(self):
        """Scroll the cursor position into view"""

        rawpos = 0
        for i in range(self.__cursor_pos):
            rawpos += self.GetFullTextExtent(self.get_text(i, i + 1))[0]
        xsize = self.Size[0] - self.padding * 2

        pos = self.xoffset + rawpos
        slop = pos - xsize
        if slop > 0:
            slop += self.GetFullTextExtent("M")[0]
            self.offset -= slop
            pos -= slop
            self.Refresh()
        elif pos < self.padding:
            self.offset = rawpos
            pos = self.padding
            self.Refresh()
        elif rawpos + self.GetFullTextExtent("M")[0] < xsize:
            self.offset = 0
            self.Refresh()
        if self.__caret is not None:
            pos = rawpos + self.xoffset
            self.__caret.Move(pos, self.padding)

    def show_caret(self):
        if self.__caret is not None and self.FindFocus() == self:
            self.adjust_scroll()
            self.__caret.Show()

    def on_key_down(self, event):
        keycode = event.GetKeyCode()
        cmd_down = event.CmdDown()
        shift_down = event.ShiftDown()
        ################
        #
        # Left Arrow
        #
        ################
        if keycode in (wx.WXK_LEFT, wx.WXK_NUMPAD_LEFT):
            if self.__cursor_pos > 0:
                self.move_cursor_pos(self.__cursor_pos - 1, not shift_down)
        ################
        #
        # Right Arrow
        #
        ################
        elif keycode in (wx.WXK_RIGHT, wx.WXK_NUMPAD_RIGHT):
            if self.__cursor_pos < len(self.__tokens):
                self.move_cursor_pos(self.__cursor_pos + 1, not shift_down)
        ################
        #
        # Down arrow (next metadata item)
        #
        ################
        elif keycode in (wx.WXK_DOWN, wx.WXK_NUMPAD_DOWN):
            pos = self.__cursor_pos
            if pos < len(self.__tokens) and isinstance(
                self.__tokens[pos], self.MetadataToken
            ):
                token = self.__tokens[pos]
                try:
                    idx = self.__metadata_choices.index(token.value) + 1
                    idx %= len(self.__metadata_choices)
                except ValueError:
                    idx = 0
                if len(self.__metadata_choices):
                    token.value = self.__metadata_choices[idx]
                    self.on_token_change()
        #################
        #
        # Up arrow (prev metadata item)
        #
        #################
        elif keycode in (wx.WXK_UP, wx.WXK_NUMPAD_UP):
            pos = self.__cursor_pos
            if pos < len(self.__tokens) and isinstance(
                self.__tokens[pos], self.MetadataToken
            ):
                token = self.__tokens[pos]
                try:
                    idx = self.__metadata_choices.index(token.value) - 1
                    if idx < 0:
                        idx = len(self.__metadata_choices) - 1
                except ValueError:
                    idx = 0
                if len(self.__metadata_choices):
                    token.value = self.__metadata_choices[idx]
                    self.on_token_change()
        #################
        #
        # Insert (add metadata item)
        #
        #################
        elif keycode in (wx.WXK_INSERT, wx.WXK_NUMPAD_INSERT):
            pos = self.__cursor_pos
            token = self.MetadataToken()
            token.value = self.__metadata_choices[0]
            self.__tokens.insert(pos, token)
            self.on_token_change()
        #################
        #
        # Backspace
        #
        #################
        elif keycode == wx.WXK_BACK:
            pos = self.__cursor_pos
            if pos > 0:
                pos -= 1
                self.move_cursor_pos(pos)
                del self.__tokens[pos]
                self.on_token_change()
        ##################
        #
        # Delete
        #
        ##################
        elif keycode in (wx.WXK_DELETE, wx.WXK_NUMPAD_DELETE):
            if self.selection[0] == self.selection[1]:
                pos = self.__cursor_pos
                if pos < len(self.__tokens):
                    del self.__tokens[pos]
                    self.on_token_change()
            else:
                self.delete_selection()
        #################
        #
        # Home
        #
        #################
        elif keycode in (wx.WXK_HOME, wx.WXK_NUMPAD_HOME):
            self.move_cursor_pos(0, not shift_down)
        #################
        #
        # End
        #
        #################
        elif keycode in (wx.WXK_END, wx.WXK_NUMPAD_END):
            self.move_cursor_pos(len(self.__tokens), not shift_down)
        #################
        #
        # Context menu
        #
        #################
        elif keycode == wx.WXK_WINDOWS_MENU:
            self.on_context_menu(event)
        #################
        #
        #  Tab
        #
        #################
        elif keycode == wx.WXK_TAB:
            #
            # Code adapted from wx.lib.calendar: author Lorne White
            #
            forward = not event.ShiftDown()
            ne = wx.NavigationKeyEvent()
            ne.SetDirection(forward)
            ne.SetCurrentFocus(self)
            ne.SetEventObject(self)
            self.GetParent().GetEventHandler().ProcessEvent(ne)
            #
            # Seems to be confused about the focus
            #
            if self.FindFocus() != self and self.__caret is not None:
                self.__caret.Hide()
                del self.__caret
                self.__caret = None
        ##################
        #
        # Paste
        #
        ##################
        elif (keycode == ord("V")) and cmd_down:
            # Cribbed from the WX drag and drop demo
            if wx.TheClipboard.Open():
                try:
                    data_object = wx.TextDataObject()
                    success = wx.TheClipboard.GetData(data_object)
                finally:
                    wx.TheClipboard.Close()
                if success:
                    self.delete_selection()
                    self.__tokens = (
                        self.__tokens[0 : self.__cursor_pos]
                        + list(data_object.GetText())
                        + self.__tokens[self.__cursor_pos :]
                    )
                    self.move_cursor_pos(self.__cursor_pos + len(data_object.GetText()))
                    self.on_token_change()
        ################
        #
        # Cut / copy
        #
        ################
        elif (keycode in (ord("C"), ord("X"))) and cmd_down:
            if self.selection[0] == self.selection[1]:
                return
            selection = list(self.selection)
            selection.sort()
            text = self.get_text(selection[0], selection[1])
            if wx.TheClipboard.Open():
                try:
                    self.__clipboard_text = wx.TextDataObject()
                    self.__clipboard_text.SetText(text)
                    wx.TheClipboard.SetData(self.__clipboard_text)
                finally:
                    wx.TheClipboard.Close()
            if keycode == ord("X"):
                self.delete_selection()
        else:
            event.Skip()

    def delete_selection(self):
        if self.selection[0] != self.selection[1]:
            selection = list(self.selection)
            selection.sort()
            del self.__tokens[selection[0] : selection[1]]
            self.move_cursor_pos(selection[0])
            self.on_token_change()

    def on_char(self, event):
        self.delete_selection()
        c = chr(event.GetUnicodeKey())
        self.__tokens.insert(self.__cursor_pos, c)
        self.move_cursor_pos(self.__cursor_pos + 1)
        self.on_token_change()

    def move_cursor_pos(self, pos, reselect=True):
        if pos == self.__cursor_pos:
            return
        self.__cursor_pos = pos
        if reselect:
            self.selection = [pos, pos]
        else:
            self.selection[1] = pos
        self.show_caret()
        self.Refresh()

    def on_token_change(self):
        event = wx.CommandEvent(wx.wxEVT_COMMAND_TEXT_UPDATED, self.GetId())
        self.GetEventHandler().ProcessEvent(event)
        self.show_caret()
        self.SetMinSize(self.DoGetBestSize())
        self.Refresh()

    def get_text(self, start_idx=0, end_idx=None):
        """Return the text representation of the tokens between the given indices

        start_idx - index of first token in string
        end_idx - index of last token or -1 for all
        """
        value = ""
        if end_idx is None:
            end_idx = len(self.__tokens)
        for token in self.__tokens[start_idx:end_idx]:
            if isinstance(token, self.MetadataToken):
                value += token.value
            else:
                value += token
        return value

    def AcceptsFocus(self):
        return True

    def AcceptsFocusFromKeyboard(self):
        return True

    def DoGetBestSize(self):
        size = self.GetFullTextExtent(self.get_text() + "M")
        size = wx.Size(size[0] + self.padding * 2, size[1] + self.padding * 2)
        return self.ClientToWindowSize(size)

    def OnSize(self, event):
        self.offset = 0
        self.adjust_scroll()
        event.Skip()

    def hit_test(self, pos):
        text = self.get_text(0, len(self.__tokens))
        dc = wx.ClientDC(self)
        dc.SetFont(self.Font)
        positions = self.get_positions(dc)
        del dc
        for i in range(len(self.__tokens)):
            if pos <= positions[i] and pos < positions[i + 1]:
                return i
        return len(self.__tokens)

    def OnLeftDown(self, event):
        if self.HitTest(event.GetPosition()) == wx.HT_WINDOW_INSIDE:
            self.__cursor_pos = self.hit_test(event.GetPosition()[0])
            if self.FindFocus() == self:
                self.show_caret()
            else:
                event.Skip()
            self.CaptureMouse()
            self.selection = [self.__cursor_pos, self.__cursor_pos]
        else:
            event.Skip()

    def OnMouseMotion(self, event):
        if not self.HasCapture():
            event.Skip()
            return
        if self.HitTest(event.GetPosition()) == wx.HT_WINDOW_INSIDE:
            self.move_cursor_pos(self.hit_test(event.GetPosition()[0]), False)

    def OnLeftUp(self, event):
        if not self.HasCapture():
            return
        self.ReleaseMouse()

    def on_right_down(self, event):
        if self.HitTest(event.GetPosition()) == wx.HT_WINDOW_INSIDE:
            index = self.hit_test(event.GetPosition()[0])
            self.__cursor_pos = index
            self.SetFocus()
            self.show_caret()
            event.Skip()

    def on_context_menu(self, event):
        menu = wx.Menu()
        index = self.__cursor_pos
        if index < len(self.__tokens) and isinstance(
            self.__tokens[index], self.MetadataToken
        ):
            heading = "Change tag"
        else:
            heading = "Insert tag"
        item = menu.Append(-1, heading)
        item.Enable(False)
        for choice in sorted(self.__metadata_choice_dict.keys()):
            menu.Append(self.__metadata_choice_dict[choice], choice)
        self.PopupMenu(menu)
        menu.Destroy()

    def select_value(self, event):
        choice = self.__metadata_choice_ids[event.GetId()]
        index = self.__cursor_pos
        if index < len(self.__tokens) and isinstance(
            self.__tokens[index], self.MetadataToken
        ):
            self.__tokens[index].value = choice
        else:
            token = self.MetadataToken()
            token.value = choice
            self.__tokens.insert(index, token)
        self.on_token_change()

    def get_positions(self, dc):
        """Get the widths of each of the tokens"""
        text = self.get_text(0, len(self.__tokens))
        raw_positions = dc.GetPartialTextExtents(text)
        positions = [self.padding]
        ptr = -1
        for i in range(len(self.__tokens) - 1):
            text = self.get_text(i, i + 1)
            ptr += len(text)
            positions.append(raw_positions[ptr] + self.padding)
        if len(raw_positions) > 0:
            positions.append(raw_positions[-1] + self.padding)
        else:
            positions.append(self.padding)
        return positions

    def OnPaint(self, event):
        dc = wx.BufferedPaintDC(self)
        try:
            dc.SetBackgroundMode(wx.PENSTYLE_SOLID)
            background_color = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW)
            metadata_color = get_primary_outline_color()
            selected_background_color = wx.SystemSettings.GetColour(
                wx.SYS_COLOUR_HIGHLIGHT
            )
            selected_color = wx.SystemSettings.GetColour(wx.SYS_COLOUR_HIGHLIGHTTEXT)
            text_color = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT)
            dc.SetBackground(wx.Brush(background_color))
            dc.SetFont(self.Font)
            dc.Clear()
            if self.native_border:
                renderer = wx.RendererNative.Get()
                style = 0
                if self.FindFocus() == self:
                    style |= wx.CONTROL_FOCUSED | wx.CONTROL_CURRENT
                if not self.Enabled:
                    style |= wx.CONTROL_DISABLED
                renderer.DrawTextCtrl(
                    self, dc, (0, 0, self.ClientSize[0], self.ClientSize[1]), style
                )
                dc.SetClippingRegion(
                    (
                        self.padding,
                        self.padding,
                        self.ClientSize[0] - 2 * self.padding,
                        self.ClientSize[1] - 2 * self.padding,
                    )
                )
            text = self.get_text(0, len(self.__tokens))
            positions = self.get_positions(dc)

            last_state = "unknown"
            text_list = []
            state_list = []
            position_list = []
            selection = None
            if self.selection is not None:
                selection = list(self.selection)
                selection.sort()
            for i, token in enumerate(self.__tokens):
                if isinstance(token, self.MetadataToken):
                    current_state = "metadata"
                elif self.selection is not None and selection[0] <= i < selection[1]:
                    current_state = "selection"
                else:
                    current_state = "boring"
                if current_state != last_state:
                    state_list.append(current_state)
                    text_list.append("")
                    last_state = current_state
                    position_list.append((positions[i], self.padding))
                text_list[-1] += self.get_text(i, i + 1)
            colors = {
                "boring": (background_color, text_color),
                "selection": (selected_background_color, selected_color),
                "metadata": (metadata_color, text_color),
            }
            background_color = [colors[state][0] for state in state_list]
            foreground_color = [colors[state][1] for state in state_list]
            dc.SetBackgroundMode(wx.PENSTYLE_SOLID)
            for text, position, background, foreground in zip(
                text_list, position_list, background_color, foreground_color
            ):
                dc.SetTextBackground(background)
                dc.SetTextForeground(foreground)
                dc.DrawText(text, position[0], position[1])
        finally:
            dc.Destroy()
