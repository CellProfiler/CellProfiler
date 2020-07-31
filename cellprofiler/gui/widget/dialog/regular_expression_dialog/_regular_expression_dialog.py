import re

import wx
import wx.stc

from ._regular_expression_state import RegularExpressionState
from ....constants.dialog.regular_expression_dialog import RE_FILENAME_GUESSES
from ....constants.dialog.regular_expression_dialog import STYLE_ERROR
from ....constants.dialog.regular_expression_dialog import STYLE_FIRST_LABEL
from ....constants.dialog.regular_expression_dialog import STYLE_NO_MATCH
from ....constants.dialog.regular_expression_dialog import TOK_BRACKET_EXP
from ....constants.dialog.regular_expression_dialog import TOK_DEFINITION
from ....constants.dialog.regular_expression_dialog import TOK_ESCAPE
from ....constants.dialog.regular_expression_dialog import TOK_GROUP
from ....constants.dialog.regular_expression_dialog import TOK_ORDINARY
from ....constants.dialog.regular_expression_dialog import TOK_REPEAT
from ....constants.dialog.regular_expression_dialog import TOK_SPECIAL
from ....utilities.dialog.regular_expression_dialog import parse


class RegularExpressionDialog(wx.Dialog):
    def __init__(self, *args, **varargs):
        varargs["title"] = "Regular expression editor"
        super(RegularExpressionDialog, self).__init__(*args, **varargs)
        self.__value = "Not initialized"
        self.__test_text = "Not initialized"
        self.__guesses = RE_FILENAME_GUESSES
        font = wx.Font(
            10, wx.FONTFAMILY_MODERN, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL
        )
        self.font = font
        self.error_font = font

        sizer = wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(hsizer, 0, wx.GROW | wx.ALL, 5)
        hsizer.Add(wx.StaticText(self, label="Regex:"), 0, wx.ALIGN_CENTER | wx.ALL, 5)

        self.regexp_display = wx.stc.StyledTextCtrl(self, -1, style=wx.BORDER_SIMPLE)
        self.regexp_display.SetBufferedDraw(True)
        o = self.regexp_display.GetFullTextExtent("".join(["M"] * 50))
        w, h = self.regexp_display.ClientToWindowSize(wx.Size(o[1], o[2]))
        self.regexp_display.SetMinSize(wx.Size(w, h))
        self.regexp_display.Text = self.value
        self.regexp_display.SetLexer(wx.stc.STC_LEX_CONTAINER)
        for key in range(31):
            self.regexp_display.StyleSetFont(key, self.font)
        self.regexp_display.StyleSetForeground(TOK_ORDINARY, wx.Colour(0, 0, 0, 255))
        self.regexp_display.StyleSetForeground(TOK_ESCAPE, wx.Colour(0, 64, 64, 255))
        self.regexp_display.StyleSetForeground(TOK_GROUP, wx.Colour(0, 0, 255, 255))
        self.regexp_display.StyleSetForeground(TOK_REPEAT, wx.Colour(0, 128, 0, 255))
        self.regexp_display.StyleSetForeground(
            TOK_BRACKET_EXP, wx.Colour(64, 64, 64, 255)
        )
        self.regexp_display.StyleSetForeground(TOK_SPECIAL, wx.Colour(128, 64, 0, 255))
        color_db = self.get_color_db()
        for i in range(1, 16):
            self.regexp_display.StyleSetForeground(
                TOK_DEFINITION - 1 + i, color_db[i % len(color_db)]
            )

        self.regexp_display.StyleSetForeground(
            STYLE_ERROR, wx.Colour(255, 64, 128, 255)
        )
        self.regexp_display.StyleSetFont(34, self.font)
        self.regexp_display.StyleSetForeground(34, wx.Colour(0, 0, 255, 255))
        self.regexp_display.StyleSetUnderline(34, True)
        self.regexp_display.StyleSetFont(35, self.font)
        self.regexp_display.StyleSetForeground(35, wx.Colour(255, 0, 0, 255))
        self.regexp_display.SetUseVerticalScrollBar(0)
        self.regexp_display.SetUseHorizontalScrollBar(0)
        self.regexp_display.SetMarginWidth(wx.stc.STC_MARGIN_NUMBER, 0)
        self.regexp_display.SetMarginWidth(wx.stc.STC_MARGIN_SYMBOL, 0)
        hsizer.Add(self.regexp_display, 1, wx.EXPAND | wx.ALL, 5)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(
            wx.StaticText(self, label="Test text:"), 0, wx.ALIGN_CENTER | wx.ALL, 5
        )
        self.test_text_ctl = wx.TextCtrl(self, value=self.__test_text)
        self.test_text_ctl.Font = self.font
        hsizer.Add(self.test_text_ctl, 1, wx.ALIGN_CENTER | wx.ALL, 5)
        sizer.Add(hsizer, 0, wx.GROW | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        style = wx.NO_BORDER
        self.test_display = wx.stc.StyledTextCtrl(self, -1, style=style)
        self.test_display.SetLexer(wx.stc.STC_LEX_CONTAINER)
        self.test_display.StyleClearAll()
        self.test_display.StyleSetFont(STYLE_NO_MATCH, self.font)
        self.test_display.StyleSetForeground(
            STYLE_NO_MATCH, wx.Colour(128, 128, 128, 255)
        )
        color_db = self.get_color_db()
        for i in range(16):
            self.test_display.StyleSetFont(STYLE_FIRST_LABEL - 1 + i, self.font)
            self.test_display.StyleSetForeground(
                STYLE_FIRST_LABEL - 1 + i, color_db[i % len(color_db)]
            )

        self.test_display.StyleSetFont(STYLE_ERROR, self.error_font)
        self.test_display.StyleSetForeground(STYLE_ERROR, wx.Colour(255, 0, 0, 255))
        self.test_display.Text = self.__test_text
        self.test_display.SetReadOnly(True)
        self.test_display.SetUseVerticalScrollBar(0)
        self.test_display.SetUseHorizontalScrollBar(0)
        self.test_display.SetMarginWidth(wx.stc.STC_MARGIN_NUMBER, 0)
        self.test_display.SetMarginWidth(wx.stc.STC_MARGIN_SYMBOL, 0)
        text_extent = self.test_display.GetFullTextExtent(self.__test_text)
        self.test_display.SetSizeHints(100, text_extent[1] + 3, maxH=text_extent[1] + 3)
        self.test_display.Enable(False)
        sizer.Add(self.test_display, 0, wx.EXPAND | wx.ALL, 5)

        line = wx.StaticLine(self, -1, size=(20, -1), style=wx.LI_HORIZONTAL)
        sizer.Add(line, 0, wx.GROW | wx.RIGHT | wx.LEFT, 5)

        hsizer = wx.StdDialogButtonSizer()
        guess_button = wx.Button(self, label="Guess")
        hsizer.Add(guess_button, 0)
        ok_button = wx.Button(self, label="Submit")
        ok_button.SetDefault()
        hsizer.Add(ok_button, 0, wx.LEFT, 5)
        cancel_button = wx.Button(self, label="Cancel")
        hsizer.Add(cancel_button, 0, wx.LEFT, 5)
        hsizer.Realize()
        sizer.Add(hsizer, 0, wx.ALIGN_RIGHT | wx.ALL, 5)

        self.Bind(wx.EVT_BUTTON, self.on_guess, guess_button)
        self.Bind(wx.EVT_BUTTON, self.on_ok_button, ok_button)
        self.Bind(wx.EVT_BUTTON, self.on_cancel_button, cancel_button)
        self.Bind(wx.EVT_TEXT, self.on_test_text_text_change, self.test_text_ctl)
        self.Bind(
            wx.stc.EVT_STC_CHANGE, self.on_editor_text_change, self.regexp_display
        )
        self.Bind(wx.stc.EVT_STC_STYLENEEDED, self.on_style_needed, self.regexp_display)
        self.regexp_display.Bind(wx.EVT_KEY_DOWN, self.on_regexp_key)
        self.SetSizer(sizer)
        self.Fit()

    @staticmethod
    def on_regexp_key(event):
        #
        # On Mac, very bad things (infinite recursion through OnPaint
        # followed by segfault) happen if you type carriage return
        #
        if event.GetKeyCode() != wx.stc.STC_KEY_RETURN:
            event.Skip()

    @staticmethod
    def get_color_db():
        color_db = [
            "BLACK",
            "RED",
            "GREEN",
            "BLUE",
            "CYAN",
            "MAGENTA",
            "SIENNA",
            "PURPLE",
        ]
        color_db = [wx.TheColourDatabase.FindColour(x) for x in color_db]
        return color_db

    def on_guess(self, event):
        sample = self.test_text_ctl.GetValue()
        for guess in self.guesses:
            m = re.match(guess, sample)
            if m is not None:
                self.regexp_display.Text = guess
                break
        else:
            wx.MessageBox(
                "None of the standard regular expressions matches the test text.",
                caption="No matching guesses",
                style=wx.OK | wx.CENTRE | wx.ICON_INFORMATION,
                parent=self,
            )

    def on_ok_button(self, event):
        self.EndModal(1)

    def on_cancel_button(self, event):
        self.__value = None
        self.EndModal(0)

    def on_editor_text_change(self, event):
        self.__value = self.regexp_display.Text
        self.refresh_text()

    def on_style_needed(self, event):
        self.refresh_regexp()

    def on_test_text_text_change(self, event):
        self.__test_text = self.test_text_ctl.GetValue()
        self.refresh_text()

    def refresh_regexp(self):
        state = RegularExpressionState()
        regexp_text = self.__value
        self.regexp_display.StartStyling(0, 0xFF)
        self.regexp_display.SetStyling(len(regexp_text), STYLE_ERROR)
        try:
            parse(regexp_text, state)
        except:
            pass
        for i in range(state.position):
            self.regexp_display.StartStyling(i, 0xFF)
            self.regexp_display.SetStyling(1, state.token_labels[i])
        pos = self.regexp_display.CurrentPos
        if state.open_expression_start is not None:
            self.regexp_display.BraceBadLight(state.open_expression_start)
        elif (
            0 < pos < len(state.matching_braces)
            and state.matching_braces[pos - 1] is not None
        ):
            self.regexp_display.BraceHighlight(state.matching_braces[pos - 1], pos - 1)
        else:
            self.regexp_display.BraceHighlight(
                wx.stc.STC_INVALID_POSITION, wx.stc.STC_INVALID_POSITION
            )

    def refresh_text(self):
        self.test_display.SetReadOnly(False)
        self.test_display.Text = self.__test_text
        try:
            parse(self.__value, RegularExpressionState())
        except ValueError as e:
            self.test_display.Text = e.args[0]
            self.test_display.StartStyling(0, 0xFF)
            self.test_display.SetStyling(len(self.test_display.Text), STYLE_ERROR)
            return
        try:
            match = re.search(self.__value, self.__test_text)
            if match:
                for i in range(len(match.groups()) + 1):
                    start = match.start(i)
                    end = match.end(i)
                    self.test_display.StartStyling(start, 0xFF)
                    self.test_display.SetStyling(end - start, i + 1)
            else:
                self.test_display.Text = "Regular expression does not match"
                self.test_display.StartStyling(0, 0xFF)
                self.test_display.SetStyling(len(self.test_display.Text), STYLE_ERROR)
        except:
            self.test_display.Text = "Regular expression is not valid"
            self.test_display.StartStyling(0, 0xFF)
            self.test_display.SetStyling(len(self.test_display.GetText()), STYLE_ERROR)
        self.test_display.SetReadOnly(True)

    def get_value(self):
        return self.__value

    def set_value(self, value):
        self.__value = value
        self.regexp_display.SetText(value)
        self.refresh_regexp()
        self.refresh_text()

    value = property(get_value, set_value)

    def get_test_text(self):
        return self.__test_text

    def set_test_text(self, test_text):
        self.__test_text = test_text
        self.test_text_ctl.SetValue(test_text)
        self.test_display.SetText(test_text)
        self.refresh_text()

    test_text = property(get_test_text, set_test_text)

    def get_guesses(self):
        """The guess regexps used when the user presses the "guess" button"""
        return self.__guesses

    def set_guesses(self, value):
        self.__guesses = value

    guesses = property(get_guesses, set_guesses)
