# coding=utf-8
"""regexp_editor - give a user feedback on their regular expression
"""

import re

import wx
import wx.stc

STYLE_NO_MATCH = 0
STYLE_MATCH = 1
STYLE_FIRST_LABEL = 2
STYLE_ERROR = 31

UUID_REGEXP = (
    "[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
)
RE_FILENAME_GUESSES = [
    # This is the generic naming convention for fluorescent microscopy images
    "^(?P<Plate>.*?)_(?P<Well>[A-Za-z]+[0-9]+)f(?P<Site>[0-9]{2})d(?P<Dye>[0-9])\\.tif$",
    # Molecular devices single site
    "^(?P<ExperimentName>.*?)_(?P<Well>[A-Za-z]+[0-9]+)_w(?P<Wavelength>[0-9])_?"
    + UUID_REGEXP
    + "\\.tif$",
    # Plate / well / site / channel without UUID
    "^(?P<Plate>.*?)_(?P<Well>[A-Za-z]+[0-9]+)_s(?P<Site>[0-9])_w(?P<Wavelength>[0-9])\\.tif$",
    # Molecular devices multi-site
    "^(?P<ExperimentName>.*?)_(?P<Well>[A-Za-z]+[0-9]+)_s(?P<Site>[0-9])_w(?P<Wavelength>[0-9])"
    + UUID_REGEXP
    + "\\.tif$",
    # Molecular devices multi-site, single wavelength
    "^(?P<ExperimentName>.*)_(?P<Well>[A-Za-z][0-9]{2})_s(?P<Site>[0-9])" + UUID_REGEXP,
    # Plate / well / [UUID]
    "^(?P<Plate>.*?)_(?P<Well>[A-Za-z]+[0-9]+)_\\[" + UUID_REGEXP + "\\]\\.tif$",
    # Cellomics
    "^(?P<ExperimentName>.*)_(?P<Well>[A-Za-z][0-9]{1,2})f(?P<Site>[0-9]{1,2})d(?P<Wavelength>[0-9])",
    # BD Pathway
    "^(?P<Wavelength>.*) - n(?P<StackSlice>[0-9]{6})",
    # GE InCell Analyzer
    r"^(?P<Row>[A-H]*) - (?P<Column>[0-9]*)\(fld (?P<Site>[0-9]*) wv (?P<Wavelength>.*) - (?P<Filter>.*)\)",
    # Phenix
    r"^r(?P<WellRow>\d{2})c(?P<WellColumn>\d{2})f(?P<Site>\d{2})p\d{2}-ch(?P<ChannelNumber>\d)",
    # GE InCell Analyzer 7.2
    r"^(?P<Row>[A-P])_(?P<Column>[0-9]*)_f(?P<Site>[0-9]*)_c(?P<Channel>[0-9]*)_x(?P<Wavelength>.*)_m("
    r"?P<Filter>.*)_z(?P<Slice>[0-9]*)_t(?P<Timepoint>[0-9]*)\.tif",
    # Please add more guesses below
]

RE_FOLDER_GUESSES = [
    # BD Pathway
    r".*[\\/](?P<Plate>[^\\/]+)[\\/](?P<Well>[A-Za-z][0-9]{2})",
    # Molecular devices
    r".*[\\/](?P<Date>\d{4}-\d{1,2}-\d{1,2})[\\/](?P<PlateID>.*)$"
    # Please add more guesses below
]


def edit_regexp(parent, regexp, test_text, guesses=None):
    if guesses is None:
        guesses = RE_FILENAME_GUESSES
    frame = RegexpDialog(parent, style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
    frame.SetMinSize((300, 200))
    frame.SetSize((600, 200))
    frame.value = regexp
    frame.test_text = test_text
    frame.guesses = guesses
    if frame.ShowModal():
        return frame.value
    return None


class RegexpDialog(wx.Dialog):
    def __init__(self, *args, **varargs):
        varargs["title"] = "Regular expression editor"
        super(RegexpDialog, self).__init__(*args, **varargs)
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
        sizer.Add(hsizer, 0, wx.GROW | wx.ALL, 5)

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
        state = RegexpState()
        regexp_text = self.__value
        self.regexp_display.StartStyling(0)
        self.regexp_display.SetStyling(len(regexp_text), STYLE_ERROR)
        try:
            parse(regexp_text, state)
        except:
            pass
        for i in range(state.position):
            self.regexp_display.StartStyling(i)
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
            parse(self.__value, RegexpState())
        except ValueError as e:
            self.test_display.Text = e.args[0]
            self.test_display.StartStyling(0)
            self.test_display.SetStyling(len(self.test_display.Text), STYLE_ERROR)
            return
        try:
            match = re.search(self.__value, self.__test_text)
            if match:
                for i in range(len(match.groups()) + 1):
                    start = match.start(i)
                    end = match.end(i)
                    self.test_display.StartStyling(start)
                    self.test_display.SetStyling(end - start, i + 1)
            else:
                self.test_display.Text = "Regular expression does not match"
                self.test_display.StartStyling(0)
                self.test_display.SetStyling(len(self.test_display.Text), STYLE_ERROR)
        except:
            self.test_display.Text = "Regular expression is not valid"
            self.test_display.StartStyling(0)
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


####################
#
# The code below parses regexps, assigning categories to tokens
#
####################

TOK_ORDINARY = 0
TOK_ESCAPE = 1
TOK_GROUP = 2
TOK_BRACKET_EXP = 3
TOK_REPEAT = 4
TOK_SPECIAL = 5
TOK_DEFINITION = 6

HARDCODE_ESCAPES = {
    r"\\",
    r"\a",
    r"\b",
    r"\d",
    r"\f",
    r"\n",
    r"\r",
    r"\s",
    r"\t",
    r"\v",
    r"\w",
    r"\A",
    r"\B",
    r"\D",
    r"\S",
    r"\W",
    r"\Z",
}
OCTAL_DIGITS = set("01234567")
DECIMAL_DIGITS = set("0123456789")
HEXIDECIMAL_DIGITS = set("0123456789ABCDEFabcdef")
REPEAT_STARTS = set("{*+?")
OTHER_SPECIAL_CHARACTERS = set(".|")

IGNORABLE_GROUPS = (r"\(\?[iLmsux]+\)", r"\(\?#.*\)")


class RegexpState(object):
    def __init__(self):
        self.__group_count = 0
        self.__group_names = []
        self.__group_depth = 0
        self.__group_starts = []
        self.__in_brackets = False
        self.__bracket_start = None
        self.__any_tokens = False
        self._is_non_grouping = False
        self.position = 0
        self.token_labels = []
        self.matching_braces = []

    def mark_tokens(self, length, label):
        self.token_labels += [label] * length
        self.matching_braces += [None] * length

    def open_group(self, length, group_name=None, is_non_grouping=False):
        """Open a grouping expression"""
        self.__group_depth += 1
        self.__group_starts.append(self.position)
        self.__any_tokens = True
        self.__group_name = group_name
        self.__is_non_grouping = is_non_grouping
        self.__any_tokens = False
        self.mark_tokens(length, TOK_GROUP)
        self.position += length

    def close_group(self):
        """Close a grouping expression returning the matching position"""
        if self.__group_depth == 0:
            raise ValueError("Unmatched closing parentheses")
        if self.__group_name is not None:
            self.__group_names.append(self.__group_name)
        self.__group_name = None
        self.__group_depth -= 1
        if self.__is_non_grouping:
            self.__group_count += 1
        matching_brace = self.__group_starts.pop()
        self.mark_tokens(1, TOK_GROUP)
        self.matching_braces[self.position] = matching_brace
        self.position += 1
        self.__any_tokens = True
        return matching_brace

    @property
    def group_count(self):
        return self.__group_count

    def get_in_brackets(self):
        """True if the state is within [] brackets"""
        return self.__in_brackets

    in_brackets = property(get_in_brackets)

    def open_brackets(self):
        self.__in_brackets = True
        self.__bracket_start = self.position
        self.mark_tokens(1, TOK_BRACKET_EXP)
        self.position += 1
        self.__any_tokens = True

    def close_brackets(self):
        if not self.in_brackets:
            raise ValueError("Unmatched closing brackets")
        self.__in_brackets = False
        self.__any_tokens = True
        self.mark_tokens(1, TOK_BRACKET_EXP)
        self.matching_braces[self.position] = self.__bracket_start
        self.position += 1
        return self.__bracket_start

    def parsed_token(self, length=1, label=TOK_ORDINARY):
        self.__any_tokens = True
        self.mark_tokens(length, label)
        self.position += length

    def parsed_special(self, length=1, label=TOK_SPECIAL):
        """Parse a token that's not repeatable"""
        self.__any_tokens = False
        self.mark_tokens(length, label)
        self.position += length

    def parsed_repeat(self, length):
        self.__any_tokens = False
        self.mark_tokens(length, TOK_REPEAT)
        self.position += length

    def is_group_name(self, x):
        return x in self.__group_names

    def group_name_index(self, x):
        if x == self.__group_name:
            return len(self.__group_names)
        return self.__group_names.index(x)

    @property
    def open_expression_start(self):
        """Return the start of the innermost open expression or None"""
        if self.__in_brackets:
            return self.__bracket_start
        elif self.__group_depth:
            return self.__group_starts[-1]

    @property
    def any_tokens(self):
        return self.__any_tokens


def looking_at_escape(s, state):
    """Return # of characters in an escape

    s - string to look at
    state - the current search state

    returns either None or the # of characters in the escape
    """
    if s[0] != "\\":
        return
    if len(s) < 2:
        raise ValueError("Unterminated escape sequence")
    if s[:2] in HARDCODE_ESCAPES:
        return 2
    if state.in_brackets:
        # within brackets, only octal supported
        if s[1] in OCTAL_DIGITS:
            for i in range(2, min(4, len(s))):
                if s[i] != OCTAL_DIGITS:
                    return i
        if s[1] in DECIMAL_DIGITS:
            raise ValueError(
                "Numeric escapes within brackets must be octal values: e.g., [\\21] for ^Q"
            )
    elif s[1] == 0:
        for i in range(2, min(4, len(s))):
            if s[i] != OCTAL_DIGITS:
                return i
    elif s[1] in DECIMAL_DIGITS:
        # A group number
        if len(s) > 2 and s[2] in DECIMAL_DIGITS:
            group_number = int(s[1:3])
            length = 2
        else:
            group_number = int(s[1])
            length = 1
        if group_number > state.group_count:
            raise ValueError("Only %d groups at this point" % state.group_count)
        return length
    if s[1] == "x":
        if s[2] in HEXIDECIMAL_DIGITS and s[3] in HEXIDECIMAL_DIGITS:
            return 4
        raise ValueError("Hexidecimal escapes are two digits long: eg. \\x1F")
    # The escape is needless, but harmless
    return 2


def looking_at_repeat(s, state):
    if s[0] not in REPEAT_STARTS:
        return None
    if state.in_brackets:
        return None
    if not state.any_tokens:
        raise ValueError("Invalid repeat placement: there is nothing to repeat")
    if s[0] == "{":
        match = re.match("{([0-9]+)(,([0-9]+))?\\}", s)
        if not match:
            raise ValueError("Incomplete or badly formatted repeat expression")
        if match.group(3) is not None:
            if int(match.group(1)) > int(match.group(3)):
                raise ValueError(
                    "Minimum # of matches in %s is greater than maximum number"
                    % match.group()
                )
        return len(match.group())
    if len(s) > 1 and s[1] == "?":
        return 2
    return 1


def handle_open_group(s, state):
    if s[0] == "(":
        if len(s) > 2 and s[1] == "?":
            if s[2] in ("=", "!", ":"):
                # a look-ahead expression or parentheses without grouping
                state.open_group(3, is_non_grouping=True)
                return 3
            elif len(s) > 3 and s[1:3] == "<=":
                # A look-ahead expression
                state.open_group(4, is_non_grouping=True)
                return 4
            elif s[2] in set("iLmsux"):
                # Setting switches
                match = re.match(r"\(\?[iLmsux]+\)", s)
                if not match:
                    raise ValueError("Incomplete or badly formatted switch expression")
                state.parsed_special(len(match.group()))
                return len(match.group())
            elif s[2] == "#":
                # comment
                match = re.match(r"\(\?#.*\)", s)
                if not match:
                    raise ValueError("Incomplete or badly formatted comment")
                state.parsed_special(len(match.group()))
                return len(match.group())
            elif s[2] == "(":
                # (?(name/id)) construct
                match = re.match(r"\(\?\(([^)]+)\)", s)
                if not match:
                    raise ValueError("Incomplete or badly formatted conditional match")
                name_or_id = match.group(1)
                if name_or_id.isdigit():
                    if int(name_or_id) > state.group_count:
                        raise ValueError(
                            "Not enough groups before conditional match: asked for %d, but only %d available"
                            % (int(name_or_id), state.group_count)
                        )
                else:
                    if not state.is_group_name(name_or_id):
                        raise ValueError(
                            'Unavailable group name, "%s", in conditional match'
                            % name_or_id
                        )
                state.open_group(len(match.group()), is_non_grouping=True)
            elif s[2] == "P" and len(s) > 3:
                if s[3] == "=":
                    # (?P=FOO) matches the previous group expression, FOO
                    match = re.match(r"\(\?P=([^)]+)\)", s)
                    if not match:
                        raise ValueError(
                            "Incomplete or badly formatted named group reference"
                        )
                    else:
                        state.parsed_token(len(match.group()), TOK_GROUP)
                        return len(match.group())
                elif s[3] == "<":
                    # Named group definition: (?P<FOO>...)
                    match = re.match(r"\(\?P<([^>]+)>", s)
                    if not match:
                        raise ValueError(
                            "Incomplete or badly formattted named group definition"
                        )
                    elif state.is_group_name(match.group(1)):
                        raise ValueError("Doubly-defined group: %s" % match.group(1))
                    else:
                        group_name = match.group(1)
                        state.open_group(
                            len(match.group()),
                            group_name=group_name,
                            is_non_grouping=True,
                        )
                        state.token_labels[-len(group_name) - 1 : -1] = [
                            TOK_DEFINITION + state.group_name_index(group_name)
                        ] * len(group_name)
                        return len(match.group())
                else:
                    raise ValueError("Incomplete or badly formatted (?P expression")
            else:
                raise ValueError("Incomplete or badly formatted (? expression")
        else:
            state.open_group(1)
            return 1


def parse(s, state):
    while state.position < len(s):
        length = looking_at_escape(s[state.position :], state)
        if length:
            state.parsed_token(length, TOK_ESCAPE)
            continue
        if state.in_brackets:
            if s[state.position] != "]":
                state.parsed_token(1, TOK_BRACKET_EXP)
            else:
                state.close_brackets()
        else:
            length = looking_at_repeat(s[state.position :], state)
            if length:
                state.parsed_repeat(length)
                continue
            if s[state.position] == "[":
                state.open_brackets()
                continue
            if s[state.position] == "^":
                if state.position:
                    raise ValueError(
                        "^ can only appear at the start of a regular expression"
                    )
                else:
                    state.parsed_special()
                continue
            if s[state.position] == "$":
                if state.position < len(s) - 1:
                    raise ValueError(
                        "$ can only appear at the end of a regular expression"
                    )
                else:
                    state.parsed_special()
                    continue
            if s[state.position] == "|":
                state.parsed_special()
                continue
            if s[state.position] == ".":
                state.parsed_token(1, TOK_SPECIAL)
                continue
            if s[state.position] == ")":
                state.close_group()
                continue
            if handle_open_group(s[state.position :], state):
                continue
            # Otherwise, assume normal character
            state.parsed_token()
    if state.open_expression_start is not None:
        state.token_labels[state.open_expression_start] = STYLE_ERROR
        raise ValueError("Incomplete expression")
    return state
