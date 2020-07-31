import re

import wx

from cellprofiler.gui.widget.dialog import RegularExpressionDialog
from cellprofiler.gui.constants.dialog.regular_expression_dialog import (
    STYLE_ERROR,
    RE_FILENAME_GUESSES,
    TOK_ESCAPE,
    TOK_GROUP,
    TOK_BRACKET_EXP,
    TOK_SPECIAL,
    TOK_DEFINITION,
    HARDCODE_ESCAPES,
    OCTAL_DIGITS,
    DECIMAL_DIGITS,
    HEXIDECIMAL_DIGITS,
    REPEAT_STARTS,
)


def edit_regexp(parent, regexp, test_text, guesses=None):
    if guesses is None:
        guesses = RE_FILENAME_GUESSES
    frame = RegularExpressionDialog(
        parent, style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER
    )
    frame.SetMinSize((300, 200))
    frame.SetSize((600, 200))
    frame.value = regexp
    frame.test_text = test_text
    frame.guesses = guesses
    if frame.ShowModal():
        return frame.value
    return None


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
