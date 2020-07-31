from cellprofiler.gui.constants.dialog.regular_expression_dialog import (
    TOK_ORDINARY,
    TOK_GROUP,
    TOK_BRACKET_EXP,
    TOK_REPEAT,
    TOK_SPECIAL,
)


class RegularExpressionState(object):
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
