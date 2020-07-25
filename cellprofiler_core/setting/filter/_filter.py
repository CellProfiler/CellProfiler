import re

from cellprofiler_core.setting import _setting
from cellprofiler_core.setting._validation_error import ValidationError


class Filter(_setting.Setting):
    """A filter that can be applied to an object

    A filter returns a value when applied to an object such as a string
    which is evaluated as either True (accept it) or False (reject it).

    The setting value is composed of tokens with a scheme-like syntax:

    (and (filename contains "_w1_") (extension is "tif"))

    Each predicate has a symbolic name which is used to find it. The predicate
    has an evaluation function and a display name. Predicates also have lists
    of the predicates that they operate on. The leftmost predicate takes two
    arguments. Other predicates, it is up to the developer. Predicates
    are called with the object of interest as the first argument and the
    evaluation value of the predicate to the right as the second argument.

    For something like "filename contains "foo"", "contains" returns a function
    that returns true if the first argument is "foo" and "filename" parses
    the first of its arguments to get the filename and returns the result of
    applying the result of "contains" to the filename.

    There are three special predicates:
    "and", "or" and "literal".
    """

    class FilterPredicate(object):
        def __init__(self, symbol, display_name, function, subpredicates, doc=None):
            self.symbol = symbol
            self.display_name = display_name
            self.function = function
            self.subpredicates = subpredicates
            self.doc = doc

        def __call__(self, *args, **kwargs):
            return self.function(*args, **kwargs)

        def test_valid(self, pipeline, *args):
            """Try running the filter on a test string"""
            self("", *args)

        @classmethod
        def encode_symbol(cls, symbol):
            """Escape encode an abritrary symbol name

            The parser needs to have special characters escaped. These are
            backslash, open and close parentheses, space and double quote.
            """
            return re.escape(symbol)

        @classmethod
        def decode_symbol(cls, symbol):
            """Decode an escape-encoded symbol"""
            s = ""
            in_escape = False
            for c in symbol:
                if in_escape:
                    in_escape = False
                    s += c
                elif c == "\\":
                    in_escape = True
                else:
                    s += c
            return s

    class CompoundFilterPredicate(FilterPredicate):
        def test_valid(self, pipeline, *args):
            for subexp in args:
                subexp[0].test_valid(pipeline, *subexp[1:])

    @classmethod
    def eval_list(cls, fn, x, *args):
        results = [v for v in [arg[0](x, *arg[1:]) for arg in args] if v is not None]
        if len(results) == 0:
            return None
        return fn(results)

    AND_PREDICATE = CompoundFilterPredicate(
        "and",
        "All",
        lambda x, *l: Filter.eval_list(all, x, *l),
        list,
        doc="All subordinate rules must be satisfied",
    )
    OR_PREDICATE = CompoundFilterPredicate(
        "or",
        "Any",
        lambda x, *l: Filter.eval_list(any, x, *l),
        list,
        doc="Any one of the subordinate rules must be satisfied",
    )
    LITERAL_PREDICATE = FilterPredicate(
        "literal", "Custom value", None, [], doc="Enter the rule's text"
    )
    CONTAINS_PREDICATE = FilterPredicate(
        "contain",
        "Contain",
        lambda x, y: x.find(y) >= 0,
        [LITERAL_PREDICATE],
        doc="The element must contain the text that you enter to the right",
    )
    STARTS_WITH_PREDICATE = FilterPredicate(
        "startwith",
        "Start with",
        lambda x, y: x.startswith(y),
        [LITERAL_PREDICATE],
        doc="The element must start with the text that you enter to the right",
    )
    ENDSWITH_PREDICATE = FilterPredicate(
        "endwith",
        "End with",
        lambda x, y: x.endswith(y),
        [LITERAL_PREDICATE],
        doc="The element must end with the text that you enter to the right",
    )

    class RegexpFilterPredicate(FilterPredicate):
        def __init__(self, display_name, subpredicates):
            super(self.__class__, self).__init__(
                "containregexp",
                display_name,
                self.regexp_fn,
                subpredicates,
                doc="The element must contain a match for the regular expression that you enter to the right",
            )

        def regexp_fn(self, x, y):
            try:
                pattern = re.compile(y)
            except:
                raise ValueError("Badly formatted regular expression: %s" % y)
            return pattern.search(x) is not None

    CONTAINS_REGEXP_PREDICATE = RegexpFilterPredicate(
        "Contain regular expression", [LITERAL_PREDICATE]
    )
    EQ_PREDICATE = FilterPredicate(
        "eq",
        "Exactly match",
        lambda x, y: x == y,
        [LITERAL_PREDICATE],
        doc="Must exactly match the text that you enter to the right",
    )

    class DoesPredicate(FilterPredicate):
        """Pass the arguments through (no-op)"""

        SYMBOL = "does"

        def __init__(
            self,
            subpredicates,
            text="Does",
            doc="The rule passes if the condition to the right holds",
        ):
            super(self.__class__, self).__init__(
                self.SYMBOL, text, lambda x, f, *l: f(x, *l), subpredicates, doc=doc
            )

    class DoesNotPredicate(FilterPredicate):
        """Negate the result of the arguments"""

        SYMBOL = "doesnot"

        def __init__(
            self,
            subpredicates,
            text="Does not",
            doc="The rule fails if the condition to the right holds",
        ):
            super(self.__class__, self).__init__(
                self.SYMBOL, text, lambda x, f, *l: not f(x, *l), subpredicates, doc=doc
            )

    def __init__(self, text, predicates, value="", **kwargs):
        super(self.__class__, self).__init__(text, value, **kwargs)
        self.predicates = predicates
        self.cached_token_string = None
        self.cached_tokens = None

    def evaluate(self, x):
        """Evaluate the value passed using the predicates"""
        try:
            tokens = self.parse()
            return tokens[0](x, *tokens[1:])
        except:
            return False

    def parse(self):
        """Parse the value into filter predicates, literals and lists

        Returns the value of the text as a list.
        """
        s = self.value_text
        if s == self.cached_token_string:
            return self.cached_tokens
        tokens = []
        predicates = self.predicates
        while len(s) > 0:
            token, s, predicates = self.parse_token(s, predicates)
            tokens.append(token)
        self.cached_tokens = list(tokens)
        self.cached_token_string = self.value_text
        return tokens

    def default(self, predicates=None):
        """A default list of tokens to use if things go horribly wrong

        We need to be able to generate a default list of tokens if the
        pipeline has been corrupted and the text can't be parsed.
        """
        tokens = []
        if predicates is None:
            predicates = self.predicates
        while len(predicates) > 0:
            token = predicates[0]
            if token is self.LITERAL_PREDICATE:
                tokens.append("")
                predicates = self.LITERAL_PREDICATE.subpredicates
            else:
                tokens.append(token)
                predicates = token.subpredicates
        return tokens

    @classmethod
    def parse_token(cls, s, predicates):
        """Parse a token out of the front of the string

        Returns the next token in the string, the rest of the string
        and the acceptable tokens for the rest of the string.
        """
        orig_predicates = predicates
        if list in predicates:
            needs_list = True
            predicates = list(predicates)
            predicates.remove(list)
        else:
            needs_list = False
        if s[0] == "(":
            if not needs_list:
                raise ValueError("List not allowed in current context")
            s = s[1:]
            result = []
            while s[0] != ")":
                token, s, predicates = cls.parse_token(s, predicates)
                result.append(token)
            if len(s) > 1 and s[1] == " ":
                return result, s[2:], orig_predicates
            return result, s[1:], orig_predicates
        elif needs_list:
            raise ValueError("List required in current context")
        if s[0] == '"':
            if cls.LITERAL_PREDICATE not in predicates:
                raise ValueError("Literal not allowed in current context")
            escape_next = False
            result = ""
            for i in range(1, len(s)):
                if escape_next:
                    result += s[i]
                    escape_next = False
                elif s[i] == "\\":
                    escape_next = True
                elif s[i] == '"':
                    return result, s[(i + 1) :], []
                else:
                    result += s[i]
            raise ValueError("Unterminated literal")
        #
        # (?:\\.|[^ )]) matches either backslash-anything or anything but
        # space and parentheses. So you can have variable names with spaces
        # and that's needed for arbitrary metadata names
        #
        match = re.match(r"^((?:\\.|[^ )])+) ?(.*)$", s)
        if match is None:
            kwd = s
            rest = ""
        else:
            kwd, rest = match.groups()
        kwd = Filter.FilterPredicate.decode_symbol(kwd)
        if kwd == cls.AND_PREDICATE.symbol:
            match = cls.AND_PREDICATE
        elif kwd == cls.OR_PREDICATE.symbol:
            match = cls.OR_PREDICATE
        else:
            matches = [x for x in predicates if x is not list and x.symbol == kwd]
            if len(matches) == 0:
                raise ValueError(
                    'The filter predicate, "%s", was not in the list of allowed predicates ("%s")'
                    % (kwd, '","'.join([x.symbol for x in predicates]))
                )
            match = matches[0]
        if match.subpredicates is list:
            predicates = [list] + predicates
        elif match.subpredicates is not None:
            predicates = match.subpredicates
        return match, rest, predicates

    @classmethod
    def encode_literal(cls, literal):
        """Encode a literal value with backslash escapes"""
        return literal.replace("\\", "\\\\").replace('"', '\\"')

    def build(self, structure):
        """Build the textual representation of a filter from its structure

        structure: the processing structure, represented using a nested list.

        The top layer of the list corresponds to the tokens in the value
        string. For instance, a list of [foo, bar, baz] where foo, bar and baz
        are filter predicates that have symbolic names of "foo", "bar" and "baz"
        will yield the string, "foo bar baz". The list [foo, bar, "baz"] will
        treat "baz" as a literal and yield the string, 'foo bar "baz"'.

        Nesting can be done using nested lists. For instance,

        [or [eq "Hello"] [eq "World"]]

        becomes

        "or (eq "Hello")(eq "World")"

        The function sets the filter's value using the generated string.
        """
        self.value = self.build_string(structure)

    @classmethod
    def build_string(cls, structure):
        """Return the text representation of structure

        This is a helper function for self.build. See self.build's
        documentation.
        """
        s = []
        for element in structure:
            if isinstance(element, Filter.FilterPredicate):
                s.append(cls.FilterPredicate.encode_symbol(str(element.symbol)))
            elif isinstance(element, str):
                s.append('"' + cls.encode_literal(element) + '"')
            else:
                s.append("(" + cls.build_string(element) + ")")
        return " ".join(s)

    def test_valid(self, pipeline):
        try:
            import javabridge as J

            J.run_script(
                """
            importPackage(Packages.org.cellprofiler.imageset.filter);
            new Filter(expr, klass);
            """,
                dict(
                    expr=self.value_text,
                    klass=J.class_for_name(
                        "org.cellprofiler.imageset.ImagePlaneDetailsStack"
                    ),
                ),
            )
        except Exception as e:
            raise ValidationError(str(e), self)

    def test_setting_warnings(self, pipeline):
        """Warn on empty literal token
        """
        super(Filter, self).test_setting_warnings(pipeline)
        self.__warn_if_blank(self.parse())

    def __warn_if_blank(self, l):
        for x in l:
            if isinstance(x, (list, tuple)):
                self.__warn_if_blank(x)
            elif x == "":
                raise ValidationError(
                    "The text entry for an expression in this filter is blank", self
                )
