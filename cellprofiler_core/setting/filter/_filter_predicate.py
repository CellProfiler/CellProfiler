import re


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
