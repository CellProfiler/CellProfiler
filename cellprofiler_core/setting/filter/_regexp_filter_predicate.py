import re

from ._filter_predicate import FilterPredicate


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
