from ._filter_predicate import FilterPredicate


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
