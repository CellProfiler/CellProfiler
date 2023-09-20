from ._filter_predicate import FilterPredicate


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
