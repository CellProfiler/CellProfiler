from ._filter_predicate import FilterPredicate


class CompoundFilterPredicate(FilterPredicate):
    def test_valid(self, pipeline, *args):
        for subexp in args:
            subexp[0].test_valid(pipeline, *subexp[1:])
