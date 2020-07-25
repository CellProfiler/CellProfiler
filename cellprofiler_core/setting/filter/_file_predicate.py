import cellprofiler_core.setting


class FilePredicate(cellprofiler_core.setting.Filter.FilterPredicate):
    """A predicate that only filters files"""

    def __init__(self):
        subpredicates = (
            cellprofiler_core.setting.Filter.CONTAINS_PREDICATE,
            cellprofiler_core.setting.Filter.CONTAINS_REGEXP_PREDICATE,
            cellprofiler_core.setting.Filter.STARTS_WITH_PREDICATE,
            cellprofiler_core.setting.Filter.ENDSWITH_PREDICATE,
            cellprofiler_core.setting.Filter.EQ_PREDICATE,
        )
        predicates = [
            cellprofiler_core.setting.Filter.DoesPredicate(subpredicates),
            cellprofiler_core.setting.Filter.DoesNotPredicate(subpredicates),
        ]
        cellprofiler_core.setting.Filter.FilterPredicate.__init__(
            self,
            "file",
            "File",
            self.fn_filter,
            predicates,
            doc="Apply the rule to files",
        )

    @staticmethod
    def fn_filter(node_type__modpath__module, *args):
        """The FilePredicate filter function

        The arg slot expects a tuple of node_type and modpath.
        The predicate returns None (= agnostic about filtering) if
        the node is not a directory, otherwise it composites the
        modpath into a file path and applies it to the rest of
        the args
        """
        (node_type, modpath, module) = node_type__modpath__module
        if node_type == cellprofiler_core.setting.FileCollectionDisplay.NODE_DIRECTORY:
            return None
        elif isinstance(modpath[-1], tuple) and len(modpath[-1]) == 3:
            filename = modpath[-2]
        else:
            filename = modpath[-1]
        return args[0](filename, *args[1:])

    def test_valid(self, pipeline, *args):
        self(
            (
                cellprofiler_core.setting.FileCollectionDisplay.NODE_FILE,
                ["/imaging", "test.tif"],
                None,
            ),
            *args,
        )
