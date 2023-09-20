import os

from ._does_not_predicate import DoesNotPredicate
from ._does_predicate import DoesPredicate
from ._file_predicate import FilePredicate
from ._filter import CONTAINS_PREDICATE
from ._filter import CONTAINS_REGEXP_PREDICATE
from ._filter import ENDSWITH_PREDICATE
from ._filter import EQ_PREDICATE
from ._filter import STARTS_WITH_PREDICATE
from ._filter_predicate import FilterPredicate
from .._file_collection_display import FileCollectionDisplay


class DirectoryPredicate(FilePredicate):
    """A predicate that only filters directories"""

    def __init__(self):
        subpredicates = (
            CONTAINS_PREDICATE,
            CONTAINS_REGEXP_PREDICATE,
            STARTS_WITH_PREDICATE,
            ENDSWITH_PREDICATE,
            EQ_PREDICATE,
        )
        predicates = [
            DoesPredicate(subpredicates),
            DoesNotPredicate(subpredicates),
        ]
        FilterPredicate.__init__(
            self,
            "directory",
            "Directory",
            self.fn_filter,
            predicates,
            doc="Apply the rule to directories",
        )

    @staticmethod
    def fn_filter(node_type__modpath__module, *args):
        """The DirectoryPredicate filter function

        The arg slot expects a tuple of node_type and modpath.
        The predicate returns None (= agnostic about filtering) if
        the node is not a directory, otherwise it composites the
        modpath into a file path and applies it to the rest of
        the args.
        """
        (node_type, modpath, module) = node_type__modpath__module
        if isinstance(modpath[-1], tuple) and len(modpath[-1]) == 3:
            path = os.path.join(*modpath[:-2])
        else:
            path = os.path.join(*modpath[:-1])
        return args[0](path, *args[1:])

    def test_valid(self, pipeline, *args):
        self(
            (FileCollectionDisplay.NODE_FILE, ["/imaging", "image.tif"], None,), *args,
        )
