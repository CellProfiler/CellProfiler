import re

from ._setting import Setting
from ._validation_error import ValidationError


class TreeChoice(Setting):
    """A tree choice chooses one path to a leaf in a tree

    Trees are represented as collections of two-tuples. The first element is
    the name of the node and the second is either None if a leaf or
    a sub-collection of two-tuples. For instance:
    (("Foo", (("1", None),("2", None))), ("Bar", None))
    is a tree for selecting ("Foo", "1"), ("Foo", "2") or ("Bar",).

    A good UI choice would be a hierarchical menu.
    """

    def __init__(self, text, value, tree, fn_is_leaf=None, **kwargs):
        """Initializer

        text - informative label

        value - the text value, e.g., as encoded by encode_path_parts

        tree - the tree to chose from

        fn_is_leaf - if defined, a function that takes a tree node and
                     returns True if that node is a leaf (a node might
                     have subnodes, but also be a leaf)
        """
        super(TreeChoice, self).__init__(text, value, **kwargs)
        self.__tree = tree
        self.fn_is_leaf = fn_is_leaf or self.default_fn_is_leaf

    @staticmethod
    def default_fn_is_leaf(node):
        return node[1] is None or len(node[1]) == 0

    def get_path_parts(self):
        """Split at |, but || escapes to |"""
        result = re.split("(?<!\\|)\\|(?!\\|)", self.get_value_text())
        return [x.replace("||", "|") for x in result]

    @staticmethod
    def encode_path_parts(value):
        """Return the setting value for a list of menu path parts"""
        return "|".join([x.replace("|", "||") for x in value])

    def get_leaves(self, path=None):
        """Get all leaf nodes of a given parent node

        path - the names of nodes traversing the path down the tree
        """
        if path is None:
            path = []
        current = self.get_tree()
        while len(path) > 0:
            idx = current.index(path[0])
            if idx == -1 or current[idx][1] is None or len(current[idx][1]) == 0:
                return []
            current = current[idx][1]
            path = path[1:]
        return [x[0] for x in current if x[1] is None or len(x[1] == 0)]

    def get_subnodes(self, path=None):
        """Get all child nodes that are not leaves for a  given parent

        path - the names of nodes traversing the path down the tree
        """
        if path is None:
            path = []
        current = self.get_tree()
        while len(path) > 0:
            idx = current.index(path[0])
            if idx == -1 or current[idx][1] is None:
                return []
            current = current[idx][1]
            path = path[1:]
        return [x[0] for x in current if x[1] is not None]

    def get_selected_leaf(self):
        """Get the leaf node of the tree for the current setting value"""
        tree = self.get_tree()
        node = None
        for item in self.get_path_parts():
            nodes = [n for n in tree if n[0] == item]
            if len(nodes) != 1:
                raise ValidationError(
                    "Unable to find command " + ">".join(self.get_path_parts()), self
                )
            node = nodes[0]
            tree = node[1]
        return node

    def test_valid(self, pipeline):
        self.get_selected_leaf()

    def get_tree(self):
        if hasattr(self.__tree, "__call__"):
            return self.__tree()
        return self.__tree
