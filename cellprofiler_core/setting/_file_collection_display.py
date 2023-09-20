import json

from ._setting import Setting


class FileCollectionDisplay(Setting):
    """A setting to be used to display directories and their files

    The FileCollectionDisplay shows directory trees with mechanisms to
    communicate directory additions and deletions to its parent module.

    The central data structure is the dictionary, "self.file_tree". The keys
    for the top-level of the dictionary are the directories managed by the
    setting. If a key represents a directory, its value is another directory.
    If a key represents a file, its value is either True (the file is included
    in the collection) or False (the file is filtered out of the collection).

    Directory dictionaries can be filtered: this is done by setting the
    special key, "None" to either True or False.

    The FileCollectionDisplay manages the tree and it should be treated as
    read-only by callers. Callers can request that nodes be added, removed,
    filtered or not filtered by calling the appropriate notification function
    with a nested collection of two-tuples and strings (modpaths). Two-tuples
    represent directories whose subdirectories or files are being operated on.
    Strings represent directories or files that are being operated on. The first
    element of the two-tuple is the directory name and the second is a
    sub-collection of two-tuples. For instance, to operate on foo/bar, send:

    ("foo", ("bar", ))

    The FileCollectionDisplay communicates events on individual files or
    directories by specifying a path as a collection of path parts. These
    can be any sort of object and it is the caller's job to maintain the
    display names of each of them and their node categories (used for
    icon display).
    """

    ADD = "ADD"
    REMOVE = "REMOVE"
    METADATA = "METADATA"
    NODE_DIRECTORY = "directory"
    NODE_COMPOSITE_IMAGE = "compositeimage"
    NODE_COLOR_IMAGE = "colorimage"
    NODE_MONOCHROME_IMAGE = "monochromeimage"
    NODE_IMAGE_PLANE = "imageplane"
    NODE_MOVIE = "movie"
    NODE_FILE = "file"
    NODE_CSV = "csv"
    BKGND_PAUSE = "pause"
    BKGND_RESUME = "resume"
    BKGND_STOP = "stop"
    BKGND_GET_STATE = "getstate"

    class DeleteMenuItem(object):
        """A placeholder in the context menu for the delete command

        The DeleteMenuItem can be placed in the context menu returned
        by fn_get_path_info so that the user can delete the selected items
        from the context menu.

        text - the text to display in the context menu
        """

        def __init__(self, text):
            self.text = text

    def __init__(
        self,
        text,
        value,
        fn_on_drop,
        fn_on_remove,
        fn_get_path_info,
        fn_on_menu_command,
        fn_on_bkgnd_control,
        hide_text="Hide filtered files",
        **kwargs,
    ):
        """Constructor

        text - the label to the left of the setting

        value - the value for the control. This is a serialization of
                the appearance (for instance, whether to show or hide
                filtered files).

        fn_on_drop - called when files are dropped. The signature is
                     fn_on_drop(pathnames, check_for_directories) The first
                     argument is a list of pathnames of the dropped files.
                     The second argument is True if the user has performed
                     a file name drop which might include directories and
                     False if the user has dropped text file names.

        fn_on_remove - called when the UI requests that files be removed. Has
                       one argument which is a collection of paths to remove.

        fn_get_path_info - called when the UI needs to know the display name,
                     icon type, context menu and tool tip for an item. These
                     are returned in a four-tuple by the callee, e.g:
                     [ "image.tif", NODE_MONOCHROME_IMAGE,
                       "image of well A01 on plate P-12345",
                       ( "Show image", "Show metadata", "Delete image")]

        fn_on_menu_command - called when the user selects a context menu
                     command. The argument is the text from the context menu or
                     None if the default command.

        fn_on_bkgnd_control - called when the UI wants to stop, pause or resume
                     all background processing. BKGND_PAUSE asks for the
                     caller to pause processing, BKGND_RESUME asks for the
                     caller to resume, BKGND_STOP asks for processing to be
                     aborted, BKGND_GET_STATE asks for the caller to
                     return its current state = BKGND_PAUSE if it is paused,
                     BKGND_RESUME if it is running or BKGND_STOP if it is
                     idle.

        hide_text - the text displayed next to the hide checkbox.
        """
        super(self.__class__, self).__init__(text, value, **kwargs)
        self.fn_on_drop = fn_on_drop
        self.fn_on_remove = fn_on_remove
        self.fn_get_path_info = fn_get_path_info
        self.fn_on_menu_command = fn_on_menu_command
        self.fn_on_bkgnd_control = fn_on_bkgnd_control
        self.hide_text = hide_text
        self.fn_update = None
        self.file_tree = {}
        self.properties = {self.SHOW_FILTERED: True}
        try:
            properties = json.loads(value)
            if isinstance(properties, dict):
                self.properties.update(properties)
        except:
            pass

    SHOW_FILTERED = "ShowFiltered"

    def update_value(self):
        """Update the setting value after changing a property"""
        self.value_text = json.dumps(self.properties)

    def update_ui(self, cmd=None, mods=None):
        if self.fn_update is not None:
            self.fn_update(cmd, mods)

    def set_update_function(self, fn_update=None):
        """Set the function that will be called when the file_tree is updated"""
        self.fn_update = fn_update

    def initialize_tree(self, mods):
        """Remove all nodes in the file tree"""
        self.file_tree = {}
        self.add_subtree(mods, self.file_tree)

    def add(self, mods):
        """Add nodes to the file tree

        mods - modification structure. See class documentation for its form.
        """
        self.add_subtree(mods, self.file_tree)
        self.update_ui(self.ADD, mods)

    def modify(self, mods):
        """Indicate a minor modification such as metadtaa change

        mods - modification structure. See class documentation for its form.
        """
        self.update_ui(self.METADATA, mods)

    @classmethod
    def is_leaf(cls, mod):
        """True if the modification structure is the leaf of a tree

        The leaves are either strings representing the last part of a path
        or 3-tuples representing image planes within an image file. Branches
        are two-tuples composed of a path part and more branches / leaves
        """
        return len(mod) != 2 or not isinstance(mod[0], str)

    def node_count(self, file_tree=None):
        """Count the # of nodes (leaves + directories) in the tree"""
        if file_tree is None:
            file_tree = self.file_tree
        count = 0
        for key in list(file_tree.keys()):
            if key is None:
                pass
            elif isinstance(file_tree[key], dict):
                count += 1 + self.node_count(file_tree[key])
            else:
                count += 1
        return count

    def get_tree_modpaths(self, path):
        """Create a modpath containing the selected node and all children

        root - list of paths to the selected node

        returns a modpath (two-tuples where the first is the key and the second
        is a list of sub-modpaths)
        """
        tree = self.file_tree
        root_modlist = sub_modlist = []
        while len(path) > 1:
            next_sub_modlist = []
            sub_modlist.append((path[0], next_sub_modlist))
            tree = tree[path[0]]
            path = path[1:]
            sub_modlist = next_sub_modlist
        if isinstance(tree[path[0]], dict):
            sub_modlist.append((path[0], self.get_all_modpaths(tree[path[0]])))
        else:
            sub_modlist.append(path[0])
        return root_modlist[0]

    def get_all_modpaths(self, tree):
        """Get all sub-modpaths from the branches of the given tree"""
        result = []
        for key in list(tree.keys()):
            if key is None:
                continue
            elif not isinstance(tree[key], dict):
                result.append(key)
            else:
                result.append((key, self.get_all_modpaths(tree[key])))
        return result

    def add_subtree(self, mods, tree):
        for mod in mods:
            if self.is_leaf(mod):
                if mod not in tree:
                    tree[mod] = True
            else:
                if mod[0] in tree and isinstance(tree[mod[0]], dict):
                    subtree = tree[mod[0]]
                else:
                    subtree = tree[mod[0]] = {}
                subtree[None] = True
                self.add_subtree(mod[1], subtree)

    def on_remove(self, mods):
        """Called when the UI wants to remove nodes

        mods - a modlist of nodes to remove
        """
        self.fn_on_remove(mods)

    def remove(self, mods):
        """Remove nodes from the file tree

        mods - modification structure. See class documentation for its form.
        """
        for mod in mods:
            self.remove_subtree(mod, self.file_tree)
        self.update_ui(self.REMOVE, mods)

    def remove_subtree(self, mod, tree):
        if not (isinstance(mod, tuple) and len(mod) == 2):
            if mod in tree:
                subtree = tree[mod]
                if isinstance(subtree, dict):
                    #
                    # Remove whole tree
                    #
                    for key in list(subtree.keys()):
                        if key is None:
                            continue
                        if isinstance(subtree[key], dict):
                            self.remove_subtree(key, subtree)
                del tree[mod]
        elif mod[0] in tree:
            root_mod = mod[0]
            subtree = tree[root_mod]
            if isinstance(subtree, dict):
                for submod in mod[1]:
                    self.remove_subtree(submod, subtree)
                #
                # Delete the subtree if the subtree is emptied
                #
                if len(subtree) == 0 or (len(subtree) == 1 and None in subtree):
                    del tree[root_mod]
            else:
                del tree[root_mod]

    def mark(self, mods, keep):
        """Mark tree nodes as filtered in or out

        mods - modification structure. See class documentation for its form.

        keep - true to mark a node as in the set, false to filter it out.
        """
        self.mark_subtree(mods, keep, self.file_tree)
        self.update_ui()

    def mark_subtree(self, mods, keep, tree):
        for mod in mods:
            if self.is_leaf(mod):
                if mod in tree:
                    if isinstance(tree[mod], dict):
                        tree[mod][None] = keep
                    else:
                        tree[mod] = keep
            else:
                if mod[0] in tree:
                    self.mark_subtree(mod[1], keep, tree[mod[0]])
        kept = [
            tree[k][None] if isinstance(tree[k], dict) else tree[k]
            for k in list(tree.keys())
            if k is not None
        ]
        tree[None] = any(kept)

    def get_node_info(self, path):
        """Get the display name, node type and tool tip for a node

        path - path to the image plane as a list of nodes

        returns a tuple of display name, node type and tool tip
        """
        display_name, node_type, tool_tip, menu = self.fn_get_path_info(path)
        return display_name, node_type, tool_tip

    def get_context_menu(self, path):
        """Get the context menu associated with a path

        path - path to the image plane

        returns a list of context menu items.
        """
        display_name, node_type, tool_tip, menu = self.fn_get_path_info(path)
        return menu

    def get_show_filtered(self):
        return self.properties[self.SHOW_FILTERED]

    def set_show_filtered(self, show_state):
        """Mark that we should show filtered files in the user interface

        show_state - true to show files / false to hide them
        """
        self.properties[self.SHOW_FILTERED] = show_state
        self.update_value()
        self.update_ui()

    show_filtered = property(get_show_filtered, set_show_filtered)
