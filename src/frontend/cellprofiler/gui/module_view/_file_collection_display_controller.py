import logging
import os
import sys
import uuid

import wx
from cellprofiler_core.preferences import report_progress
from cellprofiler_core.setting import FileCollectionDisplay

from ..pipeline import Pipeline
from ..utilities.module_view import edit_control_name
from ...icons import get_builtin_image

LOGGER = logging.getLogger(__name__)


class FileCollectionDisplayController:
    """This class provides the UI for the file collection display

    The UI has a browse button, a hide checkbox and a tree control.

    Critical attributes:

    self.walks_in_progress - this is a dictionary of keys to directory walks
                             and metadata fetches that are happening in the
                             background. The value of the dictionary entry
                             is the function to call to stop the search.

                             There's a completion callback that's called to
                             remove an entry from the dictionary. When the
                             dictionary size reaches zero, the stop and pause
                             buttons are disabled.

    self.modpath_to_item - a modpath is a collection of path parts to some file
                             handled by the controller. There's a tree item
                             for every modpath in this dictionary and the
                             dictionary can be used for fast lookup of the
                             item without traversing the entire tree.
    """

    # Don't defining constants on build, where no display available
    if wx.App.IsDisplayAvailable():
        IMAGE_LIST = wx.ImageList(16, 16, 3)
        FOLDER_IMAGE_INDEX = IMAGE_LIST.Add(
            wx.ArtProvider.GetBitmap(wx.ART_FOLDER, wx.ART_OTHER, size=(16, 16))
        )
        FOLDER_OPEN_IMAGE_INDEX = IMAGE_LIST.Add(
            wx.ArtProvider.GetBitmap(wx.ART_FOLDER_OPEN, wx.ART_OTHER, size=(16, 16))
        )
        FILE_IMAGE_INDEX = IMAGE_LIST.Add(
            wx.ArtProvider.GetBitmap(wx.ART_NORMAL_FILE, wx.ART_OTHER, size=(16, 16))
        )
        IMAGE_PLANE_IMAGE_INDEX = IMAGE_LIST.Add(
            get_builtin_image("microscope-icon_16").ConvertToBitmap()
        )
        IMAGE_PLANES_IMAGE_INDEX = IMAGE_LIST.Add(
            get_builtin_image("microscopes_16").ConvertToBitmap()
        )
        COLOR_IMAGE_INDEX = IMAGE_LIST.Add(
            get_builtin_image("microscope-color_16").ConvertToBitmap()
        )
        MOVIE_IMAGE_INDEX = IMAGE_LIST.Add(get_builtin_image("movie_16").ConvertToBitmap())

        ACTIVE_COLOR = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT)
        FILTERED_COLOR = wx.SystemSettings.GetColour(wx.SYS_COLOUR_GRAYTEXT)

    class FCDCDropTarget(wx.PyDropTarget):
        def __init__(self, file_callback_fn, text_callback_fn):
            super(self.__class__, self).__init__()
            self.file_callback_fn = file_callback_fn
            self.text_callback_fn = text_callback_fn
            self.file_data_object = wx.FileDataObject()
            self.text_data_object = wx.TextDataObject()
            self.composite_data_object = wx.DataObjectComposite()
            self.composite_data_object.Add(self.file_data_object, True)
            self.composite_data_object.Add(self.text_data_object)
            self.SetDataObject(self.composite_data_object)

        def OnDropFiles(self, x, y, filenames):
            self.file_callback_fn(x, y, filenames)

        def OnDropText(self, x, y, text):
            self.text_callback_fn(x, y, text)

        @staticmethod
        def OnEnter(x, y, d):
            return wx.DragCopy

        @staticmethod
        def OnDragOver(x, y, d):
            return wx.DragCopy

        def OnData(self, x, y, d):
            if self.GetData():
                df = self.composite_data_object.GetReceivedFormat().GetType()
                if df in (wx.DF_TEXT, wx.DF_UNICODETEXT):
                    self.OnDropText(x, y, self.text_data_object.GetText())
                elif df == wx.DF_FILENAME:
                    self.OnDropFiles(x, y, self.file_data_object.GetFilenames())
            return wx.DragCopy

        @staticmethod
        def OnDrop(x, y):
            return True

    def __init__(self, module_view, v, pipeline):
        assert isinstance(v, FileCollectionDisplay)
        self.module_view = module_view
        self.v = v
        assert isinstance(pipeline, Pipeline)
        self.pipeline = pipeline
        self.panel = wx.Panel(
            self.module_view.module_panel, -1, name=edit_control_name(v)
        )
        self.panel.controller = self
        self.panel.Sizer = wx.BoxSizer(wx.VERTICAL)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.panel.Sizer.Add(sizer, 0, wx.EXPAND)
        self.status_text = wx.StaticText(self.panel, -1)
        sizer.Add(self.status_text, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER)
        sizer.AddStretchSpacer()
        sizer.Add(
            wx.StaticText(self.panel, -1, "Drag folders and/or files here or"),
            0,
            wx.ALIGN_LEFT | wx.ALIGN_CENTER,
        )
        sizer.AddSpacer((3, 0))
        browse_button = wx.Button(self.panel, -1, "Browse...")
        sizer.Add(browse_button, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER)
        browse_button.Bind(wx.EVT_BUTTON, self.on_browse)
        tree_style = wx.TR_HIDE_ROOT | wx.TR_HAS_BUTTONS | wx.TR_MULTIPLE
        self.tree_ctrl = wx.TreeCtrl(self.panel, -1, style=tree_style)
        self.panel.Sizer.Add(self.tree_ctrl, 1, wx.EXPAND)
        self.tree_ctrl.SetImageList(self.IMAGE_LIST)
        self.tree_ctrl.Bind(wx.EVT_TREE_ITEM_MENU, self.on_tree_item_menu)
        self.tree_ctrl.Bind(wx.EVT_TREE_KEY_DOWN, self.on_tree_key_down)
        #
        # Don't auto-expand after the user collapses a node.
        #
        self.user_collapsed_a_node = False

        def on_item_collapsed(event):
            LOGGER.debug("On item collapsed")
            self.user_collapsed_a_node = True

        self.tree_ctrl.Bind(wx.EVT_TREE_ITEM_COLLAPSED, on_item_collapsed)
        self.tree_ctrl.Bind(wx.EVT_TREE_ITEM_ACTIVATED, self.on_tree_doubleclick)
        self.tree_ctrl.Bind(wx.EVT_ERASE_BACKGROUND, self.on_erase_background)

        self.panel.Bind(wx.EVT_WINDOW_DESTROY, self.on_destroy)
        self.root_item = self.tree_ctrl.AddRoot("I am the invisible root")
        self.tree_ctrl.SetPyData(self.root_item, None)
        self.tree_ctrl.SetItemImage(self.root_item, self.FOLDER_IMAGE_INDEX)
        self.tree_ctrl.SetItemImage(
            self.root_item, self.FOLDER_OPEN_IMAGE_INDEX, wx.TreeItemIcon_Expanded
        )
        self.tree_ctrl.SetMinSize((100, 300))
        self.tree_ctrl.SetMaxSize((sys.maxsize, 300))
        self.file_drop_target = self.FCDCDropTarget(
            self.on_drop_files, self.on_drop_text
        )
        self.tree_ctrl.SetDropTarget(self.file_drop_target)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.panel.Sizer.Add(sizer, 0, wx.EXPAND)
        self.hide_show_ctrl = wx.CheckBox(self.panel, -1, self.v.hide_text)
        sizer.Add(self.hide_show_ctrl, 0, wx.ALIGN_LEFT | wx.ALIGN_BOTTOM)
        self.hide_show_ctrl.Bind(wx.EVT_CHECKBOX, self.on_hide_show_checked)
        self.hide_show_ctrl.SetValue(not self.v.show_filtered)
        sizer.AddStretchSpacer()
        self.stop_button = wx.Button(self.panel, -1, "Stop")
        self.stop_button.Enable(False)
        self.stop_button.Bind(wx.EVT_BUTTON, self.on_stop)
        sizer.Add(self.stop_button, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        self.pause_button = wx.Button(self.panel, -1, "Pause")
        self.pause_button.Enable(False)
        self.pause_button.Bind(wx.EVT_BUTTON, self.on_pause_resume)
        sizer.Add(self.pause_button, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        v.set_update_function(self.request_update)
        self.needs_update = False
        self.modpath_to_item = {}
        self.request_update()

    def __del__(self):
        self.on_destroy(None)

    def on_destroy(self, event):
        self.v.set_update_function()

    def on_erase_background(self, event):
        assert isinstance(event, wx.EraseEvent)
        dc = event.GetDC()
        assert isinstance(dc, wx.DC)
        brush = wx.Brush(self.tree_ctrl.GetBackgroundColour())
        dc.SetBrush(brush)
        dc.SetPen(wx.TRANSPARENT_PEN)
        width, height = self.tree_ctrl.GetSize()
        dc.DrawRectangle(0, 0, width, height)
        if len(self.modpath_to_item) == 0:
            text = "Drop files and folders here"
            font = wx.Font(
                36, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD
            )
            dc.SetTextForeground(wx.SystemSettings.GetColour(wx.SYS_COLOUR_GRAYTEXT))
            dc.SetFont(font)
            text_width, text_height = dc.GetTextExtent(text)
            dc.DrawText(text, (width - text_width) / 2, (height - text_height) / 2)

    def on_browse(self, event):
        LOGGER.debug("Browsing for file collection directory")
        dlg = wx.DirDialog(self.panel, "Select a directory to add")
        try:
            if dlg.ShowModal() == wx.ID_OK:
                self.v.fn_on_drop([dlg.GetPath()], True)
        finally:
            dlg.Destroy()

    def on_start_received(self):
        self.pause_button.Label = "Pause"
        self.pause_button.Enable(True)
        self.stop_button.Enable(True)

    def on_stop_received(self):
        self.pause_button.Enable(False)
        self.stop_button.Enable(False)

    def on_stop(self, event):
        """Stop button pressed"""
        self.v.fn_on_bkgnd_control(self.v.BKGND_STOP)
        self.pause_button.Label = "Pause"
        self.pause_button.Enable(False)
        self.stop_button.Enable(False)

    def on_pause_resume(self, event):
        """Pause / resume pressed"""
        if self.pause_button.Label == "Pause":
            action = self.v.BKGND_PAUSE
            self.pause_button.Label = "Resume"
        else:
            action = self.v.BKGND_RESUME
            self.pause_button.Label = "Pause"
        self.v.fn_on_bkgnd_control(action)

    def add_item(self, modpath, text=None, sort=True):
        """Add an item to the tree

        modpath - a collection of path parts to the item in the tree
        text - the text to appear in the item
        """
        parent_key = tuple(modpath[:-1])
        modpath = tuple(modpath)
        if modpath in self.modpath_to_item:
            item = self.modpath_to_item[modpath]
            if text is not None:
                self.tree_ctrl.SetItemText(item, text)
            return item

        if text is None:
            text = modpath[-1]
        if len(modpath) == 1:
            parent_item = self.root_item
        elif parent_key in self.modpath_to_item:
            parent_item = self.modpath_to_item[parent_key]
        else:
            parent_item = self.add_item(parent_key, sort=sort)
            self.tree_ctrl.SetItemImage(parent_item, self.FOLDER_IMAGE_INDEX)
            self.tree_ctrl.SetItemImage(
                parent_item, self.FOLDER_OPEN_IMAGE_INDEX, wx.TreeItemIcon_Expanded
            )

        want_erase = len(self.modpath_to_item) == 0
        #
        # Put in alpha order
        #
        n_children = self.tree_ctrl.GetChildrenCount(parent_item)
        if n_children == 0 or not sort:
            item = self.tree_ctrl.AppendItem(parent_item, text)
        else:
            child, cookie = self.tree_ctrl.GetFirstChild(parent_item)
            for i in range(n_children):
                ctext = self.tree_ctrl.GetItemText(child)
                if ctext > text:
                    item = self.tree_ctrl.InsertItemBefore(parent_item, i, text)
                    break
                child = self.tree_ctrl.GetNextSibling(child)
            else:
                item = self.tree_ctrl.AppendItem(parent_item, text)

        self.tree_ctrl.SetPyData(item, modpath[-1])
        self.modpath_to_item[modpath] = item
        if want_erase:
            self.tree_ctrl.Refresh(True)
        return item

    def remove_item(self, modpath):
        modpath = tuple(modpath)
        if modpath in self.modpath_to_item:
            item = self.modpath_to_item[modpath]
            n_children = self.tree_ctrl.GetChildrenCount(item, False)
            if n_children > 0:
                child, cookie = self.tree_ctrl.GetFirstChild(item)
                child_tokens = []
                for i in range(n_children):
                    child_tokens.append(self.tree_ctrl.GetItemPyData(child))
                    child = self.tree_ctrl.GetNextSibling(child)
                for child_token in child_tokens:
                    sub_modpath = list(modpath) + [child_token]
                    self.remove_item(sub_modpath)
            self.tree_ctrl.Delete(self.modpath_to_item[modpath])
            del self.modpath_to_item[modpath]

    @classmethod
    def get_modpath(cls, path):
        """Break a path into its components"""
        result = []
        while True:
            new_path, part = os.path.split(path)
            if len(new_path) == 0 or len(part) == 0:
                result.insert(0, path)
                return result
            result.insert(0, part)
            path = new_path

    def on_drop_files(self, x, y, filenames):
        self.v.fn_on_drop(filenames, True)

    def on_drop_text(self, x, y, text):
        """Text is assumed to be one file name per line"""
        filenames = [line.strip() for line in text.split("\n") if len(line.strip()) > 0]
        self.v.fn_on_drop(filenames, False)

    def get_path_from_event(self, event):
        """Given a tree control event, find the path from the root

        event - event from tree control (e.g., EVT_TREE_ITEM_ACTIVATED)

        returns a sequence of path items from the root
        """
        item = event.GetItem()
        path = []
        while True:
            item_data = self.tree_ctrl.GetItemPyData(item)
            if item_data is None:
                break
            path.insert(0, item_data)
            item = self.tree_ctrl.GetItemParent(item)
        return path

    def on_tree_item_menu(self, event):
        LOGGER.debug("On tree item menu")
        path = self.get_path_from_event(event)
        if len(path) == 0:
            LOGGER.warning("Could not find item associated with tree event")
            return
        context_menu = self.v.get_context_menu(path)
        if len(context_menu) > 0:
            menu = wx.Menu()
            try:
                delete_menu_items = []
                for context_item in context_menu:
                    if isinstance(context_item, FileCollectionDisplay.DeleteMenuItem,):
                        delete_menu_items.append(menu.Append(-1, context_item.text).Id)
                    else:
                        menu.Append(-1, context_item)

                def on_menu(event):
                    LOGGER.debug("On menu")

                    self.pipeline.start_undoable_action()
                    try:
                        for menu_item in menu.GetMenuItems():
                            if menu_item.Id == event.Id:
                                LOGGER.debug("    Command = %s" % menu_item.Text)
                                if menu_item.Id in delete_menu_items:
                                    self.on_delete_selected(event)
                                else:
                                    self.v.fn_on_menu_command(path, menu_item.Text)
                                break
                    finally:
                        self.pipeline.stop_undoable_action()

                self.tree_ctrl.Bind(wx.EVT_MENU, on_menu)
                self.tree_ctrl.PopupMenu(menu, event.GetPoint())
                self.tree_ctrl.Unbind(wx.EVT_MENU, handler=on_menu)
            finally:
                menu.Destroy()

    def on_tree_doubleclick(self, event):
        path = self.get_path_from_event(event)
        if self.v.fn_on_menu_command(path, None):
            return True

    def on_tree_key_down(self, event):
        LOGGER.debug("On tree key down")
        key = event.GetKeyCode()
        if key == wx.WXK_DELETE:
            self.on_delete_selected(event)

    def on_delete_selected(self, event):
        mods = [self.get_item_address(item) for item in self.tree_ctrl.GetSelections()]
        mods = [x for x in mods if x is not None]
        self.v.on_remove([self.v.get_tree_modpaths(mod) for mod in mods])

    def get_item_address(self, item):
        """Get an item's address as a collection of names"""
        result = []
        while True:
            name = self.tree_ctrl.GetItemPyData(item)
            if name is None:
                break
            else:
                result.insert(0, name)
                item = self.tree_ctrl.GetItemParent(item)
        return result

    def get_item_from_modpath(self, modpath):
        """Get an item from its modpath

        returns the tree item id or None if not found.
        """
        return self.modpath_to_item.get(tuple(modpath))

    def request_update(self, hint=None, modpath=None):
        if hint == FileCollectionDisplay.BKGND_RESUME:
            self.on_start_received()
            return
        if hint == FileCollectionDisplay.BKGND_STOP:
            self.on_stop_received()
            self.status_text.Label = "Idle..."
            return
        if modpath is not None and len(modpath) > 0:
            #
            # Descend down the leftmost side of all of the tuples
            # to get something we can display
            #
            path = []
            mp = modpath[0]
            any_others = len(modpath) > 1
            if hint != FileCollectionDisplay.REMOVE:
                # It's likely that the leaf was removed and it doesn't
                # make sense to descend
                file_tree = self.v.file_tree
            is_filtered = False
            while True:
                if isinstance(mp, str) or isinstance(mp, tuple) and len(mp) == 3:
                    path.append(mp)
                    if hint != FileCollectionDisplay.REMOVE:
                        is_filtered = not file_tree[mp]
                    break
                part, mp_list = mp
                path.append(part)
                if hint != FileCollectionDisplay.REMOVE:
                    file_tree = file_tree[part]
                if len(mp_list) == 0:
                    is_filtered = not file_tree[None]
                    break
                any_others = any_others or len(mp_list) > 1
                mp = mp_list[0]
            if hint != FileCollectionDisplay.REMOVE:
                self.status_text.Label = (
                    "Processing " + path[-1] if isinstance(path[-1], str) else path[-2]
                )
            self.status_text.Update()
            if not any_others:
                #
                # It's just a modification to a single node. Try and handle
                # here.
                #
                if hint == FileCollectionDisplay.METADATA:
                    if (not self.v.show_filtered) and is_filtered:
                        return
                    item_id = self.get_item_from_modpath(path)
                    if item_id is not None:
                        text, node_type, tooltip = self.v.get_node_info(path)
                        image_id = self.get_image_id_from_nodetype(node_type)
                        self.tree_ctrl.SetItemText(item_id, text)
                        self.tree_ctrl.SetItemImage(item_id, image_id)
                        return
                elif hint == FileCollectionDisplay.ADD:
                    if self.get_item_from_modpath(path) is None:
                        text, node_type, tooltip = self.v.get_node_info(path)
                        item_id = self.add_item(path, text)
                        image_id = self.get_image_id_from_nodetype(node_type)
                        self.tree_ctrl.SetItemImage(item_id, image_id)
                        self.manage_expansion()
                        return
                elif hint == FileCollectionDisplay.REMOVE:
                    if is_filtered:
                        return
                    self.remove_item(path)
                    if len(path) > 1:
                        super_modpath = tuple(path[:-1])
                        if super_modpath in self.modpath_to_item:
                            item = self.modpath_to_item[super_modpath]
                            n_children = self.tree_ctrl.GetChildrenCount(item, False)
                            if n_children == 0:
                                self.remove_item(super_modpath)

                    return
        self.update()

    def update(self):
        operation_id = uuid.uuid4()
        total = self.v.node_count()
        if total == 0:
            return
        self.update_subtree(
            self.v.file_tree, self.root_item, False, [], operation_id, 0, total
        )
        self.manage_expansion()
        report_progress(operation_id, 1, None)

    def manage_expansion(self):
        """Handle UI expansion issues

        Make sure that the tree is auto-expanded if appropriate and that
        the root nodes are expanded.
        """
        if not self.user_collapsed_a_node:
            #
            # Expand all until we reach a node that has more than
            # one child = ambiguous choice of which to expand
            #
            item = self.root_item
            while self.tree_ctrl.GetChildrenCount(item, False) == 1:
                # Can't expand the invisible root for Mac
                if sys.platform != "darwin" or item != self.root_item:
                    self.tree_ctrl.Expand(item)
                item, cookie = self.tree_ctrl.GetFirstChild(item)
            if self.tree_ctrl.GetChildrenCount(item, False) > 0:
                self.tree_ctrl.Expand(item)
        #
        # The bottom-most nodes don't have expand buttons (why?). If you
        # have two bottom-most nodes, neither will be expanded and there
        # is no way to expand them using the UI. So, we need to make sure
        # all bottom-most nodes are expanded, no matter what.
        #
        for i in range(self.tree_ctrl.GetChildrenCount(self.root_item, False)):
            if i == 0:
                bottom_item, thing = self.tree_ctrl.GetFirstChild(self.root_item)
            else:
                bottom_item, thing = self.tree_ctrl.GetNextChild(self.root_item, thing)
            if not self.tree_ctrl.IsExpanded(bottom_item):
                self.tree_ctrl.Expand(bottom_item)

    def update_subtree(
        self, file_tree, parent_item, is_filtered, modpath, operation_id, count, total
    ):
        existing_items = {}
        show_filtered = self.v.show_filtered
        needs_sort = False
        child_count = self.tree_ctrl.GetChildrenCount(parent_item, False)
        if child_count > 0:
            child_item_id, cookie = self.tree_ctrl.GetFirstChild(parent_item)
            for i in range(child_count):
                existing_items[self.tree_ctrl.GetItemPyData(child_item_id)] = [
                    child_item_id,
                    False,
                ]
                if i < child_count - 1:
                    child_item_id = self.tree_ctrl.GetNextSibling(child_item_id)

        for x in sorted(file_tree.keys()):
            sub_modpath = modpath + [x]
            if x is None:
                continue
            text, node_type, tooltip = self.v.get_node_info(sub_modpath)
            report_progress(
                operation_id, float(count) / float(total), "Processing %s" % text
            )
            count += 1
            image_id = self.get_image_id_from_nodetype(node_type)
            if isinstance(file_tree[x], bool) or isinstance(x, tuple):
                node_is_filtered = (not file_tree[x]) or is_filtered
                if node_is_filtered and not show_filtered:
                    continue
                if x in existing_items:
                    existing_items[x][1] = True
                    item_id = existing_items[x][0]
                    self.tree_ctrl.SetItemText(item_id, text)
                else:
                    item_id = self.add_item(sub_modpath, text, sort=False)
                    existing_items[x] = (item_id, True)
                    needs_sort = True

                self.tree_ctrl.SetItemImage(item_id, image_id)
            elif isinstance(file_tree[x], dict):
                subtree = file_tree[x]
                node_is_filtered = (not subtree[None]) or is_filtered
                (
                    unfiltered_subfolders,
                    filtered_subfolders,
                    unfiltered_files,
                    filtered_files,
                ) = self.get_file_and_folder_counts(subtree)
                n_subfolders = unfiltered_subfolders + filtered_subfolders
                n_files = unfiltered_files + filtered_files
                if node_is_filtered and not show_filtered:
                    continue
                if node_type in (
                    FileCollectionDisplay.NODE_COMPOSITE_IMAGE,
                    FileCollectionDisplay.NODE_MOVIE,
                ):
                    expanded_image_id = image_id
                else:
                    image_id = self.FOLDER_IMAGE_INDEX
                    expanded_image_id = self.FOLDER_OPEN_IMAGE_INDEX
                    text = "" + x
                    if n_subfolders > 0 or n_files > 0:
                        text += " ("
                        if n_subfolders > 0:
                            if node_is_filtered:
                                text += "\t%d folders" % n_subfolders
                            else:
                                text += "\t%d of %d folders" % (
                                    unfiltered_subfolders,
                                    n_subfolders,
                                )
                            if n_files > 0:
                                text += ", "
                        if n_files > 0:
                            if node_is_filtered:
                                text += "\t%d files" % n_files
                            else:
                                text += "\t%d of %d files" % (unfiltered_files, n_files)
                        text += ")"
                if x in existing_items:
                    existing_items[x][1] = True
                    item_id = existing_items[x][0]
                    self.tree_ctrl.SetItemText(item_id, text)
                else:
                    item_id = self.add_item(sub_modpath, text, sort=False)
                    existing_items[x] = (item_id, True)
                    needs_sort = True
                self.tree_ctrl.SetItemImage(item_id, image_id)
                self.tree_ctrl.SetItemImage(
                    item_id, expanded_image_id, wx.TreeItemIcon_Expanded
                )
                has_children = n_subfolders + n_files > 0
                self.tree_ctrl.SetItemHasChildren(item_id, has_children)
                count = self.update_subtree(
                    subtree,
                    item_id,
                    node_is_filtered,
                    sub_modpath,
                    operation_id,
                    count,
                    total,
                )

            color = self.FILTERED_COLOR if node_is_filtered else self.ACTIVE_COLOR
            self.tree_ctrl.SetItemTextColour(item_id, color)
        for last_part, (item_id, keep) in list(existing_items.items()):
            if not keep:
                self.remove_item(modpath + [last_part])
        if needs_sort:
            self.tree_ctrl.SortChildren(parent_item)
        return count

    def get_image_id_from_nodetype(self, node_type):
        if node_type == FileCollectionDisplay.NODE_COLOR_IMAGE:
            image_id = self.COLOR_IMAGE_INDEX
        elif node_type == FileCollectionDisplay.NODE_COMPOSITE_IMAGE:
            image_id = self.IMAGE_PLANES_IMAGE_INDEX
        elif node_type in (
            FileCollectionDisplay.NODE_MONOCHROME_IMAGE,
            FileCollectionDisplay.NODE_IMAGE_PLANE,
        ):
            image_id = self.IMAGE_PLANE_IMAGE_INDEX
        elif node_type == FileCollectionDisplay.NODE_MOVIE:
            image_id = self.MOVIE_IMAGE_INDEX
        else:
            image_id = self.FILE_IMAGE_INDEX
        return image_id

    @classmethod
    def get_file_and_folder_counts(cls, tree):
        """Count the number of files and folders in the tree

        returns the number of immediate unfiltered and filtered subfolders
        and number of unfiltered and filtered files in the hierarchy
        """
        unfiltered_subfolders = filtered_subfolders = 0
        unfiltered_files = filtered_files = 0
        for key in tree:
            if key is None:
                continue
            if isinstance(tree[key], bool):
                if tree[key]:
                    unfiltered_files += 1
                else:
                    filtered_files += 1
            else:
                is_filtered = not tree[key][None]
                if is_filtered:
                    unfiltered_subfolders += 1
                else:
                    filtered_subfolders += 1
                ufolders, ffolders, ufiles, ffiles = cls.get_file_and_folder_counts(
                    tree[key]
                )
                filtered_files += ffiles
                if is_filtered:
                    filtered_files += ufiles
                else:
                    unfiltered_files += ufiles
        return (
            unfiltered_subfolders,
            filtered_subfolders,
            unfiltered_files,
            filtered_files,
        )

    def on_hide_show_checked(self, event):
        self.v.show_filtered = not self.hide_show_ctrl.GetValue()
        self.request_update()
