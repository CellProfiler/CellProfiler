"""PathList - the PathListCtrl displays folders and paths in a scalable way
"""

import logging
import os
import sys
import uuid
from urllib.request import url2pathname
from urllib.parse import urlparse

import wx
import wx.lib.scrolledpanel

from cellprofiler_core.pipeline import ImageFile
from cellprofiler_core.preferences import report_progress

LOGGER = logging.getLogger(__name__)

OMERO_SCHEME = "omero:"

EVT_PLC_SELECTION_CHANGED = wx.PyEventBinder(wx.NewEventType())

IMAGE_PLANE = "plane"

DROP_FILES_AND_FOLDERS_HERE = "Drop files and folders here"

ENABLED_COLOR = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT)
DISABLED_COLOR = wx.SystemSettings.GetColour(wx.SYS_COLOUR_GRAYTEXT)
FOLDER_COLOR = wx.SystemSettings.GetColour(wx.SYS_COLOUR_MENUHILIGHT)
PLANE_COLOR = ENABLED_COLOR.ChangeLightness(130)


class PathListCtrl(wx.TreeCtrl):

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.font = wx.SystemSettings.GetFont(wx.SYS_DEFAULT_GUI_FONT)
        self.SetFont(self.font)
        self.SetDoubleBuffered(True)

        # Maps folder path names to tree ID objects.
        # You must explicitly delete keys from this map when deleting from the tree
        self.folder_id_map = {}
        # Maps file URLs to tree ID objects.
        # You must also explicitly delete keys from this map when deleting from the tree.
        # Also check whether the parent folder is empty and delete that too.
        self.file_id_map = {}
        # A set of URLs which don't pass the user's file filters.
        self.disabled_urls = set()
        # TreeCtrl can't hide items, so instead we store the file objects here when they aren't shown.
        self.hidden_files = []
        # The id of the tree root. This is never shown to the user but is what folders get attached to.
        self.root_id = self.AddRoot("File List")
        # Whether we're showing disabled items on the tree
        self.show_disabled = True
        # Whether metadata has been extracted from the file list.
        self._metadata_extracted = False

        self.focus_item = None
        self.fn_delete = None
        self.fn_context_menu = None
        self.fn_do_menu_command = None
        self.fn_folder_context_menu = None
        self.fn_do_folder_menu_command = None
        self.fn_empty_context_menu = None
        self.fn_do_empty_context_menu_command = None

        # Function to add files from pasted text (from imagesetctrl).
        # Will be set by pipelinecontroller during setup.
        self.fn_add_files = None

        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_LEFT_DOWN, self.on_mouse_down)
        self.Bind(wx.EVT_MOTION, self.on_mouse_moved)
        self.Bind(wx.EVT_KEY_DOWN, self.on_key_down)
        self.Bind(wx.EVT_CONTEXT_MENU, self.on_context_menu)
        self.Bind(wx.EVT_TREE_ITEM_ACTIVATED, self.on_double_click)

        if sys.platform == "win32":
            # Choose a slightly less ugly font.
            self._plane_details_font = wx.Font(
                pointSize=11, family=wx.FONTFAMILY_TELETYPE,
                style=wx.FONTSTYLE_NORMAL, weight=wx.FONTWEIGHT_NORMAL,
                faceName="Consolas"
            )
            # Windows gets really confused by pointSize
            self._plane_details_font.SetPixelSize((0, 12))
        else:
            self._plane_details_font = wx.Font(
                11, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL
            )

        self._drop_files_font = wx.Font(
            36, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD
        )

        #
        # Compute the size of the message to display when empty
        #
        tmp = self.GetFont()
        try:
            self.SetFont(self._drop_files_font)
            self.drop_files_and_folders_text_extent = self.GetFullTextExtent(
                DROP_FILES_AND_FOLDERS_HERE
            )[:2]
        except:
            LOGGER.warning(
                'Failed to get text extend for "%s" message'
                % DROP_FILES_AND_FOLDERS_HERE,
                exc_info=True,
            )
            self.drop_files_and_folders_text_extent = (200, 30)
        finally:
            self.SetFont(tmp)
        try:
            self.SetFont(self._plane_details_font)
            _, self.details_height = self.GetFullTextExtent(
                DROP_FILES_AND_FOLDERS_HERE
            )[:2]
        except:
            LOGGER.warning(
                'Failed to get text extend for plane details font',
                exc_info=True,
            )
            self.details_height = 11
        finally:
            self.SetFont(tmp)

    def AcceptsFocus(self):
        """Tell the scrollpanel that we can accept the focus"""
        return True

    def CanPaste(self):
        return True

    def Paste(self):
        # Check keyboard data is sensible.
        if not wx.TheClipboard.IsSupported(wx.DataFormat(wx.DF_UNICODETEXT)):
            return False
        # Pull text off the clipboard into a buffer, should be one file per line.
        # wx does have clipboard file object handlers, but they don't seem to work properly on MacOS.
        text_buffer = wx.TextDataObject()
        if wx.TheClipboard.Open():
            success = wx.TheClipboard.GetData(text_buffer)
            wx.TheClipboard.Close()
        else:
            # Clipboard isn't readable for some reason.
            return False
        if not success:
            # Couldn't read whatever was on the clipboard
            return False
        contents = [s for s in text_buffer.GetText().splitlines() if self.validate_pasted_string(s)]
        if contents and self.fn_add_files is not None:
            # Send the proposed files to the file list
            # Original function expects x and y drop coords, but we don't need those.
            self.fn_add_files(None, None, contents)
        return True

    @staticmethod
    def validate_pasted_string(s):
        # Require a URL or an absolute valid path to accept a text object
        parsed = urlparse(s)
        if parsed.scheme in ('file', ''):  # Possibly a local file
            return os.path.exists(parsed.path)
        elif parsed.scheme:
            # Probably http or s3
            return True
        return False

    def set_context_menu_fn(
        self,
        fn_context_menu,
        fn_folder_menu,
        fn_empty_menu,
        fn_do_menu_command,
        fn_do_folder_menu_command,
        fn_do_empty_command,
    ):
        """Set the function to call to get context menu items

        fn_context_menu - a function that returns a list of menu items. The calling
                  signature is fn_menu(paths) and the return is a sequence
                  of two tuples of the form, (key, display_string).

        fn_folder_menu - a function that returns a list of menu items for
                  a folder. The signature is fn_folder_menu(path).

        fn_empty_menu - a function that returns a list of menu items if
                        nothing is selected

        fn_do_menu_command - a function that performs the action indicated
                  by the command. It has the signature,
                  fn_do_menu_command(paths, key) where "key" is the key from
                  fn_context_menu.

        fn_do_folder_menu_command - a function that performs the action
                  indicated by the folder command. The signature is
                  fn_do_folder_menu_command(path, key)

        fn_do_empty_menu_command - a function that performs the command from
                  the empty menu
        """
        self.fn_context_menu = fn_context_menu
        self.fn_do_menu_command = fn_do_menu_command
        self.fn_folder_context_menu = fn_folder_menu
        self.fn_do_folder_menu_command = fn_do_folder_menu_command
        self.fn_empty_context_menu = fn_empty_menu
        self.fn_do_empty_context_menu_command = fn_do_empty_command

    def set_delete_fn(self, fn_delete):
        """Set the function to call to delete items

        fn_delete - a function whose signature is fn_delete(paths)
        """
        self.fn_delete = fn_delete

    def set_show_disabled(self, show):
        """Show or hide disabled files

        show - true to show them, false to hide them
        """
        if show == self.show_disabled:
            return
        self.show_disabled = show
        if show:
            # Add hidden items back into tree.
            self.add_files(self.hidden_files)
            self.hidden_files = []
        else:
            self.do_hide_disabled(self.disabled_urls)
        self.notify_selection_changed()

    def get_show_disabled(self):
        """Return the state of the show / hide disabled flag

        returns True if we should show disabled files
        """
        return self.show_disabled

    def do_hide_disabled(self, paths):
        changed_folders = set()
        for url in paths:
            tree_id = self.file_id_map[url]
            file_object = self.GetItemData(tree_id)
            parent = self.GetItemParent(tree_id)
            changed_folders.add(parent)
            self.hidden_files.append(file_object)
            self.Delete(tree_id)
            del self.file_id_map[url]
        for changed_folder in changed_folders:
            if self.GetChildrenCount(changed_folder) == 0:
                del self.folder_id_map[self.GetItemText(changed_folder)]
                self.Delete(changed_folder)

    @staticmethod
    def splitpath(path):
        slash = path.rfind("/")
        if slash == -1:
            if path.lower().startswith(OMERO_SCHEME):
                return [path[: len(OMERO_SCHEME)], path[len(OMERO_SCHEME) :]]
            return "", path
        else:
            return path[:slash], path[(slash + 1) :]

    def add_files(self, file_objects):
        uid = uuid.uuid4()
        npaths = len(file_objects)
        added_folders = False
        changed_folders = set()
        for i, file_object in enumerate(file_objects):
            path = file_object.path
            url = file_object.url
            if i % 100 == 0:
                report_progress(uid, float(i) / npaths, "Loading %s into UI" % path)
            folder, filename = os.path.split(path)
            if folder in self.folder_id_map:
                folder_id = self.folder_id_map[folder]
            else:
                added_folders = True
                folder_id = self.AppendItem(self.root_id, folder)
                self.SetItemTextColour(folder_id, FOLDER_COLOR)
                self.folder_id_map[folder] = folder_id
            changed_folders.add(folder_id)
            if url in self.file_id_map:
                continue
            else:
                file_id = self.AppendItem(folder_id, filename, data=file_object)
                self.file_id_map[url] = file_id
                if url in self.disabled_urls:
                    self.SetItemTextColour(file_id, DISABLED_COLOR)
                if file_object.extracted:
                    for detail in file_object.plane_details_text:
                        plane_id = self.AppendItem(file_id, detail, data=file_object)
                        self.SetItemFont(plane_id, self._plane_details_font)
                        self.SetItemTextColour(plane_id, PLANE_COLOR)
        if added_folders:
            self.SortChildren(self.root_id)
        for folder_id in changed_folders:
            self.SortChildren(folder_id)
        self.ExpandAll()
        if npaths:
            report_progress(uid, 1, "Done")
        self._metadata_extracted = False
        # Needed to remove 'Drop files' text
        self.Refresh()

    def remove_files(self, urls):
        for url in urls:
            if isinstance(url, ImageFile):
                url = url.url
            if url in self.file_id_map:
                file_id = self.file_id_map[url]
                folder_id = self.GetItemParent(file_id)
                self.disabled_urls.discard(url)
                self.Delete(file_id)
                del self.file_id_map[url]
                if self.GetChildrenCount(folder_id) == 0:
                    del self.folder_id_map[self.GetItemText(folder_id)]
                    self.Delete(folder_id)

    def clear_files(self):
        self.file_id_map = {}
        for folder_id in self.folder_id_map.values():
            self.Delete(folder_id)
        self.folder_id_map = {}

    def update_metadata(self, urls):
        if len(urls) == 0:
            LOGGER.debug("No urls to update metadata with")

        for url in urls:
            file_id = self.file_id_map[url]
            file_object = self.GetItemData(file_id)
            if file_object.extracted and self.GetChildrenCount(file_id) == 0:
                for detail in file_object.plane_details_text:
                    plane_id = self.AppendItem(file_id, detail, data=file_object)
                    self.SetItemFont(plane_id, self._plane_details_font)
                    self.SetItemTextColour(plane_id, PLANE_COLOR)
                self.Expand(file_id)
        self._metadata_extracted = True

    @property
    def metadata_already_extracted(self):
        return self._metadata_extracted

    def set_metadata_extracted(self, val):
        self._metadata_extracted = val

    def enable_paths(self, paths, enabled):
        """Mark a sequence of URLs as enabled or disabled

        Set the enabled/disabled flag for the given urls.

        paths - a sequence of URLs

        enabled - True to enable them, False to disable them.
        """
        if enabled:
            for path in paths:
                if path in self.disabled_urls:
                    self.disabled_urls.remove(path)
                    self.SetItemTextColour(self.file_id_map[path], ENABLED_COLOR)
                    file_objs = [f for f in self.hidden_files if f.url == path]
                    self.add_files(file_objs)
                    # Formerly disabled URLs won't have been extracted
                    self._metadata_extracted = False
        else:
            for path in paths:
                if path not in self.disabled_urls:
                    self.disabled_urls.add(path)
                    self.SetItemTextColour(self.file_id_map[path], DISABLED_COLOR)
                    self.Collapse(self.file_id_map[path])
                    if not self.show_disabled:
                        self.do_hide_disabled([path])

    def enable_all_paths(self):
        """Mark all paths as enabled

        This puts the path list control in the appropriate state when
        filtering is disabled.
        """
        for path in self.disabled_urls:
            self.SetItemTextColour(self.file_id_map[path], ENABLED_COLOR)
        self.disabled_urls.clear()
        self.add_files(self.hidden_files)
        self.hidden_files = []
        # Formerly disabled URLs won't have been extracted
        self._metadata_extracted = False

    def expand_all(self, event=None):
        """Expand all folders"""
        self.ExpandAll()

    def collapse_all(self, event=None):
        """Collapse all folders"""
        self.CollapseAll()

    @staticmethod
    def get_folder_display_name(folder):
        """Return a path name for a URL
        For files, the user expects to see a path, not a URL
        """
        if folder.startswith("file:"):
            return url2pathname(folder[5:])
        return folder

    FLAG_ENABLED_ONLY = 1
    FLAG_SELECTED_ONLY = 2
    FLAG_FOLDERS = 4
    FLAG_RECURSE = 8
    FLAG_FOCUS_ITEM_ONLY = 16

    def get_paths(self, flags=0):
        """Return paths

        flags - PathListCtrl.FLAG_ENABLED_ONLY to only return paths marked
                as enabled, PathListCtrl.FLAG_SELECTED_ONLY to return only
                selected paths, PathListCtrl.FLAG_FOCUS_ITEM_ONLY to return
                either an empty list or the focus item's path.
        """
        paths = []
        if flags & PathListCtrl.FLAG_FOCUS_ITEM_ONLY:

            def fn_iter():
                if self.GetFocusedItem().ID is not None:
                    idx = self.GetFocusedItem()
                    yield self.GetItemData(self.GetFocusedItem()), idx

        elif flags & PathListCtrl.FLAG_SELECTED_ONLY:

            def fn_iter():
                for idx in self.GetSelections():
                    # If the folder is selected with no children, delete the whole folder.
                    # If the folder is selected but so are some children, delete the selected children only.
                    if self.is_folder(idx):
                        to_del = []
                        should_del = True
                        child_idx, cookie = self.GetFirstChild(idx)
                        while child_idx.IsOk():
                            if self.IsSelected(child_idx):
                                should_del = False
                                break
                            else:
                                to_del.append(child_idx)
                            child_idx, cookie = self.GetNextChild(idx, cookie)
                        if should_del:
                            for del_idx in to_del:
                                yield self.GetItemData(del_idx), del_idx
                        continue
                    yield self.GetItemData(idx), idx

        else:

            def fn_iter():
                for idx in self.file_id_map.values():
                    yield self.GetItemData(idx), idx

        for item, idx in fn_iter():
            if idx is None:
                continue
            if flags & PathListCtrl.FLAG_ENABLED_ONLY:
                if item.url in self.disabled_urls:
                    continue
            paths.append(item.url)
        return paths

    def get_selected_series(self):
        if self.GetFocusedItem().ID is not None:
            idx = self.GetFocusedItem()
            parent = self.GetItemParent(idx)
            if not self.is_folder(parent):
                # This is a series within a file. Find the index.
                tgt = 0
                child_idx, cookie = self.GetFirstChild(parent)
                while child_idx.IsOk():
                    if child_idx == idx:
                        # This is the series we are looking at
                        return tgt
                    child_idx, cookie = self.GetNextChild(parent, cookie)
                    tgt += 1
        return None

    def has_selections(self):
        """Return True if there are any selected items"""
        return len(self.GetSelections()) > 0

    def clear_selections(self):
        self.UnselectAll()

    def SelectAll(self):
        """Select all files in the control"""
        # Annoyingly this selects the parent folders non-recursively, but keyboard shortcuts will still work.
        self.UnselectAll()
        for node in self.folder_id_map.values():
            self.SelectItem(node)
        self.notify_selection_changed()

    def select_path(self, url):
        """Select the given URL if it is present in the list

        url - url to select if it is present

        returns True if the URL was selected
        """
        if url in self.file_id_map:
            self.SelectItem(self.file_id_map[url])
            self.notify_selection_changed()
            return True
        else:
            return False

    def notify_selection_changed(self):
        """Publish a WX event that tells the world that the selection changed"""
        event = wx.NotifyEvent(EVT_PLC_SELECTION_CHANGED.evtType[0])
        event.SetEventObject(self)
        self.GetEventHandler().ProcessEvent(event)

    def has_focus_item(self):
        """Return True if an item is focused"""
        return self.focus_item is not None

    def get_folder(self, path, flags=0):
        """Return the files or folders in the current folder.

        path - path to the folder
        flags - FLAG_ENABLED_ONLY to only return enabled files or folders
                with enabled files. FLAG_FOLDERS to return folders instead
                of files. FLAG_RECURSE to do all subfolders.
        """
        if path not in self.folder_id_map:
            raise ValueError("Invalid path", path)
        folder_id = self.folder_id_map[path]
        urls = []
        enabled_only = (flags & self.FLAG_ENABLED_ONLY) != 0
        tree_item, cookie = self.GetFirstChild(folder_id)
        while tree_item.IsOk():
            url = self.GetItemData(tree_item).url
            if not enabled_only or url not in self.disabled_urls:
                urls.append(url)
            tree_item, cookie = self.GetNextChild(folder_id, cookie)
        return urls

    def on_paint(self, event):
        if self.GetChildrenCount(self.root_id) == 0:
            """Handle the paint event"""
            assert isinstance(event, wx.PaintEvent)
            paint_dc = wx.BufferedPaintDC(self)
            paint_dc.SetBackground(
                wx.Brush(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW))
            )
            paint_dc.Clear()
            text = DROP_FILES_AND_FOLDERS_HERE
            font = self._drop_files_font
            paint_dc.SetTextForeground(
                wx.SystemSettings.GetColour(wx.SYS_COLOUR_GRAYTEXT)
            )
            paint_dc.SetFont(font)
            paint_dc.DrawLabel(
                text, wx.Bitmap(), wx.Rect(self.GetSize()), alignment=wx.ALIGN_CENTER,
            )
            paint_dc.Destroy()
        else:
            event.Skip()

    def find_invisible_above(self):
        # Returns the index of the invisible item above the last index.
        first, _ = self.GetFirstChild(self.root_id)
        if not first.IsOk() or self.IsVisible(first):
            return first
        idx = self.GetFirstVisibleItem()
        if not idx.IsOk():
            return idx
        parent = self.GetItemParent(idx)
        target_idx = parent
        assert not self.IsVisible(parent)
        child_item, cookie = self.GetFirstChild(parent)
        while child_item.IsOk():
            if self.IsVisible(child_item):
                break
            target_idx = child_item
            child_item, cookie = self.GetNextChild(parent, cookie)
        return target_idx

    def find_invisible_below(self):
        next_vis = self.GetFirstVisibleItem()
        if not next_vis.IsOk():
            return next_vis
        last_vis = next_vis
        while next_vis.IsOk():
            last_vis = next_vis
            next_vis = self.GetNextVisible(next_vis)
        if self.ItemHasChildren(last_vis) and self.IsExpanded(last_vis):
            return self.GetFirstChild(last_vis)
        elif self.GetNextSibling(last_vis).IsOk():
            return self.GetNextSibling(last_vis)
        elif self.GetNextSibling(self.GetItemParent(last_vis)).IsOk():
            return self.GetNextSibling(self.GetItemParent(last_vis))
        else:
            return self.GetNextSibling(self.GetItemParent(self.GetItemParent(last_vis)))

    def is_folder(self, idx):
        return self.GetItemText(idx) in self.folder_id_map

    def is_file(self, idx):
        if idx == self.root_id:
            return False
        return self.is_folder(self.GetItemParent(idx))

    def on_mouse_down(self, event):
        """Handle left mouse button down"""
        assert isinstance(event, wx.MouseEvent)
        self.SetFocus()
        idx = self.GetFocusedItem()
        if self.GetChildrenCount(self.root_id) == 0:
            return
        if event.ShiftDown():
            if self.GetItemText(idx) in self.folder_id_map:
                self.SelectChildren(idx)
            else:
                self.SelectItem(idx)

        elif event.ControlDown():
            self.SelectItem(idx)

        else:
            self.UnselectAll()
            if idx.IsOk():
                self.SelectItem(idx)

        event.Skip()

    def on_double_click(self, event):
        """Handle double click event"""
        idx = self.GetFocusedItem()
        if self.fn_do_menu_command is None:
            return
        if not idx.IsOk() or self.is_folder(idx):
            self.fn_do_menu_command([], None)
            return
        else:
            self.fn_do_menu_command([self.GetItemData(idx).path], None)

    def on_mouse_moved(self, event):
        """Handle mouse movement during capture"""
        if event.Dragging():
            idx, flag = self.HitTest(event.GetPosition())
            if idx.IsOk() and self.is_file(idx) and not self.IsSelected(idx):
                self.SelectItem(idx)
            self.scroll_into_view(event)

    def scroll_into_view(self, event):
        """Scroll the focus item into view"""
        _, height = self.GetSize()
        x, y = event.GetPosition()
        if y > height - 20:
            show_idx = self.find_invisible_below()
            if isinstance(show_idx, tuple):
                show_idx = show_idx[0]
            if not show_idx.IsOk():
                return
            self.ScrollTo(show_idx)
        elif y < 20:
            show_idx = self.find_invisible_above()
            if not show_idx.IsOk():
                return
            self.ScrollTo(show_idx)

    def on_key_down(self, event):
        """Handle a key press"""
        if event.GetKeyCode() == wx.WXK_DELETE and self.fn_delete is not None:
            paths = self.get_paths(self.FLAG_SELECTED_ONLY)
            self.remove_files(paths)
            self.fn_delete(paths)
            return
        event.Skip(True)

    context_menu_ids = []

    def on_context_menu(self, event):
        """Handle a context menu request"""
        index = self.GetFocusedItem()
        if index.ID is None:
            fn_context_menu = self.fn_empty_context_menu
            fn_do_menu_command = self.fn_do_empty_context_menu_command
            arg = None
        else:
            item = self.GetItemData(index)
            if item is None:
                # Is a folder
                fn_context_menu = self.fn_folder_context_menu
                fn_do_menu_command = self.fn_do_folder_menu_command
                arg = self.GetItemText(index)
            else:
                fn_context_menu = self.fn_context_menu
                fn_do_menu_command = self.fn_do_menu_command
                arg = list(set(self.GetItemData(ident).url for ident in self.GetSelections()))

        if fn_context_menu is None or fn_do_menu_command is None:
            return
        pos = event.GetPosition()
        pos = self.ScreenToClient(pos)
        item_list = fn_context_menu(arg)
        if len(self.context_menu_ids) < len(item_list):
            self.context_menu_ids += [
                wx.NewId() for _ in range(len(self.context_menu_ids), len(item_list))
            ]
        menu = wx.Menu()
        for idx, (key, display_name) in enumerate(item_list):
            menu.Append(self.context_menu_ids[idx], display_name)

        def on_menu(event):
            idx = self.context_menu_ids.index(event.Id)
            fn_do_menu_command(arg, item_list[idx][0])

        self.Bind(wx.EVT_MENU, on_menu)
        try:
            self.PopupMenu(menu, pos)
        finally:
            self.Unbind(wx.EVT_MENU, handler=on_menu)
            menu.Destroy()
