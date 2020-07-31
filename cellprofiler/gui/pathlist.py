"""PathList - the PathListCtrl displays folders and paths in a scalable way
"""

import bisect
import functools
import logging
import uuid
from urllib.request import url2pathname

import numpy
import wx
import wx.lib.scrolledpanel
from cellprofiler_core.preferences import report_progress

import cellprofiler.gui

OMERO_SCHEME = "omero:"

EVT_PLC_SELECTION_CHANGED = wx.PyEventBinder(wx.NewEventType())


class PathListCtrl(wx.ScrolledWindow):
    #
    # The width of the expander image (seems like all code samples have this
    # hardcoded)
    #
    TREEITEM_WIDTH = 16
    TREEITEM_HEIGHT = 16
    #
    # Gap between tree item and text
    #
    TREEITEM_GAP = 2

    class FolderItem(object):
        def __init__(self, ctrl, folder_name):
            self.folder_name = folder_name
            self.folder_display_name = PathListCtrl.get_folder_display_name(folder_name)
            self.display_width, _ = ctrl.GetTextExtent(self.folder_display_name)
            self.display_width += (
                PathListCtrl.TREEITEM_WIDTH + PathListCtrl.TREEITEM_GAP
            )
            self.widths = []
            self.filenames = []
            self.file_display_names = []
            self.enabled = []
            self.enabled_idxs = None
            self.opened = True

        def get_full_path(self, idx):
            """Get the full pathname for the indexed file"""
            if self.folder_name.lower() == OMERO_SCHEME:
                return self.folder_name + self.filenames[idx]
            return self.folder_name + "/" + self.filenames[idx]

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.SetFont(wx.SystemSettings.GetFont(wx.SYS_DEFAULT_GUI_FONT))
        self.SetDoubleBuffered(True)
        self.selections = set()
        self.notify_selection_changed()
        self.folder_items = []
        self.folder_names = []
        self.folder_counts = numpy.zeros(0, int)
        self.folder_idxs = numpy.zeros(0, int)
        _, height, _, _ = self.GetFullTextExtent("Wally")
        self.line_height = height
        self.leading = 0
        self.show_disabled = True
        #
        # NB: NEVER USE MAGIC!!!!!
        #
        # If I use self.dirty or even self.__dirty, something down in the bowels
        # of wx (I suspect __setattr__) intercepts my attempt to set it
        # to True. So please keep the Yiddish below or use whatever substitute
        # you want.
        #
        # And if you ever, ever, ever think about hiding a variable by
        # overriding something like __setattr__, please think of the
        # consequences of your actions. In other words, NEVER USE MAGIC.
        #
        self.schmutzy = False
        self.mouse_down_idx = None
        self.mouse_idx = None
        self.focus_item = None
        self.fn_delete = None
        self.fn_context_menu = None
        self.fn_do_menu_command = None
        self.fn_folder_context_menu = None
        self.fn_do_folder_menu_command = None
        self.fn_empty_context_menu = None
        self.fn_do_empty_context_menu_command = None
        self.EnableScrolling(True, False)
        self.SetScrollRate(1, self.line_height + self.leading)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_RIGHT_DOWN, self.on_right_mouse_down)
        self.Bind(wx.EVT_LEFT_DOWN, self.on_mouse_down)
        self.Bind(wx.EVT_LEFT_UP, self.on_mouse_up)
        self.Bind(wx.EVT_MOTION, self.on_mouse_moved)
        self.Bind(wx.EVT_MOUSE_CAPTURE_LOST, self.on_mouse_capture_lost)
        self.Bind(wx.EVT_SCROLLWIN, self.on_scroll_changed)
        self.Bind(wx.EVT_KEY_DOWN, self.on_key_down)
        self.Bind(wx.EVT_CONTEXT_MENU, self.on_context_menu)
        self.Bind(wx.EVT_SET_FOCUS, self.on_set_focus)
        self.Bind(wx.EVT_KILL_FOCUS, self.on_kill_focus)
        self.Bind(wx.EVT_LEFT_DCLICK, self.on_double_click)
        #
        # Compute the size of the message to display when empty
        #
        tmp = self.GetFont()
        try:
            self.SetFont(self.DROP_FILES_AND_FOLDERS_FONT)
            self.drop_files_and_folders_text_extent = self.GetFullTextExtent(
                self.DROP_FILES_AND_FOLDERS_HERE
            )[:2]
        except:
            logging.warn(
                'Failed to get text extend for "%s" message'
                % self.DROP_FILES_AND_FOLDERS_HERE,
                exc_info=True,
            )
            self.drop_files_and_folders_text_extent = (200, 30)
        finally:
            self.SetFont(tmp)

    def AcceptsFocus(self):
        """Tell the scrollpanel that we can accept the focus"""
        return True

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
        self.schmutzy = True
        self.selections = set()
        self.focus_item = None
        self.notify_selection_changed()
        self.Refresh(eraseBackground=False)

    def get_show_disabled(self):
        """Return the state of the show / hide disabled flag

        returns True if we should show disabled files
        """
        return self.show_disabled

    def get_path_count(self):
        """# of paths shown in UI"""
        if self.schmutzy:
            self.recalc()
        return numpy.sum(self.folder_counts)

    def get_folder_count(self):
        """# of folders shown in UI"""
        if self.schmutzy:
            self.recalc()
            self.schmutzy = False
        return len(self.folder_counts)

    def __len__(self):
        """# of lines shown in UI"""
        return self.get_path_count() + self.get_folder_count()

    def __getitem__(self, idx):
        """Return the folder and path at the index

        idx - index of item to retrieve
        """
        if self.schmutzy:
            self.recalc()
            self.schmutzy = False
        folder_idx = bisect.bisect_right(self.folder_idxs, idx) - 1
        if idx == self.folder_idxs[folder_idx]:
            return self.folder_items[folder_idx], None
        item = self.folder_items[folder_idx]
        idx = idx - self.folder_idxs[folder_idx] - 1
        if idx >= self.folder_counts[folder_idx]:
            return None, None

        if self.show_disabled:
            return item, idx
        else:
            idx = item.enabled_idxs[idx]
            return item, idx

    @staticmethod
    def splitpath(path):
        slash = path.rfind("/")
        if slash == -1:
            if path.lower().startswith(OMERO_SCHEME):
                return [path[: len(OMERO_SCHEME)], path[len(OMERO_SCHEME) :]]
            return "", path
        else:
            return path[:slash], path[(slash + 1) :]

    def add_paths(self, paths):
        """Add the given URLs to the control

        paths - a sequence of URLs
        """
        uid = uuid.uuid4()
        npaths = len(paths)
        for i, path in enumerate(paths):
            if i % 100 == 0:
                report_progress(uid, float(i) / npaths, "Loading %s into UI" % path)
            folder, filename = self.splitpath(path)
            display_name = url2pathname(filename)
            width, _, _, _ = self.GetFullTextExtent(display_name)
            idx = bisect.bisect_left(self.folder_names, folder)
            if idx >= len(self.folder_names) or self.folder_names[idx] != folder:
                folder_item = self.FolderItem(self, folder)
                self.folder_names.insert(idx, folder)
                self.folder_items.insert(idx, folder_item)
            else:
                folder_item = self.folder_items[idx]
            fp = folder_item.filenames
            pidx = bisect.bisect_left(fp, filename)
            if pidx >= len(fp) or fp[pidx] != filename:
                fp.insert(pidx, filename)
                folder_item.widths.insert(pidx, width)
                folder_item.file_display_names.insert(pidx, display_name)
                folder_item.enabled.insert(pidx, True)
        if len(paths) > 0:
            report_progress(uid, 1, "Done")
        self.schmutzy = True
        self.Refresh(eraseBackground=False)

    def enable_paths(self, paths, enabled):
        """Mark a sequence of URLs as enabled or disabled

        Set the enabled/disabled flag for the given urls.

        paths - a sequence of URLs

        enabled - True to enable them, False to disable them.
        """
        for path in paths:
            folder, filename = self.splitpath(path)
            idx = bisect.bisect_left(self.folder_names, folder)
            if idx >= len(self.folder_names) or self.folder_names[idx] != folder:
                continue
            folder_item = self.folder_items[idx]
            pidx = bisect.bisect_left(folder_item.filenames, filename)
            if (
                pidx >= len(folder_item.filenames)
                or folder_item.filenames[pidx] != filename
            ):
                continue
            folder_item.enabled[pidx] = enabled
        self.schmutzy = True
        self.Refresh(eraseBackground=False)

    def enable_all_paths(self):
        """Mark all paths as enabled

        This puts the path list control in the appropriate state when
        filtering is disabled.
        """
        for folder_item in self.folder_items:
            folder_item.enabled = [True] * len(folder_item.filenames)
        self.schmutzy = True
        self.Refresh(eraseBackground=False)

    def expand_all(self, event=None):
        """Expand all folders"""
        for folder_item in self.folder_items:
            folder_item.opened = True
        self.schmutzy = True
        self.Refresh(eraseBackground=False)

    def collapse_all(self, event=None):
        """Collapse all folders"""
        for folder_item in self.folder_items:
            folder_item.opened = False
        self.schmutzy = True
        self.Refresh(eraseBackground=False)

    @staticmethod
    def get_folder_display_name(folder):
        """Return a path name for a URL

        For files, the user expects to see a path, not a URL
        """
        if folder.startswith("file:"):
            return url2pathname(folder[5:])
        return folder

    def recalc(self):
        """Recalculate cached internals

        Call this before using any of the internals such as
        self.folder_idx
        """
        if not self.schmutzy:
            return
        if len(self.folder_items) == 0:
            max_width, total_height = self.drop_files_and_folders_text_extent
            self.folder_counts = numpy.zeros(0, int)
            self.folder_idxs = numpy.zeros(0, int)
        else:
            if self.show_disabled:
                self.folder_counts = numpy.array(
                    [len(x.filenames) if x.opened else 0 for x in self.folder_items]
                )
            else:
                for item in self.folder_items:
                    enabled_mask = numpy.array(item.enabled, bool)
                    item.enabled_idxs = numpy.arange(len(item.enabled))[enabled_mask]
                self.folder_counts = numpy.array(
                    [numpy.sum(x.enabled) if x.opened else 0 for x in self.folder_items]
                )
            self.folder_idxs = numpy.hstack(([0], numpy.cumsum(self.folder_counts + 1)))
            max_width = functools.reduce(
                max,
                [
                    max(functools.reduce(max, x.widths), x.display_width)
                    for x in self.folder_items
                ],
            )
            total_height = self.line_height * self.folder_idxs[-1]
            total_height += self.leading * (self.folder_idxs[-1] - 1)
        self.max_width = max_width
        self.total_height = total_height
        self.schmutzy = False
        self.SetVirtualSize((max_width, total_height))

    def remove_paths(self, paths):
        """Remove a sequence of URLs from the UI"""
        for path in paths:
            folder, filename = self.splitpath(path)
            idx = bisect.bisect_left(self.folder_names, folder)
            if idx < len(self.folder_names) and self.folder_names[idx] == folder:
                item = self.folder_items[idx]
                assert isinstance(item, self.FolderItem)
                fp = item.filenames
                pidx = bisect.bisect_left(fp, filename)
                if fp[pidx] == filename:
                    del fp[pidx]
                    del item.widths[pidx]
                    del item.file_display_names[pidx]
                    del item.enabled[pidx]
                    if len(fp) == 0:
                        del self.folder_names[idx]
                        del self.folder_items[idx]
        self.selections = set()  # indexes are all wrong now
        self.focus_item = None
        self.schmutzy = True
        self.notify_selection_changed()
        self.Refresh(eraseBackground=False)

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
        if self.schmutzy:
            self.recalc()
        if flags & PathListCtrl.FLAG_FOCUS_ITEM_ONLY:

            def fn_iter():
                if self.focus_item is not None:
                    yield self[self.focus_item]

        elif flags & PathListCtrl.FLAG_SELECTED_ONLY:

            def fn_iter():
                for idx in self.selections:
                    yield self[idx]

        else:

            def fn_iter():
                for item in self.folder_items:
                    for idx in range(len(item.filenames)):
                        yield item, idx

        for item, idx in fn_iter():
            if idx is None:
                continue
            if flags & PathListCtrl.FLAG_ENABLED_ONLY:
                if not item.enabled[idx]:
                    continue
            paths.append(item.get_full_path(idx))
        return paths

    def has_selections(self):
        """Return True if there are any selected items"""
        return len(self.selections) > 0

    def clear_selections(self):
        self.selections = set()
        self.schmutzy = True
        self.notify_selection_changed()
        self.Refresh(eraseBackground=False)

    def SelectAll(self):
        """Select all items in the control"""
        self.selections = set(range(len(self)))
        self.schmutzy = True
        self.notify_selection_changed()
        self.Refresh(eraseBackground=False)

    def select_path(self, url):
        """Select the given URL if it is present in the list

        url - url to select if it is present

        returns True if the URL was selected
        """
        folder, filename = self.splitpath(url)
        idx = bisect.bisect_left(self.folder_names, folder)
        if idx < len(self.folder_names) and self.folder_names[idx] == folder:
            folder_item = self.folder_items[idx]
        else:
            return False
        fp = folder_item.filenames
        pidx = bisect.bisect_left(fp, filename)
        if pidx >= len(fp) or fp[pidx] != filename:
            return False
        self.selections.add(self.folder_idxs[idx] + pidx + 1)
        self.notify_selection_changed()
        return True

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
        idx = bisect.bisect_left(self.folder_names, path)
        folders = []
        recurse = (flags & self.FLAG_RECURSE) != 0
        wants_folders = (flags & self.FLAG_FOLDERS) != 0
        enabled_only = (flags & self.FLAG_ENABLED_ONLY) != 0
        has_path = 0 <= idx < len(self.folder_names) and path == self.folder_names[idx]
        if has_path:
            if not wants_folders:
                folders.append(self.folder_items[idx])
            idx += 1
        if recurse or wants_folders:
            for idx in range(idx, len(self.folder_items)):
                if not self.folder_names[idx].startswith(path):
                    break
                rest = self.folder_names[idx][len(path)]
                if rest[0] != "/":
                    continue
                rest = rest[1:]
                if (not recurse) and "/" in rest:
                    continue
                folders.append(self.folder_items[idx])
        if wants_folders:
            return [x.folder_name for x in folders]
        else:
            result = []
            for item in folders:
                if enabled_only:
                    result += [
                        item.folder_name + "/" + item.filenames[e]
                        for e in item.enabled_idxs
                    ]
                else:
                    result += [item.folder_name + "/" + f for f in item.filenames]
            return result

    def on_scroll_changed(self, event):
        #
        # WX is buggy in the way it honors ScrolledWindow.EnableScrolling.
        # The arrow keys scroll the bitmap and the top line is scrolled down.
        #
        assert isinstance(event, wx.ScrollWinEvent)
        if event.GetOrientation() == wx.VERTICAL:
            width, _ = self.GetSize()
            r = wx.Rect(0, 0, width, (self.line_height + self.leading) * 2)
            self.Refresh(eraseBackground=False, rect=r)
        event.Skip(True)

    def on_set_focus(self, event):
        self.Refresh(eraseBackground=False)
        event.Skip(True)

    def on_kill_focus(self, event):
        self.Refresh(eraseBackground=False)
        event.Skip(True)

    DROP_FILES_AND_FOLDERS_HERE = "Drop files and folders here"
    __DROP_FILES_AND_FOLDERS_FONT = None

    @property
    def DROP_FILES_AND_FOLDERS_FONT(self):
        if self.__DROP_FILES_AND_FOLDERS_FONT is None:
            self.__DROP_FILES_AND_FOLDERS_FONT = wx.Font(
                36, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD
            )
        return self.__DROP_FILES_AND_FOLDERS_FONT

    def show_idx_as_selected(self, idx):
        """Return True if the indexed line should be shown selected"""
        if idx in self.selections:
            return True
        if self.mouse_down_idx is None:
            return False
        sel_start = min(self.mouse_down_idx, self.mouse_idx)
        sel_end = max(self.mouse_down_idx, self.mouse_idx) + 1
        return sel_start <= idx < sel_end

    def on_paint(self, event):
        """Handle the paint event"""
        assert isinstance(event, wx.PaintEvent)
        paint_dc = wx.BufferedPaintDC(self)
        if self.schmutzy:
            self.recalc()
        width, height = self.GetSize()
        rn = wx.RendererNative.Get()
        background_color = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW)
        background_brush = wx.Brush(background_color)
        paint_dc.SetBrush(background_brush)
        paint_dc.Clear()
        paint_dc.SetFont(self.GetFont())
        paint_dc.SetBackgroundMode(wx.PENSTYLE_TRANSPARENT)
        has_focus = self.FindFocus() == self
        if has_focus:
            dir_color = wx.SystemSettings.GetColour(wx.SYS_COLOUR_HOTLIGHT)
        else:
            dir_color = wx.SystemSettings.GetColour(wx.SYS_COLOUR_GRAYTEXT)

        enabled_color = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT)
        disabled_color = wx.SystemSettings.GetColour(wx.SYS_COLOUR_GRAYTEXT)
        if len(self) == 0:
            text = self.DROP_FILES_AND_FOLDERS_HERE
            font = self.DROP_FILES_AND_FOLDERS_FONT
            paint_dc.SetTextForeground(
                wx.SystemSettings.GetColour(wx.SYS_COLOUR_GRAYTEXT)
            )
            paint_dc.SetFont(font)
            text_width, text_height = paint_dc.GetTextExtent(text)
            paint_dc.DrawText(
                text, (width - text_width) / 2, (height - text_height) / 2
            )
            paint_dc.SetFont(self.GetFont())

        selected_text = wx.SystemSettings.GetColour(wx.SYS_COLOUR_HIGHLIGHTTEXT)
        try:
            x = self.GetScrollPos(wx.SB_HORIZONTAL)
            y = self.GetScrollPos(wx.SB_VERTICAL)
            line_height = self.line_height + self.leading
            yline = min(y, len(self))
            yline_max = min(
                yline + (height + line_height - 1) // line_height, len(self)
            )
            sel_width = 0
            #
            # Precompute the width of the selection rectangle
            #
            for idx in range(yline + 1, yline_max):
                item, pidx = self[idx]
                if item is None:
                    break
                if pidx is None:
                    continue
                if self.show_idx_as_selected(idx) or self.focus_item == idx:
                    item_width = paint_dc.GetTextExtent(item.file_display_names[pidx])[
                        0
                    ]
                    sel_width = max(sel_width, item_width)

            #
            # Paint the strings
            #
            for idx in range(yline, yline_max):
                yy = (idx - yline) * line_height
                item, pidx = self[idx]
                if item is None:
                    break
                if pidx is None or idx == yline:
                    # A directory
                    paint_dc.SetTextForeground(dir_color)
                    rTreeItem = wx.Rect(
                        -x, yy, self.TREEITEM_WIDTH, self.TREEITEM_HEIGHT
                    )
                    rn.DrawTreeItemButton(
                        self,
                        paint_dc,
                        rTreeItem,
                        wx.CONTROL_EXPANDED if item.opened else 0,
                    )
                    paint_dc.DrawText(
                        item.folder_display_name,
                        self.TREEITEM_WIDTH + self.TREEITEM_GAP - x,
                        yy,
                    )
                else:
                    # A file
                    selected = self.show_idx_as_selected(idx)
                    flags = wx.CONTROL_FOCUSED if has_focus else 0
                    if selected:
                        flags += wx.CONTROL_SELECTED
                    if idx == self.focus_item:
                        flags += wx.CONTROL_CURRENT
                    cellprofiler.gui.draw_item_selection_rect(
                        self,
                        paint_dc,
                        wx.Rect(
                            self.TREEITEM_WIDTH - x,
                            yy,
                            sel_width + 2 * self.TREEITEM_GAP,
                            line_height,
                        ),
                        flags,
                    )
                    if selected:
                        paint_dc.SetTextForeground(selected_text)
                    else:
                        paint_dc.SetTextForeground(
                            enabled_color if item.enabled[pidx] else disabled_color
                        )
                    paint_dc.DrawText(
                        item.file_display_names[pidx],
                        self.TREEITEM_WIDTH + self.TREEITEM_GAP - x,
                        yy,
                    )
        finally:
            paint_dc.SetBrush(wx.NullBrush)
            paint_dc.SetFont(wx.NullFont)
            background_brush.Destroy()
            paint_dc.Destroy()

    def refresh_item(self, idx):
        """Signal the window to repaint the given item

        idx - index of the item.
        """
        total_height = self.line_height + self.leading
        y = (idx - self.GetScrollPos(wx.SB_VERTICAL)) * total_height
        width, _ = self.GetSize()
        self.Refresh(eraseBackground=False, rect=wx.Rect(0, y, width, total_height))

    def get_mouse_idx(self, event):
        """Return the line index at the event's mouse coordinate"""
        if len(self.folder_items) == 0:
            return -1
        x, y = event.GetPosition()
        line_height = self.line_height + self.leading
        idx = int(y / line_height) + self.GetScrollPos(wx.SB_VERTICAL)
        idx = max(0, min(len(self) - 1, idx))
        if y < line_height:
            # It's the slightly bogus directory at the top
            self.recalc()
            folder_idx = bisect.bisect_right(self.folder_idxs, idx) - 1
            if folder_idx == -1:
                return -1
            idx = self.folder_idxs[folder_idx]
        return idx

    @staticmethod
    def get_treeitem_x():
        """Return the width of the treeitem graphic

        returns wx.SYS_SMALLICON_X if defined on the platform
        or 16 for the Mac.
        """
        treeitem_x = wx.SystemSettings.GetMetric(wx.SYS_SMALLICON_X)
        if treeitem_x < 0:
            # wx.SYS_SMALLICON_X not defined for this platform
            # (which means Mac)
            return 16
        return treeitem_x

    def on_mouse_down(self, event):
        """Handle left mouse button down"""
        assert isinstance(event, wx.MouseEvent)
        self.SetFocus()
        idx = self.get_mouse_idx(event)
        if len(self.folder_items) == 0:
            return
        item, path_idx = self[idx]
        if item is None:
            item = self.folder_items[-1]
            path_idx = len(item.filenames)

        treeitem_x = self.get_treeitem_x()

        if path_idx is None and event.GetX() < treeitem_x:
            needs_selchange_evt = len(self.selections) > 0
            self.selections = set()
            item.opened = not item.opened
            self.schmutzy = True
            if needs_selchange_evt:
                self.notify_selection_changed()
            self.Refresh(eraseBackground=False)
            return

        if event.ShiftDown() and len(self.selections) == 1:
            self.mouse_down_idx = self.selections.pop()
        else:
            self.mouse_down_idx = idx
        if not event.ControlDown():
            self.selections = set()
        self.mouse_idx = idx
        self.focus_item = idx
        self.CaptureMouse()
        self.Refresh(eraseBackground=False)

    def on_double_click(self, event):
        """Handle double click event"""
        idx = self.get_mouse_idx(event)
        if idx == -1:
            self.fn_do_menu_command([], None)
            return
        item, path_idx = self[idx]
        if item is None:
            self.fn_do_menu_command([], None)
            return
        treeitem_x = self.get_treeitem_x()
        if path_idx is None:
            if event.GetX() < treeitem_x:
                # Handle second click on tree expand/contract as
                # if the user clicked slowly
                #
                self.selections = set()
                item.opened = not item.opened
                self.schmutzy = True
                self.notify_selection_changed()
                self.Refresh(eraseBackground=False)
            return
        if self.fn_do_menu_command is not None:
            self.fn_do_menu_command([item.get_full_path(path_idx)], None)

    def on_right_mouse_down(self, event):
        """Handle right mouse button down"""
        assert isinstance(event, wx.MouseEvent)
        self.SetFocus()
        idx = self.get_mouse_idx(event)
        if idx == -1 or len(self.folder_items) == 0:
            event.Skip(True)
            return

        self.focus_item = idx
        if self[idx][1] is not None:
            self.selections.add(idx)
            self.notify_selection_changed()
        self.refresh_item(idx)
        event.Skip(True)

    def on_mouse_moved(self, event):
        """Handle mouse movement during capture"""
        if self.mouse_down_idx is None:
            return
        self.mouse_idx = self.get_mouse_idx(event)
        self.focus_item = self.mouse_idx
        self.scroll_into_view()
        self.Refresh(eraseBackground=False)

    def scroll_into_view(self):
        """Scroll the focus item into view"""
        idx_min = self.GetScrollPos(wx.SB_VERTICAL)
        current_x = self.GetScrollPos(wx.SB_HORIZONTAL)
        _, height = self.GetSize()
        height = int(height / (self.line_height + self.leading))
        idx_max = idx_min + height
        if self.focus_item <= idx_min:
            self.Scroll(current_x, self.focus_item - 1)
            self.refresh_item(self.focus_item)
            self.refresh_item(self.focus_item - 1)
        elif self.focus_item >= idx_max:
            self.Scroll(current_x, self.focus_item - height + 1)

    def on_mouse_up(self, event):
        """Handle left mouse button up event"""
        if self.mouse_down_idx is None:
            return
        if self.mouse_down_idx == self.mouse_idx:
            if self.mouse_down_idx in self.selections:
                self.selections.remove(self.mouse_down_idx)
                self.Refresh(eraseBackground=False)
            elif self[self.mouse_down_idx][1] is not None:
                self.selections.add(self.mouse_down_idx)
        else:
            start = min(self.mouse_down_idx, self.mouse_idx)
            end = max(self.mouse_down_idx, self.mouse_idx) + 1
            self.selections.update(
                [idx for idx in range(start, end) if self[idx][1] is not None]
            )
        self.mouse_down_idx = None
        self.notify_selection_changed()
        self.ReleaseMouse()

    def on_mouse_capture_lost(self, event):
        """Handle loss of mouse capture"""
        self.mouse_down_idx = None

    def on_up_down(self, event, direction):
        """Handle the up and down arrow keys

        Move the current selection up or down.

        event - key event
        direction - 1 for down,  -1 for up
        """
        needs_selchange_event = False
        if self.focus_item in self.selections and not event.ShiftDown():
            if len(self.selections) > 1:
                self.Refresh(eraseBackground=False)
            self.selections = set()
            needs_selchange_event = True
        self.refresh_item(self.focus_item)
        if direction + self.focus_item < 0 or direction + self.focus_item >= len(self):
            if needs_selchange_event:
                self.notify_selection_changed()
            return
        self.focus_item += direction
        # There should never be an empty directory, therefore, item # 1
        # should be the only item that has no precedent and we
        # should only have to skip one directory item
        if self[self.focus_item][1] is None:
            self.focus_item += direction
        self.scroll_into_view()
        self.selections.add(self.focus_item)
        self.notify_selection_changed()
        self.refresh_item(self.focus_item)

    def on_key_down(self, event):
        """Handle a key press"""
        assert isinstance(event, wx.KeyEvent)
        if event.GetKeyCode() == wx.WXK_DELETE and self.fn_delete is not None:
            paths = self.get_paths(self.FLAG_SELECTED_ONLY)
            self.fn_delete(paths)
            return
        elif (
            event.GetKeyCode() == wx.WXK_UP
            and self.focus_item is not None
            and self.focus_item > 1
        ):
            self.on_up_down(event, -1)
            return
        elif (
            event.GetKeyCode() == wx.WXK_DOWN
            and self.focus_item is not None
            and self.focus_item < len(self)
        ):
            self.on_up_down(event, 1)
            return
        event.Skip(True)

    context_menu_ids = []

    def on_context_menu(self, event):
        """Handle a context menu request"""
        if self.focus_item is None:
            fn_context_menu = self.fn_empty_context_menu
            fn_do_menu_command = self.fn_do_empty_context_menu_command
            arg = None
        else:
            item, idx = self[self.focus_item]
            if idx is None:
                fn_context_menu = self.fn_folder_context_menu
                fn_do_menu_command = self.fn_do_folder_menu_command
                arg = item.folder_name
            else:
                fn_context_menu = self.fn_context_menu
                fn_do_menu_command = self.fn_do_menu_command
                arg = self.get_paths(self.FLAG_SELECTED_ONLY)

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
