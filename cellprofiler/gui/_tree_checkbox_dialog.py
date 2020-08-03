"""treecheckboxdialog.py - tree checkbox dialog for selection of tree branches
"""

import wx


class TreeCheckboxDialog(wx.Dialog):
    """A dialog for "selecting" items on a tree by checking them"""

    def __init__(self, parent, d, *args, **kwargs):
        """Initialize the dialog

        d - dictionary representing the tree.
            Keys form the dictionary labels, values are dictionaries of subtrees
            A leaf is marked with a dictionary entry whose key is None and
            whose value is True or False, depending on whether it is
            initially selected or not.
        """
        wx.Dialog.__init__(self, parent, *args, **kwargs)

        self.bitmaps = []
        self.parent_reflects_child = True
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(sizer)
        tree_style = wx.TR_DEFAULT_STYLE
        self.tree_ctrl = wx.TreeCtrl(self, style=tree_style)
        sizer.Add(self.tree_ctrl, 1, wx.EXPAND | wx.ALL, 5)

        image_list = wx.ImageList(16, 16)
        for i, state_flag in enumerate(
            (0, wx.CONTROL_CHECKED, wx.CONTROL_UNDETERMINED)
        ):
            for j, selection_flag in enumerate((0, wx.CONTROL_CURRENT)):
                idx = image_list.Add(
                    self.get_checkbox_bitmap(state_flag | selection_flag, 16, 16)
                )
        self.tree_ctrl.SetImageList(image_list)
        self.image_list = image_list
        image_index, selected_image_index = self.img_idx(d)
        root_id = self.tree_ctrl.AddRoot("All", image_index, selected_image_index, d)
        self.tree_ctrl.SetItemImage(root_id, image_index, wx.TreeItemIcon_Normal)
        self.tree_ctrl.SetItemImage(
            root_id, selected_image_index, wx.TreeItemIcon_Selected
        )
        self.tree_ctrl.SetItemImage(root_id, image_index, wx.TreeItemIcon_Expanded)
        self.tree_ctrl.SetItemImage(
            root_id, image_index, wx.TreeItemIcon_SelectedExpanded
        )
        self.root_id = root_id
        self.tree_ctrl.SetItemHasChildren(root_id, len(d) > 1)
        self.Bind(wx.EVT_TREE_ITEM_EXPANDING, self.on_expanding, self.tree_ctrl)
        self.tree_ctrl.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
        self.tree_ctrl.Expand(root_id)
        table_sizer = wx.GridBagSizer()
        sizer.Add(table_sizer, 0, wx.EXPAND)
        table_sizer.Add(
            wx.StaticText(self, label="Key:"), (0, 0), flag=wx.LEFT | wx.RIGHT, border=3
        )
        for i, (bitmap, description) in enumerate(
            (
                (image_list.GetBitmap(0), "No subitems selected / not selected"),
                (image_list.GetBitmap(2), "All subitems selected / selected"),
                (
                    image_list.GetBitmap(4),
                    "Some subitems selected. Open tree to see selections.",
                ),
            )
        ):
            bitmap_ctrl = wx.StaticBitmap(self)
            bitmap_ctrl.SetBitmap(bitmap)
            table_sizer.Add(bitmap_ctrl, (i, 1), flag=wx.RIGHT, border=5)
            table_sizer.Add(wx.StaticText(self, label=description), (i, 2))
        table_sizer.AddGrowableCol(2)
        sizer.Add(self.CreateStdDialogButtonSizer(wx.CANCEL | wx.OK), flag=wx.CENTER)
        self.Layout()

    def set_parent_reflects_child(self, value):
        """Set the "parent_reflects_child" flag

        If you uncheck all of a parent's children, maybe that means
        that the parent should be unchecked too. But imagine the case
        where the user is checking and unchecking subdirectories. Perhaps
        they want the files in the parent, but not in the child. Set this
        to False to make the parent state be "None" if all children are False.
        This drives the parent to None instead of False, indicating that
        files should be picked up from the currenet directory, but not kids."""
        self.parent_reflects_child = value

    @staticmethod
    def img_idx(d):
        if d[None] is False:
            return 0, 1
        elif d[None] is True:
            return 2, 3
        else:
            return 4, 5

    def get_item_data(self, item_id):
        x = self.tree_ctrl.GetItemData(item_id)
        return x

    def on_expanding(self, event):
        """Populate subitems on expansion"""
        item_id = event.GetItem()
        d = self.get_item_data(item_id)
        if len(d) > 1:
            self.populate(item_id)

    def populate(self, item_id):
        """Populate the subitems of a tree"""
        try:
            d = self.get_item_data(item_id)
            assert len(d) > 1
            if self.tree_ctrl.GetChildrenCount(item_id, False) == 0:
                for key in sorted([x for x in list(d.keys()) if x is not None]):
                    d1 = d[key]
                    if hasattr(d1, "__call__"):
                        # call function to get real value
                        self.SetCursor(wx.Cursor(wx.CURSOR_WAIT))
                        d1 = d1()
                        d[key] = d1
                    image_index, selected_index = self.img_idx(d1)
                    sub_id = self.tree_ctrl.AppendItem(
                        item_id, key, image_index, selected_index, d1
                    )
                    self.tree_ctrl.SetItemImage(
                        sub_id, image_index, wx.TreeItemIcon_Normal
                    )
                    self.tree_ctrl.SetItemImage(
                        sub_id, selected_index, wx.TreeItemIcon_Selected
                    )
                    self.tree_ctrl.SetItemImage(
                        sub_id, image_index, wx.TreeItemIcon_Expanded
                    )
                    self.tree_ctrl.SetItemImage(
                        sub_id, selected_index, wx.TreeItemIcon_SelectedExpanded
                    )
                    self.tree_ctrl.SetItemHasChildren(sub_id, len(d1) > 1)
        finally:
            self.SetCursor(wx.NullCursor)

    def on_left_down(self, event):
        item_id, where = self.tree_ctrl.HitTest(event.Position)
        if where & wx.TREE_HITTEST_ONITEMICON == 0:
            event.Skip()
            return

        d = self.get_item_data(item_id)
        if d[None] is None or d[None] is False:
            state = True
        else:
            state = False
        self.set_item_state(item_id, state)
        self.set_parent_state(item_id)

    def set_parent_state(self, item_id):
        if item_id != self.root_id:
            parent_id = self.tree_ctrl.GetItemParent(item_id)
            d_parent = self.get_item_data(parent_id)
            child_id, _ = self.tree_ctrl.GetFirstChild(parent_id)
            state = self.get_item_data(child_id)[None]
            while True:
                if child_id == self.tree_ctrl.GetLastChild(parent_id):
                    break
                child_id = self.tree_ctrl.GetNextSibling(child_id)
                next_state = self.get_item_data(child_id)[None]
                if next_state != state:
                    state = None
                    break

            if d_parent[None] is not state:
                if state is False and not self.parent_reflects_child:
                    state = None
                d_parent[None] = state
                image_index, selected_index = self.img_idx(d_parent)
                self.tree_ctrl.SetItemImage(
                    parent_id, image_index, wx.TreeItemIcon_Normal
                )
                self.tree_ctrl.SetItemImage(
                    parent_id, selected_index, wx.TreeItemIcon_Selected
                )
                self.tree_ctrl.SetItemImage(
                    parent_id, image_index, wx.TreeItemIcon_Expanded
                )
                self.tree_ctrl.SetItemImage(
                    parent_id, selected_index, wx.TreeItemIcon_SelectedExpanded
                )
                self.set_parent_state(parent_id)

    def set_item_state(self, item_id, state):
        d = self.get_item_data(item_id)
        d[None] = state
        image_index, selected_index = self.img_idx(d)
        self.tree_ctrl.SetItemImage(item_id, image_index, wx.TreeItemIcon_Normal)
        self.tree_ctrl.SetItemImage(item_id, selected_index, wx.TreeItemIcon_Selected)
        self.tree_ctrl.SetItemImage(item_id, image_index, wx.TreeItemIcon_Expanded)
        self.tree_ctrl.SetItemImage(
            item_id, selected_index, wx.TreeItemIcon_SelectedExpanded
        )
        if len(d) > 1:
            if self.tree_ctrl.GetChildrenCount(item_id) == 0:
                self.populate(item_id)
            child_id, _ = self.tree_ctrl.GetFirstChild(item_id)
            while True:
                d1 = self.get_item_data(child_id)
                if d1[None] is not state:
                    self.set_item_state(child_id, state)
                if child_id == self.tree_ctrl.GetLastChild(item_id):
                    break
                child_id = self.tree_ctrl.GetNextSibling(child_id)

    def get_checkbox_bitmap(self, flags, width, height):
        """Return a bitmap with a checkbox drawn into it

        flags - rendering flags including CONTROL_CHECKED and CONTROL_UNDETERMINED
        width, height - size of bitmap to return
        """
        dc = wx.MemoryDC()
        bitmap = wx.Bitmap(width, height)
        dc.SelectObject(bitmap)
        dc.SetBrush(wx.BLACK_BRUSH)
        dc.SetTextForeground(wx.BLACK)
        try:
            dc.Clear()
            render = wx.RendererNative.Get()
            render.DrawCheckBox(self, dc, (0, 0, width, height), flags)
        finally:
            dc.SelectObject(wx.NullBitmap)
        dc.Destroy()
        self.bitmaps.append(bitmap)
        return bitmap
