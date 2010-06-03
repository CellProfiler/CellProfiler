'''treecheckboxdialog.py - tree checkbox dialog for selection of tree branches

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2010 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version__="$Revision$"

import wx

class TreeCheckboxDialog(wx.Dialog):
    '''A dialog for "selecting" items on a tree by checking them'''
    
    def __init__(self, parent, d, *args, **kwargs):
        '''Initialize the dialog
        
        d - dictionary representing the tree.
            Keys form the dictionary labels, values are dictionaries of subtrees
            A leaf is marked with a dictionary entry whose key is None and
            whose value is True or False, depending on whether it is
            initially selected or not.
        '''
        wx.Dialog.__init__(self, parent, *args, **kwargs)

        self.bitmaps = []
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(sizer)
        tree_style = wx.TR_DEFAULT_STYLE
        self.tree_ctrl = wx.TreeCtrl(self, 
                                     style = tree_style)
        sizer.Add(self.tree_ctrl, 1, wx.EXPAND | wx.ALL, 5)
        
        image_list = wx.ImageList(16, 16)
        for i, state_flag in enumerate(
            (0, wx.CONTROL_CHECKED, wx.CONTROL_UNDETERMINED)):
            for j, selection_flag in enumerate((0, wx.CONTROL_CURRENT)):
                idx = image_list.Add(
                    self.get_checkbox_bitmap(state_flag | selection_flag,
                                             16, 16))
        self.tree_ctrl.SetImageList(image_list)
        self.image_list = image_list
        image_index, selected_image_index = self.img_idx(d)
        root_id = self.tree_ctrl.AddRoot("All", image_index, 
                                         selected_image_index, 
                                         wx.TreeItemData(d))
        self.tree_ctrl.SetItemImage(root_id, image_index,
                                    wx.TreeItemIcon_Normal)
        self.tree_ctrl.SetItemImage(root_id, selected_image_index, 
                                    wx.TreeItemIcon_Selected)
        self.tree_ctrl.SetItemImage(root_id, image_index, 
                                    wx.TreeItemIcon_Expanded)
        self.tree_ctrl.SetItemImage(root_id, image_index, 
                                    wx.TreeItemIcon_SelectedExpanded)
        self.root_id = root_id
        self.tree_ctrl.SetItemHasChildren(root_id, len(d) > 1)
        self.Bind(wx.EVT_TREE_ITEM_EXPANDING, self.on_expanding, self.tree_ctrl)
        self.tree_ctrl.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
        self.tree_ctrl.Expand(root_id)
        table_sizer = wx.GridBagSizer()
        table_sizer.AddGrowableCol(1)
        sizer.Add(table_sizer, 0, wx.EXPAND)
        for i, (bitmap, description) in enumerate((
            (image_list.GetBitmap(0), "No subitems selected / not selected"),
            (image_list.GetBitmap(2), "All subitems selected / selected"),
            (image_list.GetBitmap(4), "Some subitems selected. Open tree to see selections."))):
            bitmap_ctrl = wx.StaticBitmap(self)
            bitmap_ctrl.SetBitmap(bitmap)
            table_sizer.Add(bitmap_ctrl, (i, 0))
            table_sizer.Add(wx.StaticText(self, label=description), (i, 1))
        sizer.Add(self.CreateStdDialogButtonSizer(wx.CANCEL | wx.OK), 
                  flag=wx.CENTER)
        self.Layout()
        
    def img_idx(self, d):
        if d[None] is False:
            return (0, 1)
        elif d[None] is True:
            return (2, 3)
        else:
            return (4, 5)
        
    def get_item_data(self, item_id):
        x = self.tree_ctrl.GetItemData(item_id)
        return x.GetData()
        
    def on_expanding(self, event):
        '''Populate subitems on expansion'''
        item_id = event.GetItem()
        d = self.get_item_data(item_id)
        if len(d) > 1:
            self.populate(item_id)
        
    def populate(self, item_id):
        '''Populate the subitems of a tree'''
        d = self.get_item_data(item_id)
        assert len(d) > 1
        if self.tree_ctrl.GetChildrenCount(item_id, False) == 0:
            for key in sorted([x for x in d.keys() if x is not None]):
                d1 = d[key]
                image_index, selected_index = self.img_idx(d1)
                sub_id = self.tree_ctrl.AppendItem(item_id, key, image_index,
                                                   selected_index, 
                                                   wx.TreeItemData(d1))
                self.tree_ctrl.SetItemImage(sub_id, image_index,
                                            wx.TreeItemIcon_Normal)
                self.tree_ctrl.SetItemImage(sub_id, selected_index, 
                                            wx.TreeItemIcon_Selected)
                self.tree_ctrl.SetItemImage(sub_id, image_index,
                                            wx.TreeItemIcon_Expanded)
                self.tree_ctrl.SetItemImage(sub_id, selected_index, 
                                            wx.TreeItemIcon_SelectedExpanded)
                self.tree_ctrl.SetItemHasChildren(sub_id, len(d1) > 1)
                
    def on_left_down(self, event):
        item_id, where = self.tree_ctrl.HitTest(event.Position)
        if where != wx.TREE_HITTEST_ONITEMICON:
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
                d_parent[None] = state
                image_index, selected_index = self.img_idx(d_parent)
                self.tree_ctrl.SetItemImage(parent_id, image_index, wx.TreeItemIcon_Normal)
                self.tree_ctrl.SetItemImage(parent_id, selected_index, wx.TreeItemIcon_Selected)
                self.tree_ctrl.SetItemImage(parent_id, image_index, wx.TreeItemIcon_Expanded)
                self.tree_ctrl.SetItemImage(parent_id, selected_index, wx.TreeItemIcon_SelectedExpanded)
                self.set_parent_state(parent_id)
        
    def set_item_state(self, item_id, state):
        d = self.get_item_data(item_id)
        d[None] = state
        image_index, selected_index = self.img_idx(d)
        self.tree_ctrl.SetItemImage(item_id, image_index, wx.TreeItemIcon_Normal)
        self.tree_ctrl.SetItemImage(item_id, selected_index, wx.TreeItemIcon_Selected)
        self.tree_ctrl.SetItemImage(item_id, image_index, wx.TreeItemIcon_Expanded)
        self.tree_ctrl.SetItemImage(item_id, selected_index, wx.TreeItemIcon_SelectedExpanded)
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
        '''Return a bitmap with a checkbox drawn into it
        
        flags - rendering flags including CONTROL_CHECKED and CONTROL_UNDETERMINED
        width, height - size of bitmap to return
        '''
        dc = wx.MemoryDC()
        dc.SetBrush(wx.BLACK_BRUSH)
        dc.SetTextForeground(wx.BLACK)
        bitmap = wx.EmptyBitmap(width, height)
        dc.SelectObject(bitmap)
        try:
            dc.Clear()
            render = wx.RendererNative.Get()
            render.DrawCheckBox(self, dc, (0, 0, width, height), flags)
        finally:
            dc.SelectObject(wx.NullBitmap)
        self.bitmaps.append(bitmap)
        return bitmap

if __name__ == "__main__":
    class MyApp(wx.App):
        def OnInit(self):
            d = { None:None }
            for i in range(5):
                d1 = d[str(i)] = { None: None }
                for j in range(5):
                    d2 = d1[str(j)] = { None: None }
                    for k in range(5):
                        d2[str(k)] = { None: (k & 1) != 0}
                        
            dlg = TreeCheckboxDialog(None, d, size=(640,480), 
                                     style = wx.DEFAULT_DIALOG_STYLE | 
                                     wx.RESIZE_BORDER)
            dlg.ShowModal()
            print "{"
            for i in range(5):
                d1 = d[str(i)]
                print "   %d: { None=%s," % (i, repr(d1[None]))
                for j in range(5):
                    d2 = d1[str(j)] 
                    print "       %d: { None=%s" % (j, repr(d2[None]))
                    for k in range(5):
                        d3 = d2[str(k)]
                        print "           %d: { None=%s }, " % (k, repr(d3[None]))
                    print "           },"
                print "        },"
            print "}"
            return False
    my_app = MyApp(False)
    my_app.MainLoop()
    pass
    