'''filefinderdlg.py - tree checkbox dialog wrapping the cellprofiler file finder

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
import wx
import cellprofiler.utilities.filefinder as filefinder
import os.path
import Queue

class FileFinderDialog(wx.Dialog):
    '''A dialog wrapping the cellprofiler file finder'''

    def __init__(self, parent, dirs, *args, **kwargs):
        '''Initialize the dialog

        dirs - list of directories to search
        '''
        wx.Dialog.__init__(self, parent, *args, **kwargs)

        self.bitmaps = []
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(sizer)
        self.tree_ctrl = wx.TreeCtrl(self, style = wx.TR_DEFAULT_STYLE | wx.TR_HIDE_ROOT)

        sizer.Add(self.tree_ctrl, 1, wx.EXPAND | wx.ALL, 3)

        image_list = wx.ImageList(16, 16)
        for i, state_flag in enumerate(
            (0, wx.CONTROL_CHECKED, wx.CONTROL_UNDETERMINED)):
            for j, selection_flag in enumerate((0, wx.CONTROL_CURRENT)):
                image_list.Add(self.get_checkbox_bitmap(state_flag | selection_flag, 16, 16))
        self.tree_ctrl.SetImageList(image_list)
        self.image_list = image_list
        self.root_id = self.tree_ctrl.AddRoot("All")  # hidden
        self.tree_ctrl.SetItemHasChildren(self.root_id, True)

        # for throbbers
        self.queue = Queue.Queue()
        self.progress = {}
        self.Bind(wx.EVT_TIMER, self.on_timer)
        self.timer = wx.Timer(self)
        self.timer.Start(100)

        self.key_to_itemid = {None: self.root_id}
        self.file_finder = filefinder.Locator(self.finder_cb, self.metadata_cb)
        for dir in dirs:
            self.file_finder.queue(dir)

        # self.Bind(wx.EVT_TREE_ITEM_EXPANDING, self.on_expanding, self.tree_ctrl)
        self.tree_ctrl.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)

        table_sizer = wx.GridBagSizer()
        table_sizer.AddGrowableCol(2)
        sizer.Add(table_sizer, 0, wx.EXPAND)
        table_sizer.Add(wx.StaticText(self, label='Key:'), (0, 0), flag=wx.LEFT | wx.RIGHT, border=3)
        for i, (bitmap, description) in enumerate((
            (image_list.GetBitmap(0), "No subitems selected / not selected"),
            (image_list.GetBitmap(2), "All subitems selected / selected"),
            (image_list.GetBitmap(4), "Some subitems selected. Open tree to see selections."))):
            bitmap_ctrl = wx.StaticBitmap(self)
            bitmap_ctrl.SetBitmap(bitmap)
            table_sizer.Add(bitmap_ctrl, (i, 1), flag=wx.RIGHT, border=5)
            table_sizer.Add(wx.StaticText(self, label=description), (i, 2))
        sizer.Add(self.CreateStdDialogButtonSizer(wx.CANCEL | wx.OK),
                  flag=wx.CENTER)
        self.Layout()



    def img_idx(self, checked):
        if not checked:
            return (0, 1)
        else:
            return (2, 3)

    def get_item_data(self, item_id):
        return self.tree_ctrl.GetItemPyData(item_id)

    def on_left_down(self, event):
        item_id, where = self.tree_ctrl.HitTest(event.Position)
        if where & wx.TREE_HITTEST_ONITEMICON == 0:
            event.Skip()
            return

        itemdata = self.get_item_data(item_id)
        checked, key, path = itemdata
        checked = not checked
        if not checked:
            self.file_finder.remove(key)
            self.tree_ctrl.DeleteChildren(item_id)
            self.tree_ctrl.SetItemHasChildren(item_id, False)
        else:
            # restart searching of this subtree
            _, parent_key, _ = self.get_item_data(self.tree_ctrl.GetItemParent(item_id))
            self.file_finder.put_back(key, path, parent_key)
        # store state
        itemdata[:] = checked, key, path
        self.set_item_image(item_id, checked)

    def set_item_image(self, item_id, checked):
        image_index, selected_index = self.img_idx(checked)
        self.tree_ctrl.SetItemImage(item_id, image_index, wx.TreeItemIcon_Normal)
        self.tree_ctrl.SetItemImage(item_id, selected_index, wx.TreeItemIcon_Selected)
        self.tree_ctrl.SetItemImage(item_id, image_index, wx.TreeItemIcon_Expanded)
        self.tree_ctrl.SetItemImage(item_id, selected_index, wx.TreeItemIcon_SelectedExpanded)

    def get_checkbox_bitmap(self, flags, width, height):
        '''Return a bitmap with a checkbox drawn into it

        flags - rendering flags including CONTROL_CHECKED and CONTROL_UNDETERMINED
        width, height - size of bitmap to return
        '''
        dc = wx.MemoryDC()
        bitmap = wx.EmptyBitmap(width, height)
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

    def get_children(self, item):
        child, cookie = self.tree_ctrl.GetFirstChild(item)
        while child.IsOk():
            yield child
            child, cookie = self.tree_ctrl.GetNextchild(item, cookie)

    def finder_cb(self, path, isdir, key, parent, status, data):
        # for interactivity, we queue this message, and handle them in batches
        # in on_timer(), below.
        self.queue.put((path, isdir, key, parent, status, data))

    def metadata_cb(self, path):
        return

    def update(self, path, isdir, key, parent, status, data):
        self.progress[key] = status

        if status != filefinder.FOUND:
            return

        # already have this item - was it put back?
        if key in self.key_to_itemid:
            return

        # New item - add to tree.
        # transform from filefinder ids to treeids
        tree_parent = self.key_to_itemid[parent]
        if not self.tree_ctrl.ItemHasChildren(tree_parent):
            self.tree_ctrl.SetItemHasChildren(tree_parent, True)

        tree_child = self.tree_ctrl.AppendItem(tree_parent, os.path.basename(path),
                                               image=2, selectedImage=3)
        self.key_to_itemid[key] = tree_child
        self.tree_ctrl.SetItemPyData(tree_child, [True, key, path])

    def on_timer(self, evt):
        # first, process any callbacks in the queue
        count = 0
        while not self.queue.empty():
            message = self.queue.get()
            self.update(*message)
            count += 1
        print "Processed", count

        # then update any items
        item = self.tree_ctrl.GetFirstVisibleItem()
        while item.IsOk():
            try:
                _, key, path = self.get_item_data(item)
                print path, self.progress[key]
            except:
                pass
            item = self.tree_ctrl.GetNextVisible(item)

if __name__ == "__main__":
    class MyApp(wx.App):
        def OnInit(self):
            dlg = FileFinderDialog(None, ['/Users/tjones/CellProfilerMine.git',
                                          '/Volumes/plateformes/incell/Screening Externe_BFX Projects Calls/E-003_CILS_BENMERAH/Crible_E003'],
                                   size=(640, 480),
                                   style = wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
            dlg.ShowModal()
    my_app = MyApp(False)
    my_app.MainLoop()
