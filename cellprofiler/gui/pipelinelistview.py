"""PipelineListView.py

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2010

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

from StringIO import StringIO
import time
import base64
import math
import zlib
import wx
import sys

import cellprofiler.pipeline as cpp
import cellprofiler.gui.movieslider as cpgmov
from cellprofiler.gui.cpfigure import window_name
from cellprofiler.icons import IMG_OK, IMG_ERROR, IMG_EYE, IMG_CLOSED_EYE, IMG_PAUSE, IMG_GO

NO_PIPELINE_LOADED = 'No pipeline loaded'
PADDING = 1

def plv_get_bitmap(data):
    return wx.BitmapFromImage(data)

PAUSE_COLUMN = 0
EYE_COLUMN = 1
ERROR_COLUMN = 2
MODULE_NAME_COLUMN = 3
NUM_COLUMNS = 4
ERROR = "error"
OK = "ok"
EYE = "eye"
CLOSED_EYE = "closedeye"
PAUSE = "pause"
GO = "go"
NOTDEBUG = "notdebug"

############################
#
# Image index dictionary - image names -> indexes
#
############################
image_index_dictionary = {}
def get_image_index(name):
    '''Return the index of an image in the image list'''
    global image_index_dictionary
    if not image_index_dictionary.has_key(name):
        image_index_dictionary[name] = len(image_index_dictionary)
    return image_index_dictionary[name] 

CHECK_TIMEOUT_SEC = 2

class PipelineListView(object):
    """View on a set of modules
    
    """
    def __init__(self, panel, frame):
        self.__pipeline = None
        self.__panel=panel
        self.__frame = frame
        self.__module_controls_panel = None
        self.__sizer=wx.BoxSizer(wx.HORIZONTAL)
        self.__panel.SetSizer(self.__sizer)
        self.__panel.SetAutoLayout(True)
        self.__pipeline_slider = wx.Slider(self.__panel,
                                           size=(20, -1),
                                           style=wx.SL_VERTICAL,
                                           value=0,
                                           minValue=0,
                                           maxValue=1)
        self.__pipeline_slider.SetTickFreq(1, 0)
        self.__pipeline_slider.SetBackgroundColour('white')
        self.__slider_sizer = wx.BoxSizer(wx.VERTICAL)
        self.__sizer.Add(self.__slider_sizer)
        self.__slider_sizer_item = self.__slider_sizer.Add(
            self.__pipeline_slider, 0, 
            wx.EXPAND|wx.RESERVE_SPACE_EVEN_IF_HIDDEN|wx.TOP, 10)
        self.__slider_sizer.AddStretchSpacer()
        self.make_list()
        self.set_debug_mode(False)
        wx.EVT_IDLE(panel,self.on_idle)
        self.__adjust_rows()
        self.__first_dirty_module = 0
        self.drag_underway = False
        self.drag_start = None
        self.drag_time = None
        self.list_ctrl.SetDropTarget(PipelineDropTarget(self))

    def make_list(self):
        '''Make the list control with the pipeline items in it'''
        self.list_ctrl = wx.ListCtrl(self.__panel, style = wx.LC_REPORT)
        #
        # First, make all of the bitmaps for the image list. 
        #
        d = {}
        width = 16
        height = 16
        for name, image in ((ERROR, IMG_ERROR),
                            (OK, IMG_OK),
                            (EYE, IMG_EYE),
                            (CLOSED_EYE, IMG_CLOSED_EYE),
                            (PAUSE, IMG_PAUSE),
                            (GO, IMG_GO)):
            bitmap = plv_get_bitmap(image)
            idx = get_image_index(name)
            d[idx] = bitmap
        idx = get_image_index(NOTDEBUG)
        bitmap = wx.EmptyBitmap(width, height)
        dc = wx.MemoryDC()
        dc.SelectObject(bitmap)
        dc.Clear()
        dc.SelectObject(wx.NullBitmap)
        del dc
        d[idx] = bitmap
        self.image_list = wx.ImageList(width, height)
        for i in range(len(d)):
            index = self.image_list.Add(d[i])
        
        self.list_ctrl.SetImageList(self.image_list, wx.IMAGE_LIST_SMALL)
        self.__sizer.Add(self.list_ctrl, 1, wx.EXPAND | wx.ALL, 2)
        self.list_ctrl.InsertColumn(PAUSE_COLUMN, "")
        self.list_ctrl.InsertColumn(EYE_COLUMN, "")
        self.list_ctrl.InsertColumn(ERROR_COLUMN, "")
        self.list_ctrl.InsertColumn(MODULE_NAME_COLUMN, "Module")
        for column in (PAUSE_COLUMN, EYE_COLUMN, ERROR_COLUMN):
            self.list_ctrl.SetColumnWidth(column, 20)
        self.list_ctrl.SetColumnWidth(MODULE_NAME_COLUMN, 100)
        #
        # Bind events
        #
        self.list_ctrl.Bind(wx.EVT_LIST_ITEM_SELECTED, self.__on_item_selected, self.list_ctrl)
        self.list_ctrl.Bind(wx.EVT_LIST_ITEM_DESELECTED, self.__on_item_deselected, self.list_ctrl)
        self.list_ctrl.Bind(wx.EVT_LEFT_DOWN, self.__on_list_left_down, self.list_ctrl)
        self.list_ctrl.Bind(wx.EVT_RIGHT_DOWN, self.__on_list_right_down, self.list_ctrl)
        #
        # Accelerators
        #
        from cellprofiler.gui.cpframe import ID_EDIT_DELETE
        accelerator_table = wx.AcceleratorTable([
             (wx.ACCEL_NORMAL,wx.WXK_DELETE, ID_EDIT_DELETE)])
        self.list_ctrl.SetAcceleratorTable(accelerator_table)
        if sys.platform.startswith('linux'):
            # Linux machines seem to have two subwindows (headers & list?)
            # which need mouse button down event capture instead of the
            # official window.
            for child in self.list_ctrl.GetChildren():
                child.Bind(wx.EVT_LEFT_DOWN, self.__on_list_left_down)
                child.Bind(wx.EVT_RIGHT_DOWN, self.__on_list_right_down)
        
    def set_debug_mode(self, mode):
        if self.__pipeline is not None:
            self.select_one_module(1)
        self.__debug_mode = mode
        self.__pipeline_slider.Show(mode)
        self.__sizer.Layout()
        
    def __set_min_width(self):
        """Make the minimum width of the panel be the best width
           of the list_ctrl and slider
        """
        text_width = 0
        dc = wx.ClientDC(self.list_ctrl)
        font = self.list_ctrl.Font
        for i in range(self.list_ctrl.ItemCount):
            item = self.list_ctrl.GetItem(i, MODULE_NAME_COLUMN)
            text = item.Text
            text_width = max(text_width,dc.GetFullTextExtent(text, font)[0])
        self.list_ctrl.SetColumnWidth(MODULE_NAME_COLUMN, text_width+5)

    def attach_to_pipeline(self,pipeline,controller):
        """Attach the viewer to the pipeline to allow it to listen for changes
        
        """
        self.__pipeline =pipeline
        self.__controller = controller
        pipeline.add_listener(self.notify)
        controller.attach_to_pipeline_list_view(self,self.__pipeline_slider)
        
    def attach_to_module_view(self, module_view):
        self.__module_view = module_view
        module_view.add_listener(self.__on_setting_changed_event)
        
    def notify(self,pipeline,event):
        """Pipeline event notifications come through here
        
        """
        if isinstance(event,cpp.PipelineLoadedEvent):
            self.__on_pipeline_loaded(pipeline,event)
            self.__first_dirty_module = 0
        elif isinstance(event,cpp.ModuleAddedPipelineEvent):
            self.__on_module_added(pipeline,event)
            self.__first_dirty_module = min(self.__first_dirty_module, event.module_num - 1)
        elif isinstance(event,cpp.ModuleMovedPipelineEvent):
            self.__on_module_moved(pipeline,event)
            self.__first_dirty_module = min(self.__first_dirty_module, event.module_num - 2)
        elif isinstance(event,cpp.ModuleRemovedPipelineEvent):
            self.__on_module_removed(pipeline,event)
            self.__first_dirty_module = min(self.__first_dirty_module, event.module_num - 1)
        elif isinstance(event,cpp.PipelineClearedEvent):
            self.__on_pipeline_cleared(pipeline, event)
            self.__first_dirty_module = 0
        elif isinstance(event,cpp.ModuleEditedPipelineEvent):
            self.__first_dirty_module = min(self.__first_dirty_module, event.module_num - 1)
    
    def notify_directory_change(self):
        # we can't know which modules use this information
        self.__first_dirty_module = 0

    def select_one_module(self, module_num):
        """Select only the given module number in the list box"""
        for module in self.__pipeline.modules():
            self.list_ctrl.Select(module.module_num-1, 
                                  module.module_num == module_num)
        self.__on_item_selected(None)
        
    def select_module(self,module_num,selected=True):
        """Select the given one-based module number in the list
        This is mostly for testing
        """
        self.list_ctrl.Select(module_num-1, selected)
        self.__on_item_selected(None)
        
    def get_selected_modules(self):
        return [self.__pipeline.modules()[i]
                for i in range(self.list_ctrl.ItemCount) 
                if self.list_ctrl.IsSelected(i)]
    
    def __on_list_left_down(self, event):
        if sys.platform.startswith("win"):
            item, hit_code, subitem = self.list_ctrl.HitTestSubItem(event.Position)
        else:
            # Mac's HitTestSubItem does not work. Sorry.
            #
            item, hit_code = self.list_ctrl.HitTest(event.Position)
            widths = [self.list_ctrl.GetColumnWidth(i) for i in range(4)]
            start = 0
            for subitem in range(4):
                if event.Position[0] < start + widths[subitem]:
                    break
                start += widths[subitem]
        if (item >= 0 and item < self.list_ctrl.ItemCount and
            (hit_code & wx.LIST_HITTEST_ONITEM)):
            module = self.__pipeline.modules()[item]
            if subitem == PAUSE_COLUMN and self.__debug_mode:
                module.wants_pause = not module.wants_pause
                pause_img = get_image_index(PAUSE if module.wants_pause
                                            else GO)
                self.list_ctrl.SetItemImage(item, pause_img)
            elif subitem == EYE_COLUMN:
                module.show_window = not module.show_window
                eye_img = get_image_index(EYE if module.show_window
                                          else CLOSED_EYE)
                self.set_subitem_image(item, EYE_COLUMN, eye_img)
                name = window_name(module)
                figure = self.__panel.TopLevelParent.FindWindowByName(name)
                if figure is not None:
                    figure.Close()
            else:
                if self.list_ctrl.IsSelected(item):
                    self.start_drag_operation(event)
                else:
                    event.Skip()
        else:
            event.Skip()
            
    def __on_list_right_down(self, event):
        from cellprofiler.gui.cpframe import ID_EDIT_DELETE, ID_EDIT_DUPLICATE, ID_HELP_MODULE
        
        item, hit_code = self.list_ctrl.HitTest(event.Position)
        if hit_code & wx.LIST_HITTEST_ONITEM:
            if not self.list_ctrl.IsSelected(item):
                self.select_one_module(item+1)
        if self.list_ctrl.SelectedItemCount > 0:
            menu = wx.Menu()
            menu.AppendSubMenu(self.__frame.menu_edit_add_module, "&Add")
            menu.Append(ID_EDIT_DELETE, "&Delete")
            menu.Append(ID_EDIT_DUPLICATE, "Duplicate")
            menu.Append(ID_HELP_MODULE, "&Help")
            self.__frame.PopupMenu(menu)
        else:
            self.__frame.PopupMenu(self.__frame.menu_edit_add_module)

    def start_drag_operation(self, event):
        '''Start dragging whatever is selected'''
        fd = StringIO()
        modules_to_save = [m.module_num for m in self.get_selected_modules()]
        self.__pipeline.savetxt(fd, modules_to_save)
        pipeline_data_object = PipelineDataObject()
        fd.seek(0)
        pipeline_data_object.SetData(fd.read())

        text_data_object = wx.TextDataObject()
        fd.seek(0)
        text_data_object.SetData(fd.read())
        
        data_object = wx.DataObjectComposite()
        data_object.Add(pipeline_data_object)
        data_object.Add(text_data_object)
        drop_source = wx.DropSource(self.list_ctrl)
        drop_source.SetData(data_object)
        self.drag_underway = True
        self.drag_start = event.Position
        self.drag_time = time.time()
        selected_module_ids = [m.id for m in self.get_selected_modules()]
        self.__pipeline.start_undoable_action()
        try:
            result = drop_source.DoDragDrop(wx.Drag_AllowMove)
            self.drag_underway = False
            if result == wx.DragMove:
                for id in selected_module_ids:
                    for module in self.__pipeline.modules():
                        if module.id == id:
                            self.__pipeline.remove_module(module.module_num)
                            break
        finally:
            self.__pipeline.stop_undoable_action("Drag and drop")
        
    def provide_drag_feedback(self, x, y, data):
        if self.where_to_drop(x,y)  is None:
            return False
        if self.drag_time is not None and time.time() - self.drag_time > 3:
            return True
        if self.drag_start is not None:
            distance = math.sqrt((x-self.drag_start[0])**2 +
                                 (y-self.drag_start[1])**2)
            return distance > 10
        return True
    
    def on_drop(self, x, y):
        if self.where_to_drop(x,y)  is None:
            return False
        return True
    
    def where_to_drop(self, x, y):
        nmodules = len(self.__pipeline.modules())
        if nmodules == 0:
            return 0
        else:
            last_rect = self.list_ctrl.GetItemRect(nmodules-1)
            if last_rect[1] + last_rect[3] < y:
                # Below last item. Insert after last
                return nmodules
            index, code = self.list_ctrl.HitTest(wx.Point(x,y))
            if code & wx.LIST_HITTEST_ONITEM:
                r = self.list_ctrl.GetItemRect(index)
                #
                # Put before or after depending on whether we are more or
                # less than 1/2 of the way
                #
                if y > r[1]+ r[3]/2:
                    index += 1
                return index
        return None
    
    def on_data(self, x, y, action, data):
        index = self.where_to_drop(x,y)
        if index is not None:
            #
            # Insert the new modules
            #
            wx.BeginBusyCursor()
            try:
                pipeline = cpp.Pipeline()
                pipeline.load(StringIO(data))
                for i, module in enumerate(pipeline.modules()):
                    module.module_num = i + index + 1
                    self.__pipeline.add_module(module)
            finally:
                wx.EndBusyCursor()
                    
    def __on_pipeline_loaded(self,pipeline,event):
        """Repopulate the list view after the pipeline loads
        
        """
        nrows = len(pipeline.modules())
        self.list_ctrl.DeleteAllItems()
        
        for module in pipeline.modules():
            self.__populate_row(module)
        self.__adjust_rows()
        if nrows > 0:
            self.select_one_module(1)
        self.__controller.enable_module_controls_panel_buttons()
    
    def __adjust_rows(self):
        """Adjust slider and dimensions after adding or removing rows"""
        self.__set_min_width()
        self.__pipeline_slider.Max = self.list_ctrl.ItemCount - 1
        if self.list_ctrl.ItemCount > 0:
            if self.__debug_mode:
                self.__pipeline_slider.Show()
            old_value = self.__pipeline_slider.Value
            r = self.list_ctrl.GetItemRect(0, wx.LIST_RECT_BOUNDS)
            height = r[3]
            y = r[1]
            self.__slider_sizer_item.SetBorder(int(y + height / 3))
            self.__pipeline_slider.SetMinSize(
                (20, height * self.list_ctrl.ItemCount))
            self.__pipeline_slider.Value = old_value
            #
            # Make sure that the list control is internally as big as it
            # needs to be.
            #
            r = self.list_ctrl.GetItemRect(self.list_ctrl.ItemCount -1, 
                                           wx.LIST_RECT_BOUNDS)
            min_width = self.list_ctrl.GetMinWidth()
            self.list_ctrl.SetMinSize((min_width, r[1] + r[3] + 4))
        else:
            self.__pipeline_slider.Hide()
            self.__pipeline_slider.SetMinSize((20, 10))
        self.__sizer.Layout()
        self.__panel.SetupScrolling(scroll_x=False, scroll_y=True)
    
    def set_subitem_image(self, index, column, image_number):
        item = wx.ListItem()
        item.Mask = wx.LIST_MASK_IMAGE
        item.Image = image_number
        item.Id = index
        item.Column = column
        self.list_ctrl.SetItem(item)
        
    def __populate_row(self, module):
        """Populate a row in the grid with a module."""
        row = module.module_num-1
        pause_item = wx.ListItem()
        pause_item.Mask = wx.LIST_MASK_IMAGE
        pause_item.Image = get_image_index(NOTDEBUG)
        pause_item.Column = PAUSE_COLUMN
        pause_item.Id = row
        self.list_ctrl.InsertItem(pause_item)
        
        self.set_subitem_image(row, EYE_COLUMN,
                               get_image_index(EYE if module.show_window else
                                               CLOSED_EYE))
        self.set_subitem_image(row, ERROR_COLUMN,
                               get_image_index(OK))
        
        module_name_item = wx.ListItem()
        module_name_item.Mask = wx.LIST_MASK_TEXT
        module_name_item.Text = module.module_name
        module_name_item.Column = MODULE_NAME_COLUMN
        module_name_item.Id = row
        self.list_ctrl.SetItem(module_name_item)
        
    def __on_pipeline_cleared(self,pipeline,event):
        self.list_ctrl.DeleteAllItems()
        self.__adjust_rows()
        self.__controller.enable_module_controls_panel_buttons()
        
    def __on_module_added(self,pipeline,event):
        module = pipeline.modules()[event.module_num - 1]
        self.__populate_row(module)
        self.__adjust_rows()
        self.select_one_module(event.module_num)
    
    def __on_module_removed(self,pipeline,event):
        self.list_ctrl.DeleteItem(event.module_num - 1)
        self.__adjust_rows()
        self.__module_view.clear_selection()
        self.__controller.enable_module_controls_panel_buttons()
        
    def __on_module_moved(self,pipeline,event):
        if event.direction == cpp.DIRECTION_UP:
            start = event.module_num - 1
        else:
            start = event.module_num - 2
        first_selected = self.list_ctrl.IsSelected(start)
        second_selected = self.list_ctrl.IsSelected(start+1)
        self.list_ctrl.DeleteItem(start)
        self.list_ctrl.DeleteItem(start)
        self.__populate_row(pipeline.modules()[start])
        self.__populate_row(pipeline.modules()[start+1])
        self.list_ctrl.Select(start, second_selected)
        self.list_ctrl.Select(start+1, first_selected)
        self.__adjust_rows()
        self.__controller.enable_module_controls_panel_buttons()
        self.list_ctrl.Refresh()
    
    def __on_item_selected(self,event):
        if self.__module_view:
            selections = self.get_selected_modules()
            if len(selections):
                self.__module_view.set_selection(selections[0].module_num)
        self.__controller.enable_module_controls_panel_buttons()

    def __on_item_deselected(self,event):
        self.__controller.enable_module_controls_panel_buttons()
        
    def __on_setting_changed_event(self, caller, event):
        """Handle a setting change
        
        The debugging viewer needs to rewind to rerun a module after a change
        """
        setting = event.get_setting()
        for module in self.__pipeline.modules():
            for module_setting in module.settings():
                if setting is module_setting:
                    if self.__pipeline_slider.Value >= module.module_num:
                        self.__pipeline_slider.Value = module.module_num - 1
                    return
                
    def on_stop_debugging(self):
        self.__pipeline_slider.Value = 0
        
    def on_idle(self,event):
        last_idle_time = getattr(self, "last_idle_time", 0)
        if time.time() - last_idle_time > CHECK_TIMEOUT_SEC:
            self.last_idle_time = time.time()
        else:
            return

        modules = self.__pipeline.modules()
        for idx, module in enumerate(modules):
            if module.show_window:
                eye_value = get_image_index(EYE)
            else:
                eye_value = get_image_index(CLOSED_EYE)
            target_item = self.list_ctrl.GetItem(idx, EYE_COLUMN)
            if target_item.Image != eye_value:
                self.set_subitem_image(idx, EYE_COLUMN, eye_value)

            if self.__debug_mode:
                if module.wants_pause:
                    pause_value = get_image_index(PAUSE)
                else:
                    pause_value = get_image_index(GO)
            else:
                pause_value = get_image_index(NOTDEBUG)
            
            target_item = self.list_ctrl.GetItem(idx)
            if pause_value != target_item.Image:
                self.list_ctrl.SetItemImage(idx, pause_value)

            # skip to first dirty module for validation
            if idx >= self.__first_dirty_module:
                try:
                    module.test_valid(self.__pipeline)
                    target_name = module.module_name
                    ec_value = OK
                except:
                    ec_value = ERROR
                ec_value = get_image_index(ec_value)
                target_item = self.list_ctrl.GetItem(idx, ERROR_COLUMN)
                if ec_value != target_item.Image:
                    self.set_subitem_image(idx, ERROR_COLUMN, ec_value)

        event.RequestMore(False)
        
        self.__first_dirty_module = len(modules)

PIPELINE_DATA_FORMAT = "CellProfiler.Pipeline"
class PipelineDataObject(wx.CustomDataObject):
    def __init__(self):
        super(PipelineDataObject, self).__init__(
            wx.CustomDataFormat(PIPELINE_DATA_FORMAT))
        
class PipelineDropTarget(wx.PyDropTarget):
    def __init__(self, window):
        super(PipelineDropTarget, self).__init__()
        self.window = window
        self.SetDataObject(PipelineDataObject())
        
    def OnDragOver(self, x, y, data):
        if not self.window.provide_drag_feedback(x, y, data):
            return wx.DragNone
        if wx.GetKeyState(wx.WXK_CONTROL) == 0:
            return wx.DragMove
        return wx.DragCopy
    
    def OnDrop(self, x, y):
        return self.window.on_drop(x, y)
    
    def OnData(self, x, y, action):
        if self.GetData():
            data_object = self.GetDataObject()
            self.window.on_data(x, y, action, data_object.GetDataHere(
                wx.CustomDataFormat(PIPELINE_DATA_FORMAT)))
        return action
