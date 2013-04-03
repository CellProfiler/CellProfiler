"""PipelineListView.py

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2013 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

import logging
logger = logging.getLogger(__name__)

try:
    from cStringIO import StringIO
except:
    from StringIO import StringIO
import logging
import numpy as np
import time
import base64
import math
import zlib
import wx
import sys

import cellprofiler.pipeline as cpp
import cellprofiler.gui.movieslider as cpgmov
from cellprofiler.gui.cpfigure import window_name, find_fig
from cellprofiler.icons import get_builtin_image
from cellprofiler.gui.moduleview import request_module_validation

IMG_OK = get_builtin_image('IMG_OK')
IMG_ERROR = get_builtin_image('IMG_ERROR')
IMG_EYE = get_builtin_image('IMG_EYE')
IMG_CLOSED_EYE = get_builtin_image('IMG_CLOSED_EYE')
IMG_PAUSE = get_builtin_image('IMG_PAUSE')
IMG_GO = get_builtin_image('IMG_GO')
IMG_DISABLED = get_builtin_image('IMG_DISABLED')
IMG_UNAVAILABLE = get_builtin_image('IMG_UNAVAILABLE')
BMP_WARNING = wx.ArtProvider.GetBitmap(wx.ART_WARNING,size=(16,16))

NO_PIPELINE_LOADED = 'No pipeline loaded'
PADDING = 1

def plv_get_bitmap(data):
    return wx.BitmapFromImage(data)

PAUSE_COLUMN = 0
EYE_COLUMN = 1
ERROR_COLUMN = 2
MODULE_NAME_COLUMN = 3
NUM_COLUMNS = 4
INPUT_ERROR_COLUMN = 0
INPUT_MODULE_NAME_COLUMN = 1
NUM_INPUT_COLUMNS = 2

ERROR = "error"
WARNING = "warning"
OK = "ok"
DISABLED = "disabled"
EYE = "eye"
CLOSED_EYE = "closedeye"
PAUSE = "pause"
GO = "go"
NOTDEBUG = "notdebug"
UNAVAILABLE = "unavailable"

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
CHECK_FAIL_SEC = 20

class PipelineListView(object):
    """View on a set of modules
    
    Here is the window hierarchy within the panel:
    top_sizer
        input panel
             box sizer
                 "Input" static box
                      static box sizer
                          input module list control
        "Modules" static box
            self.__sizer
                self.__pipeline_slider
                self.list_ctrl
    """
    def __init__(self, panel, frame):
        self.__pipeline = None
        self.__panel=panel
        self.__frame = frame
        self.__module_controls_panel = None
        # Map of ListCtrl.GetItemData value to module.id
        self.__module_dictionary = {}
        self.make_image_list()
        panel.Sizer = top_sizer = wx.BoxSizer(wx.VERTICAL)
        static_box = wx.StaticBox(self.__panel, label = "Create workspace")
        self.__input_controls = [static_box]
        self.__input_sizer = wx.StaticBoxSizer(static_box, wx.VERTICAL)
        top_sizer.Add(self.__input_sizer, 0, wx.EXPAND)
        self.make_input_panel()
        
        modules_box = wx.StaticBox(panel, label = "Analyze images")
        self.__sizer=wx.StaticBoxSizer(modules_box, wx.HORIZONTAL)
        top_sizer.Add(self.__sizer, 1, wx.EXPAND)
        self.__panel.SetAutoLayout(True)
        self.__pipeline_slider = wx.Slider(self.__panel,
                                           size=(20, -1),
                                           style=wx.SL_VERTICAL,
                                           value=0,
                                           minValue=0,
                                           maxValue=1)
        self.__pipeline_slider.SetTickFreq(1, 0)
        self.__pipeline_slider.SetBackgroundColour('white')
        self.__pipeline_slider.Bind(wx.EVT_LEFT_DOWN, self.__on_slider_left_down)
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
        self.__module_being_validated = 0
        self.__submission_time = 0
        self.drag_underway = False
        self.drag_start = None
        self.drag_time = None
        self.list_ctrl.SetDropTarget(PipelineDropTarget(self))
        # panel.SetDropTarget(PanelDropTarget(self))
        panel.SetupScrolling()
        #
        # The following code prevents the panel from scrolling every
        # time it gets the focus - why would anyone ever want this!
        # Remove the code to see something truly horrible.
        #
        # Thank you Mike Conley:
        # http://groups.google.com/group/wxpython-users/browse_thread/thread/5fed262dc3d144bb/2dc29b45d452c8a0?lnk=raot&fwc=2
        def defeat_its_purpose(event):
            event.Skip(False)
        panel.Bind(wx.EVT_CHILD_FOCUS, defeat_its_purpose)

    def make_image_list(self):
        '''Make the image list containing all of the graphic goodies like the eye
        '''
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
                            (GO, IMG_GO),
                            (UNAVAILABLE, IMG_UNAVAILABLE),
                            (DISABLED, IMG_DISABLED)):
            bitmap = plv_get_bitmap(image)
            idx = get_image_index(name)
            d[idx] = bitmap
        d[get_image_index(WARNING)] = BMP_WARNING
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
        
    def make_list(self):
        '''Make the list control with the pipeline items in it'''
        self.list_ctrl = wx.ListCtrl(self.__panel, style = wx.LC_REPORT)
        
        self.list_ctrl.SetImageList(self.image_list, wx.IMAGE_LIST_SMALL)
        self.__sizer.Add(self.list_ctrl, 1, wx.EXPAND | wx.ALL, 2)
        self.list_ctrl.InsertColumn(PAUSE_COLUMN, "")
        self.list_ctrl.InsertColumn(EYE_COLUMN, "")
        self.list_ctrl.InsertColumn(ERROR_COLUMN, "")
        self.list_ctrl.InsertColumn(MODULE_NAME_COLUMN, "Module")
        for column in (PAUSE_COLUMN, EYE_COLUMN, ERROR_COLUMN):
            self.list_ctrl.SetColumnWidth(column, 20)
        self.list_ctrl.SetColumnWidth(MODULE_NAME_COLUMN, -1)
        #
        # Bind events
        #
        self.list_ctrl.Bind(wx.EVT_LIST_ITEM_SELECTED, 
                            self.__on_item_selected, 
                            self.list_ctrl)
        self.list_ctrl.Bind(wx.EVT_LIST_ITEM_DESELECTED, 
                            self.__on_item_deselected, 
                            self.list_ctrl)
        self.input_list_ctrl.Bind(wx.EVT_LIST_ITEM_SELECTED,
                                  self.__on_item_selected, 
                                  self.input_list_ctrl)
        self.list_ctrl.Bind(wx.EVT_LEFT_DOWN, 
                            self.__on_list_left_down, 
                            self.list_ctrl)
        self.list_ctrl.Bind(wx.EVT_LEFT_DCLICK, 
                            self.__on_list_dclick, 
                            self.list_ctrl)
        self.list_ctrl.Bind(wx.EVT_RIGHT_DOWN, 
                            self.__on_list_right_down, 
                            self.list_ctrl)
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
                
    def make_input_panel(self):
        self.input_list_ctrl = wx.ListCtrl(
            self.__panel, style = wx.LC_REPORT | wx.LC_SINGLE_SEL)
        self.__input_controls.append(self.input_list_ctrl)
        self.input_list_ctrl.SetImageList(self.image_list, wx.IMAGE_LIST_SMALL)
        self.__input_sizer.Add(self.input_list_ctrl, 1, wx.EXPAND)
        self.input_list_ctrl.InsertColumn(INPUT_ERROR_COLUMN, "")
        self.input_list_ctrl.InsertColumn(INPUT_MODULE_NAME_COLUMN, "Module")
        #
        # The fake input list control is shown for legacy pipelines. It's
        # disabled to make it tantalizingly unavailable.
        #
        self.fake_input_list_ctrl = wx.ListCtrl(
            self.__panel, style = wx.LC_REPORT | wx.LC_SINGLE_SEL)
        self.__input_controls.append(self.fake_input_list_ctrl)
        self.fake_input_list_ctrl.SetImageList(self.image_list, wx.IMAGE_LIST_SMALL)
        self.__input_sizer.Add(self.fake_input_list_ctrl, 1, wx.EXPAND)
        self.fake_input_list_ctrl.InsertColumn(INPUT_ERROR_COLUMN, "")
        self.fake_input_list_ctrl.InsertColumn(INPUT_MODULE_NAME_COLUMN, "Module")
        self.fake_input_list_ctrl.Enable(False)
        self.fake_input_list_ctrl.SetForegroundColour(
            wx.SystemSettings.GetColour(wx.SYS_COLOUR_GRAYTEXT))
        #
        # You can't display a tooltip over a disabled window. But you can
        # display a tooltip over a transparent window in front of the disabled
        # window. 
        #
        self.transparent_window = wx.Panel(self.__panel)
        def on_background_paint(event):
            assert isinstance(event, wx.EraseEvent)
            dc = event.GetDC()
            assert isinstance(dc, wx.DC)
            # Mostly, this painting activity is for debugging, so you
            # can see how big the control is. But you need to handle
            # the event to keep the control from being painted.
            dc.SetBackgroundMode(wx.TRANSPARENT)
            dc.SetBrush(wx.TRANSPARENT_BRUSH)
            dc.SetPen(wx.TRANSPARENT_PEN)
            r = self.transparent_window.GetRect()
            dc.DrawRectangle(0, 0, r.Width, r.Height)
            return True
        self.transparent_window.Bind(wx.EVT_ERASE_BACKGROUND, on_background_paint)
        def on_fake_size(event):
            assert isinstance(event, wx.SizeEvent)
            self.transparent_window.SetSize(event.Size)
            event.Skip()
            
        self.fake_input_list_ctrl.Bind(wx.EVT_SIZE, on_fake_size)
        def on_fake_move(event):
            assert isinstance(event, wx.MoveEvent)
            self.transparent_window.Move(event.Position)
            event.Skip()
            
        self.fake_input_list_ctrl.Bind(wx.EVT_MOVE, on_fake_move)
        self.transparent_window.SetToolTipString(
            "The current pipeline is a legacy pipeline that does not use these modules")
        
        from cellprofiler.modules.images import Images
        from cellprofiler.modules.metadata import Metadata
        from cellprofiler.modules.namesandtypes import NamesAndTypes
        from cellprofiler.modules.groups import Groups
        for row, module_class in enumerate((
            Images, Metadata, NamesAndTypes, Groups)):
            error_item = wx.ListItem()
            error_item.Mask = wx.LIST_MASK_IMAGE
            error_item.Id = row
            error_item.Image = get_image_index(UNAVAILABLE)
            error_item.Column = INPUT_ERROR_COLUMN
            self.fake_input_list_ctrl.InsertItem(error_item)
            
            module_name_item = wx.ListItem()
            module_name_item.Mask = wx.LIST_MASK_TEXT
            module_name_item.Text = module_class.module_name
            module_name_item.Column = INPUT_MODULE_NAME_COLUMN
            module_name_item.Id = row
            self.fake_input_list_ctrl.SetItem(module_name_item)
        self.fake_input_list_ctrl.SetColumnWidth(INPUT_MODULE_NAME_COLUMN,
                                                 wx.LIST_AUTOSIZE)
        
        
    def show_input_panel(self, show):
        '''Show or hide the controls for input modules
        
        show - True to show the controls, False to hide
        '''
        self.input_list_ctrl.Show(show)
        self.fake_input_list_ctrl.Show(not show)
        self.transparent_window.Show(not show)
        
    def set_debug_mode(self, mode):
        if (mode == True) and (self.__pipeline is not None):
            modules = filter((lambda m:not m.is_input_module()),
                             self.__pipeline.modules())
            if len(modules) > 0:
                self.select_one_module(modules[0].module_num)
        self.__debug_mode = mode
        self.__pipeline_slider.Show(mode)
        self.__sizer.Layout()
        # force a re-check of all modules
        self.__first_dirty_module = 0

    def attach_to_pipeline(self, pipeline, controller):
        """Attach the viewer to the pipeline to allow it to listen for changes
        
        """
        self.__pipeline =pipeline
        self.__controller = controller
        pipeline.add_listener(self.notify)
        controller.attach_to_pipeline_list_view(self)
        
    def set_current_debug_module(self, module):
        assert not module.is_input_module()
        list_ctrl, index = self.get_ctrl_and_index(module)
        self.__pipeline_slider.Value = index
        self.select_one_module(module.module_num)
        
    def reset_debug_module(self):
        '''Set the pipeline slider to the first module to be debugged
        
        Skip the input modules. If there are no other modules, return None,
        otherwise return the first module
        '''
        for module in self.__pipeline.modules():
            if not module.is_input_module():
                self.set_current_debug_module(module)
                return module
        return None
        
    def get_current_debug_module(self, ignore_disabled = True):
        '''Get the current debug module according to the slider'''
        index = self.__pipeline_slider.Value
        if index >= self.list_ctrl.GetItemCount():
            return None
        data_value = self.list_ctrl.GetItemData(index)
        module_id = self.__module_dictionary[data_value]
        for module in self.__pipeline.modules(ignore_disabled):
            if module.id == module_id:
                return module
        return None
    
    def advance_debug_module(self):
        '''Move to the next debug module in the pipeline
        
        returns the module or None if we are at the end
        '''
        index = self.__pipeline_slider.Value
        while True:
            index += 1
            if index >= self.list_ctrl.GetItemCount():
                return None
            module = self.get_module_from_data_value(
                self.list_ctrl.GetItemData(index))
            if module is None:
                return None
            if module.enabled:
                break
        self.__pipeline_slider.Value = index
        self.__pipeline_slider.Refresh()
        self.set_current_debug_module(module)
        return module
        
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
            self.__first_dirty_module = \
                min(self.__first_dirty_module, event.module_num - 1)
        elif isinstance(event,cpp.ModuleMovedPipelineEvent):
            self.__on_module_moved(pipeline,event)
            self.__first_dirty_module = \
                max(0, min(self.__first_dirty_module, event.module_num - 2))
        elif isinstance(event,cpp.ModuleRemovedPipelineEvent):
            self.__on_module_removed(pipeline,event)
            self.__first_dirty_module = \
                min(self.__first_dirty_module, event.module_num - 1)
        elif isinstance(event,cpp.PipelineClearedEvent):
            self.__on_pipeline_cleared(pipeline, event)
            self.__first_dirty_module = 0
        elif isinstance(event,cpp.ModuleEditedPipelineEvent):
            if event.module_num not in [
                x.module_num for x in self.get_selected_modules()]:
                self.select_one_module(event.module_num)
            self.__first_dirty_module = \
                min(self.__first_dirty_module, event.module_num - 1)
        elif isinstance(event, cpp.ModuleEnabledEvent):
            self.__on_module_enabled(event)
            self.__first_dirty_module = min(self.__first_dirty_module, 
                                            event.module.module_num - 1)
        elif isinstance(event, cpp.ModuleDisabledEvent):
            self.__on_module_disabled(event)
            self.__first_dirty_module = min(self.__first_dirty_module, 
                                            event.module.module_num - 1)
    
    def notify_directory_change(self):
        # we can't know which modules use this information
        self.__first_dirty_module = 0
        
    def get_module_data_value(self, module):
        '''Given a module, return the integer that represents it for GetItemData
        
        module - module in question
        
        ListCtrl.GetItemData / SetItemData takes an integer data item. This
        routine returns the integer to use for a given module. Modules are
        identified by module.id which is a GUID. The ID returned is an
        integer suitable for SetItemData - a new one will be generated if
        this is the first time a module has been seen.
        '''
        for data_value, module_id in self.__module_dictionary.iteritems():
            if module_id == module.id:
                break
        else:
            all_values = np.array(sorted(self.__module_dictionary.keys()))
            if len(all_values) > 0 and all_values[0] != 1:
                data_value = 1
            else:
                not_consecutive = np.argwhere(
                    (all_values[:-1] + 1) != all_values[1:])
                if len(not_consecutive) > 0:
                    data_value = not_consecutive.flatten()[0] + 2
                else:
                    data_value = len(all_values) + 1
            self.__module_dictionary[data_value] = module.id
        return data_value
    
    def get_module_from_data_value(self, data_value):
        '''Return the module associated with the value in ListCtrl.GetItemData
        
        data_value - the value from ListCtrl.GetItemData
        '''
        module_id = self.__module_dictionary[data_value]
        modules = filter(lambda x: x.id == module_id,
                         self.__pipeline.modules(False))
        return None if len(modules) != 1 else modules[0]
    
    def iter_list_items(self):
        '''Iterate over the list items in all list controls
        
        yields a tuple of control, index for each item in
        the input and main list controls.
        '''
        for ctrl in (self.input_list_ctrl, self.list_ctrl):
            for i in range(ctrl.GetItemCount()):
                yield ctrl, i
                
    def get_ctrl_and_index(self, module):
        '''Given a module, return its list control and index
        
        module - module to find
        
        returns tuple of list control and index
        
        raises an exception if module is not in proper list control
        '''
        data_value = self.get_module_data_value(module)
        ctrl = (self.input_list_ctrl if module.is_input_module()
                else self.list_ctrl)
        assert isinstance(ctrl, wx.ListCtrl)
        for i in range(ctrl.GetItemCount()):
            if ctrl.GetItemData(i) == data_value:
                return ctrl, i
        raise IndexError("The module, %s, was not found in the list control" %
                         module.module_name)

    def select_one_module(self, module_num):
        """Select only the given module number in the list box"""
        for module in self.__pipeline.modules(False):
            if module.module_num == module_num:
                break
        else:
            logger.warn("Could not find module %d" % module_num)
            for ctrl, idx in self.iter_list_items():
                ctrl.Select(idx, False)
            self.__on_item_selected(None)
            return
        data_value = self.get_module_data_value(module)
        for ctrl, idx in self.iter_list_items():
            ctrl.Select(idx, ctrl.GetItemData(idx) == data_value)
        self.__on_item_selected(None)
        
    def select_module(self, module_num, selected=True):
        """Select the given one-based module number in the list
        This is mostly for testing
        """
        for module in self.__pipeline.modules(False):
            if module.module_num == module_num:
                break
        else:
            logger.warn("Could not find module %d" % module_num)
            return
        ctrl, idx = self.get_ctrl_and_index(module)
        ctrl.Select(idx, selected)
        self.__on_item_selected(None)
        
    def get_selected_modules(self):
        ids = set()
        for ctrl, idx in self.iter_list_items():
            if ctrl.IsSelected(idx):
                ids.add(self.__module_dictionary[ctrl.GetItemData(idx)])
                
        return [module for module in self.__pipeline.modules(False)
                if module.id in ids]
    
    def __on_slider_left_down(self, event):
        '''Handle slider mouse interaction explicitly
        
        Some modules can't be selected because they're disabled. We control
        the slider explicitly so that we can keep the user from selecting
        a disabled module.
        '''
        self.__panel.CaptureMouse()
        def unbind_all():
            self.__panel.Unbind(wx.EVT_LEFT_UP)
            self.__panel.Unbind(wx.EVT_MOTION)
            self.__panel.Unbind(wx.EVT_MOUSE_CAPTURE_LOST)
        
        def on_left_up(event):
            self.__panel.ReleaseMouse()
            unbind_all()
            
        def on_motion(event):
            screen_coords = self.__panel.ClientToScreen(event.Position)
            list_coords = self.list_ctrl.ScreenToClient(screen_coords)
            list_coords.x = self.list_ctrl.GetSize()[0] / 2
            item, hit_code = self.list_ctrl.HitTest(list_coords)
            if (item >= 0 and item < self.list_ctrl.ItemCount and
                (hit_code & wx.LIST_HITTEST_ONITEM)):
                module = self.get_module_from_data_value(
                    self.list_ctrl.GetItemData(item))
                if (module is not None and module.enabled and
                    self.__pipeline_slider.Value != item):
                    self.__pipeline_slider.Value = item
                    self.__pipeline_slider.Refresh()
        
        def on_lost_mouse(event):
            unbind_all()
        self.__panel.Bind(wx.EVT_LEFT_UP, on_left_up)
        self.__panel.Bind(wx.EVT_MOTION, on_motion)
        self.__panel.Bind(wx.EVT_MOUSE_CAPTURE_LOST, on_lost_mouse)
        
    def __on_list_dclick(self, event):
        list_ctrl = event.GetEventObject()
        if sys.platform.startswith("win"):
            item, hit_code, subitem = list_ctrl.HitTestSubItem(event.Position)
        else:
            # Mac's HitTestSubItem does not work. Sorry.
            #
            item, hit_code = list_ctrl.HitTest(event.Position)
            widths = [list_ctrl.GetColumnWidth(i) for i in range(4)]
            start = 0
            for subitem in range(4):
                if event.Position[0] < start + widths[subitem]:
                    break
                start += widths[subitem]
        
        if (item >= 0 and item < list_ctrl.ItemCount and
            (hit_code & wx.LIST_HITTEST_ONITEM) and 
            subitem == MODULE_NAME_COLUMN):
            module = self.get_module_from_data_value(list_ctrl.GetItemData(item))
            name = window_name(module)
            figure = self.__panel.TopLevelParent.FindWindowByName(name)
            if figure is not None:
                figure.Show(0)
                figure.Show(1)
                figure.SetFocus()
    
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
            module = self.get_module_from_data_value(
                self.list_ctrl.GetItemData(item))
            if subitem == PAUSE_COLUMN and self.__debug_mode:
                module.wants_pause = not module.wants_pause
                pause_img = get_image_index(PAUSE if module.wants_pause
                                            else GO)
                self.list_ctrl.SetItemImage(item, pause_img)
            elif subitem == EYE_COLUMN:
                module.show_window = not module.show_window
                eye_img = get_image_index(EYE if module.show_window
                                          else CLOSED_EYE)
                self.set_subitem_image(module, EYE_COLUMN, eye_img)
                name = window_name(module)
                figure = self.__panel.TopLevelParent.FindWindowByName(name)
                if figure is not None:
                    figure.Close()
            elif subitem == ERROR_COLUMN and not module.is_input_module():
                if module.enabled:
                    self.__pipeline.disable_module(module)
                else:
                    self.__pipeline.enable_module(module)
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
            sub_menu = wx.Menu()
            self.__controller.populate_edit_menu(sub_menu)
            menu.AppendSubMenu(sub_menu, "&Add")
            menu.Append(ID_EDIT_DELETE, "&Delete")
            menu.Append(ID_EDIT_DUPLICATE, "Duplicate")
            menu.Append(ID_HELP_MODULE, "&Help")
            self.__frame.PopupMenu(menu)
            menu.Destroy()
        else:
            self.__frame.PopupMenu(self.__frame.menu_edit_add_module)

    def start_drag_operation(self, event):
        '''Start dragging whatever is selected'''
        fd = StringIO()
        modules_to_save = [m.module_num for m in self.get_selected_modules()]
        self.__pipeline.savetxt(fd, modules_to_save, 
                                save_image_plane_details = False)
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
                    for module in self.__pipeline.modules(False):
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

    def provide_panel_drag_feedback(self, x, y, data):
        return self.where_to_drop_panel(x, y) is not None
    
    def on_drop(self, x, y):
        if self.where_to_drop(x,y)  is None:
            return False
        return True

    def on_panel_drop(self, x, y):
        if self.where_to_drop_panel(x, y) is None:
            return False
        return True
    
    def where_to_drop_panel(self, x, y):
        nmodules = self.list_ctrl.GetItemCount()
        if nmodules == 0:
            return 0
        x_screen, y_screen = self.__panel.ClientToScreenXY(x,y)
        x_lv, y_lv = self.list_ctrl.ScreenToClientXY(x_screen, y_screen)
        if y_lv < 0:
            return 0
        elif y_lv >= self.list_ctrl.Rect.Height:
            return nmodules + self.input_list_ctrl.GetItemCount()
        else:
            return None

    def where_to_drop(self, x, y):
        nmodules = self.list_ctrl.GetItemCount()
        if nmodules == 0:
            return self.input_list_ctrl.GetItemCount()
        else:
            last_rect = self.list_ctrl.GetItemRect(nmodules-1)
            last_rect_bottom = last_rect[1] + last_rect[3]
            if last_rect_bottom < y:
                # Below last item. Insert after last
                return nmodules + self.input_list_ctrl.GetItemCount()
            index, code = self.list_ctrl.HitTest(wx.Point(x,y))
            if code & wx.LIST_HITTEST_ONITEM:
                r = self.list_ctrl.GetItemRect(index)
                #
                # Put before or after depending on whether we are more or
                # less than 1/2 of the way
                #
                if y > r[1]+ r[3]/2:
                    index += 1
                return self.input_list_ctrl.GetItemCount() + index
        return None
    
    def on_data(self, x, y, action, data):
        index = self.where_to_drop(x,y)
        if index is not None:
            self.do_drop(index, action, data)

    def on_panel_data(self, x, y, action, data):
        index = self.where_to_drop_panel(x,y)
        if index is not None:
            self.do_drop(index, action, data)

    def do_drop(self, index, action, data):
        #
        # Insert the new modules
        #
        wx.BeginBusyCursor()
        try:
            pipeline = cpp.Pipeline()
            pipeline.load(StringIO(data))
            for i, module in enumerate(pipeline.modules(False)):
                module.module_num = i + index + 1
                self.__pipeline.add_module(module)
            for i in range(len(pipeline.modules(False))):
                item = self.list_ctrl.SetItemState(
                    i+index, wx.LIST_STATE_SELECTED, wx.LIST_STATE_SELECTED)
        finally:
            wx.EndBusyCursor()
                    
    def __on_pipeline_loaded(self,pipeline, event):
        """Repopulate the list view after the pipeline loads
        
        """
        self.resetItems(pipeline)
        
    def resetItems(self, pipeline):
        '''Reset the list view and repopulate the list items'''
        self.list_ctrl.DeleteAllItems()
        self.input_list_ctrl.DeleteAllItems()
        self.__module_dictionary = {}
        assert isinstance(pipeline, cpp.Pipeline)
        
        for module in pipeline.modules(False):
            self.__populate_row(module)
        self.__adjust_rows()
        self.__panel.SetupScrolling()
        self.__controller.enable_module_controls_panel_buttons()
    
    def __adjust_rows(self):
        """Adjust slider and dimensions after adding or removing rows"""
        if self.__pipeline is not None:
            for module in self.__pipeline.modules(False):
                if module.is_input_module():
                    self.show_input_panel(True)
                    break
            else:
                self.show_input_panel(False)
        for list_ctrl, column in (
            (self.list_ctrl, MODULE_NAME_COLUMN),
            (self.input_list_ctrl, INPUT_MODULE_NAME_COLUMN)):
            if list_ctrl.GetItemCount() > 0:
                list_ctrl.SetColumnWidth(column, wx.LIST_AUTOSIZE)
            else:
                list_ctrl.SetColumnWidth(
                    column, wx.LIST_AUTOSIZE_USEHEADER)
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
        if self.input_list_ctrl.ItemCount > 0:
            input_list_ctrl = self.input_list_ctrl
        else:
            input_list_ctrl = self.fake_input_list_ctrl
        r = input_list_ctrl.GetItemRect(0, wx.LIST_RECT_BOUNDS)
        height = r[3]
        y = r[1]
        min_width = input_list_ctrl.GetMinWidth()
        min_height = y + max(1, input_list_ctrl.GetItemCount()) * height + 4
        input_list_ctrl.SetMinSize((min_width, min_height))
        self.__panel.Layout()
        self.__panel.SetupScrolling(scroll_x=False, scroll_y=True, scrollToTop=False)
    
    def set_subitem_image(self, module, column, image_number):
        list_ctrl, index = self.get_ctrl_and_index(module)
        if column == 0:
            list_ctrl.SetItemImage(index, image_number)
        item = wx.ListItem()
        item.Mask = wx.LIST_MASK_IMAGE
        item.Image = image_number
        item.Id = index
        item.Column = column
        list_ctrl.SetItem(item)
        
    def __populate_row(self, module):
        """Populate a row in the grid with a module."""
        list_ctrl = (self.input_list_ctrl if module.is_input_module()
                     else self.list_ctrl)
        for row in range(list_ctrl.GetItemCount()):
            other_module = self.get_module_from_data_value(
                list_ctrl.GetItemData(row))
            if other_module.module_num > module.module_num:
                break
        else:
            row = list_ctrl.GetItemCount()
        data_value = self.get_module_data_value(module)
        if module.is_input_module():
            error_item = wx.ListItem()
            error_item.Mask = wx.LIST_MASK_IMAGE | wx.LIST_MASK_DATA
            error_item.Column = INPUT_ERROR_COLUMN
            error_item.Id = row
            error_item.Data = data_value
            self.input_list_ctrl.InsertItem(error_item)
            
            module_name_item = wx.ListItem()
            module_name_item.Mask = wx.LIST_MASK_TEXT
            module_name_item.Text = module.module_name
            module_name_item.Column = INPUT_MODULE_NAME_COLUMN
            module_name_item.Id = row
            self.input_list_ctrl.SetItem(module_name_item)
        else:
            pause_item = wx.ListItem()
            pause_item.Mask = wx.LIST_MASK_IMAGE | wx.LIST_MASK_DATA
            pause_item.Image = get_image_index(NOTDEBUG)
            pause_item.Column = PAUSE_COLUMN
            pause_item.Id = row
            pause_item.Data = data_value
            self.list_ctrl.InsertItem(pause_item)
            
            self.set_subitem_image(
                module, EYE_COLUMN,
                get_image_index(EYE if module.show_window else CLOSED_EYE))
            self.set_subitem_image(
                module, ERROR_COLUMN, 
                get_image_index(OK if module.enabled else DISABLED))
            
            module_name_item = wx.ListItem()
            module_name_item.Mask = wx.LIST_MASK_TEXT
            module_name_item.Text = module.module_name
            module_name_item.Column = MODULE_NAME_COLUMN
            module_name_item.Id = row
            self.list_ctrl.SetItem(module_name_item)
        
    def __on_pipeline_cleared(self, pipeline, event):
        self.resetItems(pipeline)
        
    def __on_module_added(self,pipeline, event):
        module = pipeline.modules(False)[event.module_num - 1]
        self.__populate_row(module)
        self.__adjust_rows()
        self.select_one_module(event.module_num)
        self.__panel.SetupScrolling(scrollToTop=False)

    def __on_module_removed(self, pipeline, event):
        all_module_ids = set([module.id for module in pipeline.modules(False)])
        missing_modules = []
        for list_ctrl, idx in self.iter_list_items():
            data_value = list_ctrl.GetItemData(idx)
            if self.__module_dictionary[data_value] not in all_module_ids:
                missing_modules.append((list_ctrl, idx, data_value))
        for list_ctrl, idx, data_value in reversed(missing_modules):
            list_ctrl.DeleteItem(idx)
            del self.__module_dictionary[data_value]
        self.__adjust_rows()
        self.__controller.enable_module_controls_panel_buttons()
        self.__panel.SetupScrolling(scrollToTop=False)
        
    def __on_module_moved(self,pipeline,event):
        module = pipeline.modules(False)[event.module_num - 1]
        list_ctrl, index = self.get_ctrl_and_index(module)
        if event.direction == cpp.DIRECTION_UP:
            # if this module was moved up, the one before it was moved down
            # and is now after
            other_module = pipeline.modules(False)[event.module_num]
            modules = [module, other_module]
        else:
            # if this module was moved down, the one after it was moved up
            # and is now before
            other_module = pipeline.modules(False)[event.module_num - 2]
            modules = [other_module, module]
            
        other_list_ctrl, other_index = self.get_ctrl_and_index(other_module)
        if other_list_ctrl != list_ctrl:
            # in different list controls, nothing changes
            return
        start = min(index, other_index)
        first_selected = list_ctrl.IsSelected(start)
        second_selected = list_ctrl.IsSelected(start+1)
        list_ctrl.DeleteItem(start)
        list_ctrl.DeleteItem(start)
        for m in modules:
            self.__populate_row(m)
        list_ctrl.Select(start, second_selected)
        list_ctrl.Select(start+1, first_selected)
        self.__adjust_rows()
        self.__controller.enable_module_controls_panel_buttons()
        list_ctrl.Refresh()
    
    def __on_module_enabled(self, event):
        self.set_subitem_image(event.module, ERROR_COLUMN, get_image_index(OK))
        if self.__debug_mode:
            self.set_subitem_image(
                event.module, PAUSE_COLUMN,
                get_image_index(PAUSE if event.module.wants_pause else GO))
    
    def __on_module_disabled(self, event):
        if event.module == self.get_current_debug_module(False):
            # Must change the current debug module to something enabled
            for module in self.__pipeline.modules():
                if module.module_num > event.module.module_num:
                    self.set_current_debug_module(module)
                    break
            else:
                for module in reversed(self.__pipeline.modules()):
                    if module.module_num < event.module.module_num:
                        self.set_current_debug_module(module)
                        break
                else:
                    self.__controller.stop_debugging()
        self.set_subitem_image(event.module, ERROR_COLUMN, 
                               get_image_index(DISABLED))
        self.set_subitem_image(event.module, PAUSE_COLUMN,
                               get_image_index(NOTDEBUG))
    
    def __on_item_selected(self, event):
        if isinstance(event, wx.Event):
            def clear_all_selections(ctrl):
                for i in range(ctrl.GetItemCount()):
                    ctrl.Select(i, False)
            if event.GetEventObject() == self.input_list_ctrl:
                clear_all_selections(self.list_ctrl)
            else:
                clear_all_selections(self.input_list_ctrl)
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
        for module in self.__pipeline.modules(False):
            for module_setting in module.settings():
                if setting is module_setting:
                    list_ctrl, index = self.get_ctrl_and_index(module)
                    if (not module.is_input_module() and
                        self.__pipeline_slider.Value > index):
                        self.__pipeline_slider.Value = index
                    return
                
    def on_stop_debugging(self):
        self.__pipeline_slider.Value = 0
        
    def on_idle(self,event):
        last_idle_time = getattr(self, "last_idle_time", 0)
        if time.time() - last_idle_time > CHECK_TIMEOUT_SEC:
            self.last_idle_time = time.time()
        else:
            return

        if self.__pipeline is None:
            event.RequestMore(False)
            return
        request_more = True
        modules = self.__pipeline.modules()
        for i, module in enumerate(modules):
            # skip to first dirty module for validation
            if (module.module_num > self.__first_dirty_module and
                 self.__submission_time + CHECK_FAIL_SEC < time.time()):
                pipeline_hash = self.__pipeline.settings_hash()
                def fn(setting_idx, message, level,
                       idx=i, settings_hash = pipeline_hash):
                    self.on_validate_module(setting_idx, message, level, idx, settings_hash)
                self.__module_being_validated = i
                self.__submission_time = time.time()
                request_module_validation(self.__pipeline, module, fn)
                
            if module.is_input_module():
                continue

            list_ctrl, idx = self.get_ctrl_and_index(module)
            if module.show_window:
                eye_value = get_image_index(EYE)
            else:
                eye_value = get_image_index(CLOSED_EYE)
            target_item = list_ctrl.GetItem(idx, EYE_COLUMN)
            if target_item.Image != eye_value:
                self.set_subitem_image(module, EYE_COLUMN, eye_value)

            if self.__debug_mode:
                if not module.enabled:
                    pause_value = get_image_index(NOTDEBUG)
                elif module.wants_pause:
                    pause_value = get_image_index(PAUSE)
                else:
                    pause_value = get_image_index(GO)
            else:
                pause_value = get_image_index(NOTDEBUG)
            
            target_item = list_ctrl.GetItem(idx)
            if pause_value != target_item.Image:
                list_ctrl.SetItemImage(idx, pause_value)

        event.RequestMore(False)

    def on_validate_module(self, setting_idx, message, level,
                           idx, settings_hash):
        if settings_hash != self.__pipeline.settings_hash():
            self.__submission_time = 0
            return
            
        module = self.__pipeline.modules()[idx]
        error_column = (INPUT_ERROR_COLUMN if module.is_input_module() 
                        else ERROR_COLUMN)
        list_ctrl, index = self.get_ctrl_and_index(module)
        target_item = list_ctrl.GetItem(index, error_column)
        if level == logging.WARNING:
            ec_value = WARNING
        elif level == logging.ERROR:
            ec_value = ERROR
        else:
            ec_value = OK
        ec_value = get_image_index(ec_value)
        if ec_value != target_item.Image:
            self.set_subitem_image(module, error_column, ec_value)
        if self.__first_dirty_module == idx:
            self.__first_dirty_module = min(self.__first_dirty_module+1,
                                            len(self.__pipeline.modules()))
            self.last_idle_time = 0
            self.__submission_time = 0
            wx.PostEvent(self.__panel, wx.IdleEvent())

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

class PanelDropTarget(wx.PyDropTarget):
    def __init__(self, window):
        super(PanelDropTarget, self).__init__()
        self.window = window
        self.SetDataObject(PipelineDataObject())
        
    def OnDragOver(self, x, y, data):
        if not self.window.provide_panel_drag_feedback(x, y, data):
            return wx.DragNone
        if wx.GetKeyState(wx.WXK_CONTROL) == 0:
            return wx.DragMove
        return wx.DragCopy
    
    def OnDrop(self, x, y):
        return self.window.on_panel_drop(x, y)
    
    def OnData(self, x, y, action):
        if self.GetData():
            data_object = self.GetDataObject()
            self.window.on_panel_data(x, y, action, data_object.GetDataHere(
                wx.CustomDataFormat(PIPELINE_DATA_FORMAT)))
        return action
