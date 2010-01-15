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
import zlib
import wx
import sys

import cellprofiler.pipeline
import cellprofiler.gui.movieslider as cpgmov
from cellprofiler.gui.cpfigure import window_name

NO_PIPELINE_LOADED = 'No pipeline loaded'
PADDING = 1

def plv_get_bitmap(data):
    raw_data = zlib.decompress(base64.b64decode(data))
    return wx.BitmapFromImage(wx.ImageFromStream(StringIO(raw_data)))

IMG_OK = ('eJzrDPBz5+WS4mJgYOD19HAJAtICIMzBBiTlP/9PBFJsSd7uLgz/QXDB3uWT'
          'gSKcBR6RxUDaA4zdTmbbgQTLChzTYfoZYo6tPczAwFTj6eIYYnF67uRYj+BC'
          'iWPzN3/eaJx+eeesXZZ3dcQvi1ycoOqp0dikNvFuRkB5YJorB5ujRqBjrETJ'
          '5VaOsqWcri4h4VWCSU2tKjsvuWRvtcx+rXP+3M1zlrf3cy/lmBx2/u3p938r'
          '5OK4/i/nOXcw/kFEkNt/YbdAhgSzqZK/vKx97pYoZXrOTq3gYpBxLf3z4eHu'
          'ypO/ssUOJlyU9ec+p7/swIUVK+wO9c6vyVJ/+z32SNTJ78wJl7MSTjk5X9Ys'
          '25Z+ydFq+amT77a98WRgtUrbu4/X4ZmYTu7WxXsfG71+X/1W6W6x6ZJvk95q'
          'Jy+Z5bPu68Mtl0/3Wnx/5bfCSUut+weDw92Gytmzra5XXCs4fPfZlq4tovnL'
          'Dz256rAhnkuNXdipi4GhdsHUncW/urZeCZXKnzg1xPWFsKjPygm5U33eH/y8'
          'rNtaOnZm0qkeB4cjXw483zhVPIh/qTNPSvLRlilG4ZW2cfnJO+32e31T2eTZ'
          'faKwXcrq7Nrd38RfJq8SFLTsm3kxoeBf3JLu7RfsdN3q7mrUNPpf9Nw6b5ZQ'
          'u+ZygZ2pWy7u6998s7s40/OpwNUQHZ+WrebOq69PLSiKmHzivZrl6hdbf2Zm'
          '6m4OyD6ne/OclqetRF/7CobAUNcD+kllBU8LU9z28XB99NXjdmRyPTRJ47Rc'
          'tup+9amOyrMCFDS0lIKYipUXPu3n3zc30fdLkJaj2dElzAx23ZPu957erib3'
          'M1xpU8uKRUqrsiLq9DT+qR86++zop+mvvj+ZX8zGoMvYoKDAldV38OxVQe7u'
          'B0k8n50XXjSN//Pq0NtQ0egXz40DeVg1l+y9HnEl+GfOvUuHN+h/8XmxiHcX'
          'w0WOS0EPN8q8WOsTXba1sOf2QbnS72J/pv5knLmR1fPZg4+WwASpWuIaUZJc'
          'lJpYkqqbAiQYjAwMLHQNDXQNLUIMLaxMzKyMTLQNLKwMDBZvvXsapiE3PyUz'
          'rZKAhj8sq/8DNTB4uvq5rHNKaAIAzkFukQ==')

IMG_ERROR = ('eJwBSQK2/YlQTkcNChoKAAAADUlIRFIAAAAQAAAAEAgGAAAAH/P/YQAAAAlwSFlz'
             'AAALEwAACxMBAJqcGAAAAAd0SU1FB9kHGxUkE6NmKioAAAHoSURBVDjLlZPB'
             'ShthFIW/O5nONIkzgxCZSu1GQ8QghkFKmjAEgkhjg3QXuhAiXXTnMBvxCVwb'
             'dGXAQsBFSVYFra9QCD6AUFwECoUWXNRASTF/N9YSyLT2wL2Ly+E7Z3OFCIXw'
             'yIC3AAN43YAv43x6FMCEo3l4DnABR0B1nE+LSPdSsFpYWNCepFJaClZD8O4N'
             'MGEvDXpid5dMs0kadBP27gUIoTgFxWy5LHo+j57Pky2XZQqKIRT5l3agewLD'
             'XqejPM9TnuepXqejTmC4A92/NghhxYWljO+L4fs4joPjOBi+T8b3xYWlEFYi'
             'ASbsZ8FIBAGiaViWhWVZiKaRCAKyYJiwPxYQwvo0pOcKBYxSCRHBtm1s20ZE'
             'MEol5goFpiEdwvoIIAQxoLF4m67pOiLyp4EImq6TCAIWwTCgEYLA7Qrh1Sy0'
             'XiwvG5NnZxCLoZSi3+8DkEwmERG4ueFqbY0P5+eDS6g34J2EEHsMV0/Bmj8+'
             '5mG1ilIKpRS1Wg2AdruNiCAi/Dg95WJjgy58/wyTmsBmHMyZXI54pXJnFBFc'
             '18V13ZFbvFJhJpcjDqbApmzD15eQmj04YKJev0v/PcAIQES4brW43NriPXyT'
             'Qxg+A3nA/+kn8BGU/gkOB/BmIuIvonQNwx40fwGX3n0IHLThEwAAAABJRU5E'
             'rkJggtPN7j0=')

IMG_EYE = ('eJwBCQP2/IlQTkcNChoKAAAADUlIRFIAAAAQAAAAEAgCAAAAkJFoNgAAAAFz'
           'UkdCAK7OHOkAAAAEZ0FNQQAAsY8L/GEFAAAAIGNIUk0AAHomAACAhAAA+gAA'
           'AIDoAAB1MAAA6mAAADqYAAAXcJy6UTwAAAKHSURBVDhPbVJfSFphFD+DGpcx'
           'oocYIXsIJxLSQw+xeuwhzIc9xQiRIREi9yGW2L9LifMhnQwpqSGSI4aN8EFk'
           'RLSl2xCDdlcjb4MkY5MMKxxuRqjVctz9bneDMXb4+Djf9/1+5/zOd8611lZ/'
           'd/edapXq6upqa2v9fv/BQYaohqhiNLLt7e2VSqV0ZTU1tLr6hTjujXhlXq+b'
           '/mf19cz09GMZAzAND0sEGel0Ond2dvL5fDqddrlcbW1tf0cADGCanPxoMj0I'
           'BAI4X1xcyHuhUMhkMvAFQejq6uro6GhpaREEnuPipNN5y+VveCsWi5eXl0Cs'
           'r6/7fD6PxwMoSsIxHo/Pz8+fn59IGUASxZ+5XC6VSi0vL9tsNoPBsLCwAIJe'
           'r1cqlRzHRaPRRCJxevpVIphM4f39XTwjmEajQSpwstkssgWDwZmZmaWlJZZl'
           'NzY2dncFSVJPzxzPI4moVqu1Wu2N6//+lN1u53ne6/VWqyWjcVEibG6u9fb2'
           '3oQxRLf676oXFQpzs2qAGt8rFf0IgJwoA3HHxxPkdgs6XSduofgk95l65nxi'
           'ebTfcBtXc+Ou4KnDNixzEomow8GT1RqFHrksqSFMnJ48bNLfb/Z76N4zzn/8'
           'o/JdVul2Ozo73TQ6+g5FT01NSd96dpYVD4l5SnqOdANNrNRT9AR9RCXb2x9Y'
           '9iVBVij0PBKJhEKhWCwmj8Cj5ObxlTM7OwsoPspqteIIOWSxvIaXTCbD4bDF'
           'YmEYBnI1SmVjQwM6oFKp0LKtrS05EMC/hy+fz5ZKhaOjzN7ep3D4hdlshL+y'
           'EhHFarEoJ/szfBjvsbHY0FBsZORtX19oYmINQhHJbI5g4RJPg4OvsAAD+BeM'
           'WKsmyxjGwwAAAABJRU5ErkJggi0SXAI=')

IMG_CLOSED_EYE = (
    'eJwBsgJN/YlQTkcNChoKAAAADUlIRFIAAAAQAAAAEAgCAAAAkJFoNgAAAARn'
    'QU1BAACxjwv8YQUAAAJpSURBVDhPbVJBSCJhFP4XSsSDePYUrMeIDkHSqUOU'
    'Jw+dPCxDBxHpkljZUBESkyuL1CAiskJIyDKUiIS4trlLmMRsDTYZwRSstBEt'
    'bi4uUaLpMvuNs8iy7ONnePO/73vve+9/LwYHIxMTL9ttotfre3t7I5HIzU2Z'
    'kB5C6hTlHB4ertfrjx3r6SF7e18ITefkjrGsn/zPDAbtxsZrFQMwmZtTCCpy'
    'bW3t4uKiUqlcXl76fL6hoaG/MwAGMGEYwW5/FY1G8d9sNtVvtVotl8vwRVEc'
    'Gxszm839/f2iyNP0AbFY2KenH4jVarVWqwXE0dFROBwOBAKABoPBg45tbm42'
    'Gj+VCiDJ8q/b21tJkra3t6HEZrPt7OwAarVaKYrK5XKpVCqfzz88fFcIdnvi'
    '+lpaX1+fnp5Gmng8zrKsx+OJxWKAwrxeL+4LhYIkiYqkycm3PI8iMgZaLBYh'
    'BiCQS6VSIpFAbohcXV2F//z8QFHvFMLJyeH4+DhiyWSS53nEwISk4+PjUCiU'
    'yWS2tray2awgFBYX88TvFy2WUYwPLQqCgE7S6bSaHl+YOgCXyxUMvsFIidv9'
    'AXoA1Wq1Op1Oo9EwDDMyMtJoNPr6+lCh+xSBADM66icezye1adAMBsM/bz0w'
    'MOB2u+/v71dWVs7OPjudKQJZHBeDeo7jzs/P1RXoGuYLPbu7u6DhEnKIy5WF'
    'd3p6CtEQCmHLy8tGoxHVaJo2mUyYKWagpgD4z/JVKl8fH6t3d+WrK7Qbdzgo'
    '+JlMUpbbtdq3bkFl+bDeCwv7s7P78/Mfp6a4paVDCEUmhyOJg0uEZmbe4wAG'
    '8G8cz71zGGIwnwAAAABJRU5ErkJggqVIPg4=')    

IMG_PAUSE = ('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')


IMG_GO = ('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')

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
    def __init__(self,panel):
        self.__pipeline = None
        self.__panel=panel
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
        self.list_ctrl.Bind(wx.EVT_LEFT_DOWN, self.__on_list_left_down, self.list_ctrl)
        
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
        if isinstance(event,cellprofiler.pipeline.PipelineLoadedEvent):
            self.__on_pipeline_loaded(pipeline,event)
            self.__first_dirty_module = 0
        elif isinstance(event,cellprofiler.pipeline.ModuleAddedPipelineEvent):
            self.__on_module_added(pipeline,event)
            self.__first_dirty_module = min(self.__first_dirty_module, event.module_num - 1)
        elif isinstance(event,cellprofiler.pipeline.ModuleMovedPipelineEvent):
            self.__on_module_moved(pipeline,event)
            self.__first_dirty_module = min(self.__first_dirty_module, event.module_num - 2)
        elif isinstance(event,cellprofiler.pipeline.ModuleRemovedPipelineEvent):
            self.__on_module_removed(pipeline,event)
            self.__first_dirty_module = min(self.__first_dirty_module, event.module_num - 1)
        elif isinstance(event,cellprofiler.pipeline.PipelineClearedEvent):
            self.__on_pipeline_cleared(pipeline, event)
            self.__first_dirty_module = 0
        elif isinstance(event,cellprofiler.pipeline.ModuleEditedPipelineEvent):
            self.__first_dirty_module = min(self.__first_dirty_module, event.module_num - 1)
    
    def notify_directory_change(self):
        # we can't know which modules use this information
        self.__first_dirty_module = 0

    def select_one_module(self, module_num):
        """Select only the given module number in the list box"""
        for module in self.__pipeline.modules():
            self.select_module(module.module_num, module.module_num == module_num)
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
        item, hit_code, subitem = self.list_ctrl.HitTestSubItem(event.Position)
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
                event.Skip()
    
    def __on_pipeline_loaded(self,pipeline,event):
        """Repopulate the list view after the pipeline loads
        
        """
        nrows = len(pipeline.modules())
        self.list_ctrl.DeleteAllItems()
        
        for module in pipeline.modules():
            self.__populate_row(module)
        self.__adjust_rows()
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
        if event.direction == cellprofiler.pipeline.DIRECTION_UP:
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
