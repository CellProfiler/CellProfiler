import wx

from cellprofiler.gui import draw_item_selection_rect
from cellprofiler.gui.constants.pipeline_list_view import BMP_WARNING
from cellprofiler.gui.constants.pipeline_list_view import ERROR
from cellprofiler.gui.constants.pipeline_list_view import ERROR_COLUMN
from cellprofiler.gui.constants.pipeline_list_view import EVT_PLV_ERROR_COLUMN_CLICKED
from cellprofiler.gui.constants.pipeline_list_view import EVT_PLV_EYE_COLUMN_CLICKED
from cellprofiler.gui.constants.pipeline_list_view import EVT_PLV_PAUSE_COLUMN_CLICKED
from cellprofiler.gui.constants.pipeline_list_view import EVT_PLV_SLIDER_MOTION
from cellprofiler.gui.constants.pipeline_list_view import EVT_PLV_STEP_COLUMN_CLICKED
from cellprofiler.gui.constants.pipeline_list_view import EYE_COLUMN
from cellprofiler.gui.constants.pipeline_list_view import IMG_CLOSED_EYE
from cellprofiler.gui.constants.pipeline_list_view import IMG_DISABLED
from cellprofiler.gui.constants.pipeline_list_view import IMG_DOWNARROW
from cellprofiler.gui.constants.pipeline_list_view import IMG_ERROR
from cellprofiler.gui.constants.pipeline_list_view import IMG_EYE
from cellprofiler.gui.constants.pipeline_list_view import IMG_GO
from cellprofiler.gui.constants.pipeline_list_view import IMG_OK
from cellprofiler.gui.constants.pipeline_list_view import IMG_PAUSE
from cellprofiler.gui.constants.pipeline_list_view import IMG_SLIDER
from cellprofiler.gui.constants.pipeline_list_view import IMG_SLIDER_ACTIVE
from cellprofiler.gui.constants.pipeline_list_view import IMG_STEP
from cellprofiler.gui.constants.pipeline_list_view import IMG_STEPPED
from cellprofiler.gui.constants.pipeline_list_view import IMG_UNAVAILABLE
from cellprofiler.gui.constants.pipeline_list_view import MODULE_NAME_COLUMN
from cellprofiler.gui.constants.pipeline_list_view import OK
from cellprofiler.gui.constants.pipeline_list_view import PAUSE_COLUMN
from cellprofiler.gui.constants.pipeline_list_view import PLV_HITTEST_SLIDER
from cellprofiler.gui.constants.pipeline_list_view import PLV_STATE_ERROR
from cellprofiler.gui.constants.pipeline_list_view import PLV_STATE_PROCEED
from cellprofiler.gui.constants.pipeline_list_view import PLV_STATE_UNAVAILABLE
from cellprofiler.gui.constants.pipeline_list_view import PLV_STATE_WARNING
from cellprofiler.gui.constants.pipeline_list_view import STEP_COLUMN
from cellprofiler.gui.constants.pipeline_list_view import WARNING
from cellprofiler.gui.utilities.pipeline_list_view import plv_get_bitmap


class PipelineListCtrl(wx.ScrolledWindow):
    """A custom widget for the pipeline module list"""

    class PipelineListCtrlItem(object):
        """An item in a pipeline list control"""

        def __init__(self, module):
            self.module = module
            self.__state = PLV_STATE_PROCEED
            self.tooltip = ""

        @property
        def module_name(self):
            """The module name of the item's module"""
            return self.module.module_name

        def set_state(self, state, state_mask):
            """Set the item's state

            state - the state bit values to set

            state_mask - the mask indicating which state bits to set.
            """
            self.__state = (self.__state & ~state_mask) | (state & state_mask)

        def is_selected(self):
            """Return True if item is selected"""
            return bool(self.__state & wx.LIST_STATE_SELECTED)

        selected = property(is_selected)

        def select(self, value=True):
            self.set_state(
                wx.LIST_STATE_SELECTED if value else 0, wx.LIST_STATE_SELECTED
            )

        def get_error_state(self):
            """Return the error state: ERROR, WARNING or OK"""
            if self.__state & PLV_STATE_ERROR:
                return ERROR
            if self.__state & PLV_STATE_WARNING:
                return WARNING
            return OK

        def can_proceed(self):
            """Return True if the pipeline can proceed past this module

            This is the state of the PLV_STATE_PROCEED flag. The pipeline
            might not be able to proceed because of an error or warning as
            well.
            """
            return (self.__state & PLV_STATE_PROCEED) == PLV_STATE_PROCEED

        error_state = property(get_error_state)

        def is_shown(self):
            """The module's frame should be shown if True"""
            return self.module.show_window

        def is_enabled(self):
            """The module should not be executed"""
            return self.module.enabled

        enabled = property(is_enabled)

        def is_paused(self):
            """The module is breakpointed in test mode"""
            return self.module.wants_pause

        def is_unavailable(self):
            """The module is unavailable = not in pipeline"""
            return bool(self.__state & PLV_STATE_UNAVAILABLE)

    def __init__(self, *args, **kwargs):
        super(PipelineListCtrl, self).__init__(*args, **kwargs)
        self.bmp_ok = plv_get_bitmap(IMG_OK)
        self.bmp_error = plv_get_bitmap(IMG_ERROR)
        self.bmp_eye = plv_get_bitmap(IMG_EYE)
        self.bmp_closed_eye = plv_get_bitmap(IMG_CLOSED_EYE)
        self.bmp_step = plv_get_bitmap(IMG_STEP)
        self.bmp_stepped = plv_get_bitmap(IMG_STEPPED)
        self.bmp_go = plv_get_bitmap(IMG_GO)
        self.bmp_pause = plv_get_bitmap(IMG_PAUSE)
        self.bmp_unavailable = plv_get_bitmap(IMG_UNAVAILABLE)
        self.bmp_disabled = plv_get_bitmap(IMG_DISABLED)
        self.bmp_slider = plv_get_bitmap(IMG_SLIDER)
        self.bmp_slider_active = plv_get_bitmap(IMG_SLIDER_ACTIVE)
        self.bmp_downarrow = plv_get_bitmap(IMG_DOWNARROW)
        # The items to display
        self.items = []
        # The current or active item (has wx.CONTROL_CURRENT display flag)
        self.active_item = None
        # The anchor of an extended selection
        self.anchor = None
        # The item highlighted by the test mode slider
        self.running_item = 0
        # The last module in the pipeline that has been run on this image set
        self.last_running_item = 0
        # True if in test mode
        self.test_mode = False
        # True if allowed to disable a module
        self.allow_disable = True
        # True to show the step column, false to hide
        self.show_step = True
        # True to show the pause column, false to hide
        self.show_go_pause = True
        # True to show the eyeball column, false to hide it
        self.show_show_frame_column = True
        # Space reserved for test mode slider
        self.slider_width = 10
        # The index at which a drop would take place, if dropping.
        self.drop_insert_point = None
        # Shading border around buttons
        self.border = 1
        # Gap between buttons
        self.gap = 1
        # Gap before and after text
        self.text_gap = 3
        # The height of one row in the display
        self.line_height = max(
            16 + self.border + self.gap, self.GetFullTextExtent("M")[1]
        )
        # The width of a icon column
        self.column_width = self.line_height
        # The row # of the currently pressed icon button
        self.pressed_row = None
        # The column of the currently pressed icon button
        self.pressed_column = None
        # True if the currently pressed button will fire when the mouse button
        # is released.
        self.button_is_active = False
        # True if the slider is currently being slid by a mouse capture action
        self.active_slider = False
        # True to draw the current item as if it were selected, even when
        # it's not.
        self.always_draw_current_as_if_selected = False
        # A pen to use to draw shadow edges on buttons
        self.shadow_pen = wx.Pen(wx.SystemSettings.GetColour(wx.SYS_COLOUR_3DSHADOW))
        # A pen to use to draw lighted edges on buttons
        self.light_pen = wx.Pen(wx.SystemSettings.GetColour(wx.SYS_COLOUR_3DLIGHT))
        self.SetScrollRate(self.line_height, self.line_height)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
        self.Bind(wx.EVT_LEFT_UP, self.on_left_up)
        self.Bind(wx.EVT_MOTION, self.on_mouse_move)
        self.Bind(wx.EVT_KEY_DOWN, self.on_key_down)
        self.Bind(wx.EVT_MOUSE_CAPTURE_LOST, self.on_capture_lost)
        self.Bind(wx.EVT_KILL_FOCUS, self.on_focus_change)
        self.Bind(wx.EVT_SET_FOCUS, self.on_focus_change)

    def AdjustScrollbars(self):
        self.SetScrollbars(1, self.line_height, self.BestSize[0], len(self.items))

    def on_focus_change(self, event):
        self.Refresh(eraseBackground=False)

    def HitTestSubItem(self, position):
        """Mimic ListCtrl's HitTestSubItem

        position - x, y position of mouse click

        returns item hit, hit code (same as ListCtrl's) and column #
        (None if no hit or hit the slider, or PAUSE_COLUMN, EYE_COLUMN,
         ERROR_COLUMN or MODULE_NAME_COLUMN)

         Hit codes:
         wx.LIST_HITTEST_NOWHERE - in the slider column, but not on the slider
         wx.LIST_HITTEST_BELOW - below any item (item #, column # not valid)
         wx.LIST_HITTEST_ONITEMLABEL - on the module name of the item
         wx.LIST_HITTEST_ONITEMICON - on one of the icons
         PLV_HITTEST_SLIDER - on the slider handle.
        """
        position = self.CalcUnscrolledPosition(position)
        x = position[0]
        y = position[1]
        x0 = self.slider_width
        if self.test_mode:
            if x < x0:
                r = self.get_slider_rect()
                if r.Contains(x, y):
                    return None, PLV_HITTEST_SLIDER, None
                return None, wx.LIST_HITTEST_NOWHERE, None
        elif x < x0:
            return None, wx.LIST_HITTEST_NOWHERE, None
        column = int((x - x0) / self.column_width)
        if not (self.show_go_pause and self.test_mode) and column == PAUSE_COLUMN:
            return None, wx.LIST_HITTEST_NOWHERE, None
        if not (self.show_go_pause and self.test_mode) and column == PAUSE_COLUMN:
            return None, wx.LIST_HITTEST_NOWHERE, None
        if not self.show_show_frame_column and column == EYE_COLUMN:
            return None, wx.LIST_HITTEST_NOWHERE, None
        row = int(y / self.line_height)
        if row >= len(self.items):
            return None, wx.LIST_HITTEST_BELOW, None
        if column < 4:
            return row, wx.LIST_HITTEST_ONITEMICON, column
        return row, wx.LIST_HITTEST_ONITEMLABEL, MODULE_NAME_COLUMN

    def HitTest(self, position):
        """Mimic ListCtrl's HitTest

        position - x, y position of mouse click

        returns hit code (see HitTestSubItem)
        """
        return self.HitTestSubItem(position)[:2]

    def GetSelectedItemCount(self):
        """Return the # of items selected"""
        return len([True for item in self.items if item.selected])

    SelectedItemCount = property(GetSelectedItemCount)

    def GetItemCount(self):
        """Return the total # of items

        """
        return len(self.items)

    ItemCount = property(GetItemCount)

    def SetItemState(self, item, state, state_mask):
        """Set the state for the given item

        item - index of item

        state - one of the list states

        """
        self.items[item].set_state(state, state_mask)
        self.Refresh(eraseBackground=False)

    def SetItemErrorToolTipString(self, item, text):
        """Set the string to display when hovering over the error indicator"""
        self.items[item].tooltip = text

    def Select(self, index, state):
        """Mark the item as selected or not

        index - index of the item to mark as selected

        state - True to be selected, False for not
        """
        self.items[index].select(state)
        if state:
            plv_event = self.make_event(wx.EVT_LIST_ITEM_SELECTED, index)
        else:
            plv_event = self.make_event(wx.EVT_LIST_ITEM_DESELECTED, index)
        self.GetEventHandler().ProcessEvent(plv_event)
        self.Refresh(eraseBackground=False)

    def CanSelect(self):
        """Return True if selection is allowed in this control"""
        return self.allow_disable

    def SelectAll(self):
        """Select all modules"""
        if self.CanSelect():
            for i in range(self.GetItemCount()):
                if not self.IsSelected(i):
                    self.Select(i, True)

    def IsSelected(self, index):
        """Return True if the item at the given index is selected"""
        return bool(self.items[index].is_selected())

    def DeleteAllItems(self):
        self.items = []
        self.active_item = None
        self.anchor = None
        self.running_item = None
        self.pressed_row = None
        self.pressed_column = None
        self.Refresh(eraseBackground=False)

    def DeleteItem(self, index):
        """Remove the item at the given index"""
        del self.items[index]

        if self.active_item is not None:
            if self.active_item == index:
                self.active_item = None
            elif self.active_item > index:
                self.active_item -= 1

        if self.anchor:
            if self.anchor == index:
                self.anchor = None
            elif self.anchor > index:
                self.anchor -= 1

        if self.running_item:
            if self.running_item > index:
                self.running_item -= 1

        self.AdjustScrollbars()

        self.Refresh(eraseBackground=False)

    def InsertItem(self, index, item):
        """Insert an item at the given index

        index - the item will appear at this slot on the list.

        item - a PipelineListCtrlItem
        """
        self.items.insert(index, item)

        if self.active_item:
            if self.active_item >= index:
                self.active_item += 1

        if self.anchor:
            if self.anchor >= index:
                self.anchor += 1

        if self.running_item:
            if self.running_item >= index:
                self.running_item += 1

        self.AdjustScrollbars()

        self.SetInitialSize(self.GetBestSize())
        self.Refresh(eraseBackground=False)

    def get_insert_index(self, position):
        """Return the index for an insert into the item list, based on position

        position - position of the mouse in the window

        returns the index that should be used when inserting a new item into
        the list.
        """
        position = self.CalcUnscrolledPosition(position)
        y = position[1]
        row = int((y + self.line_height / 2) / self.line_height)
        return max(0, min(row, self.ItemCount))

    def update_drop_insert_point(self, index):
        """Change the visual drop indication

        index - the first dropped module would end up here.
        """
        self.drop_insert_point = index
        self.Refresh(eraseBackground=False)

    def end_drop(self):
        """Remove the visual drop indication"""
        self.drop_insert_point = None
        self.Refresh(eraseBackground=False)

    def get_index(self, position):
        """Return the index of the item at the given position"""
        position = self.CalcUnscrolledPosition(position)
        y = position[1]
        row = int(y / self.line_height)
        return max(0, min(row, self.ItemCount - 1))

    def get_active_item(self):
        """Return the active PipelineListCtrlItem or None"""
        if self.active_item is not None:
            return self.items[self.active_item]
        return None

    def set_test_mode(self, mode):
        self.test_mode = mode
        self.AdjustScrollbars()
        self.Refresh(eraseBackground=False)

    def set_show_step(self, state):
        """Show or hide the go / pause test-mode icons

        state - true to show them, false to hide them
        """
        self.show_step = state
        self.AdjustScrollbars()
        self.Refresh(eraseBackground=False)

    def set_show_go_pause(self, state):
        """Show or hide the go / pause test-mode icons

        state - true to show them, false to hide them
        """
        self.show_go_pause = state
        self.AdjustScrollbars()
        self.Refresh(eraseBackground=False)

    def set_show_frame_column(self, state):
        """Show or hide the show / hide frame column

        state - True to show, False to hide
        """
        self.show_show_frame_column = state
        self.AdjustScrollbars()
        self.Refresh(eraseBackground=False)

    def set_allow_disable(self, state):
        """Allow disabling of modules / disallow disabling of modules"""
        self.allow_disable = state
        self.Refresh(eraseBackground=False)

    def set_running_item(self, idx):
        """The index of the next module to run in test mode"""
        self.running_item = idx
        if self.running_item > self.last_running_item:
            self.last_running_item = self.running_item
        self.Refresh(eraseBackground=False)

    def DoGetBestSize(self):
        x0 = self.column_width * 3 + self.slider_width + self.text_gap
        max_width = 0
        for i, item in enumerate(self.items):
            width, height, _, _ = self.GetFullTextExtent(item.module_name)
            max_width = max(width, max_width)
        total_width = x0 + max_width + self.border * 2 + self.gap + self.text_gap
        height = max((len(self.items) - 1) * self.line_height, 0)
        return total_width, height

    def DoGetVirtualSize(self):
        return self.DoGetBestSize()

    def AcceptsFocus(self):
        return True

    def get_button_rect(self, column, idx):
        x0 = column * self.column_width + self.slider_width
        if self.pressed_column == column and self.pressed_row == idx:
            x0 += 1
            y0 = 1
        else:
            y0 = 0
        return wx.Rect(x0 + self.gap, y0 + idx * self.line_height, 16, 16)

    def get_step_rect(self, idx):
        return self.get_button_rect(0, idx)

    def get_go_pause_rect(self, idx):
        return self.get_button_rect(1, idx)

    def get_eye_rect(self, idx):
        return self.get_button_rect(2, idx)

    def get_error_rect(self, idx):
        return self.get_button_rect(3, idx)

    def get_text_rect(self, index):
        x0 = self.slider_width + self.column_width * 4 + self.text_gap
        return wx.Rect(
            x0 + self.gap,
            index * self.line_height,
            self.GetSize()[0] - x0 - self.gap,
            self.line_height,
        )

    def get_slider_rect(self):
        """Return the rectangle encompassing the slider bitmap

        returns None if not displayed, otherwise the coordinates of the bitmap
        """
        if not self.test_mode:
            return None
        top = (
            self.line_height - self.bmp_slider.GetHeight()
        ) / 2 + self.line_height * self.running_item
        return wx.Rect(1, top, self.bmp_slider.GetWidth(), self.bmp_slider.GetHeight())

    def on_paint(self, event):
        assert isinstance(event, wx.PaintEvent)

        dc = wx.BufferedPaintDC(self)

        self.PrepareDC(dc)

        dc.SetBackground(wx.Brush(wx.SystemSettings.GetColour(wx.SYS_COLOUR_LISTBOX)))

        dc.Clear()

        text_color = wx.SystemSettings.GetColour(wx.SYS_COLOUR_LISTBOXTEXT)

        text_color_selected = wx.SystemSettings.GetColour(
            wx.SYS_COLOUR_LISTBOXHIGHLIGHTTEXT
        )

        for index, item in enumerate(self.items):
            item_text_color = text_color

            dc.SetFont(self.Font)

            if self.show_step and self.test_mode:
                rectangle = self.get_step_rect(index)
                bitmap = (
                    self.bmp_step if self.running_item == index else self.bmp_stepped
                )
                if self.running_item >= index and item.enabled:
                    dc.DrawBitmap(bitmap, rectangle.GetLeft(), rectangle.GetTop(), True)

            if self.show_go_pause and self.test_mode:
                rectangle = self.get_go_pause_rect(index)
                bitmap = self.bmp_pause if item.is_paused() else self.bmp_go

                dc.DrawBitmap(bitmap, rectangle.GetLeft(), rectangle.GetTop(), True)

            if self.show_show_frame_column:
                rectangle = self.get_eye_rect(index)

                bitmap = self.bmp_eye if item.is_shown() else self.bmp_closed_eye

                dc.DrawBitmap(bitmap, rectangle.GetLeft(), rectangle.GetTop(), True)

            rectangle = self.get_error_rect(index)

            bitmap = (
                self.bmp_unavailable
                if item.is_unavailable()
                else self.bmp_disabled
                if not item.enabled
                else self.bmp_error
                if item.error_state == ERROR
                else BMP_WARNING
                if item.error_state == WARNING
                else self.bmp_ok
            )

            dc.DrawBitmap(bitmap, rectangle.GetLeft(), rectangle.GetTop(), True)

            rectangle = self.get_text_rect(index)

            flags = 0 if self.Enabled else wx.CONTROL_DISABLED
            font = self.Font

            if item.selected:
                flags |= wx.CONTROL_SELECTED
                item_text_color = text_color_selected
                font = font.MakeBold()

            if self.active_item == index:
                flags |= wx.CONTROL_CURRENT
                font = font.MakeBold()

                if self.always_draw_current_as_if_selected:
                    flags |= wx.CONTROL_SELECTED
                    item_text_color = text_color_selected

            if self.FindFocus() is self:
                if item.selected or self.active_item == index:
                    flags |= wx.CONTROL_FOCUSED
            else:
                # On Windows, the highlight color is white. If focus is lost, the background color is light grey.
                # These colors together makes the font very difficult to read. The default text color is dark. Let's
                # use it instead.
                item_text_color = text_color

            dc.SetFont(font)
            draw_item_selection_rect(self, dc, rectangle, flags)

            if self.test_mode and self.running_item == index:
                dc.SetFont(font.MakeUnderlined())
                draw_item_selection_rect(
                    self, dc, rectangle, flags | wx.CONTROL_SELECTED
                )

            dc.SetBackgroundMode(wx.PENSTYLE_TRANSPARENT)

            dc.SetTextForeground(item_text_color)

            dc.DrawText(
                item.module_name,
                rectangle.GetLeft() + self.text_gap,
                rectangle.GetTop(),
            )

        if self.drop_insert_point is not None:
            y = self.line_height * self.drop_insert_point

            dc.SetPen(wx.BLACK_PEN)

            dc.DrawLine(0, y, self.GetSize()[0], y)

    def make_event(self, py_event_binder, index=None, module=None):
        event = wx.NotifyEvent(py_event_binder.evtType[0])
        event.SetEventObject(self)
        if index is not None:
            event.SetInt(index)
        if module is not None:
            event.module = module
        return event

    def on_left_down(self, event):
        assert isinstance(event, wx.MouseEvent)
        self.SetFocus()
        index, hit_test, column = self.HitTestSubItem(event.GetPosition())
        if hit_test == PLV_HITTEST_SLIDER:
            self.active_slider = True
            self.CaptureMouse()
            self.RefreshRect(self.get_slider_rect())
        elif hit_test & wx.LIST_HITTEST_ONITEMLABEL:
            item_id = None if index >= len(self.items) else id(self.items[index])
            if event.ShiftDown() and self.active_item is not None and self.CanSelect():
                # Extend the selection
                begin = min(self.active_item, index)
                end = max(self.active_item, index) + 1
                for i in range(begin, end):
                    self.Select(i, True)
                toggle_selection = False
                multiple_selection = True
                activate_before_drag = True
            else:
                activate_before_drag = not self.IsSelected(index)
                toggle_selection = True
                multiple_selection = event.ControlDown()
            if activate_before_drag:
                self.activate_item(index, toggle_selection, multiple_selection)
            plv_event = self.make_event(wx.EVT_LIST_BEGIN_DRAG, index)
            self.GetEventHandler().ProcessEvent(plv_event)
            if not activate_before_drag:
                new_item_id = (
                    None if index >= len(self.items) else id(self.items[index])
                )
                if new_item_id == item_id:
                    self.activate_item(index, toggle_selection, multiple_selection)
        elif hit_test & wx.LIST_HITTEST_ONITEMICON:
            if column != ERROR_COLUMN or self.CanSelect():
                self.pressed_row = index
                self.pressed_column = column
                self.button_is_active = True
                self.Refresh(eraseBackground=False)

    def deactivate_active_item(self):
        """Remove the activation UI indication from the current active item"""
        self.active_item = None
        self.Refresh(eraseBackground=False)

    def activate_item(
        self, index, toggle_selection, multiple_selection, anchoring=True
    ):
        """Move the active item

        index - index of item to activate

        toggle_selection - true to toggle the selection state, false to
                           always select.

        multiple_selection - true to allow multiple selections, false
                           to deselect all but the activated.

        anchoring - true to place the anchor for extended selection
                    false if the anchor should be kept where it is

        returns the selection state
        """
        self.active_item = index
        plv_event = self.make_event(wx.EVT_LIST_ITEM_ACTIVATED, index)
        self.GetEventHandler().ProcessEvent(plv_event)
        if self.allow_disable:
            if self.IsSelected(index) and toggle_selection:
                self.Select(index, False)
                self.Refresh(eraseBackground=False)
                return False
            if not multiple_selection:
                for i, item in enumerate(self.items):
                    if self.IsSelected(i):
                        self.Select(i, False)
            self.Select(index, True)
            if anchoring:
                self.anchor = index
        window_height = int(self.GetSize()[1] / self.line_height)
        #
        # Always keep the active item in view
        #
        sx = self.GetScrollPos(wx.HORIZONTAL)
        sy = self.GetScrollPos(wx.VERTICAL)
        if index < sy:
            self.Scroll(sx, index)
        elif index >= sy + window_height:
            self.Scroll(sx, index + window_height - 1)

        self.Refresh(eraseBackground=False)
        return True

    def on_left_up(self, event):
        if self.GetCapture() == self:
            self.ReleaseMouse()
        if self.active_slider:
            self.active_slider = False
            self.RefreshRect(self.get_slider_rect())
        if self.button_is_active and self.pressed_column is not None:
            if self.pressed_column == ERROR_COLUMN:
                code = EVT_PLV_ERROR_COLUMN_CLICKED
            elif self.pressed_column == STEP_COLUMN:
                code = EVT_PLV_STEP_COLUMN_CLICKED
            elif self.pressed_column == PAUSE_COLUMN:
                code = EVT_PLV_PAUSE_COLUMN_CLICKED
            else:
                code = EVT_PLV_EYE_COLUMN_CLICKED
            r = self.get_button_rect(self.pressed_column, self.pressed_row or 0)
            r.Inflate(2, 2)
            self.RefreshRect(r)
            self.button_is_active = False
            self.pressed_column = None
            plv_event = self.make_event(code, self.pressed_row)
            self.GetEventHandler().ProcessEvent(plv_event)

    def on_mouse_move(self, event):
        index, hit_test, column = self.HitTestSubItem(event.Position)
        if self.active_slider:
            index = self.get_index(event.Position)
            if self.running_item != index:
                plv_event = self.make_event(EVT_PLV_SLIDER_MOTION, index)
                self.GetEventHandler().ProcessEvent(plv_event)
                if plv_event.IsAllowed() and index <= self.last_running_item:
                    self.running_item = index
                    self.Refresh(eraseBackground=False)
        elif self.HasCapture() and self.pressed_column is not None:
            button_is_active = (
                index == self.pressed_row and column == self.pressed_column
            )
            if button_is_active != self.button_is_active:
                self.button_is_active = button_is_active
                self.Refresh(eraseBackground=False)
        else:
            tooltip_text = None
            item = (
                None
                if (index is None or index >= self.GetItemCount())
                else self.items[index]
            )
            if hit_test & wx.LIST_HITTEST_ONITEM:
                if column == EYE_COLUMN:
                    if item.module.show_window:
                        tooltip_text = (
                            "%s will show its display. Click icon to hide display"
                            % item.module.module_name
                        )
                    else:
                        tooltip_text = (
                            "%s will not show its display. Click icon to show display"
                            % item.module.module_name
                        )
                elif column == PAUSE_COLUMN:
                    if item.module.wants_pause:
                        tooltip_text = (
                            "Test mode will stop before executing %s. Click icon to change"
                            % item.module.module_name
                        )
                    else:
                        tooltip_text = (
                            "Test mode will not stop before executing %s. Click icon to change"
                            % item.module.module_name
                        )
                elif column == ERROR_COLUMN:
                    if item.get_error_state() in (WARNING, ERROR):
                        tooltip_text = item.tooltip
                    elif item.module.enabled and not item.module.is_input_module():
                        tooltip_text = (
                            "Click to disable the %s module" % item.module.module_name
                        )
                    elif not item.module.is_input_module():
                        tooltip_text = (
                            "Click to enable the %s module" % item.module.module_name
                        )
            if tooltip_text is not None:
                self.SetToolTip(tooltip_text)
            else:
                self.SetToolTip(None)

    def cancel_capture(self):
        if self.HasCapture() and self.pressed_column is not None:
            self.ReleaseMouse()
            self.pressed_column = None
            self.button_is_active = False

    def on_key_down(self, event):
        assert isinstance(event, wx.KeyEvent)
        if self.GetItemCount() > 0 and self.active_item is not None:
            multiple_selection = event.ControlDown() or event.ShiftDown()
            anchoring_selection = not event.ShiftDown()
            if (
                event.GetKeyCode() in (wx.WXK_DOWN, wx.WXK_NUMPAD_DOWN)
                and self.active_item < self.GetItemCount() - 1
            ):
                self.cancel_capture()
                # Retreating from a previous shift select
                if not anchoring_selection and self.anchor > self.active_item:
                    self.Select(self.active_item, False)
                self.activate_item(
                    self.active_item + 1,
                    False,
                    multiple_selection,
                    anchoring=anchoring_selection,
                )
            elif (
                event.GetKeyCode() in (wx.WXK_UP, wx.WXK_NUMPAD_UP)
                and self.active_item > 0
            ):
                self.cancel_capture()
                if not anchoring_selection and self.anchor < self.active_item:
                    self.Select(self.active_item, False)
                self.activate_item(
                    self.active_item - 1, False, multiple_selection, anchoring_selection
                )
            elif event.GetKeyCode() in (wx.WXK_HOME, wx.WXK_NUMPAD_HOME):
                self.cancel_capture()
                if not anchoring_selection:
                    # Extend selection to anchor / remove selection beyond anchor
                    for i in range(self.anchor + 1, self.active_item + 1):
                        self.Select(i, False)
                    for i in range(1, self.anchor):
                        self.Select(i, True)
                self.activate_item(
                    0, False, multiple_selection, anchoring=anchoring_selection
                )
            elif event.GetKeyCode() in (wx.WXK_END, wx.WXK_NUMPAD_END):
                self.cancel_capture()
                if not anchoring_selection:
                    # Extend selection from current to end / remove
                    # selection before anchor
                    for i in range(self.active_item, self.anchor):
                        self.Select(i, False)
                    for i in range(self.anchor + 1, self.GetItemCount()):
                        self.Select(i, True)
                self.activate_item(
                    self.GetItemCount() - 1,
                    False,
                    multiple_selection,
                    anchoring=anchoring_selection,
                )
            else:
                event.Skip()
        else:
            event.Skip()

    def on_capture_lost(self, event):
        self.active_slider = False
        self.button_is_active = False
        self.pressed_column = None
        self.Refresh(eraseBackground=False)
