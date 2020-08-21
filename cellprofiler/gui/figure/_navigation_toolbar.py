import matplotlib
import matplotlib.backend_bases
import matplotlib.backends.backend_wxagg
import wx

from ..constants.figure import NAV_MODE_ZOOM
from ..constants.figure import NAV_MODE_PAN
from ..constants.figure import EVT_NAV_MODE_CHANGE
from ..utilities.figure import get_crosshair_cursor


class NavigationToolbar(matplotlib.backends.backend_wxagg.NavigationToolbar2WxAgg):
    """Navigation toolbar for EditObjectsDialog"""

    def __init__(self, canvas):
        super(NavigationToolbar, self).__init__(canvas)

    def set_cursor(self, cursor):
        """Set the cursor based on the mode"""
        if cursor == matplotlib.backend_bases.cursors.SELECT_REGION:
            self.canvas.SetCursor(get_crosshair_cursor())
        else:
            matplotlib.backends.backend_wxagg.NavigationToolbar2WxAgg.set_cursor(
                self, cursor
            )

    def cancel_mode(self):
        """Toggle the current mode to off"""
        if self.mode == NAV_MODE_ZOOM:
            self.zoom()

            if "Zoom" in self.wx_ids:
                self.ToggleTool(self.wx_ids["Zoom"], False)
        elif self.mode == NAV_MODE_PAN:
            self.pan()

            if "Pan" in self.wx_ids:
                self.ToggleTool(self.wx_ids["Pan"], False)

    def zoom(self, *args):
        matplotlib.backends.backend_wxagg.NavigationToolbar2WxAgg.zoom(self, *args)

        self.__send_mode_change_event()

    def pan(self, *args):
        matplotlib.backends.backend_wxagg.NavigationToolbar2WxAgg.pan(self, *args)

        self.__send_mode_change_event()

    def reset(self):
        """Clear out the position stack"""
        # We differ from the reference implementation because we clear
        # the view stacks.
        # self._views.clear()

        # self._positions.clear()

        self.home()

    def save(self, event):
        from ._figure import Figure

        #
        # Capture any file save event and redirect it to CPFigureFrame
        # Fixes issue #829 - Mac & PC display invalid save options when
        #                    you save using the icon.
        #
        parent = self.GetTopLevelParent()

        if isinstance(parent, Figure):
            parent.on_file_save(event)
        else:
            super(NavigationToolbar, self).save(event)

    def __send_mode_change_event(self):
        event = wx.NotifyEvent(EVT_NAV_MODE_CHANGE.evtType[0])

        event.EventObject = self

        self.GetEventHandler().ProcessEvent(event)
