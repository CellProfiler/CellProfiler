import matplotlib
import matplotlib.backend_bases
import matplotlib.backends.backend_wxagg
import wx

from ..constants.figure import NAV_MODE_ZOOM
from ..constants.figure import NAV_MODE_MEASURE
from ..constants.figure import BMP_MEASURE
from ..constants.figure import NAV_MODE_PAN
from ..constants.figure import EVT_NAV_MODE_CHANGE
from ..utilities.figure import get_crosshair_cursor


class NavigationToolbar(matplotlib.backends.backend_wxagg.NavigationToolbar2WxAgg):
    """Navigation toolbar for EditObjectsDialog"""

    def __init__(self, canvas, want_measure=False):
        super(NavigationToolbar, self).__init__(canvas)
        if want_measure:
            self.bmp_error = wx.Bitmap(BMP_MEASURE)
            self.wx_ids["Measure"] = wx.NewId()
            self.Bind(wx.EVT_TOOL, self.measure, id=self.wx_ids["Measure"])
            self.measure_tool = self.CreateTool(
                self.wx_ids["Measure"],
                "MeasureLength",
                self.bmp_error,
                kind=wx.ITEM_CHECK,
                shortHelp="Measure Length"
            )
            self.InsertTool(6, self.measure_tool)
        self.Realize()

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
        elif self.mode == NAV_MODE_MEASURE:
            if "Measure" in self.wx_ids:
                self.ToggleTool(self.wx_ids["Measure"], False)

    def zoom(self, *args):
        matplotlib.backends.backend_wxagg.NavigationToolbar2WxAgg.zoom(self, *args)
        if "Measure" in self.wx_ids:
            self.ToggleTool(self.wx_ids['Measure'], False)
        self.__send_mode_change_event()

    def pan(self, *args):
        matplotlib.backends.backend_wxagg.NavigationToolbar2WxAgg.pan(self, *args)
        if "Measure" in self.wx_ids:
            self.ToggleTool(self.wx_ids['Measure'], False)
        self.__send_mode_change_event()

    def measure(self, *args):
        self.mode = NAV_MODE_MEASURE
        if "Pan" in self.wx_ids:
            self.ToggleTool(self.wx_ids['Pan'], False)
        if "Zoom" in self.wx_ids:
            self.ToggleTool(self.wx_ids['Zoom'], False)
        self.__send_mode_change_event()
        if self._active == 'MEASURE':
            self._active = None
        else:
            self._active = 'MEASURE'

        if self._idPress is not None:
            self._idPress = self.canvas.mpl_disconnect(self._idPress)
            self.mode = ''

        if self._idRelease is not None:
            self._idRelease = self.canvas.mpl_disconnect(self._idRelease)
            self.mode = ''

        if self._active:
            self.canvas.widgetlock(self)
        else:
            self.canvas.widgetlock.release(self)

        for a in self.canvas.figure.get_axes():
            a.set_navigate_mode(self._active)

        self.set_message(self.mode)

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
