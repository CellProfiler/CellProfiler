import matplotlib
import matplotlib.backend_bases
import matplotlib.backends.backend_wxagg
import wx
import wx.lib.intctrl

from ..constants.figure import NAV_MODE_ZOOM
from ..constants.figure import NAV_MODE_PAN
from ..constants.figure import EVT_NAV_MODE_CHANGE
from ..utilities.figure import get_crosshair_cursor


class NavigationToolbar(matplotlib.backends.backend_wxagg.NavigationToolbar2WxAgg):
    """Navigation toolbar for EditObjectsDialog"""

    def __init__(self, canvas):
        super(NavigationToolbar, self).__init__(canvas)
        self.volumetric = False
        self.Realize()

    def set_volumetric(self):
        if not self.volumetric:
            self.planelabel = wx.StaticText(self, label="Plane:")
            self.slider = wx.Slider(
                self,
                value=100,
                minValue=0,
                maxValue=255,
                style=wx.SL_HORIZONTAL,
                size=wx.Size(150, 22),
            )
            self.slider.SetPageSize(5)
            self.planetext = wx.lib.intctrl.IntCtrl(
                self,
                id=0,
                value=0,
                min=-0,
                max=9999,
                limited=True,
                size=wx.Size(30, 20),
                style=wx.TE_CENTRE,
            )
            self.prevplane = wx.Button(self, label="<", size=wx.Size(22, 22))
            self.nextplane = wx.Button(self, label=">", size=wx.Size(22, 22))
            self.AddSeparator()
            self.AddStretchableSpace()
            self.AddControl(self.planelabel, "Plane")
            self.AddControl(self.slider, "Plane Slider")
            self.AddControl(self.prevplane, "Previous Plane")
            self.AddControl(self.planetext, "Plane Number")
            self.AddControl(self.nextplane, "Next Plane")
            self.AddStretchableSpace()
            self.Realize()
            self.volumetric = True


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
