import abc

import wx
import wx.lib.colourselect

import cellprofiler.gui
import cellprofiler.gui.artist


class Control(abc.ABC):
    """A control of controls and a data item"""

    def __init__(self, frame, color, can_delete):
        self.frame = frame
        panel = frame.panel
        self.chooser = wx.Choice(panel)
        self.colour_select = wx.lib.colourselect.ColourSelect(panel, colour=color)
        self.show_check = wx.CheckBox(panel)
        bitmap = wx.ArtProvider.GetBitmap(wx.ART_DELETE, wx.ART_TOOLBAR, (16, 16))
        self.remove_button = wx.BitmapButton(panel, bitmap=bitmap)
        if not can_delete:
            self.remove_button.Hide()
        self.chooser.Bind(wx.EVT_CHOICE, self.on_choice)
        self.colour_select.Bind(
            wx.lib.colourselect.EVT_COLOURSELECT, self.on_color_change
        )
        self.show_check.Bind(wx.EVT_CHECKBOX, self.on_check_change)
        self.update_chooser(first=True)
        self._data = None
        self._names = []

    @staticmethod
    def bind(klass, color_select, fn_redraw):
        """Bind ImageData etc to synchronize to color select button

        data_class - ImageData, ObjectData or MaskData
        color_select - a color select button whose color synchronizes
                       to that of the data
        fn_redraw - function to be called
        """
        assert issubclass(klass, cellprofiler.gui.artist.ColorMixin)
        assert isinstance(color_select, wx.lib.colourselect.ColourSelect)

        class Binding(klass):
            def _on_color_changed(self):
                super(Binding, self)._on_color_changed()
                r, g, b = [int(x * 255) for x in self.color]
                rold, gold, bold = self.color_select.GetColour()
                if r != rold or g != gold or b != bold:
                    self.color_select.SetColour(wx.Colour(r, g, b))

        Binding.color_select = color_select

        return Binding

    @property
    def color(self):
        """The color control's current color scaled for matplotlib"""
        return tuple([float(x) / 255 for x in self.colour_select.GetColour()])

    @property
    @abc.abstractmethod
    def data(self):
        return self._data

    @data.setter
    @abc.abstractmethod
    def data(self, value):
        self._data = value

    @property
    @abc.abstractmethod
    def names(self):
        return self._names

    @names.setter
    @abc.abstractmethod
    def names(self, value):
        self._names = value

    def on_choice(self, event):
        self.data.name = self.chooser.GetStringSelection()
        self.frame.redraw()

    def on_color_change(self, event):
        self.data.color = tuple(
            [float(c) / 255.0 for c in self.colour_select.GetColour()]
        )
        self.frame.redraw()

    def on_check_change(self, event):
        self.frame.redraw()

    def update(self):
        name = self.chooser.GetStringSelection()
        names = sorted(self.names)
        image_set = self.frame.workspace.image_set
        if self.show_check.IsChecked() and name in names:
            self.data.name = name
            self.update_data(name)
            if self.data.mode == cellprofiler.gui.artist.MODE_HIDE:
                self.data.mode = self.last_mode
        elif self.data.mode != cellprofiler.gui.artist.MODE_HIDE:
            self.last_mode = self.data.get_raw_mode()
            self.data.mode = cellprofiler.gui.artist.MODE_HIDE
        self.update_chooser()

    def update_chooser(self, first=False):
        """Update the chooser with the given list of names"""
        name = self.chooser.GetStringSelection()
        names = self.names
        current_names = sorted(self.chooser.GetItems())
        if tuple(current_names) != tuple(names):
            if name not in names:
                names = sorted(list(names) + [name])
            self.chooser.SetItems(names)
            self.chooser.SetStringSelection(name)
        if first and len(names) > 0:
            name = names[0]
            self.chooser.SetStringSelection(name)
