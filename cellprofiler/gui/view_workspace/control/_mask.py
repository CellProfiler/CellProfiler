import cellprofiler.gui
from cellprofiler.gui.view_workspace.control._control import Control


def bind_data_class(data_class, color_select, fn_redraw):
    """Bind ImageData etc to synchronize to color select button

    data_class - ImageData, ObjectData or MaskData
    color_select - a color select button whose color synchronizes
                   to that of the data
    fn_redraw - function to be called
    """
    assert issubclass(data_class, cellprofiler.gui.artist.ColorMixin)
    assert isinstance(color_select, wx.lib.colourselect.ColourSelect)

    class bdc(data_class):
        def _on_color_changed(self):
            super(bdc, self)._on_color_changed()
            r, g, b = [int(x * 255) for x in self.color]
            rold, gold, bold = self.color_select.GetColour()
            if r != rold or g != gold or b != bold:
                self.color_select.SetColour(wx.Colour(r, g, b))

    bdc.color_select = color_select
    return bdc


class Mask(Control):
    """A control of controls for controlling masks"""

    def __init__(self, vw, color, can_delete):
        super(Mask, self).__init__(vw, color, can_delete)
        self.__cached_names = None
        self.update_chooser(first=True)
        name = self.chooser.GetStringSelection()
        self.data = bind_data_class(
            cellprofiler.gui.artist.MaskData, self.color_ctrl, vw.redraw
        )(
            name,
            None,
            color=self.color,
            alpha=0.5,
            mode=cellprofiler.gui.artist.MODE_HIDE,
        )
        vw.image.add(self.data)
        self.last_mode = cellprofiler.gui.artist.MODE_LINES

    def get_names(self):
        image_set = self.vw.workspace.image_set
        names = [name for name in image_set.names if image_set.get_image(name).has_mask]
        return names

    def update_data(self, name):
        """Update the image data from the workspace"""
        image_set = self.vw.workspace.image_set
        image = image_set.get_image(name)
        self.data.mask = image.mask
