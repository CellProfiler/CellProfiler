import wx

import cellprofiler.gui
from ._image import Image
from ._mask import Mask
from ._measurement import Measurement
from ._object_list import ObjectList


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
