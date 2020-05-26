import cellprofiler_core.preferences

import cellprofiler.gui
import cellprofiler.gui.artist
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


class Image(Control):
    def __init__(self, vw, color, can_delete):
        super(Image, self).__init__(vw, color, can_delete)
        image_set = vw.workspace.image_set
        name = self.chooser.GetStringSelection()

        im = cellprofiler_core.preferences.get_intensity_mode()
        if im == cellprofiler_core.preferences.INTENSITY_MODE_LOG:
            normalization = cellprofiler.gui.artist.NORMALIZE_LOG
        elif im == cellprofiler_core.preferences.INTENSITY_MODE_NORMAL:
            normalization = cellprofiler.gui.artist.NORMALIZE_LINEAR
        else:
            normalization = cellprofiler.gui.artist.NORMALIZE_RAW
        alpha = 1.0 / (len(vw.image_rows) + 1.0)
        self.data = bind_data_class(
            cellprofiler.gui.artist.ImageData, self.color_ctrl, vw.redraw
        )(
            name,
            None,
            mode=cellprofiler.gui.artist.MODE_HIDE,
            color=self.color,
            colormap=cellprofiler_core.preferences.get_default_colormap(),
            alpha=alpha,
            normalization=normalization,
        )
        vw.image.add(self.data)
        self.last_mode = cellprofiler.gui.artist.MODE_COLORIZE

    def get_names(self):
        return self.vw.workspace.image_set.names

    def update_data(self, name):
        """Update the image data from the workspace"""
        image_set = self.vw.workspace.image_set
        image = image_set.get_image(name)
        self.data.pixel_data = image.pixel_data
