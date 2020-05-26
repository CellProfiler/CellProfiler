import cellprofiler_core.preferences

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


class ObjectList(Control):
    """A control of controls for controlling objects"""

    def __init__(self, vw, color, can_delete):
        super(ObjectList, self).__init__(vw, color, can_delete)
        self.update_chooser(first=True)
        name = self.chooser.GetStringSelection()
        self.data = bind_data_class(
            cellprofiler.gui.artist.ObjectsData, self.color_ctrl, vw.redraw
        )(
            name,
            None,
            outline_color=self.color,
            colormap=cellprofiler_core.preferences.get_default_colormap(),
            alpha=0.5,
            mode=cellprofiler.gui.artist.MODE_HIDE,
        )
        vw.image.add(self.data)
        self.last_mode = cellprofiler.gui.artist.MODE_LINES

    def get_names(self):
        object_set = self.vw.workspace.object_set
        return object_set.get_object_names()

    def update_data(self, name):
        object_set = self.vw.workspace.object_set
        objects = object_set.get_objects(name)
        self.data.labels = [l for l, i in objects.get_labels()]
