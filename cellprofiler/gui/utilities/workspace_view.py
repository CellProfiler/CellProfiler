import wx
import wx.lib.colourselect

from ..artist import ColorMixin


def show_workspace_viewer(parent, workspace):
    global __the_workspace_viewer
    if __the_workspace_viewer is None:
        from cellprofiler.gui.workspace_view import WorkspaceView

        __the_workspace_viewer = WorkspaceView(parent, workspace)
    else:
        __the_workspace_viewer.set_workspace(workspace)
        __the_workspace_viewer.frame.Show()


def update_workspace_viewer(workspace):
    if __the_workspace_viewer is not None:
        __the_workspace_viewer.set_workspace(workspace)


def bind_data_class(data_class, color_select, fn_redraw):
    """Bind ImageData etc to synchronize to color select button

    data_class - ImageData, ObjectData or MaskData
    color_select - a color select button whose color synchronizes
                   to that of the data
    fn_redraw - function to be called
    """
    assert issubclass(data_class, ColorMixin)
    assert isinstance(color_select, wx.lib.colourselect.ColourSelect)

    class bdc(data_class):
        def _on_color_changed(self):
            super(bdc, self)._on_color_changed()
            r, g, b = [int(x * 255) for x in self.color]
            rold, gold, bold, alpha = self.color_select.GetColour()
            if r != rold or g != gold or b != bold:
                self.color_select.SetColour(wx.Colour(r, g, b))

    bdc.color_select = color_select
    return bdc
