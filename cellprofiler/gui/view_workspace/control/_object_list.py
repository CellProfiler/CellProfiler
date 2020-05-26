import cellprofiler_core.preferences

import cellprofiler.gui
from cellprofiler.gui.view_workspace import bind_data_class
from cellprofiler.gui.view_workspace.control._control import Control


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
