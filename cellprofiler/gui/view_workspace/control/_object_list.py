import cellprofiler_core.preferences

import cellprofiler.gui
from cellprofiler.gui.view_workspace.control._control import Control
import cellprofiler.gui.artist


class ObjectList(Control):
    """A control of controls for controlling objects"""

    def __init__(self, frame, color, can_delete):
        super(ObjectList, self).__init__(frame, color, can_delete)
        self.update_chooser(first=True)
        name = self.chooser.GetStringSelection()
        self.data = self.bind(
            cellprofiler.gui.artist.ObjectsData, self.colour_select, frame.redraw
        )(
            name,
            None,
            outline_color=self.color,
            colormap=cellprofiler_core.preferences.get_default_colormap(),
            alpha=0.5,
            mode=cellprofiler.gui.artist.MODE_HIDE,
        )
        frame.image.add(self.data)
        self.last_mode = cellprofiler.gui.artist.MODE_LINES

    def get_names(self):
        object_set = self.frame.workspace.object_set
        return object_set.get_object_names()

    def update_data(self, name):
        object_set = self.frame.workspace.object_set
        objects = object_set.get_objects(name)
        self.data.labels = [l for l, i in objects.get_labels()]
