from cellprofiler_core.preferences import get_default_colormap

from ._workspace_view_row import WorkspaceViewRow
from ..artist import MODE_HIDE
from ..artist import MODE_LINES
from ..artist import ObjectsData
from ..utilities.workspace_view import bind_data_class


class WorkspaceViewObjectsRow(WorkspaceViewRow):
    """A row of controls for controlling objects"""

    def __init__(self, workspace_view, color, can_delete):
        super(WorkspaceViewObjectsRow, self).__init__(workspace_view, color, can_delete)
        self.update_chooser(first=True)
        name = self.chooser.GetStringSelection()
        self.data = bind_data_class(
            ObjectsData, self.color_ctrl, workspace_view.redraw
        )(
            name,
            None,
            outline_color=self.color,
            colormap=get_default_colormap(),
            alpha=0.5,
            mode=MODE_HIDE,
        )
        workspace_view.image.add(self.data)
        self.last_mode = MODE_LINES

    def get_names(self):
        object_set = self.workspace_view.workspace.object_set
        return object_set.get_object_names()

    def update_data(self, name):
        object_set = self.workspace_view.workspace.object_set
        objects = object_set.get_objects(name)
        self.data.labels = [l for l, i in objects.get_labels()]
