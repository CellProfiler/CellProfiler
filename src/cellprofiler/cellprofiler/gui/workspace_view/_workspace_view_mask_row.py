from ._workspace_view_row import WorkspaceViewRow
from ..artist import MODE_HIDE
from ..artist import MODE_LINES
from ..artist import MaskData
from ..utilities.workspace_view import bind_data_class


class WorkspaceViewMaskRow(WorkspaceViewRow):
    """A row of controls for controlling masks"""

    def __init__(self, workspace_view, color, can_delete):
        super(WorkspaceViewMaskRow, self).__init__(workspace_view, color, can_delete)
        self.__cached_names = None
        self.update_chooser(first=True)
        name = self.chooser.GetStringSelection()
        self.data = bind_data_class(MaskData, self.color_ctrl, workspace_view.redraw)(
            name, None, color=self.color, alpha=0.5, mode=MODE_HIDE,
        )
        workspace_view.image.add(self.data)
        self.last_mode = MODE_LINES

    def get_names(self):
        image_set = self.workspace_view.workspace.image_set
        names = [name for name in image_set.names if image_set.get_image(name).has_mask]
        return names

    def update_data(self, name):
        """Update the image data from the workspace"""
        image_set = self.workspace_view.workspace.image_set
        image = image_set.get_image(name)
        self.data.mask = image.mask
