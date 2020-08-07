from cellprofiler_core.preferences import INTENSITY_MODE_LOG
from cellprofiler_core.preferences import INTENSITY_MODE_NORMAL
from cellprofiler_core.preferences import get_default_colormap
from cellprofiler_core.preferences import get_intensity_mode

from ._workspace_view_row import WorkspaceViewRow
from ..artist import ImageData
from ..artist import MODE_COLORIZE
from ..artist import MODE_HIDE
from ..artist import NORMALIZE_LINEAR
from ..artist import NORMALIZE_LOG
from ..artist import NORMALIZE_RAW
from ..utilities.workspace_view import bind_data_class


class WorkspaceViewImageRow(WorkspaceViewRow):
    def __init__(self, workspace_view, color, can_delete):
        super(WorkspaceViewImageRow, self).__init__(workspace_view, color, can_delete)
        image_set = workspace_view.workspace.image_set
        name = self.chooser.GetStringSelection()

        im = get_intensity_mode()
        if im == INTENSITY_MODE_LOG:
            normalization = NORMALIZE_LOG
        elif im == INTENSITY_MODE_NORMAL:
            normalization = NORMALIZE_LINEAR
        else:
            normalization = NORMALIZE_RAW
        alpha = 1.0 / (len(workspace_view.image_rows) + 1.0)
        self.data = bind_data_class(ImageData, self.color_ctrl, workspace_view.redraw)(
            name,
            None,
            mode=MODE_HIDE,
            color=self.color,
            colormap=get_default_colormap(),
            alpha=alpha,
            normalization=normalization,
        )
        workspace_view.image.add(self.data)
        self.last_mode = MODE_COLORIZE

    def get_names(self):
        return self.workspace_view.workspace.image_set.names

    def update_data(self, name):
        """Update the image data from the workspace"""
        image_set = self.workspace_view.workspace.image_set
        image = image_set.get_image(name)
        self.data.pixel_data = image.pixel_data
