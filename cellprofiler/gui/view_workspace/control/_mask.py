import cellprofiler.gui
from cellprofiler.gui.view_workspace.control._control import Control
import cellprofiler.gui.artist


class Mask(Control):
    """A control of controls for controlling masks"""

    def __init__(self, frame, color, can_delete):
        super(Mask, self).__init__(frame, color, can_delete)
        self.__cached_names = None
        self.update_chooser(first=True)
        name = self.chooser.GetStringSelection()
        self.data = self.bind(
            cellprofiler.gui.artist.MaskData, self.colour_select, frame.redraw
        )(
            name,
            None,
            color=self.color,
            alpha=0.5,
            mode=cellprofiler.gui.artist.MODE_HIDE,
        )
        frame.image.add(self.data)
        self.last_mode = cellprofiler.gui.artist.MODE_LINES

    def get_names(self):
        image_set = self.frame.workspace.image_set
        names = [name for name in image_set.names if image_set.get_image(name).has_mask]
        return names

    def update_data(self, name):
        """Update the image data from the workspace"""
        image_set = self.frame.workspace.image_set
        image = image_set.get_image(name)
        self.data.mask = image.mask
