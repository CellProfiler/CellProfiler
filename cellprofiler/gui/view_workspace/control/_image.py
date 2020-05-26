import cellprofiler_core.preferences

import cellprofiler.gui
import cellprofiler.gui.artist
from cellprofiler.gui.view_workspace.control._control import Control


class Image(Control):
    def __init__(self, frame, color, can_delete):
        super(Image, self).__init__(frame, color, can_delete)
        image_set = frame.workspace.image_set
        name = self.chooser.GetStringSelection()

        im = cellprofiler_core.preferences.get_intensity_mode()
        if im == cellprofiler_core.preferences.INTENSITY_MODE_LOG:
            normalization = cellprofiler.gui.artist.NORMALIZE_LOG
        elif im == cellprofiler_core.preferences.INTENSITY_MODE_NORMAL:
            normalization = cellprofiler.gui.artist.NORMALIZE_LINEAR
        else:
            normalization = cellprofiler.gui.artist.NORMALIZE_RAW
        alpha = 1.0 / (len(frame.image_rows) + 1.0)
        self._data = self.bind(
            cellprofiler.gui.artist.ImageData, self.colour_select, frame.redraw
        )(
            name,
            None,
            mode=cellprofiler.gui.artist.MODE_HIDE,
            color=self.color,
            colormap=cellprofiler_core.preferences.get_default_colormap(),
            alpha=alpha,
            normalization=normalization,
        )
        frame.image.add(self.data)
        self.last_mode = cellprofiler.gui.artist.MODE_COLORIZE

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, name):
        image_set = self.frame.workspace.image_set
        image = image_set.get_image(name)
        self._data.pixel_data = image.pixel_data

    @property
    def names(self):
        return self.frame.workspace.image_set.names

    def update_data(self, name):
        """Update the image data from the workspace"""
        image_set = self.frame.workspace.image_set
        image = image_set.get_image(name)
        self.data.pixel_data = image.pixel_data
