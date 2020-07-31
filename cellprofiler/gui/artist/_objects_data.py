import matplotlib
import numpy

import cellprofiler.gui
from ._outlines_mixin import OutlinesMixin
from ..constants.artist import MODE_LINES
from ..constants.artist import MODE_OUTLINES
from ..constants.artist import MODE_OVERLAY


class ObjectsData(OutlinesMixin):
    """The data needed to display objects

    name - the name of the objects
    labels - a sequence of label matrices
    outline_color - render outlines in this color. Default is
    primary color.
    line_width - the width of the line in pixels for outlines
    colormap - render overlaid labels using this colormap
    alpha - render overlaid labels using this alpha value.
    mode - the display mode: outlines, lines, overlay or hide
    scramble - True (default) to scramble the colors. False to use
                         the labels to pick from the colormap.
    """

    def __init__(
        self,
        name,
        labels,
        outline_color=None,
        line_width=None,
        colormap=None,
        alpha=None,
        mode=None,
        scramble=True,
    ):
        super(ObjectsData, self).__init__(outline_color, line_width)
        self.name = name
        self.__labels = labels
        self.colormap = colormap
        if alpha is not None:
            self.alpha = alpha
        self.mode = mode
        self.scramble = scramble
        self.__overlay = None
        self.__mask = None

    def get_labels(self):
        return self.__labels

    def set_labels(self, labels):
        self.__labels = labels
        self._flush_outlines()
        self.__overlay = None
        self.__mask = None

    def get_raw_mode(self):
        return self.mode

    labels = property(get_labels, set_labels)

    def _on_colormap_changed(self):
        super(ObjectsData, self)._on_colormap_changed()
        self.__overlay = None

    def _using_color(self):
        return self.mode in (MODE_LINES, MODE_OUTLINES)

    def _using_colormap(self):
        return self.mode == MODE_OVERLAY

    @property
    def overlay(self):
        """Return a color image of the segmentation as an overlay
        """
        if self.__overlay is not None:
            return self.__overlay
        sm = matplotlib.cm.ScalarMappable(cmap=self.colormap)
        sm.set_clim(
            vmin=1, vmax=numpy.max([numpy.max(label) for label in self.labels]) + 1
        )

        img = None
        lmin = 0
        for label in self.labels:
            if numpy.all(label == 0):
                continue
            if self.scramble:
                lmin = numpy.min(label[label != 0])
            label[label != 0] = (
                cellprofiler.gui.tools.renumber_labels_for_display(label)[label != 0]
                + lmin
            )
            lmin = numpy.max(label)
            if img is None:
                img = sm.to_rgba(label)
                img[label == 0, :] = 0
            else:
                img[label != 0, :] = sm.to_rgba(label[label != 0])
        self.__overlay = img
        return img

    @property
    def mask(self):
        """Return a mask of the labeled portion of the field of view"""
        if self.__mask is None:
            self.__mask = self.labels[0] != 0
            for l in self.labels[1:]:
                self.__mask[l != 0] = True
        return self.__mask
