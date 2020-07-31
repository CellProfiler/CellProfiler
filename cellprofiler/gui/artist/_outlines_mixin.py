import centrosome.outline
import numpy
import scipy.ndimage

from ._outline_artist import BoundaryLineCollection
from ._color_mixin import ColorMixin


class OutlinesMixin(ColorMixin):
    """Provides rendering of "labels" as outlines

    Needs self.labels to return a sequence of labels matrices and needs
    self._flush_outlines to be called when parameters change.
    """

    def __init__(self, outline_color, line_width):
        super(OutlinesMixin, self).__init__()
        self._flush_outlines()
        self.color = outline_color
        self._line_width = line_width

    def _flush_outlines(self):
        self._outlines = None
        self._points = None

    def _on_color_changed(self):
        if self._points is not None:
            self._points.set_color(self.color)

    def get_line_width(self):
        return self._line_width

    def set_line_width(self, line_width):
        self._line_width = line_width
        self._outlines = None
        if self._points is not None:
            self._points.set_linewidth(line_width)

    line_width = property(get_line_width, set_line_width)

    @property
    def outlines(self):
        """Get a mask of all the points on the border of objects"""
        if self._outlines is None:
            for i, labels in enumerate(self.labels):
                if i == 0:
                    self._outlines = centrosome.outline.outline(labels) != 0
                else:
                    self._outlines |= centrosome.outline.outline(labels) != 0
            if self.line_width > 1:
                hw = float(self.line_width) / 2
                d = scipy.ndimage.distance_transform_edt(~self._outlines)
                dti, dtj = numpy.where((d < hw + 0.5) & ~self._outlines)
                self._outlines = self._outlines.astype(numpy.float32)
                self._outlines[dti, dtj] = numpy.minimum(1, hw + 0.5 - d[dti, dtj])

        return self._outlines.astype(numpy.float32)

    @property
    def points(self):
        """Return an artist for drawing the points"""
        if self._points is None:
            self._points = BoundaryLineCollection(
                self.name, self.labels, linewidth=self.line_width, color=self.color
            )
        return self._points
