import numpy

from ._outlines_mixin import OutlinesMixin


class MaskData(OutlinesMixin):
    """The data needed to display masks

    name - name of the mask
    mask - the binary mask
    mode - the display mode: outline, lines, overlay or inverted
    color - color of outline or line or overlay
    alpha - alpha of the outline or overlay

    MODE_OVERLAY colors the part of the mask that covers the part of the
    image that the user cares about. Ironically, the user almost certainly
    wants to see that part and MODE_INVERTED with an alpha of 1 masks the
    part that the user does not want to see and ignores the part they do.
    """

    def __init__(self, name, mask, mode=None, color=None, line_width=None, alpha=None):
        super(MaskData, self).__init__(color, line_width)
        self.name = name
        self.mask = mask
        self.mode = mode
        if alpha is not None:
            self.alpha = alpha

    @property
    def labels(self):
        """Return the mask as a sequence of labels matrices"""
        return [self.mask.astype(numpy.uint8)]

    def _using_color(self):
        return True

    def get_raw_mode(self):
        return self.mode
