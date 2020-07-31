from ._color_mixin import ColorMixin
from ..constants.artist import MODE_HIDE
from ..constants.artist import MODE_RGB
from ..constants.artist import MODE_GRAYSCALE
from ..constants.artist import MODE_COLORIZE
from ..constants.artist import MODE_COLORMAP


class ImageData(ColorMixin):
    """The data and metadata needed to display an image

             name - the name of the channel
             pixel_data - a 2D or 2D * n-channel array of values
             mode - MODE_GRAYSCALE
             color - if present, render this image using values between 0,0,0
                     and this color. Default is white
             colormap - if present, use this color map for scaling
             alpha - when compositing an image on top of another, use this
                     alpha value times the color.
             normalization - one of the NORMALIZE_ constants
             vmin - the floor of the image range in raw mode - default is 0
             vmax - the ceiling of the image range in raw mode - default is 1
    """

    def __init__(
        self,
        name,
        pixel_data,
        mode=None,
        color=(1.0, 1.0, 1.0),
        colormap=None,
        alpha=None,
        normalization=None,
        vmin=0,
        vmax=1,
    ):
        super(ImageData, self).__init__()
        self.name = name
        self.pixel_data = pixel_data
        self.__mode = mode
        self.color = color
        self.colormap = colormap
        if alpha is not None:
            self.alpha = alpha
        self.normalization = normalization
        self.vmin = vmin
        self.vmax = vmax

    def set_mode(self, mode):
        self.__mode = mode

    def get_mode(self):
        if self.__mode == MODE_HIDE:
            return MODE_HIDE
        if self.pixel_data.ndim == 3:
            return MODE_RGB
        elif self.__mode is None:
            return MODE_GRAYSCALE
        else:
            return self.__mode

    def get_raw_mode(self):
        """Get the mode as set by set_mode or in the constructor"""
        return self.__mode

    mode = property(get_mode, set_mode)

    def _using_color(self):
        return self.mode == MODE_COLORIZE

    def _using_colormap(self):
        return self.mode == MODE_COLORMAP
