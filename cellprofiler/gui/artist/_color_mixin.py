import matplotlib
import numpy


class ColorMixin:
    """A mixin for burying the color, colormap and alpha in hidden attributes
    """

    def __init__(self):
        self._alpha = 1
        self._color = None
        self._colormap = None

    def _set_alpha(self, alpha):
        self._alpha = alpha
        self._on_alpha_changed()

    def _get_alpha(self):
        return self._alpha

    def get_alpha(self):
        return self._get_alpha()

    def set_alpha(self, alpha):
        self._set_alpha(alpha)

    alpha = property(get_alpha, set_alpha)

    @staticmethod
    def _on_alpha_changed():
        """Called when the alpha value is modified, default = do nothing"""
        pass

    def _get_color(self):
        """Get the color - default is the matplotlib foreground color"""
        if self._color is None:
            color = matplotlib.rcParams.get("patch.facecolor", "b")
        else:
            color = self._color
        return numpy.atleast_1d(matplotlib.colors.colorConverter.to_rgb(color))

    def _set_color(self, color):
        """Set the color"""
        self._color = color
        self._on_color_changed()

    def get_color(self):
        return ColorMixin._get_color(self)

    def set_color(self, color):
        ColorMixin._set_color(self, color)

    color = property(get_color, set_color)

    @property
    def color3(self):
        """Return the color as a 3D array"""
        return self.color[numpy.newaxis, numpy.newaxis, :]

    def _on_color_changed(self):
        """Called when the color changed, default = do nothing"""
        pass

    def _get_colormap(self):
        """Override to get the colormap"""
        if self._colormap is None:
            return matplotlib.rcParams.get("image.cmap", "jet")
        return self._colormap

    def _set_colormap(self, colormap):
        """Override to set the colormap"""
        self._colormap = colormap
        self._on_colormap_changed()

    def get_colormap(self):
        return self._get_colormap()

    def set_colormap(self, colormap):
        self._set_colormap(colormap)

    colormap = property(get_colormap, set_colormap)

    def _on_colormap_changed(self):
        """Called when the colormap changed, default = do nothing"""
        pass

    @property
    def using_alpha(self):
        """True if this data's configuration will use the alpha setting"""
        return self._using_alpha()

    @staticmethod
    def _using_alpha():
        """Override if we ever don't use the alpha setting"""
        return True

    @property
    def using_color(self):
        """True if this data's configuration will use the color setting"""
        return self._using_color()

    def _using_color(self):
        """Override this to tell the world that the data will use the color setting"""
        return False

    @property
    def using_colormap(self):
        """True if this data's configuration will use the colormap setting"""
        return self._using_colormap()

    def _using_colormap(self):
        """Override this to tell the world that the data will use the colormap setting"""
        return False
