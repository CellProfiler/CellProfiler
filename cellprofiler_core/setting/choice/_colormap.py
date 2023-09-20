import matplotlib.pyplot

from ._choice import Choice


class Colormap(Choice):
    """Represents the choice of a colormap"""

    def __init__(self, text, value="Default", *args, **kwargs):
        names = matplotlib.pyplot.colormaps()
        names.sort()
        choices = ["Default"] + names
        super(Colormap, self).__init__(text, choices, value, *args, **kwargs)
