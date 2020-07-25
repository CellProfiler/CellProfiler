import matplotlib.cm

from ._choice import Choice


class Colormap(Choice):
    """Represents the choice of a colormap"""

    def __init__(self, text, value="Default", *args, **kwargs):
        try:
            names = list(matplotlib.cm.cmapnames)
        except AttributeError:
            # matplotlib 99 does not have cmapnames
            names = [
                "Spectral",
                "copper",
                "RdYlGn",
                "Set2",
                "summer",
                "spring",
                "Accent",
                "OrRd",
                "RdBu",
                "autumn",
                "Set1",
                "PuBu",
                "Set3",
                "gist_rainbow",
                "pink",
                "binary",
                "winter",
                "jet",
                "BuPu",
                "Dark2",
                "prism",
                "Oranges",
                "gist_yarg",
                "BuGn",
                "hot",
                "PiYG",
                "YlOrBr",
                "Reds",
                "spectral",
                "RdPu",
                "Greens",
                "gist_ncar",
                "PRGn",
                "gist_heat",
                "YlGnBu",
                "RdYlBu",
                "Paired",
                "flag",
                "hsv",
                "BrBG",
                "Purples",
                "cool",
                "Pastel2",
                "gray",
                "Pastel1",
                "gist_stern",
                "GnBu",
                "YlGn",
                "Greys",
                "RdGy",
                "YlOrRd",
                "PuOr",
                "PuRd",
                "gist_gray",
                "Blues",
                "PuBuGn",
                "gist_earth",
                "bone",
            ]
        names.sort()
        choices = ["Default"] + names
        super(Colormap, self).__init__(text, choices, value, *args, **kwargs)
