import matplotlib.cm

from . import _setting
from ._validation_error import ValidationError


class Choice(_setting.Setting):
    """A setting that displays a drop-down set of choices

    """

    def __init__(
        self, text, choices, value=None, tooltips=None, choices_fn=None, *args, **kwargs
    ):
        """Initializer
        module - the module containing the setting
        text - the explanatory text for the setting
        choices - a sequence of string choices to be displayed in the drop-down
        value - the default choice or None to choose the first of the choices.
        tooltips - a dictionary of choice to tooltip
        choices_fn - a function that, if present, supplies the choices. The
                     function should have the signature, fn(pipeline).
        """
        super(Choice, self).__init__(text, value or choices[0], *args, **kwargs)
        self.__choices = choices
        self.__tooltips = tooltips
        self.__choices_fn = choices_fn

    def __internal_get_choices(self):
        """The sequence of strings that define the choices to be displayed"""
        return self.get_choices()

    def get_choices(self):
        """The sequence of strings that define the choices to be displayed"""
        return self.__choices

    choices = property(__internal_get_choices)

    def get_tooltips(self):
        """The tooltip strings for each choice"""
        return self.__tooltips

    tooltips = property(get_tooltips)

    @property
    def has_tooltips(self):
        """Return true if the choice has tooltips installed"""
        return self.__tooltips is not None

    def test_valid(self, pipeline):
        """Check to make sure that the value is among the choices"""
        if self.__choices_fn is not None:
            self.__choices = self.__choices_fn(pipeline)
        if self.value not in self.choices:
            raise ValidationError(
                "%s is not one of %s" % (self.value, ",".join(self.choices)), self
            )


class CustomChoice(Choice):
    def __init__(self, text, choices, value=None, *args, **kwargs):
        """Initializer
        text - the explanatory text for the setting
        choices - a sequence of string choices to be displayed in the drop-down
        value - the default choice or None to choose the first of the choices.
        """
        super(CustomChoice, self).__init__(text, choices, value, *args, **kwargs)

    def get_choices(self):
        """Put the custom choice at the top"""
        choices = list(super(CustomChoice, self).get_choices())
        if self.value not in choices:
            choices.insert(0, self.value)
        return choices

    def set_value(self, value):
        """Bypass the check in "Choice"."""
        _setting.Setting.set_value(self, value)


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
