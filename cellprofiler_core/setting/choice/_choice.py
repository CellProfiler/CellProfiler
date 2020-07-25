from .._setting import Setting
from .._validation_error import ValidationError


class Choice(Setting):
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
