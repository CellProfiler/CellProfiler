import functools

from .._setting import Setting
from .._validation_error import ValidationError


class MultiChoice(Setting):
    """A setting that represents selection of multiple choices from a list"""

    def __init__(self, text, choices, value=None, *args, **kwargs):
        """Initializer

        text - the explanatory text for the setting
        choices - a sequence of string choices to be selected
        value - a list of selected choices or a comma-separated string list
        """
        super(MultiChoice, self).__init__(
            text, self.parse_value(value), *args, **kwargs
        )
        self.__choices = choices

    @staticmethod
    def parse_value(value):
        if value is None:
            return ""
        elif isinstance(value, str):
            return value
        elif hasattr(value, "__getitem__"):
            return ",".join(value)
        raise ValueError("Unexpected value type: %s" % type(value))

    def __internal_get_choices(self):
        """The sequence of strings that define the choices to be displayed"""
        return self.get_choices()

    def get_choices(self):
        """The sequence of strings that define the choices to be displayed"""
        return self.__choices

    def __internal_set_choices(self, choices):
        return self.set_choices(choices)

    def set_choices(self, choices):
        self.__choices = choices

    choices = property(__internal_get_choices, __internal_set_choices)

    def set_value(self, value):
        """Set the value of a multi-choice setting

        value is either a single string, a comma-separated string of
        multiple choices or a list of strings
        """
        super(MultiChoice, self).set_value(self.parse_value(value))

    def get_selections(self):
        """Return the currently selected values"""
        value = self.get_value()
        if len(value) == 0:
            return ()
        return value.split(",")

    selections = property(get_selections)

    def test_valid(self, pipeline):
        """Ensure that the selections are among the choices"""
        for selection in self.get_selections():
            if selection not in self.choices:
                if len(self.choices) == 0:
                    raise ValidationError("No available choices", self)
                elif len(self.choices) > 25:
                    raise ValidationError(
                        "%s is not one of the choices" % selection, self
                    )
                raise ValidationError(
                    "%s is not one of %s"
                    % (
                        selection,
                        functools.reduce(lambda x, y: "%s,%s" % (x, y), self.choices),
                    ),
                    self,
                )
