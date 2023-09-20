import re

from .._text import Text
from ..._validation_error import ValidationError


class Alphanumeric(Text):
    """A setting for entering text values limited to alphanumeric + _ values

    This can be used for measurement names, object names, etc.
    """

    def __init__(self, text, value, *args, **kwargs):
        """Initializer

        text - the explanatory text for the setting UI

        value - the default / initial value

        first_must_be_alpha - True if the first character of the value must
                              be a letter or underbar.
        """
        kwargs = kwargs.copy()
        self.first_must_be_alpha = kwargs.pop("first_must_be_alpha", False)
        super(Alphanumeric, self).__init__(text, value, *args, **kwargs)

    def test_valid(self, pipeline):
        """Restrict names to legal ascii C variables

        First letter = a-zA-Z and underbar, second is that + digit.
        """
        self.validate_alphanumeric_text(self.value, self, self.first_must_be_alpha)

    @staticmethod
    def validate_alphanumeric_text(text, setting, first_must_be_alpha):
        """Validate text as alphanumeric, throwing a validation error if not

        text - text to be validated

        setting - blame this setting on failure

        first_must_be_alpha - True if the first letter has to be alpha or underbar
        """
        if first_must_be_alpha:
            pattern = "^[A-Za-z_][A-Za-z_0-9]*$"
            error = (
                'Names must start with an ASCII letter or underbar ("_")'
                " optionally followed by ASCII letters, underbars or digits."
            )
        else:
            pattern = "^[A-Za-z_0-9]+$"
            error = 'Only ASCII letters, digits and underbars ("_") can be ' "used here"

        match = re.match(pattern, text)
        if match is None:
            raise ValidationError(error, setting)
