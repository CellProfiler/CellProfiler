from ._choice import Choice
from .._setting import Setting


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
        Setting.set_value(self, value)
