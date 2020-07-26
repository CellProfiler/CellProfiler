from ._setting import Setting


class Binary(Setting):
    """A setting that is represented as either true or false
    The underlying value stored in the settings slot is "Yes" or "No"
    for historical reasons.
    """

    def __init__(self, text, value, callback=None, *args, **kwargs):
        """Initialize the binary setting with the module, explanatory
        text and value. The value for a binary setting is True or
        False.
        """
        str_value = (value and "Yes") or "No"
        super(Binary, self).__init__(text, str_value, *args, **kwargs)
        self.callback = callback

    def set_value(self, value):
        """When setting, translate true and false into yes and no"""
        if value in ("Yes", "No") or isinstance(value, str):
            super(Binary, self).set_value(value)
        else:
            str_value = (value and "Yes") or "No"
            super(Binary, self).set_value(str_value)

    def get_value(self):
        """Get the value of a binary setting as a truth value
        """
        return super(Binary, self).get_value() == "Yes"

    def eq(self, x):
        if x == "No":
            x = False
        return (self.value and x) or (not self.value and not x)

    def __bool__(self):
        """Return the value when testing for True / False"""
        return self.value

    def on_event_fired(self, selection):
        if self.callback is not None:
            self.callback(selection)
