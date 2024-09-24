import abc
import uuid

from ._validation_error import ValidationError


class Setting(abc.ABC):
    """A module setting which holds a single string value

    """


    #
    # This should be set to False for UI elements like buttons and dividers
    #
    save_to_pipeline = True

    def __init__(self, text, value, doc="", reset_view=False):
        """Initialize a setting with the enclosing module and its string value

        text   - the explanatory text for the setting
        value  - the default or initial value for the setting
        doc - documentation for the setting
        reset_view - True if miniscule editing should re-evaluate the module view
        """
        self.__text = text
        self.__value = value
        self.doc = doc
        self.__key = uuid.uuid4()
        self.reset_view = reset_view

    def to_dict(self, sanitize=False) -> dict:
        if sanitize and any(phrase in self.text.lower() for phrase in ("username", "password", "host")):
            text = "[SensitiveSetting]:*****"
        else:
            text = self.text
        return {"name": ".".join([self.__module__, self.__class__.__qualname__]),
                "text": text,
                "value": self.unicode_value}

    def from_dict(self, setting_dict: dict) -> "Setting":
        parts = setting_dict["name"].split('.') #example: "cellprofiler_core.setting._path_list_display.PathListDisplay"
        module = __import__(parts[0])
        for part in parts[1:]:
            module = getattr(module, part)
        # FIXME: Later you'll have to handle cases in which we do not have the expected (text, value) args
        try:
            # FIXME: Make sure to input choices to constructor of Choice
            if "Choice" in setting_dict["name"]:
                return module(setting_dict["text"], None, setting_dict["value"])
            return module(setting_dict["text"], setting_dict["value"])
        except (AttributeError, TypeError) as e: #We'll handle this later
            pass

    def set_value(self, value):
        self.__value = value

    def key(self):
        """Return a key that can be used in a dictionary to refer to this setting

        """
        return self.__key

    def get_text(self):
        """The explanatory text for the setting
        """
        return self.__text

    def set_text(self, value):
        self.__text = value

    text = property(get_text, set_text)

    def get_value(self):
        """The string contents of the setting"""
        return self.__value

    def __internal_get_value(self):
        """The value stored within the setting"""
        return self.get_value()

    def __internal_set_value(self, value):
        self.set_value(value)

    value = property(__internal_get_value, __internal_set_value)

    def get_value_text(self):
        """Get the underlying string value"""
        return self.__value

    def set_value_text(self, value):
        """Set the underlying string value

        Can be overridden as long as the base class set_value_text is
        called with the target value. An example is to allow the user to
        enter an invalid text value, but still maintain the last valid value
        entered.
        """
        self.__value = value

    def __internal_set_value_text(self, value):
        self.set_value_text(value)

    value_text = property(get_value_text, __internal_set_value_text)

    def __eq__(self, x):
        # we test explicitly for other Settings to prevent matching if
        # their .values are the same.
        if isinstance(x, Setting):
            return self.__key == x.__key
        return self.eq(x)

    def eq(self, x):
        """The equality test for things other than settings

        x - the thing to be compared, for instance a string

        override this to do things like compare whether an integer
        setting's value matches a given number
        """
        return self.value == str(x)

    def __ne__(self, x):
        return not self.__eq__(x)

    def get_is_yes(self):
        """Return true if the setting's value is "Yes" """
        return self.__value == "Yes"

    def set_is_yes(self, is_yes):
        """Set the setting value to Yes if true, No if false"""
        self.__value = (is_yes and "Yes") or "No"

    is_yes = property(get_is_yes, set_is_yes)

    def get_is_do_not_use(self):
        """Return true if the setting's value is Do not use"""
        return self.value == "Do not use"

    is_do_not_use = property(get_is_do_not_use)

    def test_valid(self, pipeline):
        """Throw a ValidationError if the value of this setting is inappropriate for the context"""
        pass

    def test_setting_warnings(self, pipeline):
        """Throw a ValidationError to warn the user about a setting value issue

        A setting should raise ValidationError if a setting's value is
        likely to be in error, but could possibly be correct. An example is
        a field that can be left blank, but is filled in, except for rare
        cases.
        """
        pass

    def __str__(self):
        """Return value as a string.

        NOTE: strings are deprecated, use unicode_value instead.
        """
        if isinstance(self.__value, str):
            return str(self.__value)
        if not isinstance(self.__value, str):
            raise ValidationError("%s was not a string" % self.__value, self)
        return self.__value

    @property
    def unicode_value(self):
        return self.get_unicode_value()

    def get_unicode_value(self):
        return str(self.value_text)
