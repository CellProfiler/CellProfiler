import logging

from .._text import Text
from ..._validation_error import ValidationError


LOGGER = logging.getLogger(__name__)

class Number(Text):
    """A setting that allows only numeric input
    """

    def __init__(self, text, value=0, minval=None, maxval=None, *args, **kwargs):
        if isinstance(value, str):
            text_value = value
            value = self.str_to_value(value)
        else:
            text_value = self.value_to_str(value)

        if kwargs.pop("metadata", False):
            raise ValueError("metadata=True is not a valid argument for a numeric setting.")

        super(Number, self).__init__(text, text_value, *args, **kwargs)
        self.__default = self.str_to_value(text_value)
        self.__minval = minval
        self.__maxval = maxval

    def str_to_value(self, str_value):
        """Return the value of the string passed

        Override this in a derived class to parse the numeric text or
        raise an exception if badly formatted.
        """
        raise NotImplementedError("Please define str_to_value in a subclass")

    def value_to_str(self, value):
        """Return the string representation of the value passed

        Override this in a derived class to convert a numeric value into text
        """
        raise NotImplementedError("Please define value_to_str in a subclass")

    def set_value(self, value):
        """Convert integer to string
        """
        str_value = str(value) if isinstance(value, str) else self.value_to_str(value)
        self.set_value_text(str_value)

    def get_value(self, reraise=False):
        """Return the value of the setting as a float
        """
        return self.__default

    def set_value_text(self, value_text):
        super(Number, self).set_value_text(value_text)
        try:
            self.test_valid(None)
            self.__default = self.str_to_value(value_text)
        except:
            LOGGER.debug("Number set to illegal value: %s" % value_text)

    def set_min_value(self, minval):
        """Programatically set the minimum value allowed"""
        self.__minval = minval

    def set_max_value(self, maxval):
        """Programatically set the maximum value allowed"""
        self.__maxval = maxval

    def get_min_value(self):
        """The minimum value (inclusive) that can legally be entered"""
        return self.__minval

    def get_max_value(self):
        """The maximum value (inclusive) that can legally be entered"""
        return self.__maxval

    min_value = property(get_min_value, set_min_value)
    max_value = property(get_max_value, set_max_value)

    def test_valid(self, pipeline):
        """Return true only if the text value is float
        """
        try:
            value = self.str_to_value(self.value_text)
        except ValueError:
            raise ValidationError("Value not in decimal format", self)
        if self.__minval is not None and self.__minval > value:
            raise ValidationError(
                "Must be at least %s, was %s"
                % (self.value_to_str(self.__minval), self.value_text),
                self,
            )
        if self.__maxval is not None and self.__maxval < value:
            raise ValidationError(
                "Must be at most %s, was %s"
                % (self.value_to_str(self.__maxval), self.value_text),
                self,
            )

    def eq(self, x):
        """Equal if our value equals the operand"""
        return self.value == x
