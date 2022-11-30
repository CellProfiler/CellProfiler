import logging

from .._setting import Setting
from .._validation_error import ValidationError


LOGGER = logging.getLogger(__name__)

class Range(Setting):
    """A setting representing a range between two values"""

    valid_format_text = '"%s" is formatted incorrectly'

    def __init__(self, text, value, minval=None, maxval=None, *args, **kwargs):
        """Initialize a range

        text - helpful text to be displayed to the user

        value - default value as a string, should be in the form <min>,<max>

        minval - the minimum value for the range (or None if none)

        maxval - the maximum value of the range (or None if none)
        """
        super(Range, self).__init__(text, value, *args, **kwargs)
        self._minval = minval
        self._maxval = maxval
        self.__default_min = self.min
        self.__default_max = self.max

    def str_to_value(self, value_str):
        """Convert a min/max value as a string to the native type"""
        raise NotImplementedError("str_to_value must be implemented in derived class")

    def value_to_str(self, value):
        """Convert a string to a min/max value in the native type"""
        raise NotImplementedError("value_to_str must be implemented in derived class")

    def get_value(self):
        """Return the value of this range as a min/max tuple"""
        return self.min, self.max

    def set_value(self, value):
        """Set the value of this range using either a string or a two-tuple"""
        if isinstance(value, str):
            self.set_value_text(value)
        elif hasattr(value, "__getitem__") and len(value) == 2:
            self.set_value_text(",".join([self.value_to_str(v) for v in value]))
        else:
            raise ValueError("Value for range must be a string or two-tuple")

    def get_min_text(self):
        """Get the minimum of the range as a text value"""
        return self.get_value_text().split(",")[0]

    def get_min(self):
        """Get the minimum of the range as a number"""
        try:
            value = self.str_to_value(self.get_min_text())
            if self._minval is not None and value < self._minval:
                return self._minval
            return value
        except:
            return self.__default_min

    def get_max_text(self):
        """Get the maximum of the range as a text value"""
        vv = self.get_value_text().split(",")
        if len(vv) < 2:
            return ""
        return vv[1]

    def get_max(self):
        """Get the maximum of the range as a number"""
        try:
            value = self.str_to_value(self.get_max_text())
            if self._maxval is not None and value > self._maxval:
                return self._maxval
            return value
        except:
            return self.__default_max

    def compose_min_text(self, value):
        """Return the text value that would set the minimum to the proposed value

        value - the proposed minimum value as text
        """
        return ",".join((value, self.get_max_text()))

    def set_min(self, value):
        """Set the minimum part of the value, given the minimum as a #"""
        self.set_value_text(self.compose_min_text(self.value_to_str(value)))

    def compose_max_text(self, value):
        """Return the text value that would set the maximum to the proposed value

        value - the proposed maximum value as text
        """
        return ",".join((self.get_min_text(), value))

    def set_max(self, value):
        """Set the maximum part of the value, given the maximum as a #"""
        self.set_value_text(self.compose_max_text(self.value_to_str(value)))

    min = property(get_min, set_min)
    min_text = property(get_min_text)
    max = property(get_max, set_max)
    max_text = property(get_max_text)

    def set_value_text(self, value):
        super(Range, self).set_value_text(value)
        try:
            self.test_valid(None)
            self.__default_min = self.min
            self.__default_max = self.max
        except:
            LOGGER.debug("Illegal value in range setting: %s" % value)

    def test_valid(self, pipeline):
        values = self.value_text.split(",")
        if len(values) < 2:
            raise ValidationError(
                "Minimum and maximum values must be separated by a comma", self
            )
        if len(values) > 2:
            raise ValidationError("Only two values allowed", self)
        for value in values:
            try:
                self.str_to_value(value)
            except:
                raise ValidationError(self.valid_format_text % value, self)
        v_min, v_max = [self.str_to_value(value) for value in values]
        if self._minval is not None and self._minval > v_min:
            raise ValidationError(
                "%s can't be less than %s"
                % (self.min_text, self.value_to_str(self._minval)),
                self,
            )
        if self._maxval is not None and self._maxval < v_max:
            raise ValidationError(
                "%s can't be greater than %s"
                % (self.max_text, self.value_to_str(self._maxval)),
                self,
            )
        if v_min > v_max:
            raise ValidationError(
                "%s is greater than %s" % (self.min_text, self.max_text), self
            )

    def eq(self, x):
        """If the operand is a sequence, true if it matches the min and max"""
        if hasattr(x, "__getitem__") and len(x) == 2:
            return x[0] == self.min and x[1] == self.max
        return False
