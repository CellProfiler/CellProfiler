import logging

import six

from cellprofiler_core.setting._setting import Setting
from cellprofiler_core.setting._validation_error import ValidationError

logger = logging.getLogger(__name__)


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
        if isinstance(value, six.string_types):
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
            logger.debug("Illegal value in range setting: %s" % value)

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


class IntegerRange(Range):
    """A setting that allows only integer input between two constrained values
    """

    valid_format_text = "%s must be all digits"

    def __init__(self, text, value=(0, 1), minval=None, maxval=None, *args, **kwargs):
        """Initialize an integer range
        text  - helpful text to be displayed to the user
        value - initial default value, a two-tuple as minimum and maximum
        minval - the minimum acceptable value of either
        maxval - the maximum acceptable value of either
        """
        super(IntegerRange, self).__init__(
            text, "%d,%d" % value, minval, maxval, *args, **kwargs
        )

    def str_to_value(self, value_str):
        return int(value_str)

    def value_to_str(self, value):
        return "%d" % value


class IntegerOrUnboundedRange(IntegerRange):
    """A setting that specifies an integer range where the minimum and maximum
    can be set to unbounded by the user.

    The maximum value can be relative to the far side in which case a negative
    number is returned for slicing.
    """

    def __init__(
        self, text, value=(0, "end"), minval=None, maxval=None, *args, **kwargs
    ):
        """Initialize an integer range
        text  - helpful text to be displayed to the user
        value - initial default value, a two-tuple as minimum and maximum
        minval - the minimum acceptable value of either
        maxval - the maximum acceptable value of either
        """
        Range.__init__(
            self, text, "%s,%s" % (str(value[0]), str(value[1])), *args, **kwargs
        )

    def str_to_value(self, str_value):
        if str_value == "begin":
            return 0
        elif (self.is_abs() and str_value == "end") or (
            len(str_value) > 0 and str_value[1:] == "end"
        ):
            return "end"
        return super(IntegerOrUnboundedRange, self).str_to_value(str_value)

    def value_to_str(self, value):
        if value in ("begin", "end"):
            return value
        return super(IntegerOrUnboundedRange, self).value_to_str(value)

    def get_unbounded_min(self):
        """True if there is no minimum"""
        return self.get_min() == 0

    unbounded_min = property(get_unbounded_min)

    def get_display_min(self):
        """What to display for the minimum"""
        return self.get_min_text()

    display_min = property(get_display_min)

    def get_unbounded_max(self):
        """True if there is no maximum"""
        return self.get_max_text() == "end"

    unbounded_max = property(get_unbounded_max)

    def get_display_max(self):
        """What to display for the maximum"""
        #
        # Remove the minus sign
        #
        mt = self.get_max_text()
        if self.is_abs():
            return mt
        return mt[1:]

    display_max = property(get_display_max)

    def compose_display_max_text(self, dm_value):
        """Compose a value_text value for the setting given a max text value

        dm_value - the displayed text for the maximum of the range

        Returns a text value suitable for this setting that sets the
        maximum while keeping the minimum and abs/rel the same
        """
        if self.is_abs():
            return self.compose_max_text(dm_value)
        else:
            return self.compose_max_text("-" + dm_value)

    def is_abs(self):
        """Return True if the maximum is an absolute # of pixels

        Returns False if the # of pixels is relative to the right edge.
        """
        mt = self.get_max_text()
        return len(mt) == 0 or mt[0] != "-"

    def compose_abs(self):
        """Compose a text value that uses absolute upper bounds coordinates

        Return a text value for IntegerOrUnboundedRange that keeps the min
        and the max the same, but states that the max is the distance in pixels
        from the origin.
        """
        return self.compose_max_text(self.get_display_max())

    def compose_rel(self):
        """Compose a text value that uses relative upper bounds coordinates

        Return a text value for IntegerOrUnboundedRange that keeps the min
        and the max the same, but states that the max is the distance in pixels
        from the side of the image opposite the origin.
        """
        return self.compose_max_text("-" + self.get_display_max())

    def test_valid(self, pipeline):
        values = self.value_text.split(",")
        if len(values) < 2:
            raise ValidationError(
                "Minimum and maximum values must be separated by a comma", self
            )
        if len(values) > 2:
            raise ValidationError("Only two values allowed", self)
        if (not values[0].isdigit()) and values[0] != "begin":
            raise ValidationError("%s is not an integer" % (values[0]), self)
        if len(values[1]) == 0:
            raise ValidationError("The end value is blank", self)
        if not (
            values[1] == "end"
            or values[1].isdigit()
            or (
                values[1][0] == "-"
                and (values[1][1:].isdigit() or values[1][1:] == "end")
            )
        ):
            raise ValidationError(
                "%s is not an integer or %s" % (values[1], "end"), self
            )
        if (not self.unbounded_min) and self._minval and self._minval > self.min:
            raise ValidationError(
                "%s can't be less than %d" % (self.min_text, self._minval), self
            )
        if (not self.unbounded_max) and self._maxval and self._maxval < self.max:
            raise ValidationError(
                "%d can't be greater than %d" % (self.max, self._maxval), self
            )
        if (
            (not self.unbounded_min)
            and (not self.unbounded_max)
            and self.min > self.max > 0
        ):
            raise ValidationError("%d is greater than %d" % (self.min, self.max), self)


class FloatRange(Range):
    """A setting that allows only floating point input between two constrained values
    """

    valid_format_text = "%s must be a floating-point number"

    def __init__(self, text, value=(0, 1), *args, **kwargs):
        """Initialize an integer range
        text  - helpful text to be displayed to the user
        value - initial default value, a two-tuple as minimum and maximum
        minval - the minimum acceptable value of either
        maxval - the maximum acceptable value of either
        """
        smin, smax = [("%f" % v).rstrip("0") for v in value]
        text_value = ",".join([x + "0" if x.endswith(".") else x for x in (smin, smax)])
        super(FloatRange, self).__init__(text, text_value, *args, **kwargs)

    def str_to_value(self, value_str):
        return float(value_str)

    def value_to_str(self, value):
        return "%f" % value
