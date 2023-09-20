from ._integer_range import IntegerRange
from .._range import Range
from ..._validation_error import ValidationError


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
