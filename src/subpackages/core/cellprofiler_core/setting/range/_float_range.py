from ._range import Range


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
