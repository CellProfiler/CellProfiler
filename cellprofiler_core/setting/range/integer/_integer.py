from cellprofiler_core.setting.range._range import Range


class Integer(Range):
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
        super(Integer, self).__init__(
            text, "%d,%d" % value, minval, maxval, *args, **kwargs
        )

    def str_to_value(self, value_str):
        return int(value_str)

    def value_to_str(self, value):
        return "%d" % value
