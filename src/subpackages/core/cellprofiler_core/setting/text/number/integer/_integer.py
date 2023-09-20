from .._number import Number


class Integer(Number):
    """
    A setting that allows only integer input

    Initializer:
    text - explanatory text for setting
    value - default value
    minval - minimum allowed value defaults to no minimum
    maxval - maximum allowed value defaults to no maximum
    """

    def str_to_value(self, str_value):
        return int(str_value)

    def value_to_str(self, value):
        return "%d" % value
