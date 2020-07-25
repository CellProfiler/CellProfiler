from ._number import Number


class Float(Number):
    """
    A class that only allows floating point input
    """

    def str_to_value(self, str_value):
        return float(str_value)

    def value_to_str(self, value):
        text_value = ("%f" % value).rstrip("0")
        if text_value.endswith("."):
            text_value += "0"
        return text_value
