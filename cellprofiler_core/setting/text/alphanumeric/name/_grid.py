from ._name import Name


class Grid(Name):
    """
    A setting that provides a GridInfo object
    """

    def __init__(self, text, value="Grid", *args, **kwargs):
        super(Grid, self).__init__(text, "gridgroup", value, *args, **kwargs)
