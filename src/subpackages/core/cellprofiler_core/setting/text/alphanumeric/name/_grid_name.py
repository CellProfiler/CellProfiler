from ._name import Name


class GridName(Name):
    """
    A setting that provides a GridInfo object
    """

    def __init__(self, text, value="Grid", *args, **kwargs):
        super(GridName, self).__init__(text, "gridgroup", value, *args, **kwargs)
