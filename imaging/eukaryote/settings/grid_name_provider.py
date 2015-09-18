class GridNameProvider(NameProvider):
    """A setting that provides a GridInfo object
    """

    def __init__(self, text, value="Grid", *args, **kwargs):
        super(GridNameProvider, self).__init__(text, GRID_GROUP, value,
                                               *args, **kwargs)
