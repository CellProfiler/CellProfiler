class GridNameSubscriber(NameSubscriber):
    """A setting that subscribes to grid information providers
    """

    def __init__(self, text, value=DO_NOT_USE, can_be_blank=False,
                 blank_text=LEAVE_BLANK, *args, **kwargs):
        super(GridNameSubscriber, self).__init__(text, GRID_GROUP, value,
                                                 can_be_blank, blank_text,
                                                 *args, **kwargs)
