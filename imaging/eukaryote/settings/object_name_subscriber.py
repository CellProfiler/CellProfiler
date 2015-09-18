class ObjectNameSubscriber(NameSubscriber):
    """A setting that subscribes to the list of available object names
    """

    def __init__(self, text, value=DO_NOT_USE, can_be_blank=False,
                 blank_text=LEAVE_BLANK, *args, **kwargs):
        super(ObjectNameSubscriber, self).__init__(text, OBJECT_GROUP, value,
                                                   can_be_blank, blank_text,
                                                   *args, **kwargs)
