class ExternalImageNameSubscriber(ImageNameSubscriber):
    """A setting that provides image names loaded externally (eg: from Java)"""

    def __init__(self, text, value=DO_NOT_USE, can_be_blank=False,
                 blank_text=LEAVE_BLANK, *args, **kwargs):
        super(ExternalImageNameSubscriber, self).__init__(text, value, can_be_blank,
                                                          blank_text, *args,
                                                          **kwargs)
