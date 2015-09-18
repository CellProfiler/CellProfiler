class CroppingNameSubscriber(ImageNameSubscriber):
    """A setting that provides image names that have cropping masks"""

    def __init__(self, text, value=DO_NOT_USE, can_be_blank=False,
                 blank_text=LEAVE_BLANK, *args, **kwargs):
        kwargs = kwargs.copy()
        if not kwargs.has_key(REQUIRED_ATTRIBUTES):
            kwargs[REQUIRED_ATTRIBUTES] = {}
        kwargs[REQUIRED_ATTRIBUTES][CROPPING_ATTRIBUTE] = True
        super(CroppingNameSubscriber, self).__init__(text, value, can_be_blank,
                                                     blank_text, *args,
                                                     **kwargs)
