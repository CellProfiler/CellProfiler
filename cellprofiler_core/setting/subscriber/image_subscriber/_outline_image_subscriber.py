from ._image_subscriber import ImageSubscriber


class OutlineImageSubscriber(ImageSubscriber):
    """A setting that subscribes to the list of available object outline names
    """

    def __init__(
        self,
        text,
        value="None",
        can_be_blank=False,
        blank_text="Leave blank",
        *args,
        **kwargs,
    ):
        super(OutlineImageSubscriber, self).__init__(
            text, value, can_be_blank, blank_text, *args, **kwargs
        )

    def matches(self, setting):
        """Only match OutlineNameProvider variables"""
        return isinstance(setting, OutlineImageSubscriber)
