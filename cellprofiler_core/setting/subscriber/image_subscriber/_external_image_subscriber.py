from ._image_subscriber import ImageSubscriber


class ExternalImageSubscriber(ImageSubscriber):
    """A setting that provides image names loaded externally (eg: from Java)"""

    def __init__(
        self,
        text,
        value="Do not use",
        can_be_blank=False,
        blank_text="Leave blank",
        *args,
        **kwargs,
    ):
        super(ExternalImageSubscriber, self).__init__(
            text, value, can_be_blank, blank_text, *args, **kwargs
        )
