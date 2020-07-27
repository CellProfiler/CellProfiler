from .._subscriber import Subscriber


class ImageSubscriber(Subscriber):
    """A setting that provides an image name
    """

    def __init__(
        self,
        text,
        value=None,
        can_be_blank=False,
        blank_text="Leave blank",
        *args,
        **kwargs,
    ):
        super(ImageSubscriber, self).__init__(
            text, "imagegroup", value, can_be_blank, blank_text, *args, **kwargs
        )
