from ._list_subscriber import List


class Image(List):
    """
    A setting that provides an image name
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
        super(Image, self).__init__(
            text, "imagegroup", value, can_be_blank, blank_text, *args, **kwargs
        )
