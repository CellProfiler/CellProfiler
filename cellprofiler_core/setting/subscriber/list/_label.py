class ListObjectNameSubscriber(List):
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
        super(ListObjectNameSubscriber, self).__init__(
            text, "objectgroup", value, can_be_blank, blank_text, *args, **kwargs
        )
