from ._list_subscriber import ListSubscriber


class LabelListSubscriber(ListSubscriber):
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
        super(LabelListSubscriber, self).__init__(
            text, "objectgroup", value, can_be_blank, blank_text, *args, **kwargs
        )
