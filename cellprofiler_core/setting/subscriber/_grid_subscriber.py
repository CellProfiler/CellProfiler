from ._subscriber import Subscriber


class GridSubscriber(Subscriber):
    """
    A setting that subscribes to grid information providers
    """

    def __init__(
        self,
        text,
        value="Do not use",
        can_be_blank=False,
        blank_text="Leave blank",
        *args,
        **kwargs,
    ):
        super(GridSubscriber, self).__init__(
            text, "gridgroup", value, can_be_blank, blank_text, *args, **kwargs
        )
