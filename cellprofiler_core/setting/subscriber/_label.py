class Label(Subscriber):
    """A setting that subscribes to the list of available object names
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
        super(Label, self).__init__(
            text, "objectgroup", value, can_be_blank, blank_text, *args, **kwargs
        )
