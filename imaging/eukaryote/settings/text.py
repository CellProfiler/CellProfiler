class Text(Setting):
    """A setting that displays as an edit box, accepting a string

    """

    def __init__(self, text, value, *args, **kwargs):
        kwargs = kwargs.copy()
        self.multiline_display = kwargs.pop("multiline", False)
        self.metadata_display = kwargs.pop(METADATA_ATTRIBUTE, False)
        super(Text, self).__init__(text, value, *args, **kwargs)
