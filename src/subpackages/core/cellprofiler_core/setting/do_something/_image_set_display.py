from ._do_something import DoSomething


class ImageSetDisplay(DoSomething):
    """A button that refreshes the image set display when pressed

    """

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(
            args[0], args[1], None, *args[:2], **kwargs
        )
