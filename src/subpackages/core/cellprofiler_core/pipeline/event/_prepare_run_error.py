from ._event import Event


class PrepareRunError(Event):
    """A user configuration error prevented CP from running the pipeline

    Modules use this class to report conditions that prevent construction
    of the image set list. An example would be if the user misconfigured
    LoadImages or NamesAndTypes and no images were matched.
    """

    def __init__(self, module, message):
        super(self.__class__, self).__init__()
        self.module = module
        self.message = message
