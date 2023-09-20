from ._event import Event


class IPDLoadException(Event):
    """An exception was cauaght while trying to load the image plane details

    This event is reported when an exception is thrown while loading
    the image plane details from the workspace's file list.
    """

    def __init__(self, error):
        super(self.__class__, self).__init__()
        self.error = error
        self.cancel_run = True

    def event_type(self):
        return "Image load exception"
