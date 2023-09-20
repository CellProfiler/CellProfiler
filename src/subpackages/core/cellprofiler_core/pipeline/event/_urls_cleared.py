from ._event import Event


class URLsCleared(Event):
    def __init__(self):
        super(self.__class__, self).__init__()

    def event_type(self):
        return "File list has been cleared"
