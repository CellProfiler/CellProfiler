from ._event import Event


class URLsRemoved(Event):
    def __init__(self, urls):
        super(self.__class__, self).__init__()
        self.urls = urls

    def event_type(self):
        return "URLs removed from file list"
