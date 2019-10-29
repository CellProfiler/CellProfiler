from ._event import Event


class URLsAdded(Event):
    def __init__(self, urls):
        super(self.__class__, self).__init__()
        self.urls = urls

    def event_type(self):
        return "URLs added to file list"
