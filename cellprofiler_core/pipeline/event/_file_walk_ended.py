from ._event import Event


class FileWalkEnded(Event):
    def event_type(self):
        return "File walk ended"
