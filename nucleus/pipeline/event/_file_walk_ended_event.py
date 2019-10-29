from ._event import Event


class FileWalkEndedEvent(Event):
    def event_type(self):
        return "File walk ended"
