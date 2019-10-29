from ._event import Event


class FileWalkStartedEvent(Event):
    def event_type(self):
        return "File walk started"
