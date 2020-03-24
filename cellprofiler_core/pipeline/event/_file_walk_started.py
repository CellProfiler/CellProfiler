from ._event import Event


class FileWalkStarted(Event):
    def event_type(self):
        return "File walk started"
