from ._event import Event


class EndRunEvent(Event):
    """A run ended"""

    def event_type(self):
        return "Run ended"
