from ._event import Event


class EndRun(Event):
    """A run ended"""

    def event_type(self):
        return "Run ended"
