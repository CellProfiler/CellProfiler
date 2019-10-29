from ._event import Event


class LoadException(Event):
    """An exception was caught during pipeline loading

    """

    def __init__(self, error, module, module_name=None, settings=None):
        self.error = error
        self.cancel_run = True
        self.module = module
        self.module_name = module_name
        self.settings = settings

    def event_type(self):
        return "Pipeline load exception"
