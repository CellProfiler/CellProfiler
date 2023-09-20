from ._event import Event


class ModuleShowWindow(Event):
    """A module had its "show_window" state changed

    module - the module that had its state changed
    """

    def __init__(self, module):
        super(self.__class__, self).__init__(is_pipeline_modification=True)
        self.module = module

    def event_type(self):
        return "Module show_window changed"
