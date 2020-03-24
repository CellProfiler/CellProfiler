from ._event import Event


class ModuleEnabled(Event):
    """A module was enabled

    module - the module that was enabled.
    """

    def __init__(self, module):
        """Constructor

        module - the module that was enabled
        """
        super(self.__class__, self).__init__(is_pipeline_modification=True)
        self.module = module

    def event_type(self):
        return "Module enabled"
