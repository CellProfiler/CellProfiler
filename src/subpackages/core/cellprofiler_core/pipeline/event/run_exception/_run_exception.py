from .._event import Event


class RunException(Event):
    """An exception was caught during a pipeline run

    Initializer:
    error - exception that was thrown
    module - module that was executing
    tb - traceback at time of exception, e.g from sys.exc_info
    """

    def __init__(self, error, module, tb=None):
        self.error = error
        self.cancel_run = True
        self.skip_thisset = False
        self.module = module
        self.tb = tb

    def event_type(self):
        return "Pipeline run exception"
