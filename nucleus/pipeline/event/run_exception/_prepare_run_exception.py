from ._run_exception import RunException


class PrepareRunException(RunException):
    """An event indicating an uncaught exception during the prepare_run phase"""

    def event_type(self):
        return "Prepare run exception"
