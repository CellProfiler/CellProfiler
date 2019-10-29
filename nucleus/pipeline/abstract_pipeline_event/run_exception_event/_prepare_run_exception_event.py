from ._run_exception_event import RunExceptionEvent


class PrepareRunExceptionEvent(RunExceptionEvent):
    """An event indicating an uncaught exception during the prepare_run phase"""

    def event_type(self):
        return "Prepare run exception"
