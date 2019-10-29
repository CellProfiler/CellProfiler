from ._run_exception_event import RunExceptionEvent


class PostRunExceptionEvent(RunExceptionEvent):
    """An event indicating an uncaught exception during the post_run phase"""

    def event_type(self):
        return "Post run exception"
