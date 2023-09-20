from ._run_exception import RunException


class PostRunException(RunException):
    """An event indicating an uncaught exception during the post_run phase"""

    def event_type(self):
        return "Post run exception"
