from cellprofiler_core.constants.worker import ED_STOP, ED_SKIP

from ..pipeline import RunException


class PipelineEventListener:
    """listen for pipeline events, communicate them as necessary to the
    analysis manager."""

    def __init__(self, handle_exception_fn):
        self.handle_exception_fn = handle_exception_fn
        self.image_set_number = 0
        self.should_abort = False
        self.should_skip = False

    def reset(self):
        self.should_abort = False
        self.should_skip = False

    def handle_event(self, pipeline, event):
        if isinstance(event, RunException):
            disposition = self.handle_exception_fn(
                image_set_number=self.image_set_number,
                module_name=event.module.module_name,
                exc_info=(type(event.error), event.error, event.tb),
            )
            if disposition == ED_STOP:
                self.should_abort = True
                event.cancel_run = True
            elif disposition == ED_SKIP:
                self.should_skip = True
                event.cancel_run = False
                event.skip_thisset = True
