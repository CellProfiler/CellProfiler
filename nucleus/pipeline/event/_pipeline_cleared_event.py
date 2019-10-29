from ._event import Event


class PipelineClearedEvent(Event):
    """Indicates that all modules have been removed from the pipeline

    """

    def __init__(self):
        super(PipelineClearedEvent, self).__init__(
            is_pipeline_modification=True, is_image_set_modification=True
        )

    def event_type(self):
        return "PipelineCleared"
