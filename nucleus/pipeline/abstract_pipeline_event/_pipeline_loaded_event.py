from ._abstract_pipeline_event import AbstractPipelineEvent


class PipelineLoadedEvent(AbstractPipelineEvent):
    """Indicates that the pipeline has been (re)loaded

    """

    def __init__(self):
        super(PipelineLoadedEvent, self).__init__(
            is_pipeline_modification=True, is_image_set_modification=True
        )

    def event_type(self):
        return "PipelineLoaded"