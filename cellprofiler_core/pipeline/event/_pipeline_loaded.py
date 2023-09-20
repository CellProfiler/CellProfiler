from ._event import Event


class PipelineLoaded(Event):
    """Indicates that the pipeline has been (re)loaded

    """

    def __init__(self):
        super(PipelineLoaded, self).__init__(
            is_pipeline_modification=True, is_image_set_modification=True
        )

    def event_type(self):
        return "PipelineLoaded"
