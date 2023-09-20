class Event:
    """Something that happened to the pipeline and was indicated to the listeners
    """

    def __init__(self, is_pipeline_modification=False, is_image_set_modification=False):
        self.is_pipeline_modification = is_pipeline_modification
        self.is_image_set_modification = is_image_set_modification

    def event_type(self):
        raise NotImplementedError(
            "AbstractPipelineEvent does not implement an event type"
        )


class CancelledException(Exception):
    """Exception issued by the analysis worker indicating cancellation by UI

    This is here in order to solve some import dependency problems
    """

    pass


class PipelineLoadCancelledException(Exception):
    """Exception thrown if user cancels pipeline load"""

    pass
