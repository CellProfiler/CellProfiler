from ._abstract_pipeline_event import AbstractPipelineEvent


class EndRunEvent(AbstractPipelineEvent):
    """A run ended"""

    def event_type(self):
        return "Run ended"
