from ._abstract_pipeline_event import AbstractPipelineEvent


class FileWalkStartedEvent(AbstractPipelineEvent):
    def event_type(self):
        return "File walk started"
