from ._abstract_pipeline_event import AbstractPipelineEvent


class FileWalkEndedEvent(AbstractPipelineEvent):
    def event_type(self):
        return "File walk ended"
