from ._abstract_pipeline_event import AbstractPipelineEvent


class ModuleEditedPipelineEvent(AbstractPipelineEvent):
    """A module had its settings changed

    """

    def __init__(self, module_num, is_image_set_modification=False):
        super(ModuleEditedPipelineEvent, self).__init__(
            is_pipeline_modification=True,
            is_image_set_modification=is_image_set_modification,
        )
        self.module_num = module_num

    def event_type(self):
        return "Module edited"
