from ._event import Event


class ModuleAdded(Event):
    """A module was added to the pipeline

    """

    def __init__(self, module_num, is_image_set_modification=False):
        super(ModuleAdded, self).__init__(
            is_pipeline_modification=True,
            is_image_set_modification=is_image_set_modification,
        )
        self.module_num = module_num

    def event_type(self):
        return "Module Added"
