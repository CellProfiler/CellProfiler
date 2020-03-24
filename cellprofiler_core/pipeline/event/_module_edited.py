from ._event import Event


class ModuleEdited(Event):
    """A module had its settings changed

    """

    def __init__(self, module_num, is_image_set_modification=False):
        super(ModuleEdited, self).__init__(
            is_pipeline_modification=True,
            is_image_set_modification=is_image_set_modification,
        )
        self.module_num = module_num

    def event_type(self):
        return "Module edited"
