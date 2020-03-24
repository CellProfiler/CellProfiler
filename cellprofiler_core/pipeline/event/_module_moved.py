from ._event import Event


class ModuleMoved(Event):
    """A module moved up or down

    """

    def __init__(self, module_num, direction, is_image_set_modification):
        super(ModuleMoved, self).__init__(
            is_pipeline_modification=True,
            is_image_set_modification=is_image_set_modification,
        )
        self.module_num = module_num
        self.direction = direction

    def event_type(self):
        return "Module moved"
