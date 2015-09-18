class FigureSubscriber(Setting):
    """A setting that subscribes to a figure indicator provider
    """

    def __init(self, text, value=DO_NOT_USE, *args, **kwargs):
        super(Setting, self).__init(text, value, *args, **kwargs)

    def get_choices(self, pipeline):
        choices = []
        for module in pipeline.modules():
            for setting in module.visible_settings():
                if setting.key() == self.key():
                    return choices
            choices.append("%d: %s" % (module.module_num, module.module_name))
        assert False, "Setting not among visible settings in pipeline"
