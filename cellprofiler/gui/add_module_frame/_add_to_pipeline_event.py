class AddToPipelineEvent(object):
    def __init__(self, module_name, module_loader):
        self.module_name = module_name
        self.__module_loader = module_loader

    def get_module_loader(self):
        """Return a function that, when called, will produce a module

        The function takes one argument: the module number
        """
        return self.__module_loader

    module_loader = property(get_module_loader)
