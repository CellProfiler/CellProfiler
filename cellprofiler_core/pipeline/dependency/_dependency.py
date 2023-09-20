class Dependency:
    """This class documents the dependency of one module on another

    A module is dependent on another if the dependent module requires
    data from the producer module. That data can be objects (label matrices),
    a derived image or measurements.
    """

    def __init__(
        self,
        source_module,
        destination_module,
        source_setting=None,
        destination_setting=None,
    ):
        """Constructor

        source_module - the module that produces the data
        destination_module - the module that uses the data
        source_setting - the module setting that names the item (can be None)
        destination_setting - the module setting in the destination that
        picks the setting
        """
        self.__source_module = source_module
        self.__destination_module = destination_module
        self.__source_setting = source_setting
        self.__destination_setting = destination_setting

    @property
    def source(self):
        """The source of the data item"""
        return self.__source_module

    @property
    def source_setting(self):
        """The setting that names the data item

        This can be None if it's ambiguous.
        """
        return self.__source_setting

    @property
    def destination(self):
        """The user of the data item"""
        return self.__destination_module

    @property
    def destination_setting(self):
        """The setting that picks the data item

        This can be None if it's ambiguous.
        """
        return self.__destination_setting
