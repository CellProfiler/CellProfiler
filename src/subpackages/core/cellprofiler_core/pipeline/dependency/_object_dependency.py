from ._dependency import Dependency


class ObjectDependency(Dependency):
    """A dependency on an object labeling"""

    def __init__(
        self,
        source_module,
        destination_module,
        object_name,
        source_setting=None,
        destination_setting=None,
    ):
        super(type(self), self).__init__(
            source_module, destination_module, source_setting, destination_setting
        )
        self.__object_name = object_name

    @property
    def object_name(self):
        """The name of the objects produced by the source and used by the dest"""
        return self.__object_name

    def __str__(self):
        return "Object: %s" % self.object_name
