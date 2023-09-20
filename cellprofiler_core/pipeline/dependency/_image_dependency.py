from ._dependency import Dependency


class ImageDependency(Dependency):
    """A dependency on an image"""

    def __init__(
        self,
        source_module,
        destination_module,
        image_name,
        source_setting=None,
        destination_setting=None,
    ):
        super(type(self), self).__init__(
            source_module, destination_module, source_setting, destination_setting
        )
        self.__image_name = image_name

    @property
    def image_name(self):
        """The name of the image produced by the source and used by the dest"""
        return self.__image_name

    def __str__(self):
        return "Image: %s" % self.image_name
