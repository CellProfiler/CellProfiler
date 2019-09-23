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


class MeasurementDependency(Dependency):
    """A dependency on a measurement"""

    def __init__(
        self,
        source_module,
        destination_module,
        object_name,
        feature,
        source_setting=None,
        destination_setting=None,
    ):
        """Initialize using source, destination and measurement

        source_module - module producing the measurement

        destination_module - module using the measurement

        object_name - the measurement is made on the objects with this name
        (or Image for image measurements)

        feature - the feature name for the measurement, for instance AreaShape_Area

        source_setting - the module setting that controls production of this
        measurement (very typically = None for no such thing)

        destination_setting - the module setting that chooses the measurement
        for the user of the data, for instance a MeasurementSetting
        """
        super(type(self), self).__init__(
            source_module, destination_module, source_setting, destination_setting
        )
        self.__object_name = object_name
        self.__feature = feature

    @property
    def object_name(self):
        """The objects / labels used when producing the measurement"""
        return self.__object_name

    @property
    def feature(self):
        """The name of the measurement"""
        return self.__feature

    def __str__(self):
        return "Measurement: %s.%s" % (self.object_name, self.feature)
