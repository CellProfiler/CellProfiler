from ._dependency import Dependency


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
