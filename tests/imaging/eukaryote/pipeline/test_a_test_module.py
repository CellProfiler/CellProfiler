class ATestModule(cpm.CPModule):
    module_name = "ATestModule"
    variable_revision_number = 1

    def __init__(self, settings=[], measurement_columns=[], other_providers={}):
        super(type(self), self).__init__()
        self.__settings = settings
        self.__measurement_columns = measurement_columns
        self.__other_providers = other_providers

    def settings(self):
        return self.__settings

    def get_measurement_columns(self, pipeline):
        return self.__measurement_columns

    def other_providers(self, group):
        if group not in self.__other_providers.keys():
            return []
        return self.__other_providers[group]

    def get_categories(self, pipeline, object_name):
        categories = set()
        for cobject_name, cfeature_name, ctype \
                in self.get_measurement_columns(pipeline):
            if cobject_name == object_name:
                categories.add(cfeature_name.split("_")[0])
        return list(categories)

    def get_measurements(self, pipeline, object_name, category):
        measurements = set()
        for cobject_name, cfeature_name, ctype \
                in self.get_measurement_columns(pipeline):
            ccategory, measurement = cfeature_name.split("_", 1)
            if cobject_name == object_name and category == category:
                measurements.add(measurement)
        return list(measurements)
