from ..module import Module
from ..setting.text import Text


class MeasurementFixture(Module):
    module_name = "MeasurementFixture"
    category = "Test"
    variable_revision_number = 1
    do_not_check = True

    def create_settings(self):
        self.my_variable = Text("", "")
        self.other_variable = Text("", "")

    def settings(self):
        return [self.my_variable, self.other_variable]

    def module_class(self):
        return "cellprofiler_core.modules.measurementfixture.MeasurementFixture"

    def get_measurement_columns(self, pipeline):
        return [("Image", self.my_variable.value, "varchar(255)",)]
