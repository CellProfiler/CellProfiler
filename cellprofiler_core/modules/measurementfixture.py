import cellprofiler_core.measurement
import cellprofiler_core.module
import cellprofiler_core.setting


class MeasurementFixture(cellprofiler_core.module.Module):
    module_name = "MeasurementFixture"
    category = "Test"
    variable_revision_number = 1
    do_not_check = True

    def create_settings(self):
        self.my_variable = cellprofiler_core.setting.Text("", "")
        self.other_variable = cellprofiler_core.setting.Text("", "")

    def settings(self):
        return [self.my_variable, self.other_variable]

    def module_class(self):
        return "cellprofiler_core.modules.measurementfixture.MeasurementFixture"

    def get_measurement_columns(self, pipeline):
        return [
            (cellprofiler_core.measurement.IMAGE, self.my_variable.value, "varchar(255)")
        ]