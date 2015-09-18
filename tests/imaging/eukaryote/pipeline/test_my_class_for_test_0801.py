class MyClassForTest0801(cpm.CPModule):
    def create_settings(self):
        self.my_variable = cps.Text('', '')

    def settings(self):
        return [self.my_variable]

    module_name = "MyClassForTest0801"
    variable_revision_number = 1

    def module_class(self):
        return "cellprofiler.tests.Test_Pipeline.MyClassForTest0801"

    def get_measurement_columns(self, pipeline):
        return [(cpmeas.IMAGE,
                 self.my_variable.value,
                 "varchar(255)")]
