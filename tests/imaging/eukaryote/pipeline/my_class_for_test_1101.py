class MyClassForTest1101(cpm.CPModule):
    def create_settings(self):
        self.my_variable = cps.Text('', '')

    def settings(self):
        return [self.my_variable]

    module_name = "MyClassForTest1101"
    variable_revision_number = 1

    def module_class(self):
        return "cellprofiler.tests.Test_Pipeline.MyClassForTest1101"

    def prepare_run(self, workspace, *args):
        image_set = workspace.image_set_list.get_image_set(0)
        workspace.measurements.add_measurement("Image", "Foo", 1)
        return True

    def prepare_group(self, workspace, *args):
        image_set = workspace.image_set_list.get_image_set(0)
        image = cpi.Image(np.zeros((5, 5)))
        image_set.add("dummy", image)
        return True

    def run(self, *args):
        import MySQLdb
        raise MySQLdb.OperationalError("Bogus error")
