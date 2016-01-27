'''Test the RunScript module'''

import numpy as np
import StringIO
import unittest

import cellprofiler.cpimage as cpi
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.pipeline as cpp
import cellprofiler.workspace as cpw
import cellprofiler.modules.run_script as R

class TestRunScript(unittest.TestCase):
    def make_workspace(self, 
                       script, 
                       images = None,
                       objects = None,
                       measurements = None):
        '''Make a workspace and RunScript module
        
        script - Python script to run
        
        images - a list of 3 tuples, image name, variable name and image
        
        objects - a list of 3 tuples, object name, variable name and labeling
        
        measurements - a list of 4 tuples, object name, feature name,
                       variable name and measurement or measurement vector
                       
        returns a tuple of workspace, module
        '''
        module = R.RunScript()
        module.module_num = 1
        module.script.value = script
        m = cpmeas.Measurements()
        p = cpp.Pipeline()
        p.add_module(module)
        object_set = cpo.ObjectSet()
        workspace = cpw.Workspace(p, module, m, object_set, m, None)
        
        if images is not None:
            for image_name, variable_name, image in images:
                module.add_input_image()
                group = module.input_images[-1]
                group.image_name.value = image_name
                group.variable_name.value = variable_name
                m.add(image_name, cpi.Image(image))
        if objects is not None:
            for object_name, variable_name, labeling in objects:
                module.add_input_objects()
                group = module.input_objects[-1]
                group.objects_name.value = object_name
                group.variable_name.value = variable_name
                o = cpo.Objects()
                o.segmented = labeling
                object_set.add_objects(o, object_name)
        
        if measurements is not None:
            for object_name, feature_name, variable_name, value \
                in measurements:
                module.add_input_measurement()
                group = module.input_measurements[-1]
                group.measurement_type.value = \
                    R.IO_IMAGE if object_name == cpmeas.IMAGE else R.IO_OBJECTS
                group.objects_name.value = object_name
                group.measurement.value = feature_name
                group.variable_name.value = variable_name
                m[object_name, feature_name] = value
                
        return workspace, module
    
    def test_01_00_revision_number(self):
        # If this test fails, you need to write a test for your new
        # variable revision # and increment the # here
        
        self.assertEqual(R.RunScript.variable_revision_number, 1)
        
    def test_01_01_load_v1(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20160126212053
GitHash:6b02fe9
ModuleCount:17
HasImagePlaneDetails:False
        
RunScript:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Input image count:3
    Input object count:1
    Input measurement count:1
    Output image count:3
    Output object count:1
    Output measurement count:3
    Run on which image sets?:All
    Script:import numpy as np\\u000afrom sklearn.decomposition import PCA\\u000a\\u000alabels = objects\x5B0\x5D\\u000amask = labels> 0\\u000acolor = np.dstack((red, green, blue))\\u000apca = PCA()\\u000apca.fit(color\x5Bmask, \x3A\x5D)\\u000ac = pca.transform(color.reshape(np.prod(color.shape\x5B\x3A2\x5D), color.shape\x5B2\x5D))\\u000ac1 = c\x5B\x3A, 0\x5D.reshape(red.shape)\\u000ac1\x5Bmask\x5D = c1\x5Bmask\x5D * measurement\x5Blabels\x5Bmask\x5D-1\x5D/100\\u000ac2 = c\x5B\x3A, 1\x5D.reshape(red.shape)\\u000ac3 = c\x5B\x3A, 2\x5D.reshape(red.shape)\\u000aev1, ev2, ev3 = pca.explained_variance_
    Input image:CropBlue
    Variable name:red
    Input image:CropGreen
    Variable name:green
    Input image:CropBlue
    Variable name:blue
    Input objects:Cells
    Variable name:objects
    Measurement type:Object measurement
    Measured objects:Nuclei
    Measurement:Location_Center_X
    Variable name:measurement
    Output image:c1a
    Variable name:c1
    Output image:c2a
    Variable name:c2
    Output image:c3a
    Variable name:c3
    Output objects:myobjects
    Variable name:variablename
    Measurement name:Script_ev1
    Variable name:ev1
    Measurement type:Image measurement
    Objects name:Nuclei
    Data type:Integer
    Measurement name:Script_ev2
    Variable name:ev2
    Measurement type:Image measurement
    Objects name:Nuclei
    Data type:Decimal
    Measurement name:Script_ev3
    Variable name:ev3
    Measurement type:Image measurement
    Objects name:Nuclei
    Data type:Text
"""
        pipeline = cpp.Pipeline()
        pipeline.loadtxt(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        assert isinstance(module, R.RunScript)
        self.assertEqual(module.when, R.WHEN_ALL)
        self.assertEqual(len(module.input_images), 3)
        group = module.input_images[0]
        self.assertTrue(module.script.value.startswith("import numpy"))
        self.assertEqual(group.image_name, "CropBlue")
        self.assertEqual(group.variable_name, "red")
        
        self.assertEqual(len(module.input_objects), 1)
        group = module.input_objects[0]
        self.assertEqual(group.objects_name, "Cells")
        self.assertEqual(group.variable_name, "objects")
        
        self.assertEqual(len(module.input_measurements), 1)
        group = module.input_measurements[0]
        self.assertEqual(group.measurement_type, R.IO_OBJECTS)
        self.assertEqual(group.objects_name, "Nuclei")
        self.assertEqual(group.measurement, "Location_Center_X")
        self.assertEqual(group.variable_name, "measurement")
        
        self.assertEqual(len(module.output_images), 3)
        group = module.output_images[0]
        self.assertEqual(group.image_name, "c1a")
        self.assertEqual(group.variable_name, "c1")
        
        self.assertEqual(len(module.output_objects), 1)
        group = module.output_objects[0]
        self.assertEqual(group.objects_name, "myobjects")
        self.assertEqual(group.variable_name, "variablename")
        
        self.assertEqual(len(module.output_measurements), 3)
        group = module.output_measurements[0]
        self.assertEqual(group.measurement_name, "Script_ev1")
        self.assertEqual(group.variable_name, "ev1")
        self.assertEqual(group.measurement_type, R.IO_IMAGE)
        self.assertEqual(group.objects_name, "Nuclei")
        self.assertEqual(group.data_type, R.DT_INTEGER)
        self.assertEqual(module.output_measurements[1].data_type, R.DT_DECIMAL)
        self.assertEqual(module.output_measurements[2].data_type, R.DT_TEXT)
        
    def test_02_01_script(self):
        script = """workspace.module.category = "foo"
"""
        workspace, module = self.make_workspace(script)
        module.run(workspace)
        self.assertEqual(module.category, "foo")
        
    def test_02_02_image(self):
        script = """workspace.module.category = np.prod(image.shape)
"""
        workspace, module = self.make_workspace(
            script,
            images = [("myimage", "image", np.zeros((5,5)))])
        module.run(workspace)
        self.assertEqual(module.category, 25)
        
    def test_02_03_objects(self):
        script = """workspace.module.category = np.sum(objects[0])
"""
        labels = np.zeros((5,5), int)
        labels[2,1] = 1
        labels[2,3] = 2
        workspace, module = self.make_workspace(
            script, objects = [("myobjects", "objects", labels)])
        module.run(workspace)
        self.assertEqual(module.category, 3)
        
    def test_02_04_image_measurements(self):
        script = """workspace.module.category = measurement
"""
        workspace, module = self.make_workspace(
            script, measurements = [(cpmeas.IMAGE, "foo", "measurement", 1.5)])
        module.run(workspace)
        self.assertEqual(module.category, 1.5)
        
    def test_02_05_object_measurements(self):
        script = """workspace.module.category = measurement
"""
        workspace, module = self.make_workspace(
            script, 
            measurements = [("Nuclei", "foo", "measurement", 
                             np.array([1.5, 2.6]))])
        module.run(workspace)
        self.assertSequenceEqual(module.category.tolist(), [1.5, 2.6])
        
    def test_03_01_image_output(self):
        script = """
import numpy as np
image = np.arange(9).reshape(3,3).astype(float) / 9
"""
        workspace, module = self.make_workspace(script)
        assert isinstance(module, R.RunScript)
        module.add_output_image()
        group = module.output_images[0]
        group.image_name.value = "foo"
        group.variable_name.value = "image"
        module.run(workspace)
        image = workspace.image_set.get_image("foo")
        np.testing.assert_almost_equal(
            image.pixel_data, 
            np.arange(9).reshape(3,3).astype(float)/9,
            decimal = 4)
        
    def test_03_02_object_output(self):
        script = """
import numpy as np
objects = np.zeros((5,5), int)
objects[2,1] = 1
objects[2,3] = 2
objects = [objects]
"""
        workspace, module = self.make_workspace(script)
        module.add_output_objects()
        group = module.output_objects[0]
        group.objects_name.value = "foo"
        group.variable_name.value = "objects"
        module.run(workspace)
        objects = workspace.object_set.get_objects("foo")
        self.assertEqual(objects.count, 2)
        self.assertEqual(objects.segmented[2,1], 1)
        
    def test_03_03_image_measurement_output(self):
        script = """measurement = "Hello"
"""
        workspace, module = self.make_workspace(script)
        module.add_output_measurement()
        group = module.output_measurements[0]
        group.measurement_name.value = "Script_foo"
        group.variable_name.value = "measurement"
        group.measurement_type.value = R.IO_IMAGE
        group.data_type.value = R.DT_TEXT
        module.run(workspace)
        value = workspace.measurements[cpmeas.IMAGE, "Script_foo"]
        self.assertEqual(value, "Hello")
        columns = module.get_measurement_columns()
        self.assertEqual(len(columns), 1)
        self.assertEqual(columns[0][0], cpmeas.IMAGE)
        self.assertEqual(columns[0][1], "Script_foo")
        self.assertEqual(columns[0][2], cpmeas.COLTYPE_VARCHAR)
        
    def test_03_04_object_measurement_output(self):
        script = """
import numpy as np
measurement1 = np.array([5.1, 6.2])
measurement2 = np.array([7.3, 8.4])
"""
        workspace, module = self.make_workspace(script)
        module.add_output_measurement()
        group = module.output_measurements[0]
        group.measurement_name.value = "Script_foo"
        group.variable_name.value = "measurement1"
        group.measurement_type.value = R.IO_OBJECTS
        group.objects_name.value = "Nuclei"
        group.data_type.value = R.DT_INTEGER
        group.measurement_name.value = "Script_foo"
        group.variable_name.value = "measurement2"
        group.measurement_type.value = R.IO_OBJECTS
        group.objects_name.value = "Cells"
        group.data_type.value = R.DT_DECIMAL
        module.run(workspace)
        value = workspace.measurements["Nuclei", "Script_foo"]
        self.assertSequenceEqual(value.tolist(), [5, 6] )
        value = workspace.measurements["Cells", "Script_foo"]
        self.assertSequenceEqual(value.tolist(), [7.3, 8.4] )
        columns = module.get_measurement_columns()
        self.assertEqual(len(columns), 2)
        self.assertEqual(columns[0][0], "Nuclei")
        self.assertEqual(columns[0][1], "Script_foo")
        self.assertEqual(columns[0][2], cpmeas.COLTYPE_INTEGER)
        self.assertEqual(columns[1][2], cpmeas.COLTYPE_FLOAT)
        