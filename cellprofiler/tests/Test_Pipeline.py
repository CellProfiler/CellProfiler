"""test_Pipeline.py - test the CellProfiler.Pipeline module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2010

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__ = "$Revision: 1$"

import os

import unittest
import numpy as np
import numpy.lib.index_tricks
import cStringIO

import cellprofiler.pipeline
import cellprofiler.objects
import cellprofiler.cpmodule
import cellprofiler.cpimage
import cellprofiler.settings
import cellprofiler.measurements
from cellprofiler.modules.injectimage import InjectImage

def module_directory():
    d = cellprofiler.pipeline.__file__
    d = os.path.split(d)[0] # ./CellProfiler/pyCellProfiler/cellProfiler
    d = os.path.split(d)[0] # ./CellProfiler/pyCellProfiler
    d = os.path.split(d)[0] # ./CellProfiler
    if not d:
        d = '..'
    return os.path.join(d,'Modules')

def image_with_one_cell():
    img = np.zeros((100,100))
    mgrid = np.lib.index_tricks.nd_grid()
    g=mgrid[0:100,0:100]-50                              # the manhattan distance from 50,50
    dist = g[0,:,:]*g[0,:,:]+g[1,:,:]*g[1,:,:]           # the 2-d distance (forgot the fancy name)
    img[ dist < 25] = (25.0-dist.astype(float)[dist<25])/25 # A circle in the middle of it
    return img

def exploding_pipeline(test):
    """Return a pipeline that fails if the run exception callback is called during a run
    """
    x = cellprofiler.pipeline.Pipeline()
    def fn(pipeline,event):
        if isinstance(event,cellprofiler.pipeline.RunExceptionEvent):
            test.assertFalse(event.error.message)
    x.add_listener(fn)
    return x
         
class TestPipeline(unittest.TestCase):
    
    def test_00_00_init(self):
        x = cellprofiler.pipeline.Pipeline()
    
    def test_06_01_run_pipeline(self):
        x = exploding_pipeline(self)
        module = InjectImage('OneCell',image_with_one_cell())
        module.set_module_num(1)
        x.add_module(module)
        x.run()
        
    def test_07_01_find_external_input_images(self):
        '''Check find_external_input_images'''
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9870

InputExternal:[module_num:1|svn_version:\'9859\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Give this image a name:Hi
    Give this image a name:Ho

OutputExternal:[module_num:2|svn_version:\'9859\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Select an image a name to export:Hi
 """
        pipeline = cellprofiler.pipeline.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(cStringIO.StringIO(data))
        external_inputs = pipeline.find_external_input_images()
        external_inputs.sort()
        self.assertEqual(len(external_inputs), 2)
        self.assertEqual(external_inputs[0], "Hi")
        self.assertEqual(external_inputs[1], "Ho")
        
    def test_07_02_find_external_output_images(self):
        '''Check find_external_input_images'''
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9870

InputExternal:[module_num:1|svn_version:\'9859\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Give this image a name:Hi

OutputExternal:[module_num:2|svn_version:\'9859\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Select an image a name to export:Hi
    Select an image a name to export:Ho
 """
        pipeline = cellprofiler.pipeline.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(cStringIO.StringIO(data))
        external_inputs = pipeline.find_external_output_images()
        self.assertEqual(len(external_inputs), 2)
        external_inputs.sort()
        self.assertEqual(external_inputs[0], "Hi")
        self.assertEqual(external_inputs[1], "Ho")
        
    def test_07_03_run_external(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9870

InputExternal:[module_num:1|svn_version:\'9859\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Give this image a name:Hi
    Give this image a name:Ho

OutputExternal:[module_num:2|svn_version:\'9859\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Select an image a name to export:Hi
    Select an image a name to export:Ho
 """
        pipeline = cellprofiler.pipeline.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(cStringIO.StringIO(data))
        np.random.seed(73)
        d = dict(Hi = np.random.uniform(size=(20,10)),
                 Ho = np.random.uniform(size=(20,10)))
        d_out = pipeline.run_external(d)
        for key in d.keys():
            self.assertTrue(d_out.has_key(key))
            np.testing.assert_array_almost_equal(d[key],d_out[key])
        
    def test_09_01_get_measurement_columns(self):
        '''Test the get_measurement_columns method'''
        x = cellprofiler.pipeline.Pipeline()
        module = MyClassForTest0801()
        module.module_num = 1
        module.my_variable.value = "foo"
        x.add_module(module)
        columns = x.get_measurement_columns()
        self.assertEqual(len(columns), 3)
        self.assertTrue(any([column[0] == 'Image' and 
                             column[1] == 'ModuleError_01MyClassForTest0801'
                             for column in columns]))
        self.assertTrue(any([column[0] == 'Image' and 
                             column[1] == 'ExecutionTime_01MyClassForTest0801'
                             for column in columns]))

        self.assertTrue(any([column[1] == "foo" for column in columns]))
        module.my_variable.value = "bar"
        columns = x.get_measurement_columns()
        self.assertEqual(len(columns), 3)
        self.assertTrue(any([column[1] == "bar" for column in columns]))
        module = MyClassForTest0801()
        module.module_num = 2
        module.my_variable.value = "foo"
        x.add_module(module)
        columns = x.get_measurement_columns()
        self.assertEqual(len(columns), 6)
        self.assertTrue(any([column[1] == "foo" for column in columns]))
        self.assertTrue(any([column[1] == "bar" for column in columns]))
        columns = x.get_measurement_columns(module)
        self.assertEqual(len(columns), 3)
        self.assertTrue(any([column[1] == "bar" for column in columns]))
    
    def test_10_01_all_groups(self):
        '''Test running a pipeline on all groups'''
        pipeline = exploding_pipeline(self)
        expects = ['PrepareRun',0]
        keys = ('foo','bar')
        groupings = (({'foo':'foo-A','bar':'bar-A'},(1,3)),
                     ({'foo':'foo-B','bar':'bar-B'},(2,4)))
        def prepare_run(pipeline, image_set_list, frame):
            self.assertEqual(expects[0], 'PrepareRun')
            for i in range(4):
                image_set_list.get_image_set(i)
            expects[0], expects[1] = ('PrepareGroup', 0)
            return True
        def prepare_group(pipeline, image_set_list, grouping, image_numbers):
            expects_state, expects_grouping = expects
            self.assertEqual(expects_state, 'PrepareGroup')
            for image_number in image_numbers:
                i = image_number-1
                image = cellprofiler.cpimage.Image(np.ones((10,10)) / (i+1))
                image_set = image_set_list.get_image_set(i)
                image_set.add('image', image)
            for key in keys:
                self.assertTrue(grouping.has_key(key))
                value = groupings[expects_grouping][0][key]
                self.assertEqual(grouping[key], value)
            if expects_grouping == 0:
                expects[0], expects[1] = ('Run', 1)
            else:
                expects[0], expects[1] = ('Run', 2)
            return True
        def run(workspace):
            expects_state, expects_image_number = expects
            image_number = workspace.measurements.image_set_number
            self.assertEqual(expects_state, 'Run')
            self.assertEqual(expects_image_number, image_number)
            image = workspace.image_set.get_image('image')
            self.assertTrue(np.all(image.pixel_data == 1.0 / image_number))
            if image_number == 1:
                expects[0],expects[1] = ('Run', 3)
            elif image_number == 2:
                expects[0],expects[1] = ('Run', 4)
            elif image_number == 3:
                expects[0],expects[1] = ('PostGroup', 0)
            else:
                expects[0],expects[1] = ('PostGroup', 1)
            workspace.measurements.add_image_measurement("mymeasurement",image_number)
        def post_group(workspace, grouping):
            expects_state, expects_grouping = expects
            self.assertEqual(expects_state, 'PostGroup')
            for key in keys:
                self.assertTrue(grouping.has_key(key))
                value = groupings[expects_grouping][0][key]
                self.assertEqual(grouping[key], value)
            if expects_grouping == 0:
                expects[0],expects[1] = ('PrepareGroup', 1)
            else:
                expects[0],expects[1] = ('PostRun', 0)
        def post_run(workspace):
            self.assertEqual(expects[0], 'PostRun')
            expects[0],expects[1] = ('Done', 0)
        
        module = GroupModule()
        module.setup((keys,groupings), prepare_run, prepare_group,
                     run, post_group, post_run)
        module.module_num = 1
        pipeline.add_module(module)
        measurements = pipeline.run()
        self.assertEqual(expects[0], 'Done')
        image_numbers = measurements.get_all_measurements("Image","mymeasurement")
        self.assertEqual(len(image_numbers), 4)
        self.assertTrue(np.all(image_numbers == np.array([1,3,2,4])))
         
    def test_10_02_one_group(self):
        '''Test running a pipeline on one group'''
        pipeline = exploding_pipeline(self)
        expects = ['PrepareRun',0]
        keys = ('foo','bar')
        groupings = (({'foo':'foo-A','bar':'bar-A'},(1,4)),
                     ({'foo':'foo-B','bar':'bar-B'},(2,5)),
                     ({'foo':'foo-C','bar':'bar-C'},(3,6)))
        def prepare_run(pipeline, image_set_list, frame):
            self.assertEqual(expects[0], 'PrepareRun')
            for i in range(6):
                image_set_list.get_image_set(i)
            expects[0], expects[1] = ('PrepareGroup', 1)
            return True
        def prepare_group(pipeline, image_set_list, grouping,*args):
            expects_state, expects_grouping = expects
            self.assertEqual(expects_state, 'PrepareGroup')
            for i in range(6):
                image = cellprofiler.cpimage.Image(np.ones((10,10)) / (i+1))
                image_set = image_set_list.get_image_set(i)
                image_set.add('image', image)
            for key in keys:
                self.assertTrue(grouping.has_key(key))
                value = groupings[expects_grouping][0][key]
                self.assertEqual(grouping[key], value)
            self.assertEqual(expects_grouping, 1)
            expects[0], expects[1] = ('Run', 2)
            return True
        
        def run(workspace):
            expects_state, expects_image_number = expects
            image_number = workspace.measurements.image_set_number
            self.assertEqual(expects_state, 'Run')
            self.assertEqual(expects_image_number, image_number)
            image = workspace.image_set.get_image('image')
            self.assertTrue(np.all(image.pixel_data == 1.0 / image_number))
            if image_number == 2:
                expects[0],expects[1] = ('Run', 5)
            elif image_number == 5:
                expects[0],expects[1] = ('PostGroup', 1)
            workspace.measurements.add_image_measurement("mymeasurement",image_number)

        def post_group(workspace, grouping):
            expects_state, expects_grouping = expects
            self.assertEqual(expects_state, 'PostGroup')
            for key in keys:
                self.assertTrue(grouping.has_key(key))
                value = groupings[expects_grouping][0][key]
                self.assertEqual(grouping[key], value)
            expects[0],expects[1] = ('PostRun', 0)
        def post_run(workspace):
            self.assertEqual(expects[0], 'PostRun')
            expects[0],expects[1] = ('Done', 0)
        
        module = GroupModule()
        module.setup((keys,groupings), prepare_run, prepare_group,
                     run, post_group, post_run)
        module.module_num = 1
        pipeline.add_module(module)
        measurements = pipeline.run(grouping = {'foo':'foo-B', 'bar':'bar-B'})
        self.assertEqual(expects[0], 'Done')
        image_numbers = measurements.get_all_measurements("Image","mymeasurement")
        self.assertEqual(len(image_numbers), 2)
        self.assertTrue(np.all(image_numbers == np.array([2,5])))
    
    def test_11_01_catch_operational_error(self):
        '''Make sure that a pipeline can catch an operational error
        
        This is a regression test of IMG-277
        '''
        module = MyClassForTest1101()
        module.module_num = 1
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_module(module)
        should_be_true = [False]
        def callback(caller, event):
            if isinstance(event, cellprofiler.pipeline.RunExceptionEvent):
                should_be_true[0] = True
        pipeline.add_listener(callback)
        pipeline.run()
        self.assertTrue(should_be_true[0])
        
    def test_12_01_img_286(self):
        '''Regression test for img-286: module name in class'''
        cellprofiler.modules.fill_modules()
        success = True
        all_keys = list(cellprofiler.modules.all_modules.keys())
        all_keys.sort()
        for k in all_keys:
            v = cellprofiler.modules.all_modules[k]
            try:
                v.module_name
            except:
                print "%s needs to define module_name as a class variable"%k
                success = False
        self.assertTrue(success)
        
    def test_13_01_save_pipeline(self):
        pipeline = cellprofiler.pipeline.Pipeline()
        cellprofiler.modules.fill_modules()
        module = cellprofiler.modules.instantiate_module("Align")
        module.module_num = 1
        pipeline.add_module(module)
        fd = cStringIO.StringIO()
        pipeline.save(fd)
        fd.seek(0)
        
        pipeline = cellprofiler.pipeline.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()), 1)
        module_out = pipeline.modules()[-1]
        for setting_in, setting_out in zip(module.settings(),
                                           module_out.settings()):
            self.assertEqual(setting_in.value, setting_out.value)
            
    def test_13_02_save_measurements(self):
        pipeline = cellprofiler.pipeline.Pipeline()
        cellprofiler.modules.fill_modules()
        module = cellprofiler.modules.instantiate_module("Align")
        module.module_num = 1
        pipeline.add_module(module)
        measurements = cellprofiler.measurements.Measurements()
        my_measurement = [np.random.uniform(size=np.random.randint(3,25))
                          for i in range(20)]
        my_image_measurement = [np.random.uniform() for i in range(20)]
        my_experiment_measurement = np.random.uniform()
        measurements.add_experiment_measurement("expt", my_experiment_measurement)
        for i in range(20):
            if i > 0:
                measurements.next_image_set()
            measurements.add_measurement("Foo","Bar", my_measurement[i])
            measurements.add_image_measurement(
                "img", my_image_measurement[i])
        fd = cStringIO.StringIO()
        pipeline.save_measurements(fd, measurements)
        fd.seek(0)
        measurements = cellprofiler.measurements.load_measurements(fd)
        my_measurement_out = measurements.get_all_measurements("Foo","Bar")
        self.assertEqual(len(my_measurement), len(my_measurement_out))
        for m_in, m_out in zip(my_measurement, my_measurement_out):
            self.assertEqual(len(m_in), len(m_out))
            self.assertTrue(np.all(m_in == m_out))
        my_image_measurement_out = measurements.get_all_measurements(
            "Image", "img")
        self.assertEqual(len(my_image_measurement),len(my_image_measurement_out))
        for m_in, m_out in zip(my_image_measurement, my_image_measurement_out):
            self.assertTrue(m_in == m_out)
        my_experiment_measurement_out = \
            measurements.get_experiment_measurement("expt")
        self.assertEqual(my_experiment_measurement, my_experiment_measurement_out)
            
        fd.seek(0)
        pipeline = cellprofiler.pipeline.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()), 1)
        module_out = pipeline.modules()[-1]
        for setting_in, setting_out in zip(module.settings(),
                                           module_out.settings()):
            self.assertEqual(setting_in.value, setting_out.value)
            
    def test_13_03_save_long_measurements(self):
        pipeline = cellprofiler.pipeline.Pipeline()
        cellprofiler.modules.fill_modules()
        module = cellprofiler.modules.instantiate_module("Align")
        module.module_num = 1
        pipeline.add_module(module)
        measurements = cellprofiler.measurements.Measurements()
        # m2 and m3 should go into panic mode because they differ by a cap
        m1_name = "dalkzfsrqoiualkjfrqealkjfqroupifaaalfdskquyalkhfaafdsafdsqteqteqtew"
        m2_name = "lkjxKJDSALKJDSAWQOIULKJFASOIUQELKJFAOIUQRLKFDSAOIURQLKFDSAQOIRALFAJ" 
        m3_name = "druxKJDSALKJDSAWQOIULKJFASOIUQELKJFAOIUQRLKFDSAOIURQLKFDSAQOIRALFAJ" 
        my_measurement = [np.random.uniform(size=np.random.randint(3,25))
                          for i in range(20)]
        my_other_measurement = [np.random.uniform(size=np.random.randint(3,25))
                                            for i in range(20)]
        my_final_measurement = [np.random.uniform(size=np.random.randint(3,25))
                                for i in range(20)]
        measurements.add_all_measurements("Foo",m1_name, my_measurement)
        measurements.add_all_measurements("Foo",m2_name, my_other_measurement)
        measurements.add_all_measurements("Foo",m3_name, my_final_measurement)
        fd = cStringIO.StringIO()
        pipeline.save_measurements(fd, measurements)
        fd.seek(0)
        measurements = cellprofiler.measurements.load_measurements(fd)
        reverse_mapping = cellprofiler.pipeline.map_feature_names([m1_name, m2_name, m3_name])
        mapping = {}
        for key in reverse_mapping.keys():
            mapping[reverse_mapping[key]] = key
        for name, expected in ((m1_name, my_measurement),
                               (m2_name, my_other_measurement),
                               (m3_name, my_final_measurement)):
            map_name = mapping[name]
            my_measurement_out = measurements.get_all_measurements("Foo",map_name)
            for m_in, m_out in zip(expected, my_measurement_out):
                self.assertEqual(len(m_in), len(m_out))
                self.assertTrue(np.all(m_in == m_out))
        

class MyClassForTest0801(cellprofiler.cpmodule.CPModule):
    def create_settings(self):
        self.my_variable = cellprofiler.settings.Text('','')
    def settings(self):
        return [self.my_variable]
    module_name = "MyClassForTest0801"
    variable_revision_number = 1
    
    def module_class(self):
        return "cellprofiler.tests.Test_Pipeline.MyClassForTest0801"
    
    def get_measurement_columns(self, pipeline):
        return [(cellprofiler.measurements.IMAGE,
                 self.my_variable.value,
                 "varchar(255)")]

class MyClassForTest1101(cellprofiler.cpmodule.CPModule):
    def create_settings(self):
        self.my_variable = cellprofiler.settings.Text('','')
    def settings(self):
        return [self.my_variable]
    module_name = "MyClassForTest1101"
    variable_revision_number = 1
    
    def module_class(self):
        return "cellprofiler.tests.Test_Pipeline.MyClassForTest1101"

    def prepare_run(self, pipeline, image_set_list, *args):
        image_set = image_set_list.get_image_set(0)
        return True
        
    def prepare_group(self, pipeline, image_set_list, *args):
        image_set = image_set_list.get_image_set(0)
        image = cellprofiler.cpimage.Image(np.zeros((5,5)))
        image_set.add("dummy", image)
        return True
    
    def run(self, *args):
        import MySQLdb
        raise MySQLdb.OperationalError("Bogus error")

class GroupModule(cellprofiler.cpmodule.CPModule):
    module_name = "Group"
    def setup(self, groupings, 
                 prepare_run_callback = None,
                 prepare_group_callback = None,
                 run_callback = None,
                 post_group_callback = None,
                 post_run_callback = None):
        self.prepare_run_callback = prepare_run_callback
        self.prepare_group_callback = prepare_group_callback
        self.run_callback = run_callback
        self.post_group_callback = post_group_callback
        self.post_run_callback = post_run_callback
        self.groupings = groupings
    def get_groupings(self, image_set_list):
        return self.groupings
    def prepare_run(self, *args):
        if self.prepare_run_callback is not None:
            return self.prepare_run_callback(*args)
        return True
    def prepare_group(self, *args):
        if self.prepare_group_callback is not None:
            return self.prepare_group_callback(*args)
        return True
    def run(self, *args):
        if self.run_callback is not None:
            self.run_callback(*args)
    def post_run(self, *args):
        if self.post_run_callback is not None:
            self.post_run_callback(*args)
    def post_group(self, *args):
        if self.post_group_callback is not None:
            self.post_group_callback(*args)

if __name__ == "__main__":
    unittest.main()
