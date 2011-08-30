"""test_Pipeline.py - test the CellProfiler.Pipeline module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__ = "$Revision: 1$"

import os
import cProfile
import pstats

import base64
import unittest
import numpy as np
import numpy.lib.index_tricks
import cStringIO
import zlib

import cellprofiler.pipeline as cpp
import cellprofiler.objects as cpo
import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.settings as cps
import cellprofiler.measurements as cpmeas
import cellprofiler.workspace as cpw
import cellprofiler.preferences as cpprefs
import cellprofiler.modules
from cellprofiler.modules.injectimage import InjectImage

from cellprofiler.modules.tests import example_images_directory

basedir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(basedir,'data')

def module_directory():
    d = cpp.__file__
    d = os.path.split(d)[0] # ./CellProfiler/pyCellProfiler/cellProfiler
    d = os.path.split(d)[0] # ./CellProfiler/pyCellProfiler
    d = os.path.split(d)[0] # ./CellProfiler
    if not d:
        d = '..'
    return os.path.join(d,'Modules')

def image_with_one_cell(size=(100,100)):
    img = np.zeros(size)
    mgrid = np.lib.index_tricks.nd_grid()
    g=mgrid[0:100,0:100]-50                              # the manhattan distance from 50,50
    dist = g[0,:,:]*g[0,:,:]+g[1,:,:]*g[1,:,:]           # the 2-d distance (forgot the fancy name)
    img[ dist < 25] = (25.0-dist.astype(float)[dist<25])/25 # A circle in the middle of it
    return img

def exploding_pipeline(test):
    """Return a pipeline that fails if the run exception callback is called during a run
    """
    x = cpp.Pipeline()
    def fn(pipeline,event):
        if isinstance(event, cpp.RunExceptionEvent):
            import traceback
            test.assertFalse(
                isinstance(event, cpp.RunExceptionEvent),
                "\n".join ([event.error.message] + traceback.format_tb(event.tb)))
    x.add_listener(fn)
    return x
         
class TestPipeline(unittest.TestCase):
    
    def test_00_00_init(self):
        x = cpp.Pipeline()
    def test_01_01_load_mat(self):
        '''Regression test of img-942, load a batch data pipeline with notes'''
        
        data_fi_path = os.path.join(datadir,'img-942.b64.gz')
        data_fi = open(data_fi_path,'r')
        data = data_fi.readline()
        data_fi.close()
        
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(cStringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        module = pipeline.modules()[0]
        self.assertEqual(len(module.notes), 1)
        self.assertEqual(
            module.notes[0], 
            """Excluding "_E12f03d" since it has an incomplete set of channels (and is the only one as such).""")
    
    def test_06_01_run_pipeline(self):
        x = exploding_pipeline(self)
        module = InjectImage('OneCell',image_with_one_cell())
        module.set_module_num(1)
        x.add_module(module)
        x.run()
        
    def test_06_02_memory(self):
        '''Run a pipeline and check for memory leaks'''
        from contrib.objgraph import get_objs
        
        np.random.seed(62)
        #
        # Get a size that's unlikely to be found elsewhere
        #
        size = (np.random.uniform(size=2) * 300 + 100).astype(int)
        x = exploding_pipeline(self)
        
        data_fi_path = os.path.join(datadir,'memtest.cp')
        
        x.load(data_fi_path)
        module = InjectImage('OneCell',image_with_one_cell(size),
                             release_image = True)
        module.set_module_num(1)
        x.add_module(module)
        for m in x.run_with_yield(run_in_background = False):
            pass
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        self.assertEqual(m.image_set_count, 1)
        del m
        for obj in get_objs():
            if isinstance(obj, np.ndarray) and obj.ndim > 1:
                self.assertTrue(tuple(obj.shape[:2]) != tuple(size))
        
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
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
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
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(cStringIO.StringIO(data))
        external_outputs = pipeline.find_external_output_images()
##        self.assertEqual(len(external_outputs), 2)
        external_outputs.sort()
        self.assertEqual(external_outputs[0], "Hi")
        self.assertEqual(external_outputs[1], "Ho")
        
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
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
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
        x = cpp.Pipeline()
        module = MyClassForTest0801()
        module.module_num = 1
        module.my_variable.value = "foo"
        x.add_module(module)
        columns = x.get_measurement_columns()
        self.assertEqual(len(columns), 6)
        self.assertTrue(any([column[0] == 'Image' and 
                             column[1] == 'Group_Number' and
                             column[2] == cpmeas.COLTYPE_INTEGER
                             for column in columns]))
        self.assertTrue(any([column[0] == 'Image' and 
                             column[1] == 'Group_Index' and
                             column[2] == cpmeas.COLTYPE_INTEGER
                             for column in columns]))
        self.assertTrue(any([column[0] == 'Image' and 
                             column[1] == 'ModuleError_01MyClassForTest0801'
                             for column in columns]))
        self.assertTrue(any([column[0] == 'Image' and 
                             column[1] == 'ExecutionTime_01MyClassForTest0801'
                             for column in columns]))
        self.assertTrue(any([column[0] == cpmeas.EXPERIMENT and
                             column[1] == cpp.M_PIPELINE
                             for column in columns]))

        self.assertTrue(any([column[1] == "foo" for column in columns]))
        module.my_variable.value = "bar"
        columns = x.get_measurement_columns()
        self.assertEqual(len(columns), 6)
        self.assertTrue(any([column[1] == "bar" for column in columns]))
        module = MyClassForTest0801()
        module.module_num = 2
        module.my_variable.value = "foo"
        x.add_module(module)
        columns = x.get_measurement_columns()
        self.assertEqual(len(columns), 9)
        self.assertTrue(any([column[1] == "foo" for column in columns]))
        self.assertTrue(any([column[1] == "bar" for column in columns]))
        columns = x.get_measurement_columns(module)
        self.assertEqual(len(columns), 6)
        self.assertTrue(any([column[1] == "bar" for column in columns]))
    
    def test_10_01_all_groups(self):
        '''Test running a pipeline on all groups'''
        pipeline = exploding_pipeline(self)
        expects = ['PrepareRun',0]
        keys = ('foo','bar')
        groupings = (({'foo':'foo-A','bar':'bar-A'},(1,3)),
                     ({'foo':'foo-B','bar':'bar-B'},(2,4)))
        def prepare_run(workspace):
            image_set_list = workspace.image_set_list
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
                image = cpi.Image(np.ones((10,10)) / (i+1))
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
            
        def get_measurement_columns(pipeline):
            return [(cpmeas.IMAGE, "mymeasurement", 
                     cpmeas.COLTYPE_INTEGER)]
        
        module = GroupModule()
        module.setup((keys,groupings), prepare_run, prepare_group,
                     run, post_group, post_run, get_measurement_columns)
        module.module_num = 1
        pipeline.add_module(module)
        measurements = pipeline.run()
        self.assertEqual(expects[0], 'Done')
        image_numbers = measurements.get_all_measurements("Image","mymeasurement")
        self.assertEqual(len(image_numbers), 4)
        self.assertTrue(np.all(image_numbers == np.array([1,2,3,4])))
        group_numbers = measurements.get_all_measurements("Image","Group_Number")
        self.assertTrue(np.all(group_numbers == np.array([1,2,1,2])))
        group_indexes = measurements.get_all_measurements("Image","Group_Index")
        self.assertTrue(np.all(group_indexes == np.array([1,1,2,2])))
         
    def test_10_02_one_group(self):
        '''Test running a pipeline on one group'''
        pipeline = exploding_pipeline(self)
        expects = ['PrepareRun',0]
        keys = ('foo','bar')
        groupings = (({'foo':'foo-A','bar':'bar-A'},(1,4)),
                     ({'foo':'foo-B','bar':'bar-B'},(2,5)),
                     ({'foo':'foo-C','bar':'bar-C'},(3,6)))
        def prepare_run(workspace):
            self.assertEqual(expects[0], 'PrepareRun')
            for i in range(6):
                workspace.image_set_list.get_image_set(i)
            expects[0], expects[1] = ('PrepareGroup', 1)
            return True
        def prepare_group(pipeline, image_set_list, grouping,*args):
            expects_state, expects_grouping = expects
            self.assertEqual(expects_state, 'PrepareGroup')
            for i in range(6):
                image = cpi.Image(np.ones((10,10)) / (i+1))
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
        def get_measurement_columns(pipeline):
            return [(cpmeas.IMAGE, "mymeasurement", 
                     cpmeas.COLTYPE_INTEGER)]
        
        module = GroupModule()
        module.setup((keys,groupings), prepare_run, prepare_group,
                     run, post_group, post_run, get_measurement_columns)
        module.module_num = 1
        pipeline.add_module(module)
        measurements = pipeline.run(grouping = {'foo':'foo-B', 'bar':'bar-B'})
        self.assertEqual(expects[0], 'Done')
        image_numbers = measurements.get_image_numbers()
        self.assertEqual(len(image_numbers), 2)
        self.assertTrue(np.all(image_numbers == np.array([2,5])))
    
    def test_11_01_catch_operational_error(self):
        '''Make sure that a pipeline can catch an operational error
        
        This is a regression test of IMG-277
        '''
        module = MyClassForTest1101()
        module.module_num = 1
        pipeline = cpp.Pipeline()
        pipeline.add_module(module)
        should_be_true = [False]
        def callback(caller, event):
            if isinstance(event, cpp.RunExceptionEvent):
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
        pipeline = cpp.Pipeline()
        cellprofiler.modules.fill_modules()
        module = cellprofiler.modules.instantiate_module("Align")
        module.module_num = 1
        pipeline.add_module(module)
        fd = cStringIO.StringIO()
        pipeline.save(fd)
        fd.seek(0)
        
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()), 1)
        module_out = pipeline.modules()[-1]
        for setting_in, setting_out in zip(module.settings(),
                                           module_out.settings()):
            self.assertEqual(setting_in.value, setting_out.value)
            
    def test_13_02_save_measurements(self):
        pipeline = cpp.Pipeline()
        cellprofiler.modules.fill_modules()
        module = cellprofiler.modules.instantiate_module("Align")
        module.module_num = 1
        pipeline.add_module(module)
        measurements = cpmeas.Measurements()
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
        measurements = cpmeas.load_measurements(fd)
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
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()), 1)
        module_out = pipeline.modules()[-1]
        for setting_in, setting_out in zip(module.settings(),
                                           module_out.settings()):
            self.assertEqual(setting_in.value, setting_out.value)
            
    def test_13_03_save_long_measurements(self):
        pipeline = cpp.Pipeline()
        cellprofiler.modules.fill_modules()
        module = cellprofiler.modules.instantiate_module("Align")
        module.module_num = 1
        pipeline.add_module(module)
        measurements = cpmeas.Measurements()
        # m2 and m3 should go into panic mode because they differ by a cap
        m1_name = "dalkzfsrqoiualkjfrqealkjfqroupifaaalfdskquyalkhfaafdsafdsqteqteqtew"
        m2_name = "lkjxKJDSALKJDSAWQOIULKJFASOIUQELKJFAOIUQRLKFDSAOIURQLKFDSAQOIRALFAJ" 
        m3_name = "druxKJDSALKJDSAWQOIULKJFASOIUQELKJFAOIUQRLKFDSAOIURQLKFDSAQOIRALFAJ" 
        my_measurement = [np.random.uniform(size=np.random.randint(3,25))
                          for i in range(20)]
        my_other_measurement = [np.random.uniform(size=my_measurement[i].size)
                                            for i in range(20)]
        my_final_measurement = [np.random.uniform(size=my_measurement[i].size)
                                for i in range(20)]
        measurements.add_all_measurements("Foo",m1_name, my_measurement)
        measurements.add_all_measurements("Foo",m2_name, my_other_measurement)
        measurements.add_all_measurements("Foo",m3_name, my_final_measurement)
        fd = cStringIO.StringIO()
        pipeline.save_measurements(fd, measurements)
        fd.seek(0)
        measurements = cpmeas.load_measurements(fd)
        reverse_mapping = cpp.map_feature_names([m1_name, m2_name, m3_name])
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
                
    def test_13_04_pipeline_measurement(self):
        pipeline = cpp.Pipeline()
        cellprofiler.modules.fill_modules()
        module = cellprofiler.modules.instantiate_module("Align")
        module.module_num = 1
        pipeline.add_module(module)
        m = cpmeas.Measurements()
        image_set_list = cpi.ImageSetList()
        self.assertTrue(pipeline.prepare_run(cpw.Workspace(
            pipeline, module, None, None, m, image_set_list)))
        pipeline_text = m.get_experiment_measurement(cpp.M_PIPELINE)
        pipeline_text = pipeline_text.encode("us-ascii")
        pipeline = cpp.Pipeline()
        pipeline.loadtxt(cStringIO.StringIO(pipeline_text))
        self.assertEqual(len(pipeline.modules()), 1)
        module_out = pipeline.modules()[0]
        self.assertTrue(isinstance(module_out, module.__class__))
        self.assertEqual(len(module_out.settings()), len(module.settings()))
        for m1setting, m2setting in zip(module.settings(), module_out.settings()):
            self.assertTrue(isinstance(m1setting, cps.Setting))
            self.assertTrue(isinstance(m2setting, cps.Setting))
            self.assertEqual(m1setting.value, m2setting.value)
                
    def test_14_01_unicode_save(self):
        pipeline = cpp.Pipeline()
        module = MyClassForTest0801()
        module.my_variable.value = u"\\\u2211"
        module.module_num = 1
        module.notes = u"\u03B1\\\u03B2"
        pipeline.add_module(module)
        fd = cStringIO.StringIO()
        pipeline.savetxt(fd)
        result = fd.getvalue()
        lines = result.split("\n")
        self.assertEqual(len(lines), 7)
        text, value = lines[-2].split(":")
        #
        # unicode encoding: 
        #     backslash: \\
        #     unicode character: \u
        #
        # escape encoding:
        #     backslash * 2: \\\\
        #     unicode character: \\
        #
        # result = \\\\\\u2211
        self.assertEqual(value, r"\\\\\\u2211")
        mline = lines[4]
        idx0 = mline.find("notes:")
        mline = mline[(idx0+6):]
        idx1 = mline.find("|")
        value = eval(mline[:idx1].decode('string_escape'))
        self.assertEqual(value, module.notes)
        
    def test_14_02_unicode_save_and_load(self):
        #
        # Put "MyClassForTest0801" into the module list
        #
        cellprofiler.modules.fill_modules()
        cellprofiler.modules.all_modules[MyClassForTest0801.module_name] = \
                    MyClassForTest0801
        #
        # Continue with test
        #
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        module = MyClassForTest0801()
        module.my_variable.value = u"\\\u2211"
        module.module_num = 1
        module.notes = u"\u03B1\\\u03B2"
        pipeline.add_module(module)
        fd = cStringIO.StringIO()
        pipeline.savetxt(fd)
        fd.seek(0)
        pipeline.loadtxt(fd)
        self.assertEqual(len(pipeline.modules()), 1)
        result_module = pipeline.modules()[0]
        self.assertTrue(isinstance(result_module, MyClassForTest0801))
        self.assertEqual(module.notes, result_module.notes)
        self.assertEqual(module.my_variable.value, result_module.my_variable.value)
        
    def test_15_01_profile_example_all(self):
        """
        Profile ExampleAllModulesPipeline
        
        Dependencies:
        User must have ExampleImages on their machine,
        in a location which can be found by example_images_directory().
        This directory should contain the pipeline ExampleAllModulesPipeline
        """
        example_dir = example_images_directory()
        if(not example_dir):
            import warnings
            warnings.warn('example_images_directory not found, skipping profiling of ExampleAllModulesPipeline')
            return
        pipeline_dir = os.path.join(example_dir,'ExampleAllModulesPipeline')
        pipeline_filename = os.path.join(pipeline_dir,'ExampleAllModulesPipeline.cp')
        image_dir = os.path.join(pipeline_dir,'Images')
        output_dir = pipeline_dir
       
        #Might be better to write these paths into the pipeline
        old_image_dir = cpprefs.get_default_image_directory()
        old_output_dir = cpprefs.get_default_output_directory()
        
        cpprefs.set_default_image_directory(image_dir)
        cpprefs.set_default_output_directory(output_dir)
        
        profile_pipeline(pipeline_filename)      
        
        cpprefs.set_default_image_directory(old_image_dir)  
        cpprefs.set_default_output_directory(old_output_dir)
        
def run_pipeline(pipeline_filename,image_set_start=None,image_set_end=None,groups=None,measurements_filename= None):
    pipeline = cpp.Pipeline()
    measurements = None
    pipeline.load(pipeline_filename)
    measurements = pipeline.run(
                image_set_start=image_set_start, 
                image_set_end=image_set_end,
                grouping=groups,
                measurements_filename = measurements_filename,
                initial_measurements = measurements)
        
def profile_pipeline(pipeline_filename,output_filename = None,always_run= True):
    """
    Run the provided pipeline, output the profiled results to a file.
    Pipeline is run each time by default, if canskip_rerun = True
    the pipeline is only run if the profile results filename does not exist
    
    Parameters
    --------------
    pipeline_filename: str
        Absolute path to pipeline
    output_filename: str, optional
        Output file for profiled results. Default is
        the same location&filename as pipeline_filename, with _profile
        appended
    always_run: Bool, optional
        By default, only runs if output_filename does not exist
        If always_run = True, then always runs
    """
    if(not output_filename):
        pipeline_name = os.path.basename(pipeline_filename).split('.')[0]
        pipeline_dir = os.path.dirname(pipeline_filename)
        output_filename = os.path.join(pipeline_dir,pipeline_name + '_profile')
        
    if(not os.path.exists(output_filename) or always_run):
        print 'Running %s' % (pipeline_filename)
        cProfile.runctx('run_pipeline(pipeline_filename)',globals(),locals(),output_filename)
    
    p = pstats.Stats(output_filename)
    #sort by cumulative time spent,optionally strip directory names
    to_print = p.sort_stats('cumulative')#.strip_dirs().
    to_print.print_stats(20)  
        

class MyClassForTest0801(cpm.CPModule):
    def create_settings(self):
        self.my_variable = cps.Text('','')
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

class MyClassForTest1101(cpm.CPModule):
    def create_settings(self):
        self.my_variable = cps.Text('','')
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
        
    def prepare_group(self, pipeline, image_set_list, *args):
        image_set = image_set_list.get_image_set(0)
        image = cpi.Image(np.zeros((5,5)))
        image_set.add("dummy", image)
        return True
    
    def run(self, *args):
        import MySQLdb
        raise MySQLdb.OperationalError("Bogus error")

class GroupModule(cpm.CPModule):
    module_name = "Group"
    variable_revision_number = 1
    def setup(self, groupings, 
                 prepare_run_callback = None,
                 prepare_group_callback = None,
                 run_callback = None,
                 post_group_callback = None,
                 post_run_callback = None,
                 get_measurement_columns_callback = None):
        self.prepare_run_callback = prepare_run_callback
        self.prepare_group_callback = prepare_group_callback
        self.run_callback = run_callback
        self.post_group_callback = post_group_callback
        self.post_run_callback = post_run_callback
        self.groupings = groupings
        self.get_measurement_columns_callback = get_measurement_columns_callback
    def settings(self):
        return []
    def get_groupings(self, workspace):
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
            return self.run_callback(*args)
    def post_run(self, *args):
        if self.post_run_callback is not None:
            return self.post_run_callback(*args)
    def post_group(self, *args):
        if self.post_group_callback is not None:
            return self.post_group_callback(*args)
    def get_measurement_columns(self, *args):
        if self.get_measurement_columns_callback is not None:
            return self.get_measurement_columns_callback(*args)
        return []

if __name__ == "__main__":
    unittest.main()
