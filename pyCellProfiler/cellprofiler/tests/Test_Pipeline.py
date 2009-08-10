"""test_Pipeline.py - test the CellProfiler.Pipeline module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__ = "$Revision: 1$"

import os

import unittest
import numpy
import numpy.lib.index_tricks
import cStringIO

import cellprofiler.pipeline
import cellprofiler.objects
import cellprofiler.cpmodule
import cellprofiler.cpimage
import cellprofiler.settings
import cellprofiler.measurements
from cellprofiler.modules.injectimage import InjectImage
from cellprofiler.matlab.cputils import get_matlab_instance

def module_directory():
    d = cellprofiler.pipeline.__file__
    d = os.path.split(d)[0] # ./CellProfiler/pyCellProfiler/cellProfiler
    d = os.path.split(d)[0] # ./CellProfiler/pyCellProfiler
    d = os.path.split(d)[0] # ./CellProfiler
    if not d:
        d = '..'
    return os.path.join(d,'Modules')

def get_load_images_module(module_num=0):
    module = cellprofiler.cpmodule.MatlabModule()
    module.set_module_num(module_num)
    module.create_from_file(os.path.join(module_directory(),'LoadImages.m'),module_num)
    return module

def image_with_one_cell():
    img = numpy.zeros((100,100))
    mgrid = numpy.lib.index_tricks.nd_grid()
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
    
    def test_01_01_AddMatlabModule(self):
        x = cellprofiler.pipeline.Pipeline()
        x.add_module(get_load_images_module())
    
    def test_03_01_load_pipeline_into_matlab(self):
        x = cellprofiler.pipeline.Pipeline()
        x.add_module(get_load_images_module())
        handles = x.load_pipeline_into_matlab()
        matlab = get_matlab_instance()
        self.assertEquals(matlab.num2str(matlab.isfield(handles,'Foo')),'0','Test of isfield failed - should return false for field of Foo in handles')
        for field,subfields in {'Settings':['PixelSize','ModuleNames','VariableValues','VariableInfoTypes',
                                           'VariableRevisionNumbers','ModuleRevisionNumbers','NumbersOfVariables',
                                           'ModuleNotes'],
                               'Current':['NumberOfModules','StartupDirectory','DefaultOutputDirectory','DefaultImageDirectory',
                                          'ImageToolsFilenames','ImageToolHelp','NumberOfImageSets','SetBeingAnalyzed',
                                          'SaveOutputHowOften','TimeStarted'],
                               'Pipeline':[],
                               'Preferences':['PixelSize','DefaultModuleDirectory','DefaultOutputDirectory',
                                              'DefaultImageDirectory','IntensityColorMap','LabelColorMap','StripPipeline',
                                              'SkipErrors','DisplayModeValue','FontSize','DisplayWindows'],
                               'Measurements':[]}.iteritems():
            self.assertEquals(matlab.num2str(matlab.isfield(handles,field)),'1',"Handles were missing field %s"%(field))
            for subfield in subfields:
                self.assertEquals(matlab.num2str(matlab.isfield(matlab.getfield(handles,field),subfield)),'1', "handles.%(field)s.%(subfield)s is missing"%(locals()))
        self.assertEquals(matlab.cell2mat(handles.Settings.ModuleNames[0]),'LoadImages','Did not get correct module name')
        self.assertEquals(matlab.num2str(handles.Current.NumberOfImageSets),'1','Number of image sets was not 1')
        self.assertEquals(matlab.num2str(handles.Current.SetBeingAnalyzed),'1','Set being analyzed was not 1')
        self.assertEquals(matlab.num2str(handles.Current.NumberOfModules),'1','Number of modules was not 1')
        self.assertEquals(matlab.num2str(handles.Current.SaveOutputHowOften),'1','SaveOutputHowOften was not 1')
        self.assertEquals(matlab.num2str(handles.Current.StartingImageSet),'1','Starting image set was not 1')
        
        
    def test_04_01_load_pipeline_with_images(self):
        def provide_image(image_set,image_provider):
            return cellprofiler.cpimage.Image(numpy.zeros((10,10),dtype=numpy.float64), numpy.ones((10,10),dtype=numpy.bool))
        image_set = cellprofiler.cpimage.ImageSet(1,{'number':1},{})
        image_set.providers.append(cellprofiler.cpimage.CallbackImageProvider('MyImage',provide_image))
        x = cellprofiler.pipeline.Pipeline()
        x.add_module(get_load_images_module())
        handles = x.load_pipeline_into_matlab(image_set=image_set)
        matlab = get_matlab_instance()
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Pipeline,'MyImage')),'1','handles.Pipeline.MyImage is missing')
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Pipeline,'CropMaskMyImage')),'1','handles.Pipeline.CropMaskMyImage is missing')

    def test_04_02_load_pipeline_with_objects(self):
        o = cellprofiler.objects.Objects()
        o.segmented = numpy.zeros((10,10),dtype=numpy.int32)
        o.unedited_segmented = numpy.zeros((10,10),dtype=numpy.int32)
        o.small_removed_segmented = numpy.zeros((10,10),dtype=numpy.int32)
        oset = cellprofiler.objects.ObjectSet()
        oset.add_objects(o, 'Nuclei')
        x = cellprofiler.pipeline.Pipeline()
        x.add_module(get_load_images_module())
        handles = x.load_pipeline_into_matlab(object_set = oset)
        matlab = get_matlab_instance()
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Pipeline,'SegmentedNuclei')),'1','handles.Pipeline.SegmentedNuclei is missing')
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Pipeline,'UneditedSegmentedNuclei')),'1','handles.Pipeline.UneditedSegmentedNuclei is missing')
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Pipeline,'SmallRemovedSegmentedNuclei')),'1','handles.Pipeline.SmallRemovedSegmentedNuclei is missing')
    
    def test_04_03_01_load_pipeline_with_measurements_first(self):
        m = cellprofiler.measurements.Measurements()
        m.add_measurement("Image", "FileName_OrigBlue", "/imaging/analysis/wubba-wubba-wubba")
        x = cellprofiler.pipeline.Pipeline()
        x.add_module(get_load_images_module())
        handles = x.load_pipeline_into_matlab(measurements=m)
        matlab = get_matlab_instance()
        self.assertEquals(matlab.num2str(matlab.isfield(handles,"Measurements")),'1','handles is missing Measurements')
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Measurements,'Image')),'1','handles.Measurements is missing Image')
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Measurements.Image,'FileName_OrigBlue')),'1','handles.Measurements.Image is missing FileName_OrigBlue')
        self.assertEquals(matlab.cell2mat(handles.Measurements.Image.FileName_OrigBlue[0]),'/imaging/analysis/wubba-wubba-wubba')
    
    def test_04_03_02_LoadPipelineWithMeasurementsSecond(self):
        m = cellprofiler.measurements.Measurements()
        m.add_measurement("Image", "FileName_OrigBlue", "/imaging/analysis/wubba-wubba-wubba")
        m.next_image_set()
        m.add_measurement("Image", "FileName_OrigBlue", "/imaging/analysis/w00-w00")
        x = cellprofiler.pipeline.Pipeline()
        x.add_module(get_load_images_module())
        handles = x.load_pipeline_into_matlab(measurements=m)
        matlab = get_matlab_instance()
        self.assertEquals(matlab.num2str(matlab.isfield(handles,"Measurements")),'1','handles is missing Measurements')
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Measurements,'Image')),'1','handles.Measurements is missing Image')
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Measurements.Image,'FileName_OrigBlue')),'1','handles.Measurements.Image is missing FileName_OrigBlue')
        self.assertEquals(matlab.cell2mat(handles.Measurements.Image.FileName_OrigBlue[1]),'/imaging/analysis/w00-w00')
    
    def test_04_03_03_LoadPipelineWithObjectMeasurementsFirst(self):
        m = cellprofiler.measurements.Measurements()
        numpy.random.seed(0)
        meas = numpy.random.rand(10)
        m.add_measurement("Nuclei", "Mean", meas)
        x = cellprofiler.pipeline.Pipeline()
        x.add_module(get_load_images_module())
        handles = x.load_pipeline_into_matlab(measurements=m)
        matlab = get_matlab_instance()
        self.assertEquals(matlab.num2str(matlab.isfield(handles,"Measurements")),'1','handles is missing Measurements')
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Measurements,'Nuclei')),'1','handles.Measurements is missing Nuclei')
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Measurements.Nuclei,'Mean')),'1','handles.Measurements.Image is missing Mean')
        meas2=matlab.cell2mat(handles.Measurements.Nuclei.Mean[0])
        self.assertTrue((meas.flatten()==meas2.flatten()).all())
    
    def test_04_03_04_load_pipeline_with_object_measurements_second(self):
        m = cellprofiler.measurements.Measurements()
        numpy.random.seed(0)
        meas = numpy.random.rand(10)
        m.add_measurement("Nuclei", "Mean", meas)
        m.next_image_set()
        meas = numpy.random.rand(10)
        m.add_measurement("Nuclei", "Mean", meas)
        x = cellprofiler.pipeline.Pipeline()
        x.add_module(get_load_images_module())
        handles = x.load_pipeline_into_matlab(measurements=m)
        matlab = get_matlab_instance()
        self.assertEquals(matlab.num2str(matlab.isfield(handles,"Measurements")),'1','handles is missing Measurements')
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Measurements,'Nuclei')),'1','handles.Measurements is missing Nuclei')
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Measurements.Nuclei,'Mean')),'1','handles.Measurements.Image is missing Mean')
        empty_meas = matlab.cell2mat(handles.Measurements.Nuclei.Mean[0])
        self.assertEquals(numpy.product(empty_meas.shape),0)
        meas2=matlab.cell2mat(handles.Measurements.Nuclei.Mean[1])
        self.assertTrue((meas.flatten()==meas2.flatten()).all())
    
    def test_05_01_regression(self):
        #
        # An observed bug when loading into Matlab
        #
        data = {'Image':['Count_Nuclei','Threshold_FinalThreshold_Nuclei','ModuleError_02IdentifyPrimAutomatic','PathName_CytoplasmImage','Threshold_WeightedVariance_Nuclei','FileName_CytoplasmImage',
                         'Threshold_SumOfEntropies_Nuclei','PathName_NucleusImage','Threshold_OrigThreshold_Nuclei','FileName_NucleusImage','ModuleError_01LoadImages'],
                'Nuclei':['Location_Center_Y','Location_Center_X'] }
        m=cellprofiler.measurements.Measurements()
        for key,values in data.iteritems():
            for value in values:
                m.add_measurement(key, value, 'Bogus')
        m.next_image_set()
        x = cellprofiler.pipeline.Pipeline()
        x.add_module(get_load_images_module())
        handles = x.load_pipeline_into_matlab(measurements=m)
        matlab = get_matlab_instance()
        for key,values in data.iteritems():
            for value in values:
                self.assertEquals(matlab.num2str(matlab.isfield(matlab.getfield(handles.Measurements,key),value)),'1','Did not find field handles.Measurements.%s.%s'%(key,value))
    
    def test_06_01_run_pipeline(self):
        x = exploding_pipeline(self)
        module = InjectImage('OneCell',image_with_one_cell())
        module.set_module_num(1)
        x.add_module(module)
        x.run()
    
    def test_06_01_RunPipelineWithMatlab(self): 
        x = exploding_pipeline(self)
        module = InjectImage('OneCell',image_with_one_cell())
        module.set_module_num(1)
        x.add_module(module)
        module = cellprofiler.cpmodule.MatlabModule()
        module.set_module_num(2)
        module.create_from_file(os.path.join(module_directory(),'IdentifyPrimAutomatic.m'),2)
        x.add_module(module)
        module.settings()[0].set_value('OneCell')
        module.settings()[1].set_value('Nuclei')
        measurements = x.run()
        self.assertTrue('Nuclei' in measurements.get_object_names(),"IdentifyPrimAutomatic did not create a Nuclei category")
        self.assertTrue('Location_Center_X' in measurements.get_feature_names('Nuclei'),"IdentifyPrimAutomatic did not create a Location_Center_X measurement")
        center_x = measurements.get_all_measurements('Nuclei','Location_Center_X')
        self.assertTrue(len(center_x),'There are measurements for %d image sets, should be 1'%(len(center_x)))
        self.assertEqual(numpy.product(center_x[0].shape),1,"More than one object was found")
        center = center_x[0][0,0]
        self.assertTrue(center >=45)
        self.assertTrue(center <=55)
    
    def test_07_01_InfogroupNotAfter(self):
        x = cellprofiler.pipeline.Pipeline()
        class MyClass(cellprofiler.cpmodule.CPModule):
            def __init__(self):
                super(MyClass,self).__init__()
                self.set_module_name("whatever")
                self.create_from_annotations()

            def annotations(self):
                a  = cellprofiler.settings.indep_group_annotation(1, 'independent', 'whatevergroup')
                a += cellprofiler.settings.group_annotation(2,'dependent','whatevergroup')
                return a
        module = MyClass()
        module.set_module_num(1)
        x.add_module(module)
        module.settings()[0].value = "Hello"
        choices = module.settings()[1].get_choices(x)
        self.assertEqual(len(choices),0)
         
    def test_07_02_InfogroupAfter(self):
        x = cellprofiler.pipeline.Pipeline()
        class MyClass1(cellprofiler.cpmodule.CPModule):
            def __init__(self):
                super(MyClass1,self).__init__()
                self.set_module_name("provider")
                self.create_from_annotations()

            def annotations(self):
                return cellprofiler.settings.indep_group_annotation(1, 'independent', 'whatevergroup')
        class MyClass2(cellprofiler.cpmodule.CPModule):
            def __init__(self):
                super(MyClass2,self).__init__()
                self.set_module_name("subscriber")
                self.create_from_annotations()

            def annotations(self):
                return cellprofiler.settings.group_annotation(1,'dependent','whatevergroup')
        module1 = MyClass1()
        module1.set_module_num(1)
        x.add_module(module1)
        module2 = MyClass2()
        module2.set_module_num(2)
        x.add_module(module2)
        module1.settings()[0].value = "Hello"
        choices = module2.settings()[0].get_choices(x)
        self.assertEqual(len(choices),1)
        self.assertEqual(choices[0],"Hello")
    
    def test_08_01_empty_variable(self):
        """Regression test that we can save and load the variable, ''"""
        x = cellprofiler.pipeline.Pipeline()
        module = MyClassForTest0801()
        module.set_module_num(1)
        x.add_module(module)
        fh = cStringIO.StringIO()
        x.save(fh)
        y = cellprofiler.pipeline.Pipeline()
        y.load(fh)
        self.assertEqual(len(y.modules()),1)
        module = y.module(1)
        self.assertEqual(module.my_variable.value,'')
    
    def test_09_01_get_measurement_columns(self):
        '''Test the get_measurement_columns method'''
        x = cellprofiler.pipeline.Pipeline()
        module = MyClassForTest0801()
        module.module_num = 1
        module.my_variable.value = "foo"
        x.add_module(module)
        columns = x.get_measurement_columns()
        self.assertEqual(len(columns), 2)
        self.assertTrue(any([column[0] == 'Image' and 
                             column[1] == 'ImageNumber'
                             for column in columns]))
        self.assertTrue(any([column[1] == "foo" for column in columns]))
        module.my_variable.value = "bar"
        columns = x.get_measurement_columns()
        self.assertEqual(len(columns), 2)
        self.assertTrue(any([column[1] == "bar" for column in columns]))
        module = MyClassForTest0801()
        module.module_num = 2
        module.my_variable.value = "foo"
        x.add_module(module)
        columns = x.get_measurement_columns()
        self.assertEqual(len(columns), 3)
        self.assertTrue(any([column[1] == "foo" for column in columns]))
        self.assertTrue(any([column[1] == "bar" for column in columns]))
        columns = x.get_measurement_columns(module)
        self.assertEqual(len(columns), 2)
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
                image = cellprofiler.cpimage.Image(numpy.ones((10,10)) / (i+1))
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
            image_number = workspace.measurements.get_current_image_measurement(
                'ImageNumber')
            self.assertEqual(expects_state, 'Run')
            self.assertEqual(expects_image_number, image_number)
            image = workspace.image_set.get_image('image')
            self.assertTrue(numpy.all(image.pixel_data == 1.0 / image_number))
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
        
        module = GroupModule((keys,groupings), prepare_run, prepare_group,
                             run, post_group, post_run)
        module.module_num = 1
        pipeline.add_module(module)
        measurements = pipeline.run()
        self.assertEqual(expects[0], 'Done')
        image_numbers = measurements.get_all_measurements("Image","ImageNumber")
        self.assertEqual(len(image_numbers), 4)
        self.assertTrue(numpy.all(image_numbers == numpy.array([1,3,2,4])))
         
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
                image = cellprofiler.cpimage.Image(numpy.ones((10,10)) / (i+1))
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
            image_number = workspace.measurements.get_current_image_measurement(
                'ImageNumber')
            self.assertEqual(expects_state, 'Run')
            self.assertEqual(expects_image_number, image_number)
            image = workspace.image_set.get_image('image')
            self.assertTrue(numpy.all(image.pixel_data == 1.0 / image_number))
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
        
        module = GroupModule((keys,groupings), prepare_run, prepare_group,
                             run, post_group, post_run)
        module.module_num = 1
        pipeline.add_module(module)
        measurements = pipeline.run(grouping = {'foo':'foo-B', 'bar':'bar-B'})
        self.assertEqual(expects[0], 'Done')
        image_numbers = measurements.get_all_measurements("Image","ImageNumber")
        self.assertEqual(len(image_numbers), 2)
        self.assertTrue(numpy.all(image_numbers == numpy.array([2,5])))

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

class GroupModule(cellprofiler.cpmodule.CPModule):
    module_name = "Group"
    def __init__(self, groupings, 
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
