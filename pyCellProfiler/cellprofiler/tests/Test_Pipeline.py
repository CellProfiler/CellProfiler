"""test_Pipeline.py - test the CellProfiler.Pipeline module
"""
__version__ = "$Revision: 1$"

import os

import unittest
import numpy
import numpy.lib.index_tricks

import cellprofiler.pipeline
import cellprofiler.objects
import cellprofiler.cpmodule
import cellprofiler.cpimage
import cellprofiler.variable
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
    #def test_00(self):
    #    import scipy.io.matlab.mio
    #    cells = numpy.ndarray((2),dtype='object')
    #    cells[1] = 'FOO'
    #    handles = {'foo':cells}
    #    scipy.io.matlab.mio.savemat('c:\\temp\\pyfoo.mat', handles, format='5')
    
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
        module.variables()[0].set_value('OneCell')
        module.variables()[1].set_value('Nuclei')
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
                a  = cellprofiler.variable.indep_group_annotation(1, 'independent', 'whatevergroup')
                a += cellprofiler.variable.group_annotation(2,'dependent','whatevergroup')
                return a
        module = MyClass()
        module.set_module_num(1)
        x.add_module(module)
        module.variables()[0].value = "Hello"
        choices = module.variables()[1].get_choices(x)
        self.assertEqual(len(choices),0)
         
    def test_07_02_InfogroupAfter(self):
        x = cellprofiler.pipeline.Pipeline()
        class MyClass1(cellprofiler.cpmodule.CPModule):
            def __init__(self):
                super(MyClass1,self).__init__()
                self.set_module_name("provider")
                self.create_from_annotations()

            def annotations(self):
                return cellprofiler.variable.indep_group_annotation(1, 'independent', 'whatevergroup')
        class MyClass2(cellprofiler.cpmodule.CPModule):
            def __init__(self):
                super(MyClass2,self).__init__()
                self.set_module_name("subscriber")
                self.create_from_annotations()

            def annotations(self):
                return cellprofiler.variable.group_annotation(1,'dependent','whatevergroup')
        module1 = MyClass1()
        module1.set_module_num(1)
        x.add_module(module1)
        module2 = MyClass2()
        module2.set_module_num(2)
        x.add_module(module2)
        module1.variables()[0].value = "Hello"
        choices = module2.variables()[0].get_choices(x)
        self.assertEqual(len(choices),1)
        self.assertEqual(choices[0],"Hello")
         
if __name__ == "__main__":
    unittest.main()
