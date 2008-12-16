"""test_Pipeline.py - test the CellProfiler.Pipeline module
"""
import CellProfiler.Pipeline
import CellProfiler.Objects
import CellProfiler.Image
import CellProfiler.Measurements
from CellProfiler.Modules.InjectImage import InjectImage
import unittest
import numpy
import numpy.lib.index_tricks
import os
from CellProfiler.Matlab.Utils import GetMatlabInstance

def ModuleDirectory():
    d = CellProfiler.Pipeline.__file__
    d = os.path.split(d)[0] # ./CellProfiler/pyCellProfiler/CellProfiler
    d = os.path.split(d)[0] # ./CellProfiler/pyCellProfiler
    d = os.path.split(d)[0] # ./CellProfiler
    if not d:
        d = '..'
    return os.path.join(d,'Modules')

def GetLoadImagesModule(ModuleNum=0):
    module = CellProfiler.Module.MatlabModule()
    module.SetModuleNum(ModuleNum)
    module.CreateFromFile(os.path.join(ModuleDirectory(),'LoadImages.m'),ModuleNum)
    return module

def ImageWithOneCell():
    img = numpy.zeros((100,100))
    mgrid = numpy.lib.index_tricks.nd_grid()
    g=mgrid[0:100,0:100]-50                              # the manhattan distance from 50,50
    dist = g[0,:,:]*g[0,:,:]+g[1,:,:]*g[1,:,:]           # the 2-d distance (forgot the fancy name)
    img[ dist < 25] = (25.0-dist.astype(float)[dist<25])/25 # A circle in the middle of it
    return img

def ExplodingPipeline(test):
    """Return a pipeline that fails if the run exception callback is called during a run
    """
    x = CellProfiler.Pipeline.Pipeline()
    def fn(pipeline,event):
        if isinstance(event,CellProfiler.Pipeline.RunExceptionEvent):
            test.assertFalse(event.Error.message)
    x.AddListener(fn)
    return x
         
class TestPipeline(unittest.TestCase):
    #def test_00(self):
    #    import scipy.io.matlab.mio
    #    cells = numpy.ndarray((2),dtype='object')
    #    cells[1] = 'FOO'
    #    handles = {'foo':cells}
    #    scipy.io.matlab.mio.savemat('c:\\temp\\pyfoo.mat', handles, format='5')
    
    def test_00_00_Init(self):
        x = CellProfiler.Pipeline.Pipeline()
    
    def test_01_01_AddMatlabModule(self):
        x = CellProfiler.Pipeline.Pipeline()
        x.AddModule(GetLoadImagesModule())
    
    def test_03_01_LoadPipelineIntoMatlab(self):
        x = CellProfiler.Pipeline.Pipeline()
        x.AddModule(GetLoadImagesModule())
        handles = x.LoadPipelineIntoMatlab()
        matlab = GetMatlabInstance()
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
        
        
    def test_04_01_LoadPipelineWithImages(self):
        def ProvideImage(image_set,image_provider):
            return CellProfiler.Image.Image(numpy.zeros((10,10),dtype=numpy.float64), numpy.ones((10,10),dtype=numpy.bool))
        image_set = CellProfiler.Image.ImageSet(1,{'number':1},{})
        image_set.Providers.append(CellProfiler.Image.CallbackImageProvider('MyImage',ProvideImage))
        x = CellProfiler.Pipeline.Pipeline()
        x.AddModule(GetLoadImagesModule())
        handles = x.LoadPipelineIntoMatlab(image_set=image_set)
        matlab = GetMatlabInstance()
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Pipeline,'MyImage')),'1','handles.Pipeline.MyImage is missing')
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Pipeline,'CropMaskMyImage')),'1','handles.Pipeline.CropMaskMyImage is missing')

    def test_04_02_LoadPipelineWithObjects(self):
        o = CellProfiler.Objects.Objects()
        o.Segmented = numpy.zeros((10,10),dtype=numpy.int32)
        o.UneditedSegmented = numpy.zeros((10,10),dtype=numpy.int32)
        o.SmallRemovedSegmented = numpy.zeros((10,10),dtype=numpy.int32)
        oset = CellProfiler.Objects.ObjectSet()
        oset.AddObjects(o, 'Nuclei')
        x = CellProfiler.Pipeline.Pipeline()
        x.AddModule(GetLoadImagesModule())
        handles = x.LoadPipelineIntoMatlab(object_set = oset)
        matlab = GetMatlabInstance()
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Pipeline,'SegmentedNuclei')),'1','handles.Pipeline.SegmentedNuclei is missing')
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Pipeline,'UneditedSegmentedNuclei')),'1','handles.Pipeline.UneditedSegmentedNuclei is missing')
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Pipeline,'SmallRemovedSegmentedNuclei')),'1','handles.Pipeline.SmallRemovedSegmentedNuclei is missing')
    
    def test_04_03_01_LoadPipelineWithMeasurementsFirst(self):
        m = CellProfiler.Measurements.Measurements()
        m.AddMeasurement("Image", "FileName_OrigBlue", "/imaging/analysis/wubba-wubba-wubba")
        x = CellProfiler.Pipeline.Pipeline()
        x.AddModule(GetLoadImagesModule())
        handles = x.LoadPipelineIntoMatlab(measurements=m)
        matlab = GetMatlabInstance()
        self.assertEquals(matlab.num2str(matlab.isfield(handles,"Measurements")),'1','handles is missing Measurements')
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Measurements,'Image')),'1','handles.Measurements is missing Image')
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Measurements.Image,'FileName_OrigBlue')),'1','handles.Measurements.Image is missing FileName_OrigBlue')
        self.assertEquals(matlab.cell2mat(handles.Measurements.Image.FileName_OrigBlue[0]),'/imaging/analysis/wubba-wubba-wubba')
    
    def test_04_03_02_LoadPipelineWithMeasurementsSecond(self):
        m = CellProfiler.Measurements.Measurements()
        m.AddMeasurement("Image", "FileName_OrigBlue", "/imaging/analysis/wubba-wubba-wubba")
        m.NextImageSet()
        m.AddMeasurement("Image", "FileName_OrigBlue", "/imaging/analysis/w00-w00")
        x = CellProfiler.Pipeline.Pipeline()
        x.AddModule(GetLoadImagesModule())
        handles = x.LoadPipelineIntoMatlab(measurements=m)
        matlab = GetMatlabInstance()
        self.assertEquals(matlab.num2str(matlab.isfield(handles,"Measurements")),'1','handles is missing Measurements')
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Measurements,'Image')),'1','handles.Measurements is missing Image')
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Measurements.Image,'FileName_OrigBlue')),'1','handles.Measurements.Image is missing FileName_OrigBlue')
        self.assertEquals(matlab.cell2mat(handles.Measurements.Image.FileName_OrigBlue[1]),'/imaging/analysis/w00-w00')
    
    def test_04_03_03_LoadPipelineWithObjectMeasurementsFirst(self):
        m = CellProfiler.Measurements.Measurements()
        numpy.random.seed(0)
        meas = numpy.random.rand(10)
        m.AddMeasurement("Nuclei", "Mean", meas)
        x = CellProfiler.Pipeline.Pipeline()
        x.AddModule(GetLoadImagesModule())
        handles = x.LoadPipelineIntoMatlab(measurements=m)
        matlab = GetMatlabInstance()
        self.assertEquals(matlab.num2str(matlab.isfield(handles,"Measurements")),'1','handles is missing Measurements')
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Measurements,'Nuclei')),'1','handles.Measurements is missing Nuclei')
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Measurements.Nuclei,'Mean')),'1','handles.Measurements.Image is missing Mean')
        meas2=matlab.cell2mat(handles.Measurements.Nuclei.Mean[0])
        self.assertTrue((meas.flatten()==meas2.flatten()).all())
    
    def test_04_03_04_LoadPipelineWithObjectMeasurementsSecond(self):
        m = CellProfiler.Measurements.Measurements()
        numpy.random.seed(0)
        meas = numpy.random.rand(10)
        m.AddMeasurement("Nuclei", "Mean", meas)
        m.NextImageSet()
        meas = numpy.random.rand(10)
        m.AddMeasurement("Nuclei", "Mean", meas)
        x = CellProfiler.Pipeline.Pipeline()
        x.AddModule(GetLoadImagesModule())
        handles = x.LoadPipelineIntoMatlab(measurements=m)
        matlab = GetMatlabInstance()
        self.assertEquals(matlab.num2str(matlab.isfield(handles,"Measurements")),'1','handles is missing Measurements')
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Measurements,'Nuclei')),'1','handles.Measurements is missing Nuclei')
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Measurements.Nuclei,'Mean')),'1','handles.Measurements.Image is missing Mean')
        empty_meas = matlab.cell2mat(handles.Measurements.Nuclei.Mean[0])
        self.assertEquals(numpy.product(empty_meas.shape),0)
        meas2=matlab.cell2mat(handles.Measurements.Nuclei.Mean[1])
        self.assertTrue((meas.flatten()==meas2.flatten()).all())
    
    def test_05_01_Regression(self):
        #
        # An observed bug when loading into Matlab
        #
        data = {'Image':['Count_Nuclei','Threshold_FinalThreshold_Nuclei','ModuleError_02IdentifyPrimAutomatic','PathName_CytoplasmImage','Threshold_WeightedVariance_Nuclei','FileName_CytoplasmImage',
                         'Threshold_SumOfEntropies_Nuclei','PathName_NucleusImage','Threshold_OrigThreshold_Nuclei','FileName_NucleusImage','ModuleError_01LoadImages'],
                'Nuclei':['Location_Center_Y','Location_Center_X'] }
        m=CellProfiler.Measurements.Measurements()
        for key,values in data.iteritems():
            for value in values:
                m.AddMeasurement(key, value, 'Bogus')
        m.NextImageSet()
        x = CellProfiler.Pipeline.Pipeline()
        x.AddModule(GetLoadImagesModule())
        handles = x.LoadPipelineIntoMatlab(measurements=m)
        matlab = GetMatlabInstance()
        for key,values in data.iteritems():
            for value in values:
                self.assertEquals(matlab.num2str(matlab.isfield(matlab.getfield(handles.Measurements,key),value)),'1','Did not find field handles.Measurements.%s.%s'%(key,value))
    
    def test_06_01_RunPipeline(self):
        x = ExplodingPipeline(self)
        module = InjectImage('OneCell',ImageWithOneCell())
        module.SetModuleNum(1)
        x.AddModule(module)
        x.Run()
    
    def test_06_01_RunPipelineWithMatlab(self): 
        x = ExplodingPipeline(self)
        module = InjectImage('OneCell',ImageWithOneCell())
        module.SetModuleNum(1)
        x.AddModule(module)
        module = CellProfiler.Module.MatlabModule()
        module.SetModuleNum(2)
        module.CreateFromFile(os.path.join(ModuleDirectory(),'IdentifyPrimAutomatic.m'),2)
        x.AddModule(module)
        module.Variables()[0].SetValue('OneCell')
        module.Variables()[1].SetValue('Nuclei')
        measurements = x.Run()
        self.assertTrue('Nuclei' in measurements.GetObjectNames(),"IdentifyPrimAutomatic did not create a Nuclei category")
        self.assertTrue('Location_Center_X' in measurements.GetFeatureNames('Nuclei'),"IdentifyPrimAutomatic did not create a Location_Center_X measurement")
        center_x = measurements.GetAllMeasurements('Nuclei','Location_Center_X')
        self.assertTrue(len(center_x),'There are measurements for %d image sets, should be 1'%(len(center_x)))
        self.assertEqual(numpy.product(center_x[0].shape),1,"More than one object was found")
        center = center_x[0][0,0]
        self.assertTrue(center >=45)
        self.assertTrue(center <=55)
        
if __name__ == "__main__":
    unittest.main()
