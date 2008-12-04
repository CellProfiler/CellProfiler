"""test_Pipeline.py - test the CellProfiler.Pipeline module
"""
import CellProfiler.Pipeline
import CellProfiler.Objects
import CellProfiler.Image
import CellProfiler.Measurements
import unittest
import numpy
import os
from CellProfiler.Matlab.Utils import GetMatlabInstance

def ModuleDirectory():
    d = os.path.split(__file__)[0] # ./CellProfiler/pyCellProfiler/CellProfiler/tests
    d = os.path.split(d)[0] # ./CellProfiler/pyCellProfiler/CellProfiler
    d = os.path.split(d)[0] # ./CellProfiler/pyCellProfiler
    d = os.path.split(d)[0] # ./CellProfiler
    return os.path.join(d,'Modules')
    
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
        x.AddModule(os.path.join(ModuleDirectory(),'LoadImages.m'), 0)
    
    def test_03_01_LoadPipelineIntoMatlab(self):
        x = CellProfiler.Pipeline.Pipeline()
        x.AddModule(os.path.join(ModuleDirectory(),'LoadImages.m'), 0)
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
    
    def test_04_01_LoadPipelineWithImages(self):
        def ProvideImage(image_set,image_provider):
            return CellProfiler.Image.Image(numpy.zeros((10,10),dtype=numpy.float64), numpy.ones((10,10),dtype=numpy.bool))
        image_set = CellProfiler.Image.ImageSet(1,{'number':1})
        image_set.Providers.append(CellProfiler.Image.CallbackImageProvider('MyImage',ProvideImage))
        x = CellProfiler.Pipeline.Pipeline()
        x.AddModule(os.path.join(ModuleDirectory(),'LoadImages.m'), 0)
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
        x.AddModule(os.path.join(ModuleDirectory(),'LoadImages.m'), 0)
        handles = x.LoadPipelineIntoMatlab(object_set = oset)
        matlab = GetMatlabInstance()
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Pipeline,'SegmentedNuclei')),'1','handles.Pipeline.SegmentedNuclei is missing')
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Pipeline,'UneditedSegmentedNuclei')),'1','handles.Pipeline.UneditedSegmentedNuclei is missing')
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Pipeline,'SmallRemovedSegmentedNuclei')),'1','handles.Pipeline.SmallRemovedSegmentedNuclei is missing')
    
    def test_04_03_01_LoadPipelineWithMeasurementsFirst(self):
        m = CellProfiler.Measurements.Measurements()
        m.AddMeasurement("Image", "FileName_OrigBlue", "/imaging/analysis/wubba-wubba-wubba")
        x = CellProfiler.Pipeline.Pipeline()
        x.AddModule(os.path.join(ModuleDirectory(),'LoadImages.m'), 0)
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
        x.AddModule(os.path.join(ModuleDirectory(),'LoadImages.m'), 0)
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
        x.AddModule(os.path.join(ModuleDirectory(),'LoadImages.m'), 0)
        handles = x.LoadPipelineIntoMatlab(measurements=m)
        matlab = GetMatlabInstance()
        self.assertEquals(matlab.num2str(matlab.isfield(handles,"Measurements")),'1','handles is missing Measurements')
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Measurements,'Nuclei')),'1','handles.Measurements is missing Nuclei')
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Measurements.Nuclei,'Mean')),'1','handles.Measurements.Image is missing Mean')
        meas2=matlab.cell2mat(handles.Measurements.Nuclei.Mean[0])
        self.assertTrue((meas==meas2).all())
    
    def test_04_03_04_LoadPipelineWithObjectMeasurementsSecond(self):
        m = CellProfiler.Measurements.Measurements()
        numpy.random.seed(0)
        meas = numpy.random.rand(10)
        m.AddMeasurement("Nuclei", "Mean", meas)
        m.NextImageSet()
        meas = numpy.random.rand(10)
        m.AddMeasurement("Nuclei", "Mean", meas)
        x = CellProfiler.Pipeline.Pipeline()
        x.AddModule(os.path.join(ModuleDirectory(),'LoadImages.m'), 0)
        handles = x.LoadPipelineIntoMatlab(measurements=m)
        matlab = GetMatlabInstance()
        self.assertEquals(matlab.num2str(matlab.isfield(handles,"Measurements")),'1','handles is missing Measurements')
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Measurements,'Nuclei')),'1','handles.Measurements is missing Nuclei')
        self.assertEquals(matlab.num2str(matlab.isfield(handles.Measurements.Nuclei,'Mean')),'1','handles.Measurements.Image is missing Mean')
        meas2=matlab.cell2mat(handles.Measurements.Nuclei.Mean[1])
        self.assertTrue((meas==meas2).all())
    
if __name__ == "__main__":
    unittest.main()
