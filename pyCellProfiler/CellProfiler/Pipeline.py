"""Pipeline.py - an ordered set of modules to be executed

    $Revision$
"""
import numpy
import scipy.io.matlab
import os
import sys
import CellProfiler.Module
import CellProfiler.Preferences
from CellProfiler.Matlab.Utils import NewStringCellArray,GetMatlabInstance,SCellFun,MakeCellStructDType,LoadIntoMatlab,GetIntFromMatlab,EncapsulateStringsInArrays
import CellProfiler.VariableChoices
import CellProfiler.Image
import CellProfiler.Measurements
import CellProfiler.Objects
import tempfile
import datetime
import traceback

CURRENT = 'Current'
NUMBER_OF_IMAGE_SETS     = 'NumberOfImageSets'
NUMBER_OF_MODULES        = 'NumberOfModules'
SET_BEING_ANALYZED       = 'SetBeingAnalyzed'
SAVE_OUTPUT_HOW_OFTEN    = 'SaveOutputHowOften'
TIME_STARTED             = 'TimeStarted'
STARTING_IMAGE_SET       = 'StartingImageSet'
STARTUP_DIRECTORY        = 'StartupDirectory'
DEFAULT_MODULE_DIRECTORY = 'DefaultModuleDirectory'
DEFAULT_IMAGE_DIRECTORY  = 'DefaultImageDirectory'
DEFAULT_OUTPUT_DIRECTORY = 'DefaultOutputDirectory'
IMAGE_TOOLS_FILENAMES    = 'ImageToolsFilenames'
IMAGE_TOOL_HELP          = 'ImageToolHelp'
PREFERENCES              = 'Preferences'
PIXEL_SIZE               = 'PixelSize'
SKIP_ERRORS              = 'SkipErrors'
INTENSITY_COLOR_MAP      = 'IntensityColorMap'
LABEL_COLOR_MAP          = 'LabelColorMap'
STRIP_PIPELINE           = 'StripPipeline'
DISPLAY_MODE_VALUE       = 'DisplayModeValue'
DISPLAY_WINDOWS          = 'DisplayWindows'
FONT_SIZE                = 'FontSize'
IMAGES                   = 'Images'
MEASUREMENTS             = 'Measurements'
PIPELINE                 = 'Pipeline'    
SETTINGS                  = 'Settings'
VARIABLE_VALUES           = 'VariableValues'
VARIABLE_INFO_TYPES       = 'VariableInfoTypes'
MODULE_NAMES              = 'ModuleNames'
PIXEL_SIZE                = 'PixelSize'
NUMBERS_OF_VARIABLES      = 'NumbersOfVariables'
VARIABLE_REVISION_NUMBERS = 'VariableRevisionNumbers'
MODULE_REVISION_NUMBERS   = 'ModuleRevisionNumbers'
MODULE_NOTES              = 'ModuleNotes'
CURRENT_MODULE_NUMBER     = 'CurrentModuleNumber'
SETTINGS_DTYPE = numpy.dtype([(VARIABLE_VALUES, '|O4'), 
                            (VARIABLE_INFO_TYPES, '|O4'), 
                            (MODULE_NAMES, '|O4'), 
                            (NUMBERS_OF_VARIABLES, '|O4'), 
                            (PIXEL_SIZE, '|O4'), 
                            (VARIABLE_REVISION_NUMBERS, '|O4'), 
                            (MODULE_REVISION_NUMBERS, '|O4'), 
                            (MODULE_NOTES, '|O4')])
CURRENT_DTYPE = MakeCellStructDType([ NUMBER_OF_IMAGE_SETS,SET_BEING_ANALYZED,NUMBER_OF_MODULES, 
                                     SAVE_OUTPUT_HOW_OFTEN,TIME_STARTED, STARTING_IMAGE_SET,
                                     STARTUP_DIRECTORY, DEFAULT_OUTPUT_DIRECTORY, DEFAULT_IMAGE_DIRECTORY, 
                                     IMAGE_TOOLS_FILENAMES, IMAGE_TOOL_HELP])
PREFERENCES_DTYPE = MakeCellStructDType([PIXEL_SIZE, DEFAULT_MODULE_DIRECTORY, DEFAULT_OUTPUT_DIRECTORY, 
                                         DEFAULT_IMAGE_DIRECTORY, INTENSITY_COLOR_MAP, LABEL_COLOR_MAP,
                                         STRIP_PIPELINE, SKIP_ERRORS, DISPLAY_MODE_VALUE, FONT_SIZE,
                                         DISPLAY_WINDOWS])
 
def AddMatlabImages(handles,image_set):
    """Add any images from the handles to the image set
    Generally, the handles have images added as they get returned from a Matlab module.
    You can use this to update the image set and capture them.
    """
    matlab = GetMatlabInstance()
    pipeline_fields = matlab.fields(handles.Pipeline)
    provider_set = set([x.Name() for x in image_set.Providers])
    image_fields = set()
    crop_fields = set()
    for i in range(0,int(matlab.length(pipeline_fields)[0,0])):
        field = matlab.cell2mat(pipeline_fields[i])
        if field.startswith('CropMask'):
            crop_fields.add(field)
        elif field.startswith('Segmented') or field.startswith('UneditedSegmented') or field.startswith('SmallRemovedSegmented'):
            continue
        elif field.startswith('Pathname') or field.startswith('FileList') or field.startswith('Filename'):
            if not image_set.LegacyFields.has_key(field):
                value = matlab.getfield(handles.Pipeline,field)
                if not (isinstance(value,str) or isinstance(value,unicode)):
                    # The two supported types: string/unicode or cell array of strings
                    count = matlab.length(value)
                    new_value = numpy.ndarray((1,count),dtype='object')
                    for j in range(0,count):
                        new_value[0,j] = matlab.cell2mat(value[j])
                    value = new_value 
                image_set.LegacyFields[field] = value
        elif not field in provider_set:
            image_fields.add(field)
    for field in image_fields:
        image = CellProfiler.Image.Image()
        image.Image = matlab.getfield(handles.Pipeline,field)
        crop_field = 'CropMask'+field
        if crop_field in crop_fields:
            image.Mask = matlab.getfield(handles.Pipeline,crop_field)
        image_set.Providers.append(CellProfiler.Image.VanillaImageProvider(field,image))
    number_of_image_sets = GetIntFromMatlab(handles.Current.NumberOfImageSets)
    if (not image_set.LegacyFields.has_key(NUMBER_OF_IMAGE_SETS)) or number_of_image_sets < image_set.LegacyFields[NUMBER_OF_IMAGE_SETS]:
        image_set.LegacyFields[NUMBER_OF_IMAGE_SETS] = number_of_image_sets

def AddMatlabObjects(handles,object_set):
    """Add any objects from the handles to the object set
    You can use this to update the object set after calling a matlab module
    """
    matlab = GetMatlabInstance()
    pipeline_fields = matlab.fields(handles.Pipeline)
    objects_names = set(object_set.GetObjectNames())
    segmented_fields = set()
    unedited_segmented_fields = set()
    small_removed_segmented_fields = set()
    for i in range(0,int(matlab.length(pipeline_fields)[0,0])):
        field = matlab.cell2mat(pipeline_fields[i])
        if field.startswith('Segmented'):
            segmented_fields.add(field)
        elif field.startswith('UneditedSegmented'):
            unedited_segmented_fields.add(field)
        elif field.startswith('SmallRemovedSegmented'):
            small_removed_segmented_fields.add(field)
    for field in segmented_fields:
        object_name = field.replace('Segmented','')
        if object_name in object_set.GetObjectNames():
            continue
        objects = CellProfiler.Objects.Objects()
        objects.Segmented = matlab.getfield(handles.Pipeline,field)
        unedited_field ='Unedited'+field
        small_removed_segmented_field = 'SmallRemoved'+field 
        if unedited_field in unedited_segmented_fields:
            objects.UneditedSegmented = matlab.getfield(handles.Pipeline,unedited_field)
        if small_removed_segmented_field in small_removed_segmented_fields:
            objects.SmallRemovedSegmented = matlab.getfield(handles.Pipeline,small_removed_segmented_field)
        object_set.AddObjects(objects,object_name)

def AddMatlabMeasurements(handles, measurements):
    """Get measurements made by Matlab and put them into our Python measurements object
    """
    matlab = GetMatlabInstance()
    measurement_fields = matlab.fields(handles.Measurements)
    for i in range(0,int(matlab.length(measurement_fields)[0,0])):
        field = matlab.cell2mat(measurement_fields[i])
        object_measurements = matlab.getfield(handles.Measurements,field)
        object_fields = matlab.fields(object_measurements)
        for j in range(0,int(matlab.length(object_fields)[0,0])):
            feature = matlab.cell2mat(object_fields[j])
            if not measurements.HasCurrentMeasurements(field,feature):
                value = matlab.cell2mat(matlab.getfield(object_measurements,feature)[measurements.ImageSetNumber])
                if not isinstance(value,numpy.ndarray) or numpy.product(value.shape) > 0:
                    # It's either not a numpy array (it's a string) or it's not the empty numpy array
                    # so add it to the measurements
                    measurements.AddMeasurement(field,feature,value)

def AddAllImages(handles,image_set, object_set):
    """ Add all images to the handles structure passed
    
    Add images to the handles structure, for example in the Python sandwich.
    """
    images = {}
    for provider in image_set.Providers:
        name = provider.Name()
        image = image_set.GetImage(name)
        images[name] = image.Image
        if image.HasMask:
            images['CropMask'+name] = image.Mask
    
    for object_name in object_set.ObjectNames:
        objects = object_set.GetObjects(object_name)
        images['Segmented'+object_name] = objects.Segmented
        if objects.HasUneditedSegmented():
            images['UneditedSegmented'+object_name] = objects.UneditedSegmented
        if objects.HasSmallRemovedSegmented():
            images['SmallRemovedSegmented'+object_name] = objects.SmallRemovedSegmented
    
    npy_images = numpy.ndarray((1,1),dtype=MakeCellStructDType(images.keys()))
    for key,image in images.iteritems():
        npy_images[key][0,0] = image
    handles[PIPELINE]=npy_images

def AddAllMeasurements(handles, measurements):
    """Add all measurements from our measurements object into the numpy structure passed
    
    """
    measurements_dtype = MakeCellStructDType(measurements.GetObjectNames())
    npy_measurements = numpy.ndarray((1,1),dtype=measurements_dtype)
    handles[MEASUREMENTS]=npy_measurements
    for object_name in measurements.GetObjectNames():
        object_dtype = MakeCellStructDType(measurements.GetFeatureNames(object_name))
        object_measurements = numpy.ndarray((1,1),dtype=object_dtype)
        npy_measurements[object_name][0,0] = object_measurements
        for feature_name in measurements.GetFeatureNames(object_name):
            feature_measurements = numpy.ndarray((1,measurements.ImageSetNumber+1),dtype='object')
            object_measurements[feature_name][0,0] = feature_measurements
            for i in range(0,measurements.ImageSetNumber+1):
                data = measurements.GetCurrentMeasurement(object_name,feature_name)
                if data != None:
                    feature_measurements[0,i] = data

class Pipeline:
    """A pipeline represents the modules that a user has put together
    to analyze their images.
    
    """
    def __init__(self):
        self.__modules = [];
        self.__listeners = [];
        self.__infogroups = {};
        self.__variable_choices = {}
    
    def CreateFromHandles(self,handles):
        """Read a pipeline's modules out of the handles structure
        
        """
        self.__modules = [];
        self.__variable_choices = {}
        Settings = handles[SETTINGS][0,0]
        module_names = Settings[MODULE_NAMES]
        module_count = module_names.shape[1]
        for ModuleNum in range(1,module_count+1):
            idx = ModuleNum-1
            module_name = module_names[0,idx][0]
            module = self.InstantiateModule(module_name)
            module.CreateFromHandles(handles, ModuleNum)
            self.__modules.append(module)
            self.__HookModuleVariables(module)
        self.NotifyListeners(PipelineLoadedEvent())
    
    def InstantiateModule(self,module_name):
        if module_name.find('.') != -1:
            parts     = module_name.split('.')
            pkg_name  = '.'.join(parts[:-1])
            pkg       = __import__(pkg_name)
            module    = eval("%s()"%(module_name))
        else:
            module = CellProfiler.Module.MatlabModule()
        return module
        
    def SaveToHandles(self):
        """Create a numpy array representing this pipeline
        
        """
        settings = numpy.ndarray(shape=[1,1],dtype=SETTINGS_DTYPE)
        handles = {SETTINGS:settings }
        setting = settings[0,0]
        # The variables are a (modules,max # of variables) array of cells (objects)
        # where an empty cell is a (1,0) array of float64
        variable_count = max([len(module.Variables()) for module in self.Modules()])
        module_count = len(self.Modules())
        setting[VARIABLE_VALUES] =          NewStringCellArray((module_count,variable_count))
        # The variable info types are similarly shaped
        setting[VARIABLE_INFO_TYPES] =      NewStringCellArray((module_count,variable_count))
        setting[MODULE_NAMES] =             NewStringCellArray((1,module_count))
        setting[NUMBERS_OF_VARIABLES] =     numpy.ndarray(shape=(1,module_count),dtype=numpy.dtype('uint8'))
        setting[PIXEL_SIZE] =               CellProfiler.Preferences.GetPixelSize()
        setting[VARIABLE_REVISION_NUMBERS] =numpy.ndarray(shape=(1,module_count),dtype=numpy.dtype('uint8'))
        setting[MODULE_REVISION_NUMBERS] =  numpy.ndarray(shape=(1,module_count),dtype=numpy.dtype('uint16'))
        setting[MODULE_NOTES] =             NewStringCellArray((1,module_count))
        for module in self.Modules():
            module.SaveToHandles(handles)
        return handles
    
    def LoadPipelineIntoMatlab(self, image_set=None, object_set=None, measurements=None):
        """Load the pipeline into the Matlab singleton and return the handles structure
        
        The handles structure has all of the goodies needed to run the pipeline including
        * Settings
        * Current (set up to run the first image with the first module
        * Measurements - filled in from measurements (TO_DO)
        * Pipeline - filled in from the image set (TO_DO)
        Returns the handles proxy
        """
        handles = self.BuildMatlabHandles(image_set, object_set, measurements)
        mat_handles = LoadIntoMatlab(handles)
        
        matlab = CellProfiler.Matlab.Utils.GetMatlabInstance()
        if not handles.has_key(MEASUREMENTS):
            mat_handles.Measurements = matlab.struct()
        if not handles.has_key(PIPELINE):
            mat_handles.Pipeline = matlab.struct()
        return mat_handles
    
    def BuildMatlabHandles(self, image_set = None, object_set = None, measurements=None):
        handles = self.SaveToHandles()
        image_tools_dir = os.path.join(CellProfiler.Preferences.CellProfilerRootDirectory(),'ImageTools')
        image_tools = [str(os.path.split(os.path.splitext(filename)[0])[1])
                       for filename in os.listdir(image_tools_dir)
                       if os.path.splitext(filename)[1] == '.m']
        image_tools.insert(0,'Image tools')
        npy_image_tools = numpy.ndarray((1,len(image_tools)),dtype=numpy.dtype('object'))
        for tool,idx in zip(image_tools,range(0,len(image_tools))):
            npy_image_tools[0,idx] = tool
            
        current = numpy.ndarray(shape=[1,1],dtype=CURRENT_DTYPE)
        handles[CURRENT]=current
        current[NUMBER_OF_IMAGE_SETS][0,0]     = [(image_set != None and image_set.LegacyFields.has_key(NUMBER_OF_IMAGE_SETS) and image_set.LegacyFields[NUMBER_OF_IMAGE_SETS]) or 1]
        current[SET_BEING_ANALYZED][0,0]       = [(measurements and measurements.ImageSetNumber + 1) or 1]
        current[NUMBER_OF_MODULES][0,0]        = [len(self.__modules)]
        current[SAVE_OUTPUT_HOW_OFTEN][0,0]    = [1]
        current[TIME_STARTED][0,0]             = str(datetime.datetime.now())
        current[STARTING_IMAGE_SET][0,0]       = [1]
        current[STARTUP_DIRECTORY][0,0]        = CellProfiler.Preferences.CellProfilerRootDirectory()
        current[DEFAULT_OUTPUT_DIRECTORY][0,0] = CellProfiler.Preferences.GetDefaultOutputDirectory()
        current[DEFAULT_IMAGE_DIRECTORY][0,0]  = CellProfiler.Preferences.GetDefaultImageDirectory()
        current[IMAGE_TOOLS_FILENAMES][0,0]    = npy_image_tools
        current[IMAGE_TOOL_HELP][0,0]          = []

        preferences = numpy.ndarray(shape=(1,1),dtype=PREFERENCES_DTYPE)
        handles[PREFERENCES] = preferences
        preferences[PIXEL_SIZE][0,0]               = CellProfiler.Preferences.GetPixelSize()
        preferences[DEFAULT_MODULE_DIRECTORY][0,0] = CellProfiler.Preferences.ModuleDirectory()
        preferences[DEFAULT_OUTPUT_DIRECTORY][0,0] = CellProfiler.Preferences.GetDefaultOutputDirectory()
        preferences[DEFAULT_IMAGE_DIRECTORY][0,0]  = CellProfiler.Preferences.GetDefaultImageDirectory()
        preferences[INTENSITY_COLOR_MAP][0,0]      = 'gray'
        preferences[LABEL_COLOR_MAP][0,0]          = 'jet'
        preferences[STRIP_PIPELINE][0,0]           = 'Yes'                  # TODO - get from preferences
        preferences[SKIP_ERRORS][0,0]              = 'No'                   # TODO - get from preferences
        preferences[DISPLAY_MODE_VALUE][0,0]       = [1]                    # TODO - get from preferences
        preferences[FONT_SIZE][0,0]                = [10]                   # TODO - get from preferences
        preferences[DISPLAY_WINDOWS][0,0]          = [1 for module in self.__modules] # TODO - UI allowing user to choose whether to display a window
        
        images = {}
        if image_set:
            for provider in image_set.Providers:
                image = image_set.GetImage(provider.Name())
                if image.Image != None:
                    images[provider.Name()]=image.Image
                if image.Mask != None:
                    images['CropMask'+provider.Name()]=image.Mask
            for key,value in image_set.LegacyFields.iteritems():
                if key != NUMBER_OF_IMAGE_SETS:
                    images[key]=value
                
        if object_set:
            for name,objects in object_set.AllObjects:
                images['Segmented'+name]=objects.Segmented
                if objects.HasUneditedSegmented():
                    images['UneditedSegmented'+name] = objects.UneditedSegmented
                if objects.HasSmallRemovedSegmented():
                    images['SmallRemovedSegmented'+name] = objects.SmallRemovedSegmented
                    
        if len(images):
            pipeline_dtype = MakeCellStructDType(images.keys())
            pipeline = numpy.ndarray((1,1),dtype=pipeline_dtype)
            handles[PIPELINE] = pipeline
            for name,image in images.items():
                pipeline[name][0,0] = images[name]

        no_measurements = (measurements == None or len(measurements.GetObjectNames())==0)
        if not no_measurements:
            measurements_dtype = MakeCellStructDType(measurements.GetObjectNames())
            npy_measurements = numpy.ndarray((1,1),dtype=measurements_dtype)
            handles['Measurements']=npy_measurements
            for object_name in measurements.GetObjectNames():
                object_dtype = MakeCellStructDType(measurements.GetFeatureNames(object_name))
                object_measurements = numpy.ndarray((1,1),dtype=object_dtype)
                npy_measurements[object_name][0,0] = object_measurements
                for feature_name in measurements.GetFeatureNames(object_name):
                    feature_measurements = numpy.ndarray((1,measurements.ImageSetNumber+1),dtype='object')
                    object_measurements[feature_name][0,0] = feature_measurements
                    data = measurements.GetCurrentMeasurement(object_name,feature_name)
                    feature_measurements.fill(numpy.ndarray((0,),dtype=numpy.float64))
                    if data != None:
                        feature_measurements[0,measurements.ImageSetNumber] = data
        return handles
    
    def Run(self,frame = None):
        """Run the pipeline
        
        Run the pipeline, returning the measurements made
        """
        matlab = CellProfiler.Matlab.Utils.GetMatlabInstance()
        self.SetMatlabPath()
        DisplaySize = (1024,768)
        image_set_list = CellProfiler.Image.ImageSetList()
        measurements = CellProfiler.Measurements.Measurements()
        
        for module in self.Modules():
            try:
                module.PrepareRun(self, image_set_list)
            except Exception,instance:
                traceback.print_exc()
                event = RunExceptionEvent(instance,module)
                self.NotifyListeners(event)
                if event.CancelRun:
                    return None
            
        first_set = True
        while first_set or \
            image_set_list.Count()>measurements.ImageSetNumber+1 or \
            (image_set_list.LegacyFields.has_key(NUMBER_OF_IMAGE_SETS) and image_set_list.LegacyFields[NUMBER_OF_IMAGE_SETS] > measurements.ImageSetNumber+1):
            if not first_set:
                measurements.NextImageSet()
            NumberofWindows = 0;
            SlotNumber = 0
            object_set = CellProfiler.Objects.ObjectSet()
            image_set = image_set_list.GetImageSet(measurements.ImageSetNumber)
            for module in self.Modules():
                module_error_measurement = 'ModuleError_%02d%s'%(module.ModuleNum(),module.ModuleName())
                failure = 1
                try:
                    module.Run(self,image_set,object_set,measurements, frame)
                    failure = 0
                except Exception,instance:
                    traceback.print_exc()
                    event = RunExceptionEvent(instance,module)
                    self.NotifyListeners(event)
                    if event.CancelRun:
                        return None
                if module.ModuleName() != 'Restart':
                    measurements.AddMeasurement('Image',module_error_measurement,failure);
            first_set = False
        return measurements

    def ExperimentalRun(self,frame = None):
        """Run the pipeline - experimental, uses yield
        
        Run the pipeline, returning the measurements made
        """
        matlab = CellProfiler.Matlab.Utils.GetMatlabInstance()
        self.SetMatlabPath()
        DisplaySize = (1024,768)
        image_set_list = CellProfiler.Image.ImageSetList()
        measurements = CellProfiler.Measurements.Measurements()
        
        for module in self.Modules():
            try:
                module.PrepareRun(self, image_set_list)
            except Exception,instance:
                traceback.print_exc()
                event = RunExceptionEvent(instance,module)
                self.NotifyListeners(event)
                if event.CancelRun:
                    return
            
        first_set = True
        while first_set or \
            image_set_list.Count()>measurements.ImageSetNumber+1 or \
            (image_set_list.LegacyFields.has_key(NUMBER_OF_IMAGE_SETS) and image_set_list.LegacyFields[NUMBER_OF_IMAGE_SETS] > measurements.ImageSetNumber+1):
            if not first_set:
                measurements.NextImageSet()
            NumberofWindows = 0;
            SlotNumber = 0
            object_set = CellProfiler.Objects.ObjectSet()
            image_set = image_set_list.GetImageSet(measurements.ImageSetNumber)
            for module in self.Modules():
                module_error_measurement = 'ModuleError_%02d%s'%(module.ModuleNum(),module.ModuleName())
                failure = 1
                try:
                    module.Run(self,image_set,object_set,measurements, frame)
                    failure = 0
                except Exception,instance:
                    traceback.print_exc()
                    event = RunExceptionEvent(instance,module)
                    self.NotifyListeners(event)
                    if event.CancelRun:
                        return
                if module.ModuleName() != 'Restart':
                    measurements.AddMeasurement('Image',module_error_measurement,failure);
                yield measurements
            first_set = False

    def SetMatlabPath(self):
        matlab = CellProfiler.Matlab.Utils.GetMatlabInstance()
        matlab.path(os.path.join(CellProfiler.Preferences.CellProfilerRootDirectory(),'DataTools'),matlab.path())
        matlab.path(os.path.join(CellProfiler.Preferences.CellProfilerRootDirectory(),'ImageTools'),matlab.path())
        matlab.path(os.path.join(CellProfiler.Preferences.CellProfilerRootDirectory(),'CPsubfunctions'),matlab.path())
        matlab.path(CellProfiler.Preferences.ModuleDirectory(),matlab.path())
        matlab.path(CellProfiler.Preferences.CellProfilerRootDirectory(),matlab.path())

    def Clear(self):
        old_modules = self.__modules
        self.__modules = []
        self.__variable_choices = {}
        for module in old_modules:
            module.Delete()
        self.NotifyListeners(PipelineClearedEvent())
    
    def MoveModule(self,ModuleNum,direction):
        """Move module # ModuleNum either DIRECTION_UP or DIRECTION_DOWN in the list
        
        Move the 1-indexed module either up one or down one in the list, displacing
        the other modules in the list
        """
        idx=ModuleNum-1
        if direction == DIRECTION_DOWN:
            if ModuleNum >= len(self.__modules):
                raise ValueError('%(ModuleNum)d is at or after the last module in the pipeline and can''t move down'%(locals()))
            module = self.__modules[idx]
            NewModuleNum = ModuleNum+1
            module.SetModuleNum(ModuleNum+1)
            next_module = self.__modules[idx+1]
            next_module.SetModuleNum(ModuleNum)
            self.__modules[idx]=next_module
            self.__modules[idx+1]=module
        elif direction == DIRECTION_UP:
            if ModuleNum <= 1:
                raise ValueError('The module is at the top of the pipeline and can''t move up')
            module = self.__modules[idx]
            prev_module = self.__modules[idx-1]
            NewModuleNum = prev_module.ModuleNum()
            module.SetModuleNum(NewModuleNum)
            prev_module.SetModuleNum(ModuleNum)
            self.__modules[idx]=self.__modules[idx-1]
            self.__modules[idx-1]=module
        else:
            raise ValueError('Unknown direction: %s'%(direction))    
        self.NotifyListeners(ModuleMovedPipelineEvent(NewModuleNum,direction))
        
    def Modules(self):
        return self.__modules
    
    def Module(self,ModuleNum):
        module = self.__modules[ModuleNum-1]
        assert module.ModuleNum()==ModuleNum,'Misnumbered module. Expected %d, got %d'%(ModuleNum,module.ModuleNum())
        return module
    
    def AddModule(self,new_module):
        """Insert a module into the pipeline with the given module #
        
        Insert a module into the pipeline with the given module #. 
        'file_name' - the path to the file containing the variables for the module.
        ModuleNum - the one-based index for the placement of the module in the pipeline
        """
        ModuleNum = new_module.ModuleNum()
        idx = ModuleNum-1
        self.__modules = self.__modules[:idx]+[new_module]+self.__modules[idx:]
        for module,mn in zip(self.__modules[idx+1:],range(ModuleNum+1,len(self.__modules)+1)):
            module.SetModuleNum(mn)
        self.__HookModuleVariables(new_module)
        self.NotifyListeners(ModuleAddedPipelineEvent(ModuleNum))
    
    def RemoveModule(self,ModuleNum):
        """Remove a module from the pipeline
        
        Remove a module from the pipeline
        ModuleNum - the one-based index of the module
        """
        idx =ModuleNum-1
        module = self.__modules[idx]
        self.__modules = self.__modules[:idx]+self.__modules[idx+1:]
        for variable in module.Variables():
            if self.__variable_choices.has_key(variable.Key()):
                self.__variable_choices.pop(variable.Key())
        module.Delete()
        for module in self.__modules[idx:]:
            module.SetModuleNum(module.ModuleNum()-1)
        self.NotifyListeners(ModuleRemovedPipelineEvent(ModuleNum))
    
    def __HookModuleVariables(self,module):
        """Create whatever VariableChoices are needed
        to represent variable dependencies, groups, etc.
        
        """
        all_variable_notes = []
        for variable in module.Variables():
            annotations = module.VariableAnnotations(variable.VariableNumber())
            # variable_notes stores things we find out about variables as we
            # go along so we can refer back to them for subsequent variables
            variable_notes = {'dependency':None, 'popuptype':None, 'variable':variable }
            if annotations.has_key('inputtype'):
                split = annotations['inputtype'][0].Value.split(' ')
                if split[0] == 'popupmenu' and len(split) > 1:
                    variable_notes['popuptype'] = split[1]
            # Handle both info type producers and consumers
            if annotations.has_key('infotype'):
                info = annotations['infotype'][0].Value.split(' ')
                if not self.__infogroups.has_key(info[0]):
                    self.__infogroups[info[0]] = CellProfiler.VariableChoices.InfoGroupVariableChoices(self)
                if len(info) > 1 and info[-1] == 'indep':
                    self.__infogroups[info[0]].AddIndepVariable(variable)
                else:
                    variable_notes['dependency'] = info[0]
            elif (variable_notes['popuptype'] == 'category' and
                  len(all_variable_notes) > 0 and
                  all_variable_notes[-1]['dependency'] == 'objectgroup'):
                # A category popup with an objectgroup ahead of it.
                # We guess here that we're looking for categories
                # of measurements on the selected object
                vc = CellProfiler.VariableChoices.CategoryVariableChoices(self, all_variable_notes[-1]['variable'])
                self.__variable_choices[variable.Key()] = vc
            elif (variable_notes['popuptype'] == 'measurement' and
                  len(all_variable_notes) > 1 and
                  all_variable_notes[-1]['popuptype'] == 'category' and
                  all_variable_notes[-2]['dependency'] == 'objectgroup'):
                # A measurement popup that follows an objectgroup variable and a category variable
                vc = CellProfiler.VariableChoices.MeasurementVariableChoices(self,
                                                                             all_variable_notes[-2]['variable'],
                                                                             all_variable_notes[-1]['variable'])
                self.__variable_choices[variable.Key()] = vc
            all_variable_notes.append(variable_notes)
    
    def GetVariableChoices(self,variable):
        """Get the variable choices instance that provides choices for this variable. Return None if not a choices variable
        """
        module = variable.Module()
        annotations = module.VariableAnnotations(variable.VariableNumber())
        if annotations.has_key('infotype'):
            info = annotations['infotype'][0].Value.split(' ')
            if info[-1] != 'indep':
                return self.__infogroups[info[0]]
        elif annotations.has_key('choice'):
            choices = [annotation.Value for annotation in annotations['choice']]
            return CellProfiler.VariableChoices.StaticVariableChoices(choices)
        elif self.__variable_choices.has_key(variable.Key()):
            return self.__variable_choices[variable.Key()]
        
    def NotifyListeners(self,event):
        """Notify listeners of an event that happened to this pipeline
        
        """
        for listener in self.__listeners:
            listener(self,event)
    
    def AddListener(self,listener):
        self.__listeners.append(listener)
        
    def RemoveListener(self,listener):
        self.__listeners.remove(listener)

class AbstractPipelineEvent:
    """Something that happened to the pipeline and was indicated to the listeners
    """
    def EventType(self):
        raise NotImplementedError("AbstractPipelineEvent does not implement an event type")

class PipelineLoadedEvent(AbstractPipelineEvent):
    """Indicates that the pipeline has been (re)loaded
    
    """
    def EventType(self):
        return "PipelineLoaded"

class PipelineClearedEvent(AbstractPipelineEvent):
    """Indicates that all modules have been removed from the pipeline
    
    """
    def EventType(self):
        return "PipelineCleared"

DIRECTION_UP = "up"
DIRECTION_DOWN = "down"
class ModuleMovedPipelineEvent(AbstractPipelineEvent):
    """A module moved up or down
    
    """
    def __init__(self,ModuleNum, direction):
        self.ModuleNum = ModuleNum
        self.Direction = direction
    
    def EventType(self):
        return "Module moved"

class ModuleAddedPipelineEvent(AbstractPipelineEvent):
    """A module was added to the pipeline
    
    """
    def __init__(self,ModuleNum):
        self.ModuleNum = ModuleNum
    
    def EventType(self):
        return "Module Added"
    
class ModuleRemovedPipelineEvent(AbstractPipelineEvent):
    """A module was removed from the pipeline
    
    """
    def __init__(self,ModuleNum):
        self.ModuleNum = ModuleNum
        
    def EventType(self):
        return "Module deleted"

class RunExceptionEvent(AbstractPipelineEvent):
    """An exception was caught during a pipeline run
    
    """
    def __init__(self,error,module):
        self.Error     = error
        self.CancelRun = True
        self.Module    = module
    
    def EventType(self):
        return "Pipeline run exception"

def AddHandlesImages(handles,image_set):
    """Add any images from the handles to the image set
    Generally, the handles have images added as they get returned from a Matlab module.
    You can use this to update the image set and capture them.
    """
    hpipeline = handles['Pipeline'][0,0]
    pipeline_fields = hpipeline.dtype.fields.keys()
    provider_set = set([x.Name() for x in image_set.Providers])
    image_fields = set()
    crop_fields = set()
    for field in pipeline_fields:
        if field.startswith('CropMask'):
            crop_fields.add(field)
        elif field.startswith('Segmented') or field.startswith('UneditedSegmented') or field.startswith('SmallRemovedSegmented'):
            continue
        elif field.startswith('Pathname') or field.startswith('FileList') or field.startswith('Filename'):
            if not image_set.LegacyFields.has_key(field):
                value = hpipeline[field]
                if value.dtype.kind in ['U','S']:
                    image_set.LegacyFields[field] = value[0]
                else:
                    image_set.LegacyFields[field] = value
        elif not field in provider_set:
            image_fields.add(field)
    for field in image_fields:
        image = CellProfiler.Image.Image()
        image.Image = hpipeline[field]
        crop_field = 'CropMask'+field
        if crop_field in crop_fields:
            image.Mask = hpipeline[crop_field]
        image_set.Providers.append(CellProfiler.Image.VanillaImageProvider(field,image))
    number_of_image_sets = int(handles[CURRENT][0,0][NUMBER_OF_IMAGE_SETS][0,0])
    if (not image_set.LegacyFields.has_key(NUMBER_OF_IMAGE_SETS)) or number_of_image_sets < image_set.LegacyFields[NUMBER_OF_IMAGE_SETS]:
        image_set.LegacyFields[NUMBER_OF_IMAGE_SETS] = number_of_image_sets

def AddHandlesObjects(handles,object_set):
    """Add any objects from the handles to the object set
    You can use this to update the object set after calling a matlab module
    """
    hpipeline = handles['Pipeline'][0,0]
    pipeline_fields = hpipeline.dtype.fields.keys()
    objects_names = set(object_set.GetObjectNames())
    segmented_fields = set()
    unedited_segmented_fields = set()
    small_removed_segmented_fields = set()
    for field in pipeline_fields:
        if field.startswith('Segmented'):
            segmented_fields.add(field)
        elif field.startswith('UneditedSegmented'):
            unedited_segmented_fields.add(field)
        elif field.startswith('SmallRemovedSegmented'):
            small_removed_segmented_fields.add(field)
    for field in segmented_fields:
        object_name = field.replace('Segmented','')
        if object_name in object_set.GetObjectNames():
            continue
        objects = CellProfiler.Objects.Objects()
        objects.Segmented = hpipeline[field]
        unedited_field ='Unedited'+field
        small_removed_segmented_field = 'SmallRemoved'+field 
        if unedited_field in unedited_segmented_fields:
            objects.UneditedSegmented = hpipeline[unedited_field]
        if small_removed_segmented_field in small_removed_segmented_fields:
            objects.SmallRemovedSegmented = hpipeline[small_removed_segmented_field]
        object_set.AddObjects(objects,object_name)

def AddHandlesMeasurements(handles, measurements):
    """Get measurements made by Matlab and put them into our Python measurements object
    """
    measurement_fields = handles[MEASUREMENTS].dtype.fields.keys()
    set_being_analyzed = handles[CURRENT][0,0][SET_BEING_ANALYZED][0,0]
    for field in measurement_fields:
        object_measurements = handles[MEASUREMENTS][0,0][field][0,0]
        object_fields = object_measurements.dtype.fields.keys()
        for feature in object_fields:
            if not measurements.HasCurrentMeasurements(field,feature):
                value = object_measurements[feature][0,set_being_analyzed-1]
                if not isinstance(value,numpy.ndarray) or numpy.product(value.shape) > 0:
                    # It's either not a numpy array (it's a string) or it's not the empty numpy array
                    # so add it to the measurements
                    measurements.AddMeasurement(field,feature,value)

debug_matlab_run = None

def DebugMatlabRun(value):
    global debug_matlab_run
    debug_matlab_run = value
     
def MatlabRun(handles):
    """Run a Python module, given a Matlab handles structure
    """
    if debug_matlab_run:
        import wx.py
        import wx
        class MyPyCrustApp(wx.App):
            locals = {}
            def OnInit(self):
                wx.InitAllImageHandlers()
                Frame = wx.Frame(None,-1,"MatlabRun explorer")
                sizer = wx.BoxSizer()
                Frame.SetSizer(sizer)
                crust = wx.py.crust.Crust(Frame,-1,locals=self.locals);
                sizer.Add(crust,1,wx.EXPAND)
                Frame.Fit()
                self.SetTopWindow(Frame)
                Frame.Show()
                return 1

    if debug_matlab_run == u"init":
        app = MyPyCrustApp(0)
        app.locals["handles"] = handles
        app.MainLoop()
        
    EncapsulateStringsInArrays(handles)
    if debug_matlab_run == u"enc":
        app = MyPyCrustApp(0)
        app.locals["handles"] = handles
        app.MainLoop()
    orig_handles = handles
    handles = handles[0,0]
    #
    # Get all the pieces you need to run a module:
    # pipeline, image set and set list, measurements and object_set
    #
    pipeline = Pipeline()
    pipeline.CreateFromHandles(handles)
    image_set_list = CellProfiler.Image.ImageSetList()
    image_set = image_set_list.GetImageSet(0)
    measurements = CellProfiler.Measurements.Measurements()
    object_set = CellProfiler.Objects.ObjectSet()
    #
    # Get the values for the current image_set, making believe this is the first image set
    #
    AddHandlesImages(handles, image_set)
    AddHandlesObjects(handles,object_set)
    AddHandlesMeasurements(handles, measurements)
    current_module = int(handles[CURRENT][0,0][CURRENT_MODULE_NUMBER][0])
    #
    # Get and run the module
    #
    module = pipeline.Module(current_module)
    if debug_matlab_run == u"ready":
        app = MyPyCrustApp(0)
        app.locals["handles"] = handles
        app.MainLoop()
    module.Run(pipeline, image_set, object_set, measurements)
    #
    # Add everything to the handles
    #
    AddAllImages(handles, image_set, object_set)
    AddAllMeasurements(handles, measurements)
    if debug_matlab_run == u"run":
        app = MyPyCrustApp(0)
        app.locals["handles"] = handles
        app.MainLoop()

    return orig_handles
    
if __name__ == "__main__":
    handles = scipy.io.matlab.loadmat('c:\\temp\\mh.mat',struct_as_record=True)['handles']
    handles[0,0][CURRENT][0,0][CURRENT_MODULE_NUMBER][0] = str(int(handles[0,0][CURRENT][0,0][CURRENT_MODULE_NUMBER][0])+1)
    MatlabRun(handles)
