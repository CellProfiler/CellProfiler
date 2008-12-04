"""Pipeline.py - an ordered set of modules to be executed

    $Revision$
"""
import numpy
import scipy.io.matlab.mio
import os
import CellProfiler.Module
import CellProfiler.Preferences
from CellProfiler.Matlab.Utils import NewStringCellArray,GetMatlabInstance,SCellFun,MakeCellStructDType,LoadIntoMatlab
import CellProfiler.VariableChoices
import CellProfiler.Image
import CellProfiler.Measurements
import CellProfiler.Objects
import tempfile
import datetime

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
MEASUREMENTS             = 'Measurements'
PIPELINE                 = 'Pipeline'    
SETTINGS = 'Settings'
VARIABLE_VALUES = 'VariableValues'
VARIABLE_INFO_TYPES = 'VariableInfoTypes'
MODULE_NAMES = 'ModuleNames'
PIXEL_SIZE = 'PixelSize'
NUMBERS_OF_VARIABLES = 'NumbersOfVariables'
VARIABLE_REVISION_NUMBERS = 'VariableRevisionNumbers'
MODULE_REVISION_NUMBERS = 'ModuleRevisionNumbers'
MODULE_NOTES = 'ModuleNotes'
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
        for ModuleNum,module_name in zip(range(1,module_count+1),module_names):
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
            py_module = getattr(pkg,parts[-2])
            py_class  = getattr(py_module,parts[-1])
            module    = py_class()
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
        current[NUMBER_OF_IMAGE_SETS][0,0]     = 1
        current[SET_BEING_ANALYZED][0,0]       = 1
        current[NUMBER_OF_MODULES][0,0]        = len(self.__modules)
        current[SAVE_OUTPUT_HOW_OFTEN][0,0]    = 1
        current[TIME_STARTED][0,0]             = str(datetime.datetime.now())
        current[STARTING_IMAGE_SET][0,0]       = 1
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
        preferences[DISPLAY_MODE_VALUE][0,0]       = 1                      # TODO - get from preferences
        preferences[FONT_SIZE][0,0]                = 10                     # TODO - get from preferences
        preferences[DISPLAY_WINDOWS][0,0]          = [1 for module in self.__modules] # TODO - UI allowing user to choose whether to display a window
        
        images = {}
        if image_set:
            for provider in image_set.Providers:
                image = image_set.GetImage(provider.Name())
                if image.Image != None:
                    images[provider.Name()]=image.Image
                if image.Mask != None:
                    images['CropMask'+provider.Name()]=image.Mask
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
                    feature_measurements.fill(numpy.ndarray((0,),dtype=numpy.float64))
                    data = measurements.GetCurrentMeasurement(object_name,feature_name)
                    if data != None:
                        feature_measurements[0,measurements.ImageSetNumber] = data
        
        matlab = CellProfiler.Matlab.Utils.GetMatlabInstance()
        handles = LoadIntoMatlab(handles)
        
        if no_measurements:
            handles.Measurements = matlab.struct()
        if not len(images):
            handles.Pipeline = matlab.struct()
        return handles
    
    def Run(self):
        """Run the pipeline
        
        Run the pipeline, returning the measurements made
        """
        matlab = CellProfiler.Matlab.Utils.GetMatlabInstance()
        self.SetMatlabPath()
        DisplaySize = (1024,768)
        image_set_list = CellProfiler.Image.ImageSetList()
        measurements = CellProfiler.Measurements.Measurements()
        first_set = True
        while measurements.ImageSetNumber==0 or image_set_list.Count()>measurements.ImageSetNumber:
            if not first_set:
                measurements.NextImageSet()
            NumberofWindows = 0;
            SlotNumber = 0
            for module in self.Modules():
                handles.Current.CurrentModuleNumber = str(module.ModuleNum())
                if handles.Current.SetBeingAnalyzed == 1:
                    figure_field = 'FigureNumberForModule%d'%(module.ModuleNum())
                    if handles.Preferences.DisplayWindows[SlotNumber] == 0:
                        # Make up a fake figure for the module if we're not displaying its window
                        unused_figure_handle = math.ceil(max(matlab.findobj()))+1 
                        handles.Current = matlab.setfield(handles.Current,figure_field,unused_figure_handle)
                        figure = unused_figure_handle
                    else:
                        NumberofWindows = NumberofWindows+1;
                        LeftPos = DisplaySize.width * ((NumberofWindows-1)%12)/12;
                        figure = matlab.CPfigure(handles,'',
                                                 'Name','%s Display, cycle # '%(module.ModuleName()),
                                                 'Position',[LeftPos,DisplaySize.height-522, 560, 442])
                        handles.Current = matlab.setfield(handles.Current, figure_field, figure)
                module_error_measurement = 'ModuleError_%02d%s'%(module.ModuleNum(),module.ModuleName())
                failure = 1
                try:
                    handles = module.Run(handles)
                    failure = 0
                except Exception,instance:
                    self.__frame.DisplayError('Failed during run of module %s (module # %d)'%(module.ModuleName(),module.ModuleNum()),instance)
                if module.ModuleName() != 'Restart':
                    handles = matlab.CPaddmeasurements(handles,'Image',module_error_measurement,failure);
                SlotNumber+=1
            first_set = False
        return measurements

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
    
    def AddModule(self,file_name,ModuleNum):
        """Insert a module into the pipeline with the given module #
        
        Insert a module into the pipeline with the given module #. 
        'file_name' - the path to the file containing the variables for the module.
        ModuleNum - the one-based index for the placement of the module in the pipeline
        """
        new_module = CellProfiler.Module.MatlabModule()
        new_module.CreateFromFile(file_name, ModuleNum)
        idx = ModuleNum-1
        self.__modules = self.__modules[:idx]+[new_module]+self.__modules[idx:]
        for module in self.__modules[idx+1:]:
            module.SetModuleNum(module.ModuleNum())
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

    
