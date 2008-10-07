"""Pipeline.py - an ordered set of modules to be executed

    $Revision$
"""
import numpy
import scipy.io.matlab.mio
import os
import CellProfiler.Module
import CellProfiler.Preferences
from CellProfiler.Matlab.Utils import NewStringCellArray,GetMatlabInstance
import CellProfiler.VariableChoices
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
class Pipeline:
    """A pipeline represents the modules that a user has put together
    to analyze their images.
    
    """
    def __init__(self):
        self.__modules = [];
        self.__listeners = [];
        self.__infogroups = {};
    
    def CreateFromHandles(self,handles):
        """Read a pipeline's modules out of the handles structure
        
        """
        self.__modules = [];
        Settings = handles[SETTINGS][0,0]
        module_names = Settings[MODULE_NAMES]
        module_count = module_names.shape[1]
        for ModuleNum in range(1,module_count+1):
            module = CellProfiler.Module.MatlabModule()
            module.CreateFromHandles(handles, ModuleNum)
            self.__modules.append(module)
            self.__HookModuleVariables(module)
        self.NotifyListeners(PipelineLoadedEvent())
        
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
    
    def LoadPipelineIntoMatlab(self):
        """Load the pipeline into the Matlab singleton and return the handles structure
        
        The handles structure has all of the goodies needed to run the pipeline including
        * Settings
        * Current (set up to run the first image with the first module
        * Measurements (blank, but set up to take measurements)
        * Pipeline (blank, but set up to save images)
        Returns the handles proxy
        """
        handles = self.SaveToHandles()
        (matfd,matpath) = tempfile.mkstemp('.mat')
        matfh = os.fdopen(matfd,'w')
        closed = False
        try:
            scipy.io.matlab.mio.savemat(matfh,handles,format='5')
            matfh.close()
            closed = True
            matlab = GetMatlabInstance()
            matlab.handles = matlab.load(matpath)
        finally:
            if not closed:
                matfh.close()
            os.unlink(matpath)
        image_tools_dir = os.path.join(CellProfiler.Preferences.CellProfilerRootDirectory(),'ImageTools')
        image_tools = [os.path.split(os.path.splitext(filename)[0])[1]
                       for filename in os.listdir(image_tools_dir)
                       if os.path.splitext(filename) == '.m']
        matlab.handles.Current = matlab.struct(NUMBER_OF_IMAGE_SETS,1,
                                               SET_BEING_ANALYZED,1,
                                               NUMBER_OF_MODULES, len(self.__modules),
                                               SAVE_OUTPUT_HOW_OFTEN,1,
                                               TIME_STARTED, str(datetime.datetime.now()),
                                               STARTING_IMAGE_SET,1,
                                               STARTUP_DIRECTORY, CellProfiler.Preferences.CellProfilerRootDirectory(),
                                               DEFAULT_OUTPUT_DIRECTORY, CellProfiler.Preferences.GetDefaultOutputDirectory(),
                                               DEFAULT_IMAGE_DIRECTORY, CellProfiler.Preferences.GetDefaultImageDirectory(),
                                               IMAGE_TOOLS_FILENAMES, image_tools,
                                               IMAGE_TOOL_HELP,[]
                                               )

        matlab.handles.Preferences = matlab.struct(PIXEL_SIZE, CellProfiler.Preferences.GetPixelSize(),
                                                   DEFAULT_MODULE_DIRECTORY, CellProfiler.Preferences.ModuleDirectory(),
                                                   DEFAULT_OUTPUT_DIRECTORY, CellProfiler.Preferences.GetDefaultOutputDirectory(),
                                                   DEFAULT_IMAGE_DIRECTORY, CellProfiler.Preferences.GetDefaultImageDirectory(),
                                                   INTENSITY_COLOR_MAP, 'gray',              # TODO - get from preferences
                                                   LABEL_COLOR_MAP, 'jet',                   # TODO - get from preferences
                                                   STRIP_PIPELINE, 'Yes',                    # TODO - get from preferences
                                                   SKIP_ERRORS, 'No',                        # TODO - get from preferences
                                                   DISPLAY_MODE_VALUE, 1,                    # TODO - get from preferences
                                                   FONT_SIZE, 10,                            # TODO - get from preferences
                                                   DISPLAY_WINDOWS,[1 for module in self.__modules] # TODO - UI allowing user to choose whether to display a window
                                                   )
        matlab.handles.Measurements = matlab.struct()
        matlab.handles.Pipeline = matlab.struct()
        return matlab.handles

    def Clear(self):
        old_modules = self.__modules
        self.__modules = []
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
        module.Delete()
        for module in self.__modules[idx:]:
            module.SetModuleNum(module.ModuleNum()-1)
        self.NotifyListeners(ModuleRemovedPipelineEvent(ModuleNum))
    
    def __HookModuleVariables(self,module):
        """Create whatever VariableChoices are needed
        to represent variable dependencies, groups, etc.
        
        """
        for variable in module.Variables():
            annotations = module.VariableAnnotations(variable.VariableNumber())
            if annotations.has_key('infotype'):
                info = annotations['infotype'][0].Value.split(' ')
                if not self.__infogroups.has_key(info[0]):
                    self.__infogroups[info[0]] = CellProfiler.VariableChoices.InfoGroupVariableChoices(self)
                if len(info) > 1 and info[-1] == 'indep':
                    self.__infogroups[info[0]].AddIndepVariable(variable)
    
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

    
