"""Pipeline.py - an ordered set of modules to be executed

    $Revision$
"""
import numpy
import CellProfiler.Module
import CellProfiler.Preferences
from CellProfiler.Matlab.Utils import NewStringCellArray

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
        module = CellProfiler.Module.MatlabModule()
        module.CreateFromFile(file_name, ModuleNum)
        idx = ModuleNum-1
        self.__modules = self.__modules[:idx]+[module]+self.__modules[idx:]
        for module in self.__modules[idx+1:]:
            module.SetModuleNum(module.ModuleNum())
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

    def NotifyListeners(self,event):
        """Notify listeners of an event that happened to this pipeline
        
        """
        for listener in self.__listeners:
            listener.Notify(self,event)
    
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

    