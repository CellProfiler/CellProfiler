"""Pipeline.py - an ordered set of modules to be executed

    $Revision$
"""
import CellProfiler.Module
import CellProfiler.Preferences

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
        Settings = handles['Settings']
        for ModuleNum in range(1,len(Settings.ModuleNames)+1):
            module = CellProfiler.Module.MatlabModule()
            module.CreateFromHandles(handles, ModuleNum)
            self.__modules.append(module)
        self.NotifyListeners(PipelineLoadedEvent())
    
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
        NotifyListeners(ModuleRemovedPipelineEvent(ModuleNum))

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

    