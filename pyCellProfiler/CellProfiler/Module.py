"""Module.py - represents a CellProfiler pipeline module
    
    $Revision$
    """
import CellProfiler.Variable
import re
import os

class AbstractModule:
    """ Derive from the abstract module class to create your own module in Python
    
    """
    
    def __init__(self):
        self.__module_num = -1
        self.__variables = []
        self.__notes = []
        self.__variable_revision_number = 0
        self.__module_name = 'unknown'
        
    def CreateFromHandles(self,handles,ModuleNum):
        """Fill a module with the information stored in the handles structure for module # ModuleNum 
        
        """
        Settings = handles['Settings']
        self.__module_num = ModuleNum
        idx = ModuleNum-1
        self.__module_name = Settings.ModuleNames[idx]
        if 'ModuleNotes' in Settings._fieldnames:
            #
            # There are two cases - for a single line, text comes across as a Unicode string
            # and for multiple lines, text comes across as the class, 'numpy.ndarray'
            #
            if type(Settings.ModuleNotes[idx]) == 'numpy.ndarray':
                self.__notes = [line for line in Settings.ModuleNotes[idx]]
            else:
                self.__notes = [Settings.ModuleNotes[idx]]
        else:
            self.__notes = []
        self.__variable_revision_number = Settings.VariableRevisionNumbers[idx]
        variable_values = [v for v in Settings.VariableValues[idx][0:Settings.NumbersOfVariables[idx]]]
        self.__variables = [CellProfiler.Variable.Variable(self,VariableIdx+1,variable_values[VariableIdx])
                            for VariableIdx in range(0,Settings.NumbersOfVariables[idx])]
        
        filename = os.path.join(CellProfiler.Preferences.ModuleDirectory(),self.ModuleName()+CellProfiler.Preferences.ModuleExtension())
        file = open(filename)
        try:
            (self.__annotations, self.__variable_revision_number) = self.__read_annotations(file)
        finally:
            file.close()
        
    def CreateFromFile(self,file_path,ModuleNum):
        """Parse a file to get the default variables for a module
        """
        
    def ModuleNum(self):
        """Get the module's index number
        
        The module's index number or ModuleNum is a one-based index of its
        execution position in the pipeline. It can be used to predict what
        modules have been run (creating whatever images and measurements
        those modules create) previous to a given module.
        """
        if self.__module_num == -1:
            raise(Exception('Module has not been created'))
        return self.__module_num
    
    def SetModuleNum(self,ModuleNum):
        """Return the module's one-based index number in the pipeline
        
        """
        self.__module_num = ModuleNum
    
    def ModuleName(self):
        """Generally, the name corresponds to the .m file for the module
        
        """
        return self.__module_name
    

    def VariableRevisionNumber(self):
        """The version number, as parsed out of the .m file, saved in the handles or rewritten using an import rule
        """
        return self.__variable_revision_number
    
    def Variables(self):
        """A module's variables
        
        """
        return self.__variables
    
    def Annotations(self):
        """Return the variable annotations, as read out of the module file.
        
        Return the variable annotations, as read out of the module file.
        Each annotation is an instance of the CellProfiler.Variable.Annotation
        class.
        """
        return self.__annotations
    def Delete(self):
        """Delete the module, notifying listeners that it's going away
        
        """
        for variable in self.__variables:
            variable.NotifyListeners(CellProfiler.Variable.DELETE_NOTIFICATION)
    
    def Notes(self):
        """The user-entered notes for a module
        """
        return self.__notes
    
    def SetNotes(self,Notes):
        """Give the module new user-entered notes
        
        """
        return self.__notes
    
    def WriteToHandles(self,handles):
        """Write out the module's state to the handles
        
        """
    
    def WriteToText(self,file):
        """Write the module's state, informally, to a text file
        """
        
    def Run(self,handles):
        """Run the module (abstract method)
        
        """
        raise(NotImplementedError("Please implement the Run method to do whatever your module does, or use the MatlabModule class for Matlab modules"));
    
    def __read_annotations(self,file):
        """Read and return the annotations and variable revision # from a file
        
        """
        annotations = []
        variable_revision_number = 0
        for line in file:
            try:
                annotations.append(CellProfiler.Variable.Annotation(line))
            except:
                # Might be something else...
                match = re.match('^%%%VariableRevisionNumber = ([0-9]+)',line)
                if match:
                    variable_revision_number = int(match.groups()[0]) 
                    break
        return annotations,variable_revision_number

class MatlabModule(AbstractModule):
    def Run(self,handles):
        """Run the module in Matlab
        
        """