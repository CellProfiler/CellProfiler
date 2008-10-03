"""Module.py - represents a CellProfiler pipeline module
    
    $Revision$
    
    TO-DO: capture and save module revision #s in the handles
    """
import numpy
import CellProfiler.Variable
import re
import os
import CellProfiler.Pipeline

class AbstractModule:
    """ Derive from the abstract module class to create your own module in Python
    
    """
    
    def __init__(self):
        self.__module_num = -1
        self.__variables = []
        self.__notes = []
        self.__variable_revision_number = 0
        self.__module_name = 'unknown'
        self.__annotation_dict = {}
        
    def CreateFromHandles(self,handles,ModuleNum):
        """Fill a module with the information stored in the handles structure for module # ModuleNum 
        
        """
        Settings = handles['Settings'][0,0]
        self.__module_num = ModuleNum
        idx = ModuleNum-1
        self.__module_name = Settings['ModuleNames'][0,idx][0]
        if Settings.dtype.fields.has_key('ModuleNotes'):
            n=Settings['ModuleNotes'][0,idx]
            self.__notes = [n[i,0][0] for i in range(0,n.size)]
        else:
            self.__notes = []
        variable_count=Settings['NumbersOfVariables'][0,idx]
        self.__variable_revision_number = Settings['VariableRevisionNumbers'][0,idx]
        variable_values = [Settings['VariableValues'][idx,i][0] for i in range(0,variable_count)]
        self.__variables = [CellProfiler.Variable.Variable(self,VariableIdx+1,variable_values[VariableIdx])
                            for VariableIdx in range(0,variable_count)]
        
        filename = os.path.join(CellProfiler.Preferences.ModuleDirectory(),self.ModuleName()+CellProfiler.Preferences.ModuleExtension())
        print filename
        file = open(filename)
        try:
            (self.__annotations, self.__variable_revision_number) = self.__read_annotations(file)
        finally:
            file.close()
        self.__annotation_dict = CellProfiler.Variable.GetAnnotationsAsDictionary(self.Annotations()) 
        
    def CreateFromFile(self,file_path,ModuleNum):
        """Parse a file to get the default variables for a module
        """
        self.__module_num = ModuleNum
        self.__module_name = os.path.splitext(os.path.split(file_path)[1])[0]
        fid = open(file_path,'r')
        try:
            (self.__annotations, self.__variable_revision_number) = self.__read_annotations(fid)
        finally:
            fid.close()
        self.__annotation_dict = CellProfiler.Variable.GetAnnotationsAsDictionary(self.Annotations()) 
        variable_dict = {}
        max_variable = 0
        for annotation in self.__annotations:
            vn = annotation.VariableNumber
            if annotation.Kind == 'default':
                variable_dict[vn] = annotation.Value
            elif annotation.Kind == 'choice' and not variable_dict.has_key(vn):
                variable_dict[vn] = annotation.Value
            if vn > max_variable:
                max_variable = vn
        self.__variables=[CellProfiler.Variable.Variable(self,i,'') for i in range(1,max_variable+1)]
        for key in variable_dict.keys():
            self.__variables[key-1].SetValue(variable_dict[key])
            
    def SaveToHandles(self,handles):
        module_idx = self.ModuleNum()-1
        setting = handles[CellProfiler.Pipeline.SETTINGS][0,0]
        setting[CellProfiler.Pipeline.MODULE_NAMES][0,module_idx] = unicode(self.ModuleName())
        setting[CellProfiler.Pipeline.MODULE_NOTES][0,module_idx] = numpy.ndarray(shape=(len(self.Notes()),1),dtype='object')
        for i in range(0,len(self.Notes())):
            setting[CellProfiler.Pipeline.MODULE_NOTES][0,module_idx][i,0]=self.Notes()[i]
        setting[CellProfiler.Pipeline.NUMBERS_OF_VARIABLES][0,module_idx] = len(self.Variables())
        for i in range(0,len(self.Variables())):
            variable = self.Variables()[i]
            setting[CellProfiler.Pipeline.VARIABLE_VALUES][module_idx,i] = unicode(variable.Value())
            vn = variable.VariableNumber()
            annotations = self.VariableAnnotations(vn)
            if annotations.has_key('infotype'):
                setting[CellProfiler.Pipeline.VARIABLE_INFO_TYPES][module_idx,i] = unicode(annotations['infotype'][0].Value)
        setting[CellProfiler.Pipeline.VARIABLE_REVISION_NUMBERS][0,module_idx] = self.__variable_revision_number
        setting[CellProfiler.Pipeline.MODULE_REVISION_NUMBERS][0,module_idx] = 0
    
    def VariableAnnotations(self,VariableNum):
        """Return annotations for the variable with the given number
        
        """
        if self.__annotation_dict.has_key(VariableNum):
            return self.__annotation_dict[VariableNum]
        return {}
    
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
            variable.NotifyListeners(CellProfiler.Variable.DeleteVariableEvent())
    
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
