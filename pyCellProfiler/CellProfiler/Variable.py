""" Variable.py - represents a module variable

   $Revision$
   """
import re

BEFORE_CHANGE_NOTIFICATION = 'BeforeChange'
AFTER_CHANGE_NOTIFICATION = 'AfterChange'
DELETE_NOTIFICATION = 'Delete'

class Variable:
    """A module variable which holds a single string value
    
    """
    def __init__(self,module,VariableNumber,value):
        """Initialize a variable with the enclosing module and its string value
        """
        self.__listeners = []
        self.__annotations = []
        self.__module = module;
        self.__variable_number = VariableNumber;
        self.__value = value
    
    def SetValue(self,value):
        self.NotifyListeners(BEFORE_CHANGE_NOTIFICATION)
        self.__value=value
        self.NotifyListeners(AFTER_CHANGE_NOTIFICATION)

    def Value(self):
        return self.__value
    
    def VariableNumber(self):
        return self.__variable_number
    
    def NotifyListeners(self,event):
        """Notify listeners of an event happening to a variable
        
        """
        for listener in self.__listeners:
            listener.Notify(self,event)
        
    def AddListener(self,listener):
        """Add a variable listener
        
        """
        listeners.append(listener)
        
    def RemoveListener(self,listener):
        """Remove a variable listener
        
        """
        listeners.remove(listener)
        
    def Module(self):
        """Return the enclosing module for this variable
        
        """
        return self.__module

class AbstractVariableListener:
    """Implement this when implementing a listener to a variable
    
    """
    def Notify(self, variable, event):
        raise("notify not implemented")

class Annotation:
    """Annotations are the bits of comments parsed out of a .m file that provide metadata on a variable
    
    """
    def __init__(self,line):
        m=re.match("^%([a-z]+)VAR([0-9]+) = (.+)$",line)
        if not m:
            raise(ValueError('Not a variable annotation comment: %s)'%(line)))
        self.Kind = m.groups()[0]
        self.VariableNumber = int(m.groups()[1])
        self.Value = m.groups()[2]

def GetAnnotationsAsDictionary(annotations):
    """Return a multilevel dictionary based on the annotations
    
    Return a multilevel dictionary based on the annotations. The first level
    is the variable number. The second level is the variable kind. The value
    of the second level is an array containing all annotations of that kind
    and variable number.
    """
    dict = {}
    for annotation in annotations:
        vn = annotation.VariableNumber
        if not dict.has_key(vn):
            dict[vn]={}
        if not dict[vn].has_key(annotation.Kind):
            dict[vn][annotation.Kind] = []
        dict[vn][annotation.Kind].append(annotation)
    return dict

def GetVariableAnnotations(annotations,VariableNumber):
    variable_annotations = []
    for annotation in annotations:
        if annotation.VariableNumber == VariableNumber:
            variable_annotations.append(annotation)
    return variable_annotations

def GetVariableText(annotations, VariableNumber):
    for annotation in annotations:
        if annotation.VariableNumber == VariableNumber and annotation.Kind=='text':
            return annotation.Value
    return None
