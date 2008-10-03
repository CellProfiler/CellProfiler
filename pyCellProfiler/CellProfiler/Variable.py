""" Variable.py - represents a module variable

   $Revision$
   """
import re

DO_NOT_USE = 'Do not use'

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
        old_value = self.__value
        before_change_event = BeforeChangeVariableEvent(old_value,value)
        self.NotifyListeners(before_change_event)
        if not before_change_event.AllowChange():
            return False
        self.__value=value
        self.NotifyListeners(AfterChangeVariableEvent(old_value,value))
        return True

    def Value(self):
        return self.__value
    
    def VariableNumber(self):
        return self.__variable_number
    
    def NotifyListeners(self,event):
        """Notify listeners of an event happening to a variable
        
        """
        for listener in self.__listeners:
            listener(self,event)
        
    def AddListener(self,listener):
        """Add a variable listener
        
        """
        self.__listeners.append(listener)
        
    def RemoveListener(self,listener):
        """Remove a variable listener
        
        """
        self.__listeners.remove(listener)
        
    def Module(self):
        """Return the enclosing module for this variable
        
        """
        return self.__module

class AbstractVariableListener:
    """Implement this when implementing a listener to a variable
    
    """
    def Notify(self, variable, event):
        raise("notify not implemented")

class ChangeVariableEvent:
    """Abstract class representing either the event that a variable will be
    changed or has been changed
    
    """
    def __init__(self,old_value, new_value):
        self.__old_value = old_value
        self.__new_value = new_value

class BeforeChangeVariableEvent(ChangeVariableEvent):
    """Indicates that a variable is about to change, allows a listener to cancel the change
    
    """
    def __init__(self,old_value,new_value):
        ChangeVariableEvent.__init__(self,old_value,new_value)
        self.__allow_change = True
        
    def CancelChange(self):
        self.__allow_change = False
    
    def AllowChange(self):
        return self.__allow_change
    
class AfterChangeVariableEvent(ChangeVariableEvent):
    """Indicates that a variable has changed its value
    
    """
    def __init__(self,old_value,new_value):
        ChangeVariableEvent.__init__(self,old_value,new_value)

class DeleteVariableEvent():
    def __init__(self):
        pass

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
