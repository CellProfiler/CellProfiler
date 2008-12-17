""" Variable.py - represents a module variable

   $Revision$
   """
import re
import uuid

DO_NOT_USE = 'Do not use'
AUTOMATIC = "Automatic"
YES = 'Yes'
NO = 'No'

class Variable(object):
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
        self.__key = uuid.uuid1() 
    
    def SetValue(self,value):
        old_value = self.__value
        before_change_event = BeforeChangeVariableEvent(old_value,value)
        self.NotifyListeners(before_change_event)
        if not before_change_event.AllowChange():
            raise ValueError(before_change_event.CancelReason)
        self.__value=value
        self.NotifyListeners(AfterChangeVariableEvent(old_value,value))

    def Key(self):
        """Return a key that can be used in a dictionary to refer to this variable
        
        """
        return self.__key
    def GetValue(self):
        """The string contents of the variable"""
        return self.__value
    
    Value = property(GetValue,SetValue)
    
    def GetIsYes(self):
        """Return true if the variable's value is "Yes" """
        return self.Value == YES
    
    def SetIsYes(self,is_yes):
        """Set the variable value to Yes if true, No if false"""
        self.Value = (is_yes and YES) or NO
    
    IsYes = property(GetIsYes,SetIsYes)
    
    def GetIsDoNotUse(self):
        """Return true if the variable's value is Do not use"""
        return self.Value == DO_NOT_USE
    
    IsDoNotUse = property(GetIsDoNotUse)
    
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

def ValidateIntegerVariable(variable, event, lower_bound = None, upper_bound = None, cancel_reason = "The value must be an integer"):
    """A listener that validates integer variables"""
    if isinstance(event,BeforeChangeVariableEvent):
        if not event.NewValue.isdigit():
            event.CancelChange(cancel_reason)
        value = int(event.NewValue)
        if lower_bound != None and value < lower_bound:
            event.CancelChange(cancel_reason)
        if upper_bound != None and value > upper_bound:
            event.CancelChange(cancel_reason)

def ValidateIntegerVariableListener(lower_bound = None, upper_bound = None, cancel_reason = "The value must be an integer",allow_automatic=False):
    """Return a lambda representation of ValidateIntegerVariable, with arguments bound"""
    if allow_automatic:
        return lambda variable,event: ValidateIntegerOrAutomatic(variable, event, lower_bound, upper_bound, cancel_reason)
    return lambda variable, event: ValidateIntegerVariable(variable, event, lower_bound, upper_bound, cancel_reason)

def ValidateIntegerRange(variable, event, lower_bound = None, upper_bound = None):
    if isinstance(event,BeforeChangeVariableEvent):
        values = event.NewValue.split(',')
        if len(values) != 2:
            event.CancelChange("There must be two integer values, separated by commas")
        elif not (values[0].isdigit() and values[1].isdigit()):
            event.CancelChange("The values must be integers")
        elif lower_bound != None and int(values[0]) < lower_bound:
            event.CancelChange("The lower value must be at least %d"%(lower_bound))
        elif upper_bound != None and int(values[1]) > upper_bound:
            event.CancelChange("The upper value must be at most %d"%(upper_bound))
        elif int(values[0]) > int(values[1]):
            event.CancelChange("The lower value must be less than the upper value")
    
def ValidateIntegerRangeListener(lower_bound = None, upper_bound = None):
    """Return a lambda representation of ValidateIntegerRange, with arguments bound"""
    return lambda variable, event: ValidateIntegerRange(variable, event, lower_bound, upper_bound)

def ValidateRealVariable(variable, event, lower_bound = None, upper_bound = None, cancel_reason = "The value must be a real number"):
    """A listener that validates floats and such"""
    if isinstance(event,BeforeChangeVariableEvent):
        try:
            value = float(event.NewValue)
        except ValueError:
            event.CancelChange(cancel_reason)
            return
        if lower_bound != None and value < lower_bound:
            event.CancelChange(cancel_reason)
        if upper_bound != None and value > upper_bound:
            event.CancelChange(cancel_reason)

def ValidateRealVariableListener(lower_bound = None, upper_bound = None, cancel_reason = "The value must be an integer",allow_automatic=False):
    """Return a lambda representation of ValidateRealVariable, with arguments bound"""
    if allow_automatic:
        return lambda variable, event: ValidateRealOrAutomatic(variable, event, lower_bound, upper_bound, cancel_reason)
    return lambda variable, event: ValidateRealVariable(variable, event, lower_bound, upper_bound, cancel_reason)

def ValidateRealRange(variable, event, lower_bound = None, upper_bound = None):
    if isinstance(event,BeforeChangeVariableEvent):
        values = event.NewValue.split(',')
        if len(values) != 2:
            event.CancelChange("There must be two integer values, separated by commas")
            return
        lower = float(values[0])
        upper = float(values[1])
        if lower_bound != None and lower < lower_bound:
            event.CancelChange("The lower value must be at least %d"%(lower_bound))
        elif upper_bound != None and upper > upper_bound:
            event.CancelChange("The upper value must be at most %d"%(upper_bound))
        elif lower > upper:
            event.CancelChange("The lower value must be less than the upper value")
    
def ValidateRealRangeListener(lower_bound = None, upper_bound = None):
    """Return a lambda representation of ValidateRealRange, with arguments bound"""
    return lambda variable, event: ValidateRealRange(variable, event, lower_bound, upper_bound)

def ValidateIntegerOrAutomatic(variable, event, lower_bound = None, upper_bound = None, cancel_reason = """The value must be an integer or "Automatic"."""):
    if isinstance(event,BeforeChangeVariableEvent):
        if event.NewValue == AUTOMATIC:
            return
        ValidateIntegerVariable(variable, event, lower_bound, upper_bound, cancel_reason)
 
def ValidateRealOrAutomatic(variable, event, lower_bound = None, upper_bound = None, cancel_reason = """The value must be a real or "Automatic"."""):
    if isinstance(event,BeforeChangeVariableEvent):
        if event.NewValue == AUTOMATIC:
            return
        ValidateRealVariable(variable, event, lower_bound, upper_bound, cancel_reason)
        
class ChangeVariableEvent(object):
    """Abstract class representing either the event that a variable will be
    changed or has been changed
    
    """
    def __init__(self,old_value, new_value):
        self.__old_value = old_value
        self.__new_value = new_value
    
    def GetOldValue(self):
        return self.__old_value
    
    OldValue=property(GetOldValue)
    
    def GetNewValue(self):
        return self.__new_value
    
    NewValue=property(GetNewValue)

class BeforeChangeVariableEvent(ChangeVariableEvent):
    """Indicates that a variable is about to change, allows a listener to cancel the change
    
    """
    def __init__(self,old_value,new_value):
        ChangeVariableEvent.__init__(self,old_value,new_value)
        self.__allow_change = True
        self.__cancel_reason = None
        
    def CancelChange(self,reason=None):
        self.__allow_change = False
        self.__cancel_reason = reason
    
    def AllowChange(self):
        return self.__allow_change
    
    def GetCancelReason(self):
        return self.__cancel_reason
    
    CancelReason = property(GetCancelReason)
    
class AfterChangeVariableEvent(ChangeVariableEvent):
    """Indicates that a variable has changed its value
    
    """
    def __init__(self,old_value,new_value):
        ChangeVariableEvent.__init__(self,old_value,new_value)

class DeleteVariableEvent():
    def __init__(self):
        pass

# Valid Kind arguments to annotation
ANN_TEXT   = 'text'
ANN_CHOICE = 'choice'
ANN_DEFAULT = 'default'
ANN_INFOTYPE = 'infotype'
ANN_INPUTTYPE = 'inputtype'
ANN_PATHNAMETEXT = 'pathnametext'
ANN_FILENAMETEXT = 'filenametext'
DO_NOT_USE = 'Do not use'

class Annotation:
    """Annotations are the bits of comments parsed out of a .m file that provide metadata on a variable
    
    """
    def __init__(self,*args,**kwargs):
        """Initialize either matlab-style with a line of text that is regexp-parsed or more explicitly
        
        args - should be a single line that is parsed
        kind - the kind of annotation it is. Legal values are "text", "choice","default","infotype","inputtype",
               "pathnametext" and "filenametext"
        variable_number - the one-indexed index of the variable in the module's set of variables
        value - the value of the annotation
        """
        if len(args) == 1:
            line = args[0]
            m=re.match("^%([a-z]+)VAR([0-9]+) = (.+)$",line)
            if not m:
                raise(ValueError('Not a variable annotation comment: %s)'%(line)))
            self.Kind = m.groups()[0]
            self.VariableNumber = int(m.groups()[1])
            self.Value = m.groups()[2]
        else:
            self.Kind = kwargs['kind']
            self.VariableNumber = kwargs['variable_number']
            self.Value = kwargs['value']
        if self.Kind not in [ANN_TEXT,ANN_CHOICE,ANN_DEFAULT,ANN_INFOTYPE,ANN_INPUTTYPE, ANN_PATHNAMETEXT, ANN_FILENAMETEXT]:
            raise ValueError("Unrecognized annotation: %s"%(self.Kind))

def TextAnnotation(variable_number, value):
    """Create a text annotation
    """
    return Annotation(kind=ANN_TEXT,variable_number = variable_number, value = value)

def ChoiceAnnotations(variable_number, values):
    """Create choice annotations for a variable
    
    variable_number - the one-indexed variable number
    values - a sequence of possible values for the variable
    """
    return [Annotation(kind=ANN_CHOICE,variable_number=variable_number,value=value) for value in values]

def DefaultAnnotation(variable_number,value):
    """Create a default value annotation
    """
    return Annotation(kind=ANN_DEFAULT,variable_number=variable_number,value=value)

def InfotypeProviderAnnotation(variable_number,value):
    """Create an infotype provider that provides a certain class of thing (e.g. imagegroup or objectgroup)
    
    variable_number - one-based variable number for the annotation
    value - infotype such as object
    """
    return Annotation(kind=ANN_INFOTYPE, variable_number = variable_number, value="%s indep"%(value))

def InfotypeClientAnnotation(variable_number,value):
    """Create an infotype provider that needs a certain class of thing (e.g. imagegroup or objectgroup)
    
    variable_number - one-based variable number for the annotation
    value - infotype such as object
    """
    return Annotation(kind=ANN_INFOTYPE, variable_number = variable_number, value=value)

def InputTypeAnnotation(variable_number,value):
    """Create an input type annotation, such as popupmenu
    """
    return Annotation(kind=ANN_INPUTTYPE, variable_number= variable_number, value=value)

def ChoicePopupAnnotation(variable_number, text, values, customizable=False):
    """Create all the pieces needed for a choice popup variable
    
    variable_number - the one-based index of the variable
    text - what the user sees to the left of the popup
    values - a sequence containing the allowed values
    """
    return [TextAnnotation(variable_number,text)] + \
            ChoiceAnnotations(variable_number, values) +\
            [InputTypeAnnotation(variable_number,(customizable and 'menupopup custom)') or 'menupopup')]

def IndepGroupAnnotation(variable_number, text, group, default=DO_NOT_USE):
    """Create all the pieces needed for an edit box for a variable defining a member of a particular group
    
    variable_number - the one-based index of the variable
    text - what the user sees to the left of the edit box
    group - the group, for instance imagegroup or objectgroup
    default - the default value that appears when the variable is created
    """
    return EditBoxAnnotation(variable_number, text, default)+ \
           [InfotypeProviderAnnotation(variable_number,group)]

def GroupAnnotation(variable_number, text, group):
    """Create the pieces needed for a dependent group popup menu
    
    variable_number - one-based index of the variable
    text - the text to the left of the drop-down
    group - the group, forinstance imagegroup or objectgroup
    """
    return [TextAnnotation(variable_number, text), \
            InfotypeClientAnnotation(variable_number, group),
            InputTypeAnnotation(variable_number,'menupopup')] 

def EditBoxAnnotation(variable_number, text, default=DO_NOT_USE):
    """Create a text annotation and a default annotation to define a variable that uses an edit box
    
    variable_number - the one-based index of the variable
    text - what the user sees to the left of the edit box
    default - the default value for the box
    """
    return [TextAnnotation(variable_number,text),
            DefaultAnnotation(variable_number, default)]

def CheckboxAnnotation(variable_number, text, default=False):
    """Create a checkbox annotation
    
    The checkbox annotation has choice values = 'Yes' and 'No' but
    gets translated by the Gui code into a checkbox.
    variable_number - the one-based index of the variable
    text - the text to display to the user
    default - whether the box should be checked initially (True) or unchecked (False)
    """
    if default:
        choices = [YES,NO]
    else:
        choices = [NO,YES]
    return ChoicePopupAnnotation(variable_number, text, choices)
    
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
