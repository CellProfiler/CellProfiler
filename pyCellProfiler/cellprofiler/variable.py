""" Variable.py - represents a module variable

"""
__version__="$Revision$"

import re
import uuid
import variablechoices

DO_NOT_USE = 'Do not use'
AUTOMATIC = "Automatic"
YES = 'Yes'
NO = 'No'

class Variable(object):
    """A module variable which holds a single string value
    
    """
    def __init__(self,text,value):
        """Initialize a variable with the enclosing module and its string value
        
        module - the module containing this variable
        text   - the explanatory text for the variable
        value  - the default or initial value for the variable
        """
        self.__annotations = []
        self.__text = text
        self.__value = value
        self.__key = uuid.uuid1() 
    
    def set_value(self,value):
        self.__value=value

    def key(self):
        """Return a key that can be used in a dictionary to refer to this variable
        
        """
        return self.__key
    
    def get_text(self):
        """The explanatory text for the variable
        """
        return self.__text
    
    text = property(get_text)
    
    def get_value(self):
        """The string contents of the variable"""
        return self.__value
    
    def __internal_get_value(self):
        """The value stored within the variable"""
        return self.get_value()
    
    def __internal_set_value(self,value):
        self.set_value(value)
    
    value = property(__internal_get_value,__internal_set_value)
    
    def __eq__(self, x):
        return self.value == str(x)
    
    def __ne__(self, x):
        return not self.__eq__(x)
    
    def get_is_yes(self):
        """Return true if the variable's value is "Yes" """
        return self.__value == YES
    
    def set_is_yes(self,is_yes):
        """Set the variable value to Yes if true, No if false"""
        self.__value = (is_yes and YES) or NO
    
    is_yes = property(get_is_yes,set_is_yes)
    
    def get_is_do_not_use(self):
        """Return true if the variable's value is Do not use"""
        return self.value == DO_NOT_USE
    
    is_do_not_use = property(get_is_do_not_use)
    
    def test_valid(self, pipeline):
        """Throw a ValueError if the value of this variable is inappropriate for the context"""
        pass
    
    def __str__(self):
        if not isinstance(self.__value,str):
            raise ValueError("%s was not a string"%(self.__value))
        return self.__value
    
class Text(Variable):
    """A variable that displays as an edit box, accepting a string
    
    """
    def __init__(self,text,value):
        super(Text,self).__init__(text,value)

class DirectoryPath(Text):
    """A variable that displays a filesystem path name
    """
    def __init__(self,text,value):
        super(DirectoryPath,self).__init__(text,value)

class FilenameText(Text):
    """A variable that displays a file name
    """
    def __init__(self,text,value):
        super(FilenameText,self).__init__(text,value)

class Integer(Text):
    """A variable that allows only integer input
    """
    def __init__(self,text,value=0,minval=None, maxval=None):
        super(Integer,self).__init__(text,str(value))
        self.__minval = minval
        self.__maxval = maxval
    
    def set_value(self,value):
        """Convert integer to string
        """
        str_value = str(value)
        super(Integer,self).set_value(str_value)
        
    def get_value(self):
        """Return the value of the variable as an integer
        """
        return int(super(Integer,self).get_value())
    
    def test_valid(self,pipeline):
        """Return true only if the text value is an integer
        """
        if not str(self).isdigit():
            raise ValueError('Must be an integer value, was "%s"'%(str(self)))
        if self.__minval != None and self.__minval > self.value:
            raise ValueError('Must be at least %d, was %d'%(self.__minval, self.value))
        if self.__maxval != None and self.__maxval < self.value:
            raise ValueError('Must be at most %d, was %d'%(self.__maxval, self.value))
        
    def __eq__(self,x):
        if super(Integer,self).__eq__(x):
            return True
        return self.value == x

class IntegerRange(Variable):
    """A variable that allows only integer input between two constrained values
    """
    def __init__(self,text,value=(0,1),minval=None, maxval=None):
        """Initialize an integer range
        text  - helpful text to be displayed to the user
        value - initial default value, a two-tuple as minimum and maximum
        minval - the minimum acceptable value of either
        maxval - the maximum acceptable value of either
        """
        super(IntegerRange,self).__init__(text,"%d,%d"%value)
        self.__minval = minval
        self.__maxval = maxval
        
    
    def set_value(self,value):
        """Convert integer tuples to string
        """
        try: 
            if len(value) == 2:
                super(IntegerRange,self).set_value("%d,%d"%(value[0],value[1]))
                return
        except: 
            pass
        super(IntegerRange,self).set_value(value)
    
    def get_value(self):
        """Convert the underlying string to a two-tuple"""
        values = str(self).split(',')
        return (int(values[0]),int(values[1]))
    
    def get_min(self):
        """The minimum value of the range"""
        return self.value[0]
    
    min = property(get_min)
    
    def get_max(self):
        """The maximum value of the range"""
        return self.value[1]
    
    max = property(get_max) 
    
    def test_valid(self, pipeline):
        values = str(self).split(',')
        if len(values) < 2:
            raise ValueError("Minimum and maximum values must be separated by a comma")
        if len(values) > 2:
            raise ValueError("Only two values allowed")
        for value in values:
            if not value.isdigit():
                raise ValueError("%s is not an integer"%(value))
        if self.__minval > self.min:
            raise ValueError("%d can't be less than %d"%(self.min,self.__minval))
        if self.__maxval < self.max:
            raise ValueError("%d can't be greater than %d"%(self.max,self.__maxval))
        if self.min > self.max:
            raise ValueError("%d is greater than %d"%(self.min, self.max))

class Float(Text):
    """A variable that allows only floating point input
    """
    def __init__(self,text,value=0,minval=None, maxval=None):
        super(Float,self).__init__(text,str(value))
        self.__minval = minval
        self.__maxval = maxval
    
    def set_value(self,value):
        """Convert integer to string
        """
        str_value = str(value)
        super(Float,self).set_value(str_value)
        
    def get_value(self):
        """Return the value of the variable as an integer
        """
        return float(super(Float,self).get_value())
    
    def test_valid(self,pipeline):
        """Return true only if the text value is float
        """
        # Raises value error inside self.value if not a float
        if self.__minval != None and self.__minval > self.value:
            raise ValueError('Must be at least %d, was %d'%(self.__minval, self.value))
        if self.__maxval != None and self.__maxval < self.value:
            raise ValueError('Must be at most %d, was %d'%(self.__maxval, self.value))
        
    def __eq__(self,x):
        if super(Float,self).__eq__(x):
            return True
        return self.value == x

class FloatRange(Variable):
    """A variable that allows only floating point input between two constrained values
    """
    def __init__(self,text,value=(0,1),minval=None, maxval=None):
        """Initialize an integer range
        text  - helpful text to be displayed to the user
        value - initial default value, a two-tuple as minimum and maximum
        minval - the minimum acceptable value of either
        maxval - the maximum acceptable value of either
        """
        super(FloatRange,self).__init__(text,"%f,%f"%value)
        self.__minval = minval
        self.__maxval = maxval
    
    def set_value(self,value):
        """Convert integer tuples to string
        """
        try: 
            if len(value) == 2:
                super(FloatRange,self).set_value("%f,%f"%(value[0],value[1]))
                return
        except: 
            pass
        super(FloatRange,self).set_value(value)
    
    def get_value(self):
        """Convert the underlying string to a two-tuple"""
        values = str(self).split(',')
        return (float(values[0]),float(values[1]))
    
    def get_min(self):
        """The minimum value of the range"""
        return self.value[0]
    
    min = property(get_min)
    
    def get_max(self):
        """The maximum value of the range"""
        return self.value[1]
    
    max = property(get_max) 
    
    def test_valid(self, pipeline):
        values = str(self).split(',')
        if len(values) < 2:
            raise ValueError("Minimum and maximum values must be separated by a comma")
        if len(values) > 2:
            raise ValueError("Only two values allowed")
        for value in values:
            float(value)
        if self.__minval > self.min:
            raise ValueError("%f can't be less than %f"%(self.min,self.__minval))
        if self.__maxval < self.max:
            raise ValueError("%f can't be greater than %f"%(self.max,self.__maxval))
        if self.min > self.max:
            raise ValueError("%f is greater than %f"%(self.min, self.max))

class NameProvider(Text):
    """A variable that provides a named object
    """
    def __init__(self,text,group,value=DO_NOT_USE):
        super(NameProvider,self).__init__(text,value)
        self.__group = group
    
    def get_group(self):
        """This variable provides a name to this group
        
        Returns a group name, e.g. imagegroup or objectgroup
        """
        return self.__group
    
    group = property(get_group)

class ImageNameProvider(NameProvider):
    """A variable that provides an image name
    """
    def __init__(self,text,value=DO_NOT_USE):
        super(ImageNameProvider,self).__init__(text,'imagegroup',value)

class ObjectNameProvider(NameProvider):
    """A variable that provides an image name
    """
    def __init__(self,text,value=DO_NOT_USE):
        super(ImageNameProvider,self).__init__(text,'objectgroup',value)

class NameSubscriber(Variable):
    """A variable that takes its value from one made available by name providers
    """
    def __init__(self,text,group,value='None'):
        super(NameSubscriber,self).__init__(text,value)
    
        self.__group = group
    
    def get_group(self):
        """This variable provides a name to this group
        
        Returns a group name, e.g. imagegroup or objectgroup
        """
        return self.__group
    
    group = property(get_group)
    
    def get_choices(self,pipeline):
        choices = []
        for module in pipeline.modules():
            module_choices = []
            for variable in module.visible_variables():
                if variable.key() == self.key():
                    return choices
                if isinstance(variable, NameProvider) and variable != DO_NOT_USE:
                    module_choices.append(variable.value)
            choices += module_choices
        assert False, "Variable not among visible variables in pipeline"
    
    def test_valid(self,pipeline):
        if len(self.get_choices(pipeline)) == 0:
            raise ValueError("No prior instances of %s were defined"%(self.group))
        if self.value not in self.get_choices(pipeline):
            raise ValueError("%s not in %s"%(self.value,reduce(lambda x,y: "%s,%s"%(x,y),self.get_choices(pipeline))))

class ImageNameSubscriber(NameSubscriber):
    """A variable that provides an image name
    """
    def __init__(self,text,value=DO_NOT_USE):
        super(ImageNameSubscriber,self).__init__(text,'imagegroup',value)

class ObjectNameSubscriber(NameSubscriber):
    """A variable that provides an image name
    """
    def __init__(self,text,value=DO_NOT_USE):
        super(ObjetNameSubscriber,self).__init__(text,'objectgroup',value)

class Binary(Variable):
    """A variable that is represented as either true or false
    The underlying value stored in the variables slot is "Yes" or "No"
    for historical reasons.
    """
    def __init__(self,text,value):
        """Initialize the binary variable with the module, explanatory text and value
        
        The value for a binary variable is True or False
        """
        str_value = (value and YES) or NO
        super(Binary,self).__init__(text, str_value)
    
    def set_value(self,value):
        """When setting, translate true and false into yes and no"""
        if value == YES or value == NO or\
           isinstance(value,str) or isinstance(value,unicode):
            super(Binary,self).set_value(value)
        else: 
            str_value = (value and YES) or NO
            super(Binary,self).set_value(str_value)
    
    def get_value(self):
        """Get the value of a binary variable as a truth value
        """
        return super(Binary,self).get_value() == YES 
    
    def __eq__(self,x):
        if x == NO:
            x = False
        return (self.value and x) or ((not self.value) and (not x)) 
    
class Choice(Variable):
    """A variable that displays a drop-down set of choices
    
    """
    def __init__(self,text,choices,value=None):
        """Initializer
        module - the module containing the variable
        text - the explanatory text for the variable
        choices - a sequence of string choices to be displayed in the drop-down
        value - the default choice or None to choose the first of the choices.
        """
        super(Choice,self).__init__(text, value or choices[0])
        self.__choices = choices
    
    def __internal_get_choices(self):
        """The sequence of strings that define the choices to be displayed"""
        return self.get_choices()
    
    def get_choices(self):
        """The sequence of strings that define the choices to be displayed"""
        return self.__choices
    
    choices = property(__internal_get_choices)
    
    def test_valid(self,pipeline):
        """Check to make sure that the value is among the choices"""
        if self.value not in self.choices:
            raise ValueError("%s is not one of %s"%(self.value, reduce(lambda x,y: "%s,%s"%(x,y),self.choices)))

class CustomChoice(Choice):
    def __init__(self,text,choices,value=None):
        """Initializer
        module - the module containing the variable
        text - the explanatory text for the variable
        choices - a sequence of string choices to be displayed in the drop-down
        value - the default choice or None to choose the first of the choices.
        """
        super(CustomChoice,self).__init__(text, choices, value)
    
    def get_choices(self):
        """Put the custom choice at the top"""
        choices = list(super(CustomChoice,self).get_choices())
        if self.value not in choices:
            choices.insert(0,self.value)
        return choices
    
    def set_value(self,value):
        """Bypass the check in "Choice"."""
        Variable.set_value(self, value)
    
class DoSomething(Variable):
    """Do something in response to a button press
    """
    def __init__(self,text,label,callback,*args):
        super(DoSomething,self).__init__(text,'n/a')
        self.__label = label
        self.__callback = callback
        self.__args = args
    
    def get_label(self):
        """Return the text label for the button"""
        return self.__label
    
    label = property(get_label)
    
    def on_event_fired(self):
        """Call the callback in response to the user's request to do something"""
        self.__callback(*self.__args)

def validate_integer_variable(variable, event, lower_bound = None, upper_bound = None, cancel_reason = "The value must be an integer"):
    """A listener that validates integer variables"""
    if isinstance(event,BeforeChangeVariableEvent):
        if not event.new_value.isdigit():
            event.cancel_change(cancel_reason)
        value = int(event.new_value)
        if lower_bound != None and value < lower_bound:
            event.cancel_change(cancel_reason)
        if upper_bound != None and value > upper_bound:
            event.cancel_change(cancel_reason)

def validate_integer_variable_listener(lower_bound = None, upper_bound = None, cancel_reason = "The value must be an integer",allow_automatic=False):
    """Return a lambda representation of ValidateIntegerVariable, with arguments bound"""
    if allow_automatic:
        return lambda variable,event: validate_integer_or_automatic(variable, event, lower_bound, upper_bound, cancel_reason)
    return lambda variable, event: validate_integer_variable(variable, event, lower_bound, upper_bound, cancel_reason)

def validate_integer_range(variable, event, lower_bound = None, upper_bound = None):
    if isinstance(event,BeforeChangeVariableEvent):
        values = event.new_value.split(',')
        if len(values) != 2:
            event.cancel_change("There must be two integer values, separated by commas")
        elif not (values[0].isdigit() and values[1].isdigit()):
            event.cancel_change("The values must be integers")
        elif lower_bound != None and int(values[0]) < lower_bound:
            event.cancel_change("The lower value must be at least %d"%(lower_bound))
        elif upper_bound != None and int(values[1]) > upper_bound:
            event.cancel_change("The upper value must be at most %d"%(upper_bound))
        elif int(values[0]) > int(values[1]):
            event.cancel_change("The lower value must be less than the upper value")
    
def validate_integer_range_listener(lower_bound = None, upper_bound = None):
    """Return a lambda representation of ValidateIntegerRange, with arguments bound"""
    return lambda variable, event: validate_integer_range(variable, event, lower_bound, upper_bound)

def validate_real_variable(variable, event, lower_bound = None, upper_bound = None, cancel_reason = "The value must be a real number"):
    """A listener that validates floats and such"""
    if isinstance(event,BeforeChangeVariableEvent):
        try:
            value = float(event.new_value)
        except ValueError:
            event.cancel_change(cancel_reason)
            return
        if lower_bound != None and value < lower_bound:
            event.cancel_change(cancel_reason)
        if upper_bound != None and value > upper_bound:
            event.cancel_change(cancel_reason)

def validate_real_variable_listener(lower_bound = None, upper_bound = None, cancel_reason = "The value must be an integer",allow_automatic=False):
    """Return a lambda representation of ValidateRealVariable, with arguments bound"""
    if allow_automatic:
        return lambda variable, event: validate_real_or_automatic(variable, event, lower_bound, upper_bound, cancel_reason)
    return lambda variable, event: validate_real_variable(variable, event, lower_bound, upper_bound, cancel_reason)

def validate_real_range(variable, event, lower_bound = None, upper_bound = None):
    if isinstance(event,BeforeChangeVariableEvent):
        values = event.new_value.split(',')
        if len(values) != 2:
            event.cancel_change("There must be two integer values, separated by commas")
            return
        lower = float(values[0])
        upper = float(values[1])
        if lower_bound != None and lower < lower_bound:
            event.cancel_change("The lower value must be at least %d"%(lower_bound))
        elif upper_bound != None and upper > upper_bound:
            event.cancel_change("The upper value must be at most %d"%(upper_bound))
        elif lower > upper:
            event.cancel_change("The lower value must be less than the upper value")
    
def validate_real_range_listener(lower_bound = None, upper_bound = None):
    """Return a lambda representation of ValidateRealRange, with arguments bound"""
    return lambda variable, event: validate_real_range(variable, event, lower_bound, upper_bound)

def validate_integer_or_automatic(variable, event, lower_bound = None, upper_bound = None, cancel_reason = """The value must be an integer or "Automatic"."""):
    if isinstance(event,BeforeChangeVariableEvent):
        if event.new_value == AUTOMATIC:
            return
        validate_integer_variable(variable, event, lower_bound, upper_bound, cancel_reason)
 
def validate_real_or_automatic(variable, event, lower_bound = None, upper_bound = None, cancel_reason = """The value must be a real or "Automatic"."""):
    if isinstance(event,BeforeChangeVariableEvent):
        if event.new_value == AUTOMATIC:
            return
        validate_real_variable(variable, event, lower_bound, upper_bound, cancel_reason)
        
class ChangeVariableEvent(object):
    """Abstract class representing either the event that a variable will be
    changed or has been changed
    
    """
    def __init__(self,old_value, new_value):
        self.__old_value = old_value
        self.__new_value = new_value
    
    def get_old_value(self):
        return self.__old_value
    
    old_value=property(get_old_value)
    
    def get_new_value(self):
        return self.__new_value
    
    new_value=property(get_new_value)

class BeforeChangeVariableEvent(ChangeVariableEvent):
    """Indicates that a variable is about to change, allows a listener to cancel the change
    
    """
    def __init__(self,old_value,new_value):
        ChangeVariableEvent.__init__(self,old_value,new_value)
        self.__allow_change = True
        self.__cancel_reason = None
        
    def cancel_change(self,reason=None):
        self.__allow_change = False
        self.__cancel_reason = reason
    
    def allow_change(self):
        return self.__allow_change
    
    def get_cancel_reason(self):
        return self.__cancel_reason
    
    cancel_reason = property(get_cancel_reason)
    
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
            self.kind = m.groups()[0]
            self.variable_number = int(m.groups()[1])
            self.value = m.groups()[2]
        else:
            self.kind = kwargs['kind']
            self.variable_number = kwargs['variable_number']
            self.value = kwargs['value']
        if self.kind not in [ANN_TEXT,ANN_CHOICE,ANN_DEFAULT,ANN_INFOTYPE,ANN_INPUTTYPE, ANN_PATHNAMETEXT, ANN_FILENAMETEXT]:
            raise ValueError("Unrecognized annotation: %s"%(self.Kind))

def text_annotation(variable_number, value):
    """Create a text annotation
    """
    return Annotation(kind=ANN_TEXT,variable_number = variable_number, value = value)

def choice_annotations(variable_number, values):
    """Create choice annotations for a variable
    
    variable_number - the one-indexed variable number
    values - a sequence of possible values for the variable
    """
    return [Annotation(kind=ANN_CHOICE,variable_number=variable_number,value=value) for value in values]

def default_annotation(variable_number,value):
    """Create a default value annotation
    """
    return Annotation(kind=ANN_DEFAULT,variable_number=variable_number,value=value)

def infotype_provider_annotation(variable_number,value):
    """Create an infotype provider that provides a certain class of thing (e.g. imagegroup or objectgroup)
    
    variable_number - one-based variable number for the annotation
    value - infotype such as object
    """
    return Annotation(kind=ANN_INFOTYPE, variable_number = variable_number, value="%s indep"%(value))

def infotype_client_annotation(variable_number,value):
    """Create an infotype provider that needs a certain class of thing (e.g. imagegroup or objectgroup)
    
    variable_number - one-based variable number for the annotation
    value - infotype such as object
    """
    return Annotation(kind=ANN_INFOTYPE, variable_number = variable_number, value=value)

def input_type_annotation(variable_number,value):
    """Create an input type annotation, such as popupmenu
    """
    return Annotation(kind=ANN_INPUTTYPE, variable_number= variable_number, value=value)

def choice_popup_annotation(variable_number, text, values, customizable=False):
    """Create all the pieces needed for a choice popup variable
    
    variable_number - the one-based index of the variable
    text - what the user sees to the left of the popup
    values - a sequence containing the allowed values
    """
    return [text_annotation(variable_number,text)] + \
            choice_annotations(variable_number, values) +\
            [input_type_annotation(variable_number,(customizable and 'menupopup custom)') or 'menupopup')]

def indep_group_annotation(variable_number, text, group, default=DO_NOT_USE):
    """Create all the pieces needed for an edit box for a variable defining a member of a particular group
    
    variable_number - the one-based index of the variable
    text - what the user sees to the left of the edit box
    group - the group, for instance imagegroup or objectgroup
    default - the default value that appears when the variable is created
    """
    return edit_box_annotation(variable_number, text, default)+ \
           [infotype_provider_annotation(variable_number,group)]

def group_annotation(variable_number, text, group):
    """Create the pieces needed for a dependent group popup menu
    
    variable_number - one-based index of the variable
    text - the text to the left of the drop-down
    group - the group, forinstance imagegroup or objectgroup
    """
    return [text_annotation(variable_number, text), \
            infotype_client_annotation(variable_number, group),
            input_type_annotation(variable_number,'menupopup')] 

def edit_box_annotation(variable_number, text, default=DO_NOT_USE):
    """Create a text annotation and a default annotation to define a variable that uses an edit box
    
    variable_number - the one-based index of the variable
    text - what the user sees to the left of the edit box
    default - the default value for the box
    """
    return [text_annotation(variable_number,text),
            default_annotation(variable_number, default)]

def checkbox_annotation(variable_number, text, default=False):
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
    return choice_popup_annotation(variable_number, text, choices)
    
def get_annotations_as_dictionary(annotations):
    """Return a multilevel dictionary based on the annotations
    
    Return a multilevel dictionary based on the annotations. The first level
    is the variable number. The second level is the variable kind. The value
    of the second level is an array containing all annotations of that kind
    and variable number.
    """
    dict = {}
    for annotation in annotations:
        vn = annotation.variable_number
        if not dict.has_key(vn):
            dict[vn]={}
        if not dict[vn].has_key(annotation.kind):
            dict[vn][annotation.kind] = []
        dict[vn][annotation.kind].append(annotation)
    return dict

def get_variable_annotations(annotations,variable_number):
    variable_annotations = []
    for annotation in annotations:
        if annotation.variable_number == variable_number:
            variable_annotations.append(annotation)
    return variable_annotations

def get_variable_text(annotations, variable_number):
    for annotation in annotations:
        if annotation.variable_number == variable_number and annotation.kind=='text':
            return annotation.value
    return None
