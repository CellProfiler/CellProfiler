""" Setting.py - represents a module setting

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import matplotlib.cm
import numpy as np
import re
import uuid

DO_NOT_USE = 'Do not use'
AUTOMATIC = "Automatic"
YES = 'Yes'
NO = 'No'
LEAVE_BLANK = 'Leave blank'
DEFAULT = 'Default'

class Setting(object):
    """A module setting which holds a single string value
    
    """
    def __init__(self,text,value,doc=None):
        """Initialize a setting with the enclosing module and its string value
        
        module - the module containing this setting
        text   - the explanatory text for the setting
        value  - the default or initial value for the setting
        """
        self.__annotations = []
        self.__text = text
        self.__value = value
        self.doc = doc
        self.__key = uuid.uuid1() 
    
    def set_value(self,value):
        self.__value=value

    def key(self):
        """Return a key that can be used in a dictionary to refer to this setting
        
        """
        return self.__key
    
    def get_text(self):
        """The explanatory text for the setting
        """
        return self.__text
    
    def set_text(self, value):
        self.__text = value
    
    text = property(get_text, set_text)
    
    def get_value(self):
        """The string contents of the setting"""
        return self.__value
    
    def __internal_get_value(self):
        """The value stored within the setting"""
        return self.get_value()
    
    def __internal_set_value(self,value):
        self.set_value(value)
    
    value = property(__internal_get_value,__internal_set_value)
    
    def __eq__(self, x):
        return self.value == str(x)
    
    def __ne__(self, x):
        return not self.__eq__(x)
    
    def get_is_yes(self):
        """Return true if the setting's value is "Yes" """
        return self.__value == YES
    
    def set_is_yes(self,is_yes):
        """Set the setting value to Yes if true, No if false"""
        self.__value = (is_yes and YES) or NO
    
    is_yes = property(get_is_yes,set_is_yes)
    
    def get_is_do_not_use(self):
        """Return true if the setting's value is Do not use"""
        return self.value == DO_NOT_USE
    
    is_do_not_use = property(get_is_do_not_use)
    
    def test_valid(self, pipeline):
        """Throw a ValidationError if the value of this setting is inappropriate for the context"""
        pass
    
    def __str__(self):
        if not isinstance(self.__value,str):
            raise ValidationError("%s was not a string"%(self.__value),self)
        return self.__value

class HiddenCount(Setting):
    """A setting meant only for saving an item count
    
    The HiddenCount setting should never be in the visible settings.
    It should be tied to a sequence variable which gives the number of
    items which is the value for this variable.
    """
    def __init__(self, sequence):
        super(HiddenCount,self).__init__("Hidden", str(len(sequence)))
        self.__sequence = sequence

    def set_value(self, value):
        if not value.isdigit():
            raise ValueError("The value must be an integer")
        count = int(value)
        if count == len(self.__sequence):
            # The value was "inadvertantly" set, but is correct
            return
        raise NotImplementedError("The count should be inferred, not set")

    def get_value(self):
        return len(self.__sequence)

    def set_sequence(self, sequence):
        '''Set the sequence used to maintain the count'''
        self.__sequence = sequence
    
    def __str__(self):
        return str(len(self.__sequence))

class Text(Setting):
    """A setting that displays as an edit box, accepting a string
    
    """
    def __init__(self, text, value, *args, **kwargs):
        super(Text,self).__init__(text, value, *args, **kwargs)

class RegexpText(Setting):
    """A setting with a regexp button on the side
    """
    def __init__(self, text, value, *args, **kwargs):
        super(RegexpText,self).__init__(text, value, *args, **kwargs)

class DirectoryPath(Text):
    """A setting that displays a filesystem path name
    """
    def __init__(self, text, value, *args, **kwargs):
        super(DirectoryPath,self).__init__(text, value, *args, **kwargs)

class FilenameText(Text):
    """A setting that displays a file name
    """
    def __init__(self, text, value, *args, **kwargs):
        super(FilenameText,self).__init__(text, value, *args, **kwargs)

class Integer(Text):
    """A setting that allows only integer input
    
    Initializer:
    text - explanatory text for setting
    value - default value
    minval - minimum allowed value defaults to no minimum
    maxval - maximum allowed value defaults to no maximum
    """
    def __init__(self, text, value=0, minval=None, maxval=None, *args, 
                 **kwargs):
        super(Integer,self).__init__(text, str(value), *args, **kwargs)
        self.__minval = minval
        self.__maxval = maxval
    
    def set_value(self,value):
        """Convert integer to string
        """
        str_value = str(value)
        super(Integer,self).set_value(str_value)
        
    def get_value(self):
        """Return the value of the setting as an integer
        """
        return int(super(Integer,self).get_value())
    
    def test_valid(self,pipeline):
        """Return true only if the text value is an integer
        """
        if not str(self).isdigit():
            raise ValidationError('Must be an integer value, was "%s"'%(str(self)),self)
        if self.__minval != None and self.__minval > self.value:
            raise ValidationError('Must be at least %d, was %d'%(self.__minval, self.value),self)
        if self.__maxval != None and self.__maxval < self.value:
            raise ValidationError('Must be at most %d, was %d'%(self.__maxval, self.value),self)
        
    def __eq__(self,x):
        if super(Integer,self).__eq__(x):
            return True
        return self.value == x

class IntegerRange(Setting):
    """A setting that allows only integer input between two constrained values
    """
    def __init__(self,text,value=(0,1),minval=None, maxval=None, *args, 
                 **kwargs):
        """Initialize an integer range
        text  - helpful text to be displayed to the user
        value - initial default value, a two-tuple as minimum and maximum
        minval - the minimum acceptable value of either
        maxval - the maximum acceptable value of either
        """
        super(IntegerRange,self).__init__(text, "%d,%d"%value, *args, **kwargs)
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
        if values[0].isdigit():
            min = int(values[0])
        else:
            min = None
        if len(values) > 1  and values[1].isdigit():
            max = int(values[1])
        else:
            max = None
        return (min,max)
    
    def get_min(self):
        """The minimum value of the range"""
        return self.value[0]
    
    def set_min(self, value):
        self.set_value((value, self.max))
        
    min = property(get_min, set_min)
    
    def get_max(self):
        """The maximum value of the range"""
        return self.value[1]
    
    def set_max(self, value):
        self.set_value((self.min, value))
        
    max = property(get_max, set_max)
    
    def test_valid(self, pipeline):
        values = str(self).split(',')
        if len(values) < 2:
            raise ValidationError("Minimum and maximum values must be separated by a comma",self)
        if len(values) > 2:
            raise ValidationError("Only two values allowed",self)
        for value in values:
            if not value.isdigit():
                raise ValidationError("%s is not an integer"%(value),self)
        if self.__minval and self.__minval > self.min:
            raise ValidationError("%d can't be less than %d"%(self.min,self.__minval),self)
        if self.__maxval and self.__maxval < self.max:
            raise ValidationError("%d can't be greater than %d"%(self.max,self.__maxval),self)
        if self.min > self.max:
            raise ValidationError("%d is greater than %d"%(self.min, self.max),self)

class Coordinates(Setting):
    """A setting representing X and Y coordinates on an image
    """
    def __init__(self, text, value=(0,0), *args, **kwargs):
        """Initialize an integer range
        text  - helpful text to be displayed to the user
        value - initial default value, a two-tuple as x and y
        """
        super(Coordinates,self).__init__(text, "%d,%d"%value, *args, **kwargs)
    
    def set_value(self,value):
        """Convert integer tuples to string
        """
        try: 
            if len(value) == 2:
                super(Coordinates,self).set_value("%d,%d"%(value[0],value[1]))
                return
        except: 
            pass
        super(Coordinates,self).set_value(value)
    
    def get_value(self):
        """Convert the underlying string to a two-tuple"""
        values = str(self).split(',')
        if values[0].isdigit():
            x = int(values[0])
        else:
            x = None
        if len(values) > 1  and values[1].isdigit():
            y = int(values[1])
        else:
            y = None
        return (x,y)
    
    def get_x(self):
        """The x coordinate"""
        return self.value[0]
    
    x = property(get_x)
    
    def get_y(self):
        """The y coordinate"""
        return self.value[1]
    
    y = property(get_y)
    
    def test_valid(self, pipeline):
        values = str(self).split(',')
        if len(values) < 2:
            raise ValidationError("X and Y values must be separated by a comma",self)
        if len(values) > 2:
            raise ValidationError("Only two values allowed",self)
        for value in values:
            if not value.isdigit():
                raise ValidationError("%s is not an integer"%(value),self)

BEGIN = "begin"
END = "end"

class IntegerOrUnboundedRange(Setting):
    """A setting that specifies an integer range where the minimum and maximum
    can be set to unbounded by the user.
    
    The maximum value can be relative to the far side in which case a negative
    number is returned for slicing.
    """
    def __init__(self, text, value=(0,END), minval=None, maxval=None,
                 *args, **kwargs):
        """Initialize an integer range
        text  - helpful text to be displayed to the user
        value - initial default value, a two-tuple as minimum and maximum
        minval - the minimum acceptable value of either
        maxval - the maximum acceptable value of either
        """
        super(IntegerOrUnboundedRange,self).__init__(text, 
                                                     "%s,%s"% (str(value[0]),
                                                               str(value[1])),
                                                     *args, **kwargs)
        self.__minval = minval
        self.__maxval = maxval
        
    
    def set_value(self,value):
        """Convert integer tuples to string
        """
        try:
            if len(value) == 2:
                values = value
            else:
                values = value.split(",")
            min_value = str(values[0])
            max_value = str(values[1])
            super(IntegerOrUnboundedRange,self).set_value("%s,%s"%
                                                          (min_value,
                                                           max_value))
            return
        except: 
            pass
        super(IntegerOrUnboundedRange,self).set_value(value)
    
    def get_value(self):
        """Convert the underlying string to a two-tuple"""
        values = str(self).split(',')
        if values[0].isdigit():
            min = int(values[0])
        else:
            min = None
        if len(values) > 1:  
            if values[1].isdigit():
                max = int(values[1])
            elif values[1] == END:
                max = END
            else:
                max = None
        else:
            max = None
        return (min,max)
    
    def get_unbounded_min(self):
        """True if there is no minimum"""
        return self.get_value()[0]==0
    
    unbounded_min = property(get_unbounded_min)
    
    def get_min(self):
        """The minimum value of the range"""
        return self.value[0]
    
    min = property(get_min)
    
    def get_display_min(self):
        """What to display for the minimum"""
        return str(self.min)
    display_min = property(get_display_min)
    
    def get_unbounded_max(self):
        """True if there is no maximum"""
        return self.get_value()[1] == END
    
    unbounded_max = property(get_unbounded_max)
    
    def get_max(self):
        """The maximum value of the range"""
        return self.value[1]
    
    max = property(get_max) 
    
    def get_display_max(self):
        """What to display for the maximum"""
        if self.unbounded_max:
            return "0"
        return str(abs(self.max))
    display_max = property(get_display_max)
    
    def test_valid(self, pipeline):
        values = str(self).split(',')
        if len(values) < 2:
            raise ValidationError("Minimum and maximum values must be separated by a comma",self)
        if len(values) > 2:
            raise ValidationError("Only two values allowed",self)
        if not values[0].isdigit():
            raise ValidationError("%s is not an integer"%(values[0]))
        if not (values[1] == END or
                values[1].isdigit() or
                (values[1][0]=='-' and values[1][1:].isdigit())):
                raise ValidationError("%s is not an integer or %s"%(values[1], END),self)
        if ((not self.unbounded_min) and 
            self.__minval and
            self.__minval > self.min):
            raise ValidationError("%d can't be less than %d"%(self.min,self.__minval),self)
        if ((not self.unbounded_max) and 
            self.__maxval and 
            self.__maxval < self.max):
            raise ValidationError("%d can't be greater than %d"%(self.max,self.__maxval),self)
        if ((not self.unbounded_min) and (not self.unbounded_max) and 
            self.min > self.max and self.max > 0):
            raise ValidationError("%d is greater than %d"%(self.min, self.max),self)

class Float(Text):
    """A setting that allows only floating point input
    """
    def __init__(self, text, value=0, minval=None, maxval=None, *args,
                 **kwargs):
        super(Float,self).__init__(text, str(value), *args, **kwargs)
        self.__minval = minval
        self.__maxval = maxval
    
    def set_value(self,value):
        """Convert integer to string
        """
        str_value = str(value)
        super(Float,self).set_value(str_value)
        
    def get_value(self):
        """Return the value of the setting as an integer
        """
        return float(super(Float,self).get_value())
    
    def test_valid(self,pipeline):
        """Return true only if the text value is float
        """
        try:
            # Raises value error inside self.value if not a float
            if self.__minval != None and self.__minval > self.value:
                raise ValidationError('Must be at least %d, was %d'%(self.__minval, self.value),self)
        except ValueError:
            raise ValidationError('Value not in decimal format', self)
        if self.__maxval != None and self.__maxval < self.value:
            raise ValidationError('Must be at most %d, was %d'%(self.__maxval, self.value),self)
        
    def __eq__(self,x):
        if super(Float,self).__eq__(x):
            return True
        return self.value == x

class FloatRange(Setting):
    """A setting that allows only floating point input between two constrained values
    """
    def __init__(self, text, value=(0,1), minval=None, maxval=None, *args,
                 **kwargs):
        """Initialize an integer range
        text  - helpful text to be displayed to the user
        value - initial default value, a two-tuple as minimum and maximum
        minval - the minimum acceptable value of either
        maxval - the maximum acceptable value of either
        """
        super(FloatRange,self).__init__(text, "%f,%f"%value, *args, **kwargs)
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
        return float(str(self).split(',')[0])
    
    def set_min(self, value):
        self.set_value((value, self.max))
        
    min = property(get_min, set_min)
    
    def get_max(self):
        """The maximum value of the range"""
        return float(str(self).split(',')[1])
    
    def set_max(self, value):
        self.set_value((self.min, value))
        
    max = property(get_max, set_max)
    
    def test_valid(self, pipeline):
        values = str(self).split(',')
        if len(values) < 2:
            raise ValidationError("Minimum and maximum values must be separated by a comma",self)
        if len(values) > 2:
            raise ValidationError("Only two values allowed",self)
        for value in values:
            try:
                float(value)
            except ValueError:
                raise ValidationError("%s is not in decimal format"%(value),self)
        if self.__minval and self.__minval > self.min:
            raise ValidationError("%f can't be less than %f"%(self.min,self.__minval),self)
        if self.__maxval and self.__maxval < self.max:
            raise ValidationError("%f can't be greater than %f"%(self.max,self.__maxval),self)
        if self.min > self.max:
            raise ValidationError("%f is greater than %f"%(self.min, self.max),self)

class NameProvider(Text):
    """A setting that provides a named object
    """
    def __init__(self, text, group, value=DO_NOT_USE, *args, **kwargs):
        super(NameProvider,self).__init__(text, value, *args, **kwargs)
        self.__group = group
    
    def get_group(self):
        """This setting provides a name to this group
        
        Returns a group name, e.g. imagegroup or objectgroup
        """
        return self.__group
    
    group = property(get_group)

class ImageNameProvider(NameProvider):
    """A setting that provides an image name
    """
    def __init__(self, text, value=DO_NOT_USE, *args, **kwargs):
        super(ImageNameProvider,self).__init__(text, 'imagegroup', value,
                                               *args, **kwargs)

class FileImageNameProvider(ImageNameProvider):
    """A setting that provides an image name where the image has an associated file"""
    def __init__(self, text, value=DO_NOT_USE, *args, **kwargs):
        super(FileImageNameProvider,self).__init__(text, value, *args,
                                                   **kwargs)

class CroppingNameProvider(ImageNameProvider):
    """A setting that provides an image name where the image has a cropping mask"""
    def __init__(self, text, value=DO_NOT_USE, *args, **kwargs):
        super(CroppingNameProvider,self).__init__(text, value, *args, **kwargs)
    
class ObjectNameProvider(NameProvider):
    """A setting that provides an image name
    """
    def __init__(self, text, value=DO_NOT_USE, *args, **kwargs):
        super(ObjectNameProvider,self).__init__(text, 'objectgroup', value,
                                                *args, **kwargs)

class OutlineNameProvider(NameProvider):
    '''A setting that provides an object outline name
    '''
    def __init__(self, text, value=DO_NOT_USE, *args, **kwargs):
        super(OutlineNameProvider,self).__init__(text, 'outlinegroup', *args,
                                                 **kwargs)

class NameSubscriber(Setting):
    """A setting that takes its value from one made available by name providers
    """
    def __init__(self, text, group, value=None,
                 can_be_blank=False, blank_text=LEAVE_BLANK, *args, **kwargs):
        if value==None:
            value = (can_be_blank and blank_text) or "None"
        super(NameSubscriber,self).__init__(text, value, *args, **kwargs)
    
        self.__group = group
        self.__can_be_blank = can_be_blank
        self.__blank_text = blank_text
    
    def get_group(self):
        """This setting provides a name to this group
        
        Returns a group name, e.g. imagegroup or objectgroup
        """
        return self.__group
    
    group = property(get_group)
    
    def get_choices(self,pipeline):
        choices = []
        if self.__can_be_blank:
            choices.append(self.__blank_text)
        for module in pipeline.modules():
            module_choices = module.get_name_providers(self.group)
            for setting in module.visible_settings():
                if setting.key() == self.key():
                    choices = np.unique(choices).tolist()
                    choices.sort()
                    return choices
                if (isinstance(setting, NameProvider) and 
                    setting != DO_NOT_USE and
                    self.matches(setting)):
                    module_choices.append(setting.value)
            choices += module_choices
        assert False, "Setting not among visible settings in pipeline"
    
    def get_is_blank(self):
        """True if the selected choice is the blank one"""
        return self.__can_be_blank and self.value == self.__blank_text
    is_blank = property(get_is_blank)
    
    def matches(self, setting):
        """Return true if this subscriber matches the category of the provider"""
        return self.group == setting.group
    
    def test_valid(self,pipeline):
        if len(self.get_choices(pipeline)) == 0:
            raise ValidationError("No prior instances of %s were defined"%(self.group),self)
        if self.value not in self.get_choices(pipeline):
            raise ValidationError("%s not in %s"%(self.value,reduce(lambda x,y: "%s,%s"%(x,y),self.get_choices(pipeline))),self)

class ImageNameSubscriber(NameSubscriber):
    """A setting that provides an image name
    """
    def __init__(self, text, value=None, can_be_blank = False,
                 blank_text=LEAVE_BLANK, *args, **kwargs):
        super(ImageNameSubscriber,self).__init__(text, 'imagegroup', value,
                                                 can_be_blank, blank_text,
                                                 *args, **kwargs)

class FileImageNameSubscriber(ImageNameSubscriber):
    """A setting that provides image names loaded from files"""
    def __init__(self, text, value=DO_NOT_USE, can_be_blank = False,
                 blank_text=LEAVE_BLANK, *args, **kwargs):
        super(FileImageNameSubscriber,self).__init__(text, value, can_be_blank,
                                                     blank_text, *args,
                                                     **kwargs)
    
    def matches(self,setting):
        """Only match FileImageNameProvider variables"""
        return isinstance(setting, FileImageNameProvider)

class CroppingNameSubscriber(ImageNameSubscriber):
    """A setting that provides image names that have cropping masks"""
    def __init__(self, text, value=DO_NOT_USE, can_be_blank=False,
                 blank_text=LEAVE_BLANK, *args, **kwargs):
        super(CroppingNameSubscriber,self).__init__(text, value, can_be_blank,
                                                    blank_text, *args, 
                                                    **kwargs)
    
    def matches(self,setting):
        """Only match CroppingNameProvider variables"""
        return isinstance(setting, CroppingNameProvider)

class ObjectNameSubscriber(NameSubscriber):
    """A setting that provides an image name
    """
    def __init__(self, text, value=DO_NOT_USE, can_be_blank=False,
                 blank_text=LEAVE_BLANK, *args, **kwargs):
        super(ObjectNameSubscriber,self).__init__(text, 'objectgroup', value,
                                                  can_be_blank, blank_text,
                                                  *args, **kwargs)

class OutlineNameSubscriber(NameSubscriber):
    '''A setting that provides a list of available object outline names
    '''
    def __init__(self, text, value="None", can_be_blank=False, 
                 blank_text=LEAVE_BLANK, *args, **kwargs):
        super(OutlineNameSubscriber, self).__init__(text, 'outlinegroup', 
                                                    value, can_be_blank,
                                                    blank_text, *args,
                                                    **kwargs)

class FigureSubscriber(Setting):
    """A setting that provides a figure indicator
    """
    def __init(self,text,value=DO_NOT_USE, *args, **kwargs):
        super(Setting,self).__init(text, value, *args, **kwargs)
    
    def get_choices(self,pipeline):
        choices = []
        for module in pipeline.modules():
            for setting in module.visible_settings():
                if setting.key() == self.key():
                    return choices
            choices.append("%d: %s"%(module.module_num, module.module_name))
        assert False, "Setting not among visible settings in pipeline"

class Binary(Setting):
    """A setting that is represented as either true or false
    The underlying value stored in the settings slot is "Yes" or "No"
    for historical reasons.
    """
    def __init__(self, text, value, *args, **kwargs):
        """Initialize the binary setting with the module, explanatory
        text and value. The value for a binary setting is True or
        False.
        """
        str_value = (value and YES) or NO
        super(Binary,self).__init__(text, str_value, *args, **kwargs)
    
    def set_value(self,value):
        """When setting, translate true and false into yes and no"""
        if value == YES or value == NO or\
           isinstance(value,str) or isinstance(value,unicode):
            super(Binary,self).set_value(value)
        else: 
            str_value = (value and YES) or NO
            super(Binary,self).set_value(str_value)
    
    def get_value(self):
        """Get the value of a binary setting as a truth value
        """
        return super(Binary,self).get_value() == YES 
    
    def __eq__(self,x):
        if x == NO:
            x = False
        return (self.value and x) or ((not self.value) and (not x)) 
    
class Choice(Setting):
    """A setting that displays a drop-down set of choices
    
    """
    def __init__(self, text, choices, value=None, tooltips=None, *args,
                 **kwargs):
        """Initializer
        module - the module containing the setting
        text - the explanatory text for the setting
        choices - a sequence of string choices to be displayed in the drop-down
        value - the default choice or None to choose the first of the choices.
        tooltips - a dictionary of choice to tooltip
        """
        super(Choice,self).__init__(text, value or choices[0], *args, **kwargs)
        self.__choices = choices
        self.__tooltips = tooltips
    
    def __internal_get_choices(self):
        """The sequence of strings that define the choices to be displayed"""
        return self.get_choices()
    
    def get_choices(self):
        """The sequence of strings that define the choices to be displayed"""
        return self.__choices
    
    choices = property(__internal_get_choices)
    
    def get_tooltips(self):
        '''The tooltip strings for each choice''' 
        return self.__tooltips
    
    tooltips = property(get_tooltips)
    
    @property
    def has_tooltips(self):
        '''Return true if the choice has tooltips installed'''
        return self.__tooltips is not None
    
    def test_valid(self,pipeline):
        """Check to make sure that the value is among the choices"""
        if self.value not in self.choices:
            raise ValidationError("%s is not one of %s"%(self.value, reduce(lambda x,y: "%s,%s"%(x,y),self.choices)),self)

class CustomChoice(Choice):
    def __init__(self, text, choices, value=None, *args, **kwargs):
        """Initializer
        module - the module containing the setting
        text - the explanatory text for the setting
        choices - a sequence of string choices to be displayed in the drop-down
        value - the default choice or None to choose the first of the choices.
        """
        super(CustomChoice,self).__init__(text, choices, value, *args, 
                                          **kwargs)
    
    def get_choices(self):
        """Put the custom choice at the top"""
        choices = list(super(CustomChoice,self).get_choices())
        if self.value not in choices:
            choices.insert(0,self.value)
        return choices
    
    def set_value(self,value):
        """Bypass the check in "Choice"."""
        Setting.set_value(self, value)
    
class DoSomething(Setting):
    """Do something in response to a button press
    """
    def __init__(self, text, label, callback, *args, **kwargs):
        super(DoSomething,self).__init__(text, 'n/a', *args, **kwargs)
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

class Measurement(Setting):
    '''A measurement done on a class of objects (or Experiment or Image)
    
    A measurement represents a fully-qualified name of a measurement taken
    on an object. Measurements have categories and feature names and
    may or may not have a secondary image (for instance, the image used
    to measure an intensity), secondary object (for instance, the parent
    object when relating two classes of objects or the object name when
    aggregating object measurements over an image) or scale.
    '''
    def __init__(self, text, object_fn, value = "None", *args, **kwargs):
        '''Construct the measurement category subscriber setting
        
        text - Explanatory text that appears to the side of the setting
        object_fn - a function that returns the measured object when called
        value - the initial value of the setting
        '''
        super(Measurement, self).__init__(text, value, *args, **kwargs)
        self.__object_fn = object_fn
    
    def construct_value(self, category, feature_name, image_name, scale):
        '''Construct a value that might represent a partially complete value'''
        if category is None:
            value='None'
        elif feature_name is None:
            value = category
        else:
            parts = [category, feature_name]
            if not image_name is None:
                parts.append(image_name)
            if not scale is None:
                parts.append(scale)
            value = '_'.join(parts)
        return str(value)
        
    def get_category_choices(self, pipeline):
        '''Find the categories of measurements available from the object '''
        object_name = self.__object_fn()
        categories = set()
        for module in pipeline.modules():
            if self.key() in [x.key() for x in module.settings()]:
                break
            categories.update(module.get_categories(pipeline, object_name))
        result = list(categories)
        result.sort()
        return result
    
    def get_category(self, pipeline):
        '''Return the currently chosen category'''
        categories = self.get_category_choices(pipeline)
        for category in categories:
            if (self.value.startswith(category+'_') or
                self.value == category):
                return category
        return None
    
    def get_feature_name_choices(self, pipeline):
        '''Find the feature name choices available for the chosen category'''
        object_name = self.__object_fn()
        category = self.get_category(pipeline)
        if category is None:
            return []
        feature_names = set()
        for module in pipeline.modules():
            if self.key() in [x.key() for x in module.settings()]:
                break
            feature_names.update(module.get_measurements(pipeline, object_name,
                                                         category))
        result = list(feature_names)
        result.sort()
        return result
    
    def get_feature_name(self, pipeline):
        '''Return the currently selected feature name'''
        category = self.get_category(pipeline)
        if category is None:
            return None
        feature_names = self.get_feature_name_choices(pipeline)
        for feature_name in feature_names:
            head = '_'.join((category, feature_name))
            if (self.value.startswith(head+'_') or
                self.value == head):
                return feature_name
        return None
    
    def get_image_name_choices(self, pipeline):
        '''Find the secondary image name choices available for a feature
        
        A measurement can still be valid, even if there are no available
        image name choices. The UI should not offer image name choices
        if no choices are returned.
        '''
        object_name = self.__object_fn()
        category = self.get_category(pipeline)
        feature_name = self.get_feature_name(pipeline)
        if category is None or feature_name is None:
            return []
        image_names = set()
        for module in pipeline.modules():
            if self.key() in [x.key() for x in module.settings()]:
                break
            image_names.update(module.get_measurement_images(pipeline,
                                                             object_name,
                                                             category,
                                                             feature_name))
        result = list(image_names)
        result.sort()
        return result
    
    def get_image_name(self, pipeline):
        '''Return the currently chosen image name'''
        object_name = self.__object_fn()
        category = self.get_category(pipeline)
        if category is None:
            return None
        feature_name = self.get_feature_name(pipeline)
        if feature_name is None:
            return None
        image_names = self.get_image_name_choices(pipeline)
        for image_name in image_names:
            head = '_'.join((category, feature_name, image_name))
            if (self.value.startswith(head+'_') or
                self.value == head):
                return image_name
        return None
    
    def get_scale_choices(self, pipeline):
        '''Return the measured scales for the currently chosen measurement
        
        The setting may still be valid, even though there are no scale choices.
        In this case, the UI should not offer the user a scale choice.
        '''
        object_name = self.__object_fn()
        category = self.get_category(pipeline)
        feature_name = self.get_feature_name(pipeline)
        image_name = self.get_image_name(pipeline)
        if category is None or feature_name is None:
            return []
        scales = set()
        for module in pipeline.modules():
            if self.key() in [x.key() for x in module.settings()]:
                break
            scales.update(module.get_measurement_scales(pipeline,
                                                        object_name,
                                                        category,
                                                        feature_name,
                                                        image_name))
        result = list(scales)
        result.sort()
        return result
        
    def get_scale(self, pipeline):
        '''Return the currently chosen scale'''
        object_name = self.__object_fn()
        category = self.get_category(pipeline)
        feature_name = self.get_feature_name(pipeline)
        image_name = self.get_image_name(pipeline)
        if category is None or feature_name is None:
            return None
        if image_name is None:
            head = '_'.join((category, feature_name))
        else:
            head = '_'.join((category, feature_name, image_name))
        for scale in self.get_scale_choices(pipeline):
            if self.value == '_'.join((head, scale)):
                return scale
        return None 
    
    def test_valid(self, pipeline):
        if self.get_category(pipeline) is None:
            raise ValidationError("%s has an unavailable measurement category" %
                                  self.value, self)
        if self.get_feature_name(pipeline) is None:
            raise ValidationError("%s has an unmeasured feature name" %
                                  self.value, self)
        if (self.get_image_name(pipeline) is None and
            len(self.get_image_name_choices(pipeline))):
            raise ValidationError("%s has an unavailable image name" %
                                  self.value, self)
        if self.get_scale(pipeline) and len(self.get_scale_choices(pipeline)):
            raise ValidationError("%s has an unavailable scale" %
                                  self.value, self)

class Colormap(Choice):
    '''Represents the choice of a colormap'''
    def __init__(self, text, value=DEFAULT, *args, **kwargs):
        names = list(matplotlib.cm.cmapnames)
        names.sort()
        choices = [DEFAULT] + names
        super(Colormap,self).__init__(text, choices, value, *args, **kwargs)
    
class ChangeSettingEvent(object):
    """Abstract class representing either the event that a setting will be
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

class BeforeChangeSettingEvent(ChangeSettingEvent):
    """Indicates that a setting is about to change, allows a listener to cancel the change
    
    """
    def __init__(self,old_value,new_value):
        ChangeSettingEvent.__init__(self,old_value,new_value)
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
    
class AfterChangeSettingEvent(ChangeSettingEvent):
    """Indicates that a setting has changed its value
    
    """
    def __init__(self,old_value,new_value):
        ChangeSettingEvent.__init__(self,old_value,new_value)

class DeleteSettingEvent():
    def __init__(self):
        pass

class ValidationError(ValueError):
    """An exception indicating that a setting's value prevents the pipeline from running
    """
    def __init__(self,message,setting):
        """Initialize with an explanatory message and the setting that caused the problem
        """
        super(ValidationError,self).__init__(message)
        self.__setting = setting
    
    def get_setting(self):
        """The setting responsible for the problem
        
        This might be one of several settings partially responsible
        for the problem.
        """
        return self.__setting
    
    setting = property(get_setting)
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
    """Annotations are the bits of comments parsed out of a .m file that provide metadata on a setting
    
    """
    def __init__(self,*args,**kwargs):
        """Initialize either matlab-style with a line of text that is regexp-parsed or more explicitly
        
        args - should be a single line that is parsed
        kind - the kind of annotation it is. Legal values are "text", "choice","default","infotype","inputtype",
               "pathnametext" and "filenametext"
        setting_number - the one-indexed index of the setting in the module's set of settings
        value - the value of the annotation
        """
        if len(args) == 1:
            line = args[0]
            m=re.match("^%([a-z]+)VAR([0-9]+) = (.+)$",line)
            if not m:
                raise(ValueError('Not a setting annotation comment: %s)'%(line)))
            self.kind = m.groups()[0]
            self.setting_number = int(m.groups()[1])
            self.value = m.groups()[2]
        else:
            self.kind = kwargs['kind']
            self.setting_number = kwargs['setting_number']
            self.value = kwargs['value']
        if self.kind not in [ANN_TEXT,ANN_CHOICE,ANN_DEFAULT,ANN_INFOTYPE,ANN_INPUTTYPE, ANN_PATHNAMETEXT, ANN_FILENAMETEXT]:
            raise ValueError("Unrecognized annotation: %s"%(self.Kind))

def text_annotation(setting_number, value):
    """Create a text annotation
    """
    return Annotation(kind=ANN_TEXT,setting_number = setting_number, value = value)

def choice_annotations(setting_number, values):
    """Create choice annotations for a setting
    
    setting_number - the one-indexed setting number
    values - a sequence of possible values for the setting
    """
    return [Annotation(kind=ANN_CHOICE,setting_number=setting_number,value=value) for value in values]

def default_annotation(setting_number,value):
    """Create a default value annotation
    """
    return Annotation(kind=ANN_DEFAULT,setting_number=setting_number,value=value)

def infotype_provider_annotation(setting_number,value):
    """Create an infotype provider that provides a certain class of thing (e.g. imagegroup or objectgroup)
    
    setting_number - one-based setting number for the annotation
    value - infotype such as object
    """
    return Annotation(kind=ANN_INFOTYPE, setting_number = setting_number, value="%s indep"%(value))

def infotype_client_annotation(setting_number,value):
    """Create an infotype provider that needs a certain class of thing (e.g. imagegroup or objectgroup)
    
    setting_number - one-based setting number for the annotation
    value - infotype such as object
    """
    return Annotation(kind=ANN_INFOTYPE, setting_number = setting_number, value=value)

def input_type_annotation(setting_number,value):
    """Create an input type annotation, such as popupmenu
    """
    return Annotation(kind=ANN_INPUTTYPE, setting_number= setting_number, value=value)

def choice_popup_annotation(setting_number, text, values, customizable=False):
    """Create all the pieces needed for a choice popup setting
    
    setting_number - the one-based index of the setting
    text - what the user sees to the left of the popup
    values - a sequence containing the allowed values
    """
    return [text_annotation(setting_number,text)] + \
            choice_annotations(setting_number, values) +\
            [input_type_annotation(setting_number,(customizable and 'menupopup custom)') or 'menupopup')]

def indep_group_annotation(setting_number, text, group, default=DO_NOT_USE):
    """Create all the pieces needed for an edit box for a setting defining a member of a particular group
    
    setting_number - the one-based index of the setting
    text - what the user sees to the left of the edit box
    group - the group, for instance imagegroup or objectgroup
    default - the default value that appears when the setting is created
    """
    return edit_box_annotation(setting_number, text, default)+ \
           [infotype_provider_annotation(setting_number,group)]

def group_annotation(setting_number, text, group):
    """Create the pieces needed for a dependent group popup menu
    
    setting_number - one-based index of the setting
    text - the text to the left of the drop-down
    group - the group, forinstance imagegroup or objectgroup
    """
    return [text_annotation(setting_number, text), \
            infotype_client_annotation(setting_number, group),
            input_type_annotation(setting_number,'menupopup')] 

def edit_box_annotation(setting_number, text, default=DO_NOT_USE):
    """Create a text annotation and a default annotation to define a setting that uses an edit box
    
    setting_number - the one-based index of the setting
    text - what the user sees to the left of the edit box
    default - the default value for the box
    """
    return [text_annotation(setting_number,text),
            default_annotation(setting_number, default)]

def checkbox_annotation(setting_number, text, default=False):
    """Create a checkbox annotation
    
    The checkbox annotation has choice values = 'Yes' and 'No' but
    gets translated by the Gui code into a checkbox.
    setting_number - the one-based index of the setting
    text - the text to display to the user
    default - whether the box should be checked initially (True) or unchecked (False)
    """
    if default:
        choices = [YES,NO]
    else:
        choices = [NO,YES]
    return choice_popup_annotation(setting_number, text, choices)
    
def get_annotations_as_dictionary(annotations):
    """Return a multilevel dictionary based on the annotations
    
    Return a multilevel dictionary based on the annotations. The first level
    is the setting number. The second level is the setting kind. The value
    of the second level is an array containing all annotations of that kind
    and setting number.
    """
    dict = {}
    for annotation in annotations:
        vn = annotation.setting_number
        if not dict.has_key(vn):
            dict[vn]={}
        if not dict[vn].has_key(annotation.kind):
            dict[vn][annotation.kind] = []
        dict[vn][annotation.kind].append(annotation)
    return dict

def get_setting_annotations(annotations,setting_number):
    setting_annotations = []
    for annotation in annotations:
        if annotation.setting_number == setting_number:
            setting_annotations.append(annotation)
    return setting_annotations

def get_setting_text(annotations, setting_number):
    for annotation in annotations:
        if annotation.setting_number == setting_number and annotation.kind=='text':
            return annotation.value
    return None
