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

'''Matlab CellProfiler uses this string for settings to be excluded'''
DO_NOT_USE = 'Do not use'
'''Matlab CellProfiler uses this string for automatically calculated settings'''
AUTOMATIC = "Automatic"
'''Value to store for boolean True settings'''
YES = 'Yes'
'''Value to store for boolean False settings'''
NO = 'No'
LEAVE_BLANK = 'Leave blank'
DEFAULT = 'Default'

'''Names providers and subscribers of images'''
IMAGE_GROUP = 'imagegroup'

'''Names providers and subscribers of objects'''
OBJECT_GROUP = 'objectgroup'

class Setting(object):
    """A module setting which holds a single string value
    
    """
    def __init__(self,text,value,doc=None):
        """Initialize a setting with the enclosing module and its string value
        
        module - the module containing this setting
        text   - the explanatory text for the setting
        value  - the default or initial value for the setting
        doc - documentation for the setting
        """
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
        if isinstance(self.__value, unicode):
            return str(self.__value)
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
        try:
            min = int(values[0])
        except:
            min = None
        if len(values) > 1:  
            if values[1] == END:
                max = END
            else:
                try:
                    max = int(values[1])
                except:
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
        elif self.max is not None:
            return str(abs(self.max))
        else:
            return str(self.max).split(',')[1]
    display_max = property(get_display_max)
    
    def test_valid(self, pipeline):
        values = str(self).split(',')
        if len(values) < 2:
            raise ValidationError("Minimum and maximum values must be separated by a comma",self)
        if len(values) > 2:
            raise ValidationError("Only two values allowed",self)
        if not values[0].isdigit():
            raise ValidationError("%s is not an integer"%(values[0]),self)
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
        super(ImageNameProvider,self).__init__(text, IMAGE_GROUP, value,
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
        super(ObjectNameProvider,self).__init__(text, OBJECT_GROUP, value,
                                                *args, **kwargs)

class OutlineNameProvider(ImageNameProvider):
    '''A setting that provides an object outline name
    '''
    def __init__(self, text, value=DO_NOT_USE, *args, **kwargs):
        super(OutlineNameProvider,self).__init__(text, value, *args,
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
        return choices + get_name_provider_choices(pipeline, self, self.group)
    
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

def get_name_provider_choices(pipeline, last_setting, group):
    '''Scan the pipeline to find name providers for the given group
    
    pipeline - pipeline to scan
    last_setting - scan the modules in order until you arrive at this setting
    group - the name of the group of providers to scan
    returns a list of provider values
    '''
    choices = []
    for module in pipeline.modules():
        module_choices = module.other_providers(group)
        for setting in module.visible_settings():
            if setting.key() == last_setting.key():
                choices = np.unique(choices).tolist()
                choices.sort()
                return choices
            if (isinstance(setting, NameProvider) and 
                setting != DO_NOT_USE and
                last_setting.matches(setting)):
                module_choices.append(setting.value)
        choices += module_choices
    assert False, "Setting not among visible settings in pipeline"
        
class ImageNameSubscriber(NameSubscriber):
    """A setting that provides an image name
    """
    def __init__(self, text, value=None, can_be_blank = False,
                 blank_text=LEAVE_BLANK, *args, **kwargs):
        super(ImageNameSubscriber,self).__init__(text, IMAGE_GROUP, value,
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
        super(ObjectNameSubscriber,self).__init__(text, OBJECT_GROUP, value,
                                                  can_be_blank, blank_text,
                                                  *args, **kwargs)

class OutlineNameSubscriber(ImageNameSubscriber):
    '''A setting that provides a list of available object outline names
    '''
    def __init__(self, text, value="None", can_be_blank=False, 
                 blank_text=LEAVE_BLANK, *args, **kwargs):
        super(OutlineNameSubscriber, self).__init__(text, 
                                                    value, can_be_blank,
                                                    blank_text, *args,
                                                    **kwargs)
    def matches(self, setting):
        '''Only match OutlineNameProvider variables'''
        return isinstance(setting, OutlineNameProvider)

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
    
    def __nonzero__(self):
        '''Return the value when testing for True / False'''
        return self.value
    
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

class MultiChoice(Setting):
    '''A setting that represents selection of multiple choices from a list'''
    def __init__(self, text, choices, value=None, *args, **kwargs):
        '''Initializer
        
        text - the explanatory text for the setting
        choices - a sequence of string choices to be selected
        value - a list of selected choices or a comma-separated string list
        '''
        super(MultiChoice,self).__init__(text, 
                                         self.parse_value(value),
                                         *args, **kwargs)
        self.__choices = choices

    def parse_value(self, value):
        if value is None:
            return ''
        elif isinstance(value, str) or isinstance(value, unicode):
            return value
        elif hasattr(value, "__getitem__"):
            return ','.join(value)
        raise ValueError("Unexpected value type: %s"%type(value))
        
    def __internal_get_choices(self):
        """The sequence of strings that define the choices to be displayed"""
        return self.get_choices()
    
    def get_choices(self):
        """The sequence of strings that define the choices to be displayed"""
        return self.__choices

    def __internal_set_choices(self, choices):
        return self.set_choices(choices)
    
    def set_choices(self, choices):
        self.__choices = choices
    
    choices = property(__internal_get_choices,__internal_set_choices)
    
    def set_value(self, value):
        '''Set the value of a multi-choice setting
        
        value is either a single string, a comma-separated string of
        multiple choices or a list of strings
        '''
        super(MultiChoice,self).set_value(self.parse_value(value))
    
    def get_selections(self):
        '''Return the currently selected values'''
        value = self.get_value()
        if len(value) == 0:
            return ()
        return value.split(',')
    
    selections = property(get_selections)
    
    def test_valid(self, pipeline):
        '''Ensure that the selections are among the choices'''
        for selection in self.get_selections():
            if selection not in self.choices:
                raise ValidationError("%s is not one of %s" % 
                                      (selection, 
                                       reduce(lambda x,y: "%s,%s" % 
                                              (x,y),self.choices)),
                                      self)

class SubscriberMultiChoice(MultiChoice):
    '''A multi-choice setting that gets its choices through providers
    
    This setting operates similarly to the name subscribers. It gets
    its choices from the name providers for the subscriber's group.
    It displays a list of choices and the user can select multiple
    choices.
    '''
    def __init__(self, text, group, value=None, *args, **kwargs):
        super(SubscriberMultiChoice,self).__init__(text, [], value,
                                                   *args, **kwargs)
        self.group = group
    
    def load_choices(self, pipeline):
        '''Get the choice list from name providers'''
        self.choices = get_name_provider_choices(pipeline, self, self.group)
    
    def matches(self, provider):
        '''Return true if the provider is compatible with this subscriber
        
        This method can be used to be more particular about the providers
        that are selected. For instance, if you want a list of only
        FileImageNameProviders (images loaded from files), you can
        check that here.
        '''
        return provider.group == self.group
    
    def test_valid(self, pipeline):
        self.load_choices(pipeline)
        super(SubscriberMultiChoice, self).test_valid(pipeline)
    
class ObjectSubscriberMultiChoice(SubscriberMultiChoice):
    '''A multi-choice setting that displays objects
    
    This setting displays a list of objects taken from ObjectNameProviders.
    '''
    def __init__(self, text, value=None, *args, **kwargs):
        super(ObjectSubscriberMultiChoice, self).__init__(text, OBJECT_GROUP,
                                                          value, *args, **kwargs)
        
class DoSomething(Setting):
    """Do something in response to a button press
    """
    def __init__(self, text, label, callback, *args, **kwargs):
        super(DoSomething,self).__init__(text, 'n/a', **kwargs)
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
        
class RemoveSettingButton(DoSomething):
    '''A button whose only purpose is to remove something from a list.'''
    def __init__(self, text, label, list, entry):
        super(RemoveSettingButton, self).__init__(text, label, lambda: list.remove(entry))

class Divider(Setting):
    """The divider setting inserts a vertical space, possibly with a horizontal line, in the GUI"""
    def __init__(self, text = "", line=True, doc=None):
        super(Divider, self).__init__(text, 'n/a', doc=doc)
        self.line=line

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
    
    def construct_value(self, category, feature_name, image_name, object_name, 
                        scale):
        '''Construct a value that might represent a partially complete value'''
        if category is None:
            value='None'
        elif feature_name is None:
            value = category
        else:
            parts = [category, feature_name]
            if not image_name is None:
                parts.append(image_name)
            if object_name is not None:
                parts.append(object_name)
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
        result = [str(scale) for scale in scales]
        result.sort()
        return result
        
    def get_scale(self, pipeline):
        '''Return the currently chosen scale'''
        object_name = self.__object_fn()
        category = self.get_category(pipeline)
        feature_name = self.get_feature_name(pipeline)
        image_name = self.get_image_name(pipeline)
        sub_object_name = self.get_object_name(pipeline)
        if category is None or feature_name is None:
            return None
        if image_name is not None:
            head = '_'.join((category, feature_name, image_name))
        elif sub_object_name is not None:
            head = '_'.join((category, feature_name, sub_object_name))
        else:
            head = '_'.join((category, feature_name))
        for scale in self.get_scale_choices(pipeline):
            if self.value == '_'.join((head, scale)):
                return scale
        return None 
    
    def get_object_name_choices(self, pipeline):
        '''Return a list of objects for a particular feature
        
        Typically these are image features measured on the objects in the image
        '''
        object_name = self.__object_fn()
        category = self.get_category(pipeline)
        feature_name = self.get_feature_name(pipeline)
        objects = set()
        for module in pipeline.modules():
            if self.key in [x.key() for x in module.settings()]:
                break
            objects.update(module.get_measurement_objects(pipeline,
                                                          object_name,
                                                          category,
                                                          feature_name))
        result = list(objects)
        result.sort()
        return result
    
    def get_object_name(self, pipeline):
        '''Return the currently chosen image name'''
        object_name = self.__object_fn()
        category = self.get_category(pipeline)
        if category is None:
            return None
        feature_name = self.get_feature_name(pipeline)
        if feature_name is None:
            return None
        object_names = self.get_object_name_choices(pipeline)
        for object_name in object_names:
            head = '_'.join((category, feature_name, object_name))
            if (self.value.startswith(head+'_') or
                self.value == head):
                return object_name
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
        if (self.get_object_name(pipeline) is None and
            len(self.get_object_name_choices(pipeline))):
            raise ValidationError("%s has an unavailable object name" %
                                  self.value, self)
        if (self.get_scale(pipeline) not in self.get_scale_choices(pipeline)
            and len(self.get_scale_choices(pipeline)) > 0):
            raise ValidationError("%s has an unavailable scale" %
                                  self.value, self)

class Colormap(Choice):
    '''Represents the choice of a colormap'''
    def __init__(self, text, value=DEFAULT, *args, **kwargs):
        try:
            names = list(matplotlib.cm.cmapnames)
        except AttributeError:
            # matplotlib 99 does not have cmapnames
            names = ['Spectral', 'copper', 'RdYlGn', 'Set2', 'summer', 'spring', 'Accent', 'OrRd', 'RdBu', 'autumn', 'Set1', 'PuBu', 'Set3', 'gist_rainbow', 'pink', 'binary', 'winter', 'jet', 'BuPu', 'Dark2', 'prism', 'Oranges', 'gist_yarg', 'BuGn', 'hot', 'PiYG', 'YlOrBr', 'Reds', 'spectral', 'RdPu', 'Greens', 'gist_ncar', 'PRGn', 'gist_heat', 'YlGnBu', 'RdYlBu', 'Paired', 'flag', 'hsv', 'BrBG', 'Purples', 'cool', 'Pastel2', 'gray', 'Pastel1', 'gist_stern', 'GnBu', 'YlGn', 'Greys', 'RdGy', 'YlOrRd', 'PuOr', 'PuRd', 'gist_gray', 'Blues', 'PuBuGn', 'gist_earth', 'bone']
        names.sort()
        choices = [DEFAULT] + names
        super(Colormap,self).__init__(text, choices, value, *args, **kwargs)

class SettingsGroup(object):
    '''A group of settings that are managed together in the UI.
    Particulary useful when used with a RemoveSettingButton.
    Individual settings can be added with append(), and their value
    fetched from the group using the name given in append.
    '''

    def __init__(self):
        self.settings = []

    def append(self, name, setting):
        '''Add a new setting to the group, with a name.  The setting
        will then be available as group.name
        '''
        assert name not in self.__dict__, "%s already in SettingsGroup (previous setting or built in attribute)"%(name)
        self.__setattr__(name, setting)
        self.settings.append(setting)

    def unpack_group(self):
        '''Return a list of the settings in the group, in the order
        they were added to the group.
        '''
        return self.settings
        

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
