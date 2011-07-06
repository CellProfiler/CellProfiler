""" Setting.py - represents a module setting

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import matplotlib.cm
import numpy as np
import os
import sys
import re
import uuid
from cellprofiler.preferences import \
     DEFAULT_INPUT_FOLDER_NAME, DEFAULT_OUTPUT_FOLDER_NAME,\
     DEFAULT_INPUT_SUBFOLDER_NAME, DEFAULT_OUTPUT_SUBFOLDER_NAME, \
     ABSOLUTE_FOLDER_NAME, URL_FOLDER_NAME, NO_FOLDER_NAME,\
     get_default_image_directory, get_default_output_directory, \
     standardize_default_folder_names

from cellprofiler.utilities.utf16encode import utf16encode

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
NONE = 'None'

'''Names providers and subscribers of images'''
IMAGE_GROUP = 'imagegroup'

'''Names providers and subscribers of objects'''
OBJECT_GROUP = 'objectgroup'

'''Names providers and subscribers of grid information'''
GRID_GROUP = 'gridgroup'

'''Indicates that the image comes from a cropping operation'''
CROPPING_ATTRIBUTE = "cropping_image"
'''Indicates that the image was loaded from a file and has a file name and path'''
FILE_IMAGE_ATTRIBUTE = "file_image"
'''Indicates that the image is external (eg: from Java)'''
EXTERNAL_IMAGE_ATTRIBUTE = "external_image"
'''Indicates that the image is the result of an aggregate operation'''
AGGREGATE_IMAGE_ATTRIBUTE = "aggregate_image"
'''Indicates that the image is only available on the last cycle'''
AVAILABLE_ON_LAST_ATTRIBUTE = "available_on_last"
'''Indicates that the control can contain metadata tags'''
METADATA_ATTRIBUTE = "metadata"

class Setting(object):
    """A module setting which holds a single string value
    
    """
    #
    # This should be set to False for UI elements like buttons and dividers
    #
    save_to_pipeline = True
    def __init__(self, text, value, doc=None, reset_view = False):
        """Initialize a setting with the enclosing module and its string value
        
        module - the module containing this setting
        text   - the explanatory text for the setting
        value  - the default or initial value for the setting
        doc - documentation for the setting
        reset_view - True if miniscule editing should re-evaluate the module view
        """
        self.__text = text
        self.__value = value
        self.doc = doc
        self.__key = uuid.uuid1() 
        self.reset_view = reset_view
    
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
    
    def get_value_text(self):
        '''Get the underlying string value'''
        return self.__value
    
    def set_value_text(self, value):
        '''Set the underlying string value'''
        self.__value = value
        
    value_text = property(get_value_text, set_value_text)
    
    def __eq__(self, x):
        # we test explicitly for other Settings to prevent matching if
        # their .values are the same.
        if isinstance(x, Setting):
            return self .__key == x.__key
        return self.value == unicode(x)
    
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
        '''Return value as a string. 
        
        NOTE: strings are deprecated, use unicode_value instead.
        '''
        if isinstance(self.__value, unicode):
            return str(utf16encode(self.__value))
        if not isinstance(self.__value,str):
            raise ValidationError("%s was not a string"%(self.__value),self)
        return self.__value
    
    @property
    def unicode_value(self):
        return self.get_unicode_value()
    
    def get_unicode_value(self):
        return unicode(self.value_text)
    
class HiddenCount(Setting):
    """A setting meant only for saving an item count
    
    The HiddenCount setting should never be in the visible settings.
    It should be tied to a sequence variable which gives the number of
    items which is the value for this variable.
    """
    def __init__(self, sequence, text = "Hidden"):
        super(HiddenCount,self).__init__(text, str(len(sequence)))
        self.__sequence = sequence

    def set_value(self, value):
        if not value.isdigit():
            raise ValueError("The value must be an integer")
        count = int(value)
        if count == len(self.__sequence):
            # The value was "inadvertantly" set, but is correct
            return
        raise NotImplementedError("The count should be inferred, not set  - actual: %d, set: %d"%(len(self.__sequence), count))

    def get_value(self):
        return len(self.__sequence)

    def set_sequence(self, sequence):
        '''Set the sequence used to maintain the count'''
        self.__sequence = sequence
    
    def __str__(self):
        return str(len(self.__sequence))
    
    def get_unicode_value(self):
        return unicode(len(self.__sequence))

class Text(Setting):
    """A setting that displays as an edit box, accepting a string
    
    """
    def __init__(self, text, value, *args, **kwargs):
        kwargs = kwargs.copy()
        self.multiline_display = kwargs.pop("multiline", False)
        self.metadata_display = kwargs.pop(METADATA_ATTRIBUTE, False)
        super(Text,self).__init__(text, value, *args, **kwargs)

class RegexpText(Setting):
    """A setting with a regexp button on the side
    """
    def __init__(self, text, value, *args, **kwargs):
        kwargs = kwargs.copy()
        self.get_example_fn = kwargs.pop("get_example_fn",None)
        super(RegexpText,self).__init__(text, value, *args, **kwargs)

    def test_valid(self, pipeline):
        try:
            # Convert Matlab to Python
            pattern = re.sub('(\\(\\?)([<].+?[>])','\\1P\\2',self.value)
            re.search('(|(%s))'%(pattern), '')
        except re.error, v:
            raise ValidationError("Invalid regexp: %s"%(v), self)

class DirectoryPath(Text):
    """A setting that displays a filesystem path name
    """
    DIR_ALL = [DEFAULT_INPUT_FOLDER_NAME, DEFAULT_OUTPUT_FOLDER_NAME,
               ABSOLUTE_FOLDER_NAME, DEFAULT_INPUT_SUBFOLDER_NAME,
               DEFAULT_OUTPUT_SUBFOLDER_NAME]
    def __init__(self, text, value = None, dir_choices = None, 
                 allow_metadata = True, support_urls = False,
                 *args, **kwargs):
        if dir_choices is None:
            dir_choices = DirectoryPath.DIR_ALL
        if support_urls and not (URL_FOLDER_NAME in dir_choices):
            dir_choices = dir_choices + [URL_FOLDER_NAME]
        if value is None:
            value = DirectoryPath.static_join_string(
                dir_choices[0], "None")
        self.dir_choices = dir_choices
        self.allow_metadata = allow_metadata
        super(DirectoryPath,self).__init__(text, value, *args, **kwargs)
        
    def split_parts(self):
        '''Return the directory choice and custom path as a tuple'''
        return tuple(self.value.split('|',1))
    
    @staticmethod
    def split_string(value):
        return tuple(value.split('|',1))
    
    def join_parts(self, dir_choice = None, custom_path = None):
        '''Join the directory choice and custom path to form a value'''
        self.value = self.join_string(dir_choice, custom_path)
        
    def join_string(self, dir_choice = None, custom_path = None):
        '''Return the value string composed of a directory choice & path'''
        return self.static_join_string(
            dir_choice if dir_choice is not None 
            else self.dir_choice, 
            custom_path if custom_path is not None
            else self.custom_path)
    
    @staticmethod
    def static_join_string(dir_choice, custom_path):
        return '|'.join((dir_choice, custom_path))
    
    @staticmethod
    def upgrade_setting(value):
        dir_choice, custom_path = DirectoryPath.split_string(value)
        dir_choice = standardize_default_folder_names([dir_choice], 0)[0]
        return DirectoryPath.static_join_string(dir_choice, custom_path)
        
    def get_dir_choice(self):
        '''The directory selection method'''
        return self.split_parts()[0]
    
    def set_dir_choice(self, choice):
        self.join_parts(dir_choice = choice)
        
    dir_choice = property(get_dir_choice, set_dir_choice)
    
    def get_custom_path(self):
        '''The custom path relative to the directory selection method'''
        return self.split_parts()[1]
    
    def set_custom_path(self, custom_path):
        self.join_parts(custom_path = custom_path)
        
    custom_path = property(get_custom_path, set_custom_path)
    
    @property
    def is_custom_choice(self):
        '''True if the current dir_choice requires a custom path'''
        return self.dir_choice in [
            ABSOLUTE_FOLDER_NAME, DEFAULT_INPUT_SUBFOLDER_NAME,
            DEFAULT_OUTPUT_SUBFOLDER_NAME, URL_FOLDER_NAME]
    
    def get_absolute_path(self, measurements=None, image_set_number = None):
        '''Return the absolute path specified by the setting
        
        Concoct an absolute path based on the directory choice,
        the custom path and metadata taken from the measurements.
        '''
        if self.dir_choice == DEFAULT_INPUT_FOLDER_NAME:
            return get_default_image_directory()
        if self.dir_choice == DEFAULT_OUTPUT_FOLDER_NAME:
            return get_default_output_directory()
        if self.dir_choice == DEFAULT_INPUT_SUBFOLDER_NAME:
            root_directory = get_default_image_directory()
        elif self.dir_choice == DEFAULT_OUTPUT_SUBFOLDER_NAME:
            root_directory = get_default_output_directory()
        elif self.dir_choice == ABSOLUTE_FOLDER_NAME:
            root_directory = os.curdir
        elif self.dir_choice == URL_FOLDER_NAME:
            root_directory = ''
        elif self.dir_choice == NO_FOLDER_NAME:
            return ''
        else:
            raise ValueError("Unknown directory choice: %s" % self.dir_choice)
        if self.allow_metadata:
            if measurements is not None:
                custom_path = measurements.apply_metadata(self.custom_path,
                                                          image_set_number)
            else:
                # For UI, get the path up to the metadata.
                custom_path = self.custom_path
                md_start = custom_path.find("\\g<")
                if md_start != -1:
                    custom_path = custom_path[:md_start]
                    custom_path = os.path.split(custom_path)[0]
        else:
            custom_path = self.custom_path
        if self.dir_choice == URL_FOLDER_NAME:
            return custom_path
        path = os.path.join(root_directory, custom_path)
        return os.path.abspath(path)
    
    def get_parts_from_path(self, path):
        '''Figure out how to set up dir_choice and custom path given a path'''
        path = os.path.abspath(path)
        custom_path = self.custom_path
        img_dir = get_default_image_directory()
        out_dir = get_default_output_directory()
        if sys.platform.startswith("win"):
            # set to lower-case for comparisons
            cmp_path = path.lower()
            img_dir = img_dir.lower()
            out_dir = out_dir.lower()
        else:
            cmp_path = path
        seps = [os.path.sep]
        if hasattr(os, 'altsep'):
            seps += [os.altsep]
        if cmp_path == img_dir:
            dir_choice = DEFAULT_INPUT_FOLDER_NAME
        elif cmp_path == out_dir:
            dir_choice = DEFAULT_OUTPUT_FOLDER_NAME
        elif (cmp_path.startswith(img_dir) and 
              cmp_path[len(img_dir)] in seps):
            dir_choice = DEFAULT_INPUT_SUBFOLDER_NAME
            custom_path = path[len(img_dir)+1:]
        elif (cmp_path.startswith(out_dir) and 
              cmp_path[len(out_dir)] in seps):
            dir_choice = DEFAULT_OUTPUT_SUBFOLDER_NAME
            custom_path = path[len(out_dir)+1:]
        else:
            dir_choice = ABSOLUTE_FOLDER_NAME
            custom_path = path
        return dir_choice, custom_path
    
    def alter_for_create_batch_files(self, fn_alter_path):
        '''Call this to alter the setting appropriately for batch execution'''
        custom_path = self.custom_path
        regexp_substitution =custom_path.find(r"\g<") != -1
        if custom_path.startswith("\g<") and sys.platform.startswith("win"):
            # So ugly, the "\" sets us up for the root directory during
            # os.path.join, so we need r".\\" at start to fake everyone out
            custom_path = r".\\" + custom_path
    
        if self.dir_choice == DEFAULT_INPUT_FOLDER_NAME:
            self.dir_choice = ABSOLUTE_FOLDER_NAME
            self.custom_path = fn_alter_path(get_default_image_directory())
        elif self.dir_choice == DEFAULT_OUTPUT_FOLDER_NAME:
            self.dir_choice = ABSOLUTE_FOLDER_NAME
            self.custom_path = fn_alter_path(get_default_output_directory())
        elif self.dir_choice == ABSOLUTE_FOLDER_NAME:
            self.custom_path = fn_alter_path(
                self.custom_path, regexp_substitution = regexp_substitution)
        elif self.dir_choice == DEFAULT_INPUT_SUBFOLDER_NAME:
            self.dir_choice = ABSOLUTE_FOLDER_NAME
            self.custom_path = fn_alter_path(
                os.path.join(get_default_image_directory(), custom_path),
                regexp_substitution=regexp_substitution)
        elif self.dir_choice == DEFAULT_OUTPUT_SUBFOLDER_NAME:
            self.dir_choice = ABSOLUTE_FOLDER_NAME
            self.custom_path = fn_alter_path(
                os.path.join(get_default_output_directory(), custom_path), 
                regexp_substitution = regexp_substitution)
        
    def test_valid(self, pipeline):
        if self.dir_choice not in self.dir_choices + [NO_FOLDER_NAME]:
            raise ValidationError("Unsupported directory choice: %s" %
                                  self.dir_choice, self)
        if (not self.allow_metadata and self.is_custom_choice and
            self.custom_path.find(r"\g<") != -1):
            raise ValidationError("Metadata not supported for this setting",
                                  self)

class FilenameText(Text):
    """A setting that displays a file name
    
    optional arguments - 
       get_directory_fn is a function that gets the initial directory
           for the browse button
       set_directory_fn is a function that sets the directory after browsing
       browse_msg - message at top of file browser
       exts - a list of tuples where the first is the user-displayed text
       and the second is the file filter for an extension, like 
       [("Pipeline (*.cp)","*.cp")]
    """
    def __init__(self, text, value, *args, **kwargs):
        kwargs = kwargs.copy()
        self.get_directory_fn = kwargs.pop("get_directory_fn", None)
        self.set_directory_fn = kwargs.pop("set_directory_fn", None)
        self.browse_msg = kwargs.pop("browse_msg", "Choose a file")
        self.exts = kwargs.pop("exts", None)
        super(FilenameText,self).__init__(text, value, *args, **kwargs)
        self.browsable = True
 
    def set_browsable(self, val):
        self.browsable = val

class ImageFileSpecifier(Text):
    """A setting for choosing an image file, including switching between substring, file globbing, and regular expressions,
    and choosing different directories (or common defaults).
    """
    def __init__(self, text, value, *args, **kwargs):
        if 'regexp' in kwargs:
            self.regexp = kwargs['regexp']
            del kwargs['regexp']
        if 'default_dir' in kwargs:
            self.default_dir = kwargs['default_dir']
            del kwargs['default_dir']
        super(ImageFileSpecifier,self).__init__(text, value, *args, **kwargs)

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
        super(Integer,self).__init__(text, unicode(value), *args, **kwargs)
        self.__default = int(value)
        self.__minval = minval
        self.__maxval = maxval
    
    def set_value(self,value):
        """Convert integer to string
        """
        str_value = unicode(value)
        super(Integer,self).set_value(str_value)
        
    def get_value(self):
        """Return the value of the setting as an integer
        """
        try:
            self.__default = int(super(Integer,self).get_value())
        except ValueError:
            pass
        return self.__default
    
    def test_valid(self,pipeline):
        """Return true only if the text value is an integer
        """
        try:
            int(unicode(self))
        except ValueError:
            raise ValidationError('Must be an integer value, was "%s"'%(self.value_text),self)
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
        values = self.value_text.split(',')
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
        values = self.value_text.split(',')
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
        values = self.value_text.split(',')
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
        values = self.value_text.split(',')
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
        values = self.value_text.split(',')
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
            return END
        elif self.max is not None:
            return str(abs(self.max))
        else:
            return self.value_text.split(',')[1]
    display_max = property(get_display_max)
    
    def test_valid(self, pipeline):
        values = self.value_text.split(',')
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
        super(Float,self).__init__(text, unicode(value), *args, **kwargs)
        self.__default = float(value)
        self.__minval = minval
        self.__maxval = maxval
    
    def set_value(self,value):
        """Convert integer to string
        """
        str_value = unicode(value)
        super(Float,self).set_value(str_value)
        
    def get_value(self, reraise=False):
        """Return the value of the setting as a float
        """
        try:
            str_value = super(Float,self).get_value()
            if str_value.endswith("%"):
                self.__default = float(str_value[:-1]) / 100.0
            else:
                self.__default = float(str_value)
        except ValueError:
            if reraise:
                raise
            pass
        return self.__default
    
    def test_valid(self,pipeline):
        """Return true only if the text value is float
        """
        try:
            # Raises value error inside self.value if not a float
            value = self.get_value(reraise=True)
        except ValueError:
            raise ValidationError('Value not in decimal format', self)
        if self.__minval != None and self.__minval > value:
            raise ValidationError('Must be at least %d, was %d'%(self.__minval, self.value),self)
        if self.__maxval != None and self.__maxval < value:
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
        values = self.value_text.split(',')
        return (float(values[0]),float(values[1]))
    
    def get_min(self):
        """The minimum value of the range"""
        return float(self.value_text.split(',')[0])
    
    def set_min(self, value):
        self.set_value_text("%s,%s" % (value, self.max))
        
    min = property(get_min, set_min)
    
    def get_max(self):
        """The maximum value of the range"""
        return float(self.value_text.split(',')[1])
    
    def set_max(self, value):
        self.set_value("%s,%s" % (self.min, value))
        
    max = property(get_max, set_max)
    
    def test_valid(self, pipeline):
        values = self.value_text.split(',')
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

PROVIDED_ATTRIBUTES = "provided_attributes"
class NameProvider(Text):
    """A setting that provides a named object
    """
    def __init__(self, text, group, value=DO_NOT_USE, *args, **kwargs):
        self.__provided_attributes = { "group":group }
        if kwargs.has_key("provided_attributes"):
            self.__provided_attributes.update(kwargs["provided_attributes"])
            kwargs = kwargs.copy()
            del kwargs[PROVIDED_ATTRIBUTES]
        super(NameProvider,self).__init__(text, value, *args, **kwargs)
    
    def get_group(self):
        """This setting provides a name to this group
        
        Returns a group name, e.g. imagegroup or objectgroup
        """
        return self.__provided_attributes["group"]
    
    group = property(get_group)
    
    @property
    def provided_attributes(self):
        '''Return the dictionary of attributes of this provider
        
        These are things like the group ("objectgroup" for instance) and
        hints about the thing itself, such as that it is an image
        that was loaded from  a file.
        '''
        return self.__provided_attributes

class ImageNameProvider(NameProvider):
    """A setting that provides an image name
    """
    def __init__(self, text, value=DO_NOT_USE, *args, **kwargs):
        super(ImageNameProvider,self).__init__(text, IMAGE_GROUP, value,
                                               *args, **kwargs)

class FileImageNameProvider(ImageNameProvider):
    """A setting that provides an image name where the image has an associated file"""
    def __init__(self, text, value=DO_NOT_USE, *args, **kwargs):
        kwargs = kwargs.copy()
        if not kwargs.has_key(PROVIDED_ATTRIBUTES):
            kwargs[PROVIDED_ATTRIBUTES] = {}
        kwargs[PROVIDED_ATTRIBUTES][FILE_IMAGE_ATTRIBUTE] = True
        super(FileImageNameProvider,self).__init__(text, value, *args,
                                                   **kwargs)
        
class ExternalImageNameProvider(ImageNameProvider):
    """A setting that provides an image name where the image is loaded 
    externally. (eg: from Java)"""
    def __init__(self, text, value=DO_NOT_USE, *args, **kwargs):
        kwargs = kwargs.copy()
        if not kwargs.has_key(PROVIDED_ATTRIBUTES):
            kwargs[PROVIDED_ATTRIBUTES] = {}
        kwargs[PROVIDED_ATTRIBUTES][EXTERNAL_IMAGE_ATTRIBUTE] = True
        super(ExternalImageNameProvider,self).__init__(text, value, *args,
                                                   **kwargs)

class CroppingNameProvider(ImageNameProvider):
    """A setting that provides an image name where the image has a cropping mask"""
    def __init__(self, text, value=DO_NOT_USE, *args, **kwargs):
        kwargs = kwargs.copy()
        if not kwargs.has_key(PROVIDED_ATTRIBUTES):
            kwargs[PROVIDED_ATTRIBUTES] = {}
        kwargs[PROVIDED_ATTRIBUTES][CROPPING_ATTRIBUTE] = True
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
        super(OutlineNameProvider,self).__init__(text, value, 
                                                 *args, **kwargs)

class GridNameProvider(NameProvider):
    """A setting that provides a GridInfo object
    """
    def __init__(self, text, value="Grid", *args, **kwargs):
        super(GridNameProvider, self).__init__(text, GRID_GROUP, value, 
                                               *args, **kwargs)

REQUIRED_ATTRIBUTES = "required_attributes"
class NameSubscriber(Setting):
    """A setting that takes its value from one made available by name providers
    """
    def __init__(self, text, group, value=None,
                 can_be_blank=False, blank_text=LEAVE_BLANK, *args, **kwargs):
        if value==None:
            value = (can_be_blank and blank_text) or "None"
        self.__required_attributes = { "group":group }
        if kwargs.has_key(REQUIRED_ATTRIBUTES):
            self.__required_attributes.update(kwargs[REQUIRED_ATTRIBUTES])
            kwargs = kwargs.copy()
            del kwargs[REQUIRED_ATTRIBUTES]
        self.__can_be_blank = can_be_blank
        self.__blank_text = blank_text
        super(NameSubscriber,self).__init__(text, value, *args, **kwargs)
    
    
    def get_group(self):
        """This setting provides a name to this group
        
        Returns a group name, e.g. imagegroup or objectgroup
        """
        return self.__required_attributes["group"]
    
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
        return all([setting.provided_attributes.get(key, None) ==
                    self.__required_attributes[key]
                    for key in self.__required_attributes.keys()])
    
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

def get_name_providers(pipeline, last_setting):
    '''Scan the pipeline to find name providers matching the name given in the setting
    
    pipeline - pipeline to scan
    last_setting - scan the modules in order until you arrive at this setting
    returns a list of providers that provide a correct "thing" with the
    same name as that of the subscriber
    '''
    choices = []
    for module in pipeline.modules():
        module_choices = []
        for setting in module.visible_settings():
            if setting.key() == last_setting.key():
                return choices
            if (isinstance(setting, NameProvider) and 
                setting != DO_NOT_USE and
                last_setting.matches(setting) and
                setting.value == last_setting.value):
                module_choices.append(setting)
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
        kwargs = kwargs.copy()
        if not kwargs.has_key(REQUIRED_ATTRIBUTES):
            kwargs[REQUIRED_ATTRIBUTES] = {}
        kwargs[REQUIRED_ATTRIBUTES][FILE_IMAGE_ATTRIBUTE] = True
        super(FileImageNameSubscriber,self).__init__(text, value, can_be_blank,
                                                     blank_text, *args,
                                                     **kwargs)
    
class CroppingNameSubscriber(ImageNameSubscriber):
    """A setting that provides image names that have cropping masks"""
    def __init__(self, text, value=DO_NOT_USE, can_be_blank=False,
                 blank_text=LEAVE_BLANK, *args, **kwargs):
        kwargs = kwargs.copy()
        if not kwargs.has_key(REQUIRED_ATTRIBUTES):
            kwargs[REQUIRED_ATTRIBUTES] = {}
        kwargs[REQUIRED_ATTRIBUTES][CROPPING_ATTRIBUTE] = True
        super(CroppingNameSubscriber,self).__init__(text, value, can_be_blank,
                                                    blank_text, *args, 
                                                    **kwargs)

class ExternalImageNameSubscriber(ImageNameSubscriber):
    """A setting that provides image names loaded externally (eg: from Java)"""
    def __init__(self, text, value=DO_NOT_USE, can_be_blank = False,
                 blank_text=LEAVE_BLANK, *args, **kwargs):
        super(ExternalImageNameSubscriber,self).__init__(text, value, can_be_blank,
                                                     blank_text, *args,
                                                     **kwargs)

class ObjectNameSubscriber(NameSubscriber):
    """A setting that subscribes to the list of available object names
    """
    def __init__(self, text, value=DO_NOT_USE, can_be_blank=False,
                 blank_text=LEAVE_BLANK, *args, **kwargs):
        super(ObjectNameSubscriber,self).__init__(text, OBJECT_GROUP, value,
                                                  can_be_blank, blank_text,
                                                  *args, **kwargs)

class OutlineNameSubscriber(ImageNameSubscriber):
    '''A setting that subscribes to the list of available object outline names
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
    """A setting that subscribes to a figure indicator provider
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

class GridNameSubscriber(NameSubscriber):
    """A setting that subscribes to grid information providers
    """
    def __init__(self, text, value=DO_NOT_USE, can_be_blank=False,
                 blank_text=LEAVE_BLANK, *args, **kwargs):
        super(GridNameSubscriber,self).__init__(text, GRID_GROUP, value,
                                                  can_be_blank, blank_text,
                                                  *args, **kwargs)
        
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
    def __init__(self, text, choices, value=None, tooltips=None, 
                 choices_fn=None, *args, **kwargs):
        """Initializer
        module - the module containing the setting
        text - the explanatory text for the setting
        choices - a sequence of string choices to be displayed in the drop-down
        value - the default choice or None to choose the first of the choices.
        tooltips - a dictionary of choice to tooltip
        choices_fn - a function that, if present, supplies the choices. The
                     function should have the signature, fn(pipeline).
        """
        super(Choice,self).__init__(text, value or choices[0], *args, **kwargs)
        self.__choices = choices
        self.__tooltips = tooltips
        self.__choices_fn = choices_fn
    
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
        if self.__choices_fn is not None:
            self.__choices = self.__choices_fn(pipeline)
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
                if len(self.choices) == 0:
                    raise ValidationError("No available choices", self)
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
        self.__required_attributes = { "group":group }
        if kwargs.has_key(REQUIRED_ATTRIBUTES):
            self.__required_attributes.update(kwargs[REQUIRED_ATTRIBUTES])
            kwargs = kwargs.copy()
            del kwargs[REQUIRED_ATTRIBUTES]
        super(SubscriberMultiChoice,self).__init__(text, [], value,
                                                   *args, **kwargs)
    
    def load_choices(self, pipeline):
        '''Get the choice list from name providers'''
        self.choices = get_name_provider_choices(pipeline, self, self.group)
    
    @property
    def group(self):
        return self.__required_attributes["group"]
    
    def matches(self, provider):
        '''Return true if the provider is compatible with this subscriber
        
        This method can be used to be more particular about the providers
        that are selected. For instance, if you want a list of only
        FileImageNameProviders (images loaded from files), you can
        check that here.
        '''
        return all([provider.provided_attributes.get(key, None) ==
                    self.__required_attributes[key]
                    for key in self.__required_attributes])
    
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

class ImageNameSubscriberMultiChoice(SubscriberMultiChoice):
    '''A multi-choice setting that displays images
    
    This setting displays a list of images taken from ImageNameProviders.
    '''
    def __init__(self, text, value=None, *args, **kwargs):
        super(ImageNameSubscriberMultiChoice, self).__init__(text, IMAGE_GROUP,
                                                          value, *args, **kwargs)
        
class MeasurementMultiChoice(MultiChoice):
    '''A multi-choice setting for selecting multiple measurements'''
    def __init__(self, text, value='', *args, **kwargs):
        '''Initialize the measurement multi-choice
        
        At initialization, the choices are empty because the measurements
        can't be fetched here. It's done (bit of a hack) in test_valid.
        '''
        super(MeasurementMultiChoice, self).__init__(text, [], value, *args, **kwargs)
    
    def encode_object_name(self, object_name):
        '''Encode object name, escaping |'''
        return object_name.replace('|','||')
    
    def decode_object_name(self, object_name):
        '''Decode the escaped object name'''
        return object_name.replace('||','|')
    
    def split_choice(self, choice):
        '''Split object and feature within a choice'''
        subst_choice = choice.replace('||','++')
        loc = subst_choice.find('|')
        assert loc != -1
        return (choice[:loc], choice[(loc+1):])
    
    def get_measurement_object(self, choice):
        return self.decode_object_name(self.split_choice(choice)[0])
    
    def get_measurement_feature(self, choice):
        return self.split_choice(choice)[1]
    
    def make_measurement_choice(self, object_name, feature):
        return self.encode_object_name(object_name) + "|" + feature
    
    @staticmethod
    def get_value_string(choices):
        '''Return the string value representing the choices made
        
        choices - a collection of choices as returned by make_measurement_choice
        '''
        return ','.join(choices)
    
    def test_valid(self, pipeline):
        '''Get the choices here and call the superclass validator'''
        #
        # Find our module
        #
        for module in pipeline.modules():
            for setting in module.visible_settings():
                if id(setting) == id(self):
                    break
        columns = pipeline.get_measurement_columns(module)
        self.set_choices([self.make_measurement_choice(c[0], c[1])
                          for c in columns])
        super(MeasurementMultiChoice, self).test_valid(pipeline)
        
class SubdirectoryFilter(MultiChoice):
    '''A setting that indicates which subdirectories should be excluded from an operation
    
    The subdirectory filter holds a collection of subdirectories that
    should be excluded from a file discovery operation that scans
    subdirectories.
    '''
    def __init__(self, text, value='', directory_path=None, **kwargs):
        '''Initialize the setting
        
        text - a tag for the setting that briefly indicates its purpose
        
        value - the value for the setting, as saved in the pipeline
        
        directory_path - an optional DirectoryPath setting that can be used
                         to find the root of the subdirectory tree.
        '''
        super(SubdirectoryFilter, self).__init__(text, value, **kwargs)
        assert (directory_path is None) or isinstance(directory_path, DirectoryPath)
        self.directory_path = directory_path

    @staticmethod
    def get_value_string(choices):
        '''Return the string value representing the choices made
        
        choices - a collection of choices as returned by make_measurement_choice
        '''
        return ','.join(choices)
        
    def alter_for_create_batch_files(self, fn_alter_path):
        selections = [fn_alter_path(selection) 
                      for selection in self.get_selections()]
        self.value = self.get_value_string(selections)
        
    def test_valid(self, pipeline):
        if self.directory_path is not None:
            root = self.directory_path.get_absolute_path()
            for subdirectory in self.get_selections():
                path = os.path.join(root, subdirectory)
                if not os.path.isdir(path):
                    raise ValidationError("%s is not a valid directory" % path,
                                          self)
        
class DoSomething(Setting):
    """Do something in response to a button press
    """
    save_to_pipeline = False
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
    def __init__(self, text, label, list, entry, **kwargs):
        super(RemoveSettingButton, self).__init__(text, label, 
                                                  lambda: list.remove(entry),
                                                  **kwargs)

class Divider(Setting):
    """The divider setting inserts a vertical space, possibly with a horizontal line, in the GUI"""
    save_to_pipeline = False
    def __init__(self, text = "", line=True, doc=None):
        super(Divider, self).__init__(text, 'n/a', doc=doc)
        self.line = line and (text == "")

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
    
    def construct_value(self, category, feature_name, object_or_image_name, 
                        scale):
        '''Construct a value that might represent a partially complete value'''
        if category is None:
            value='None'
        elif feature_name is None:
            value = category
        else:
            parts = [category, feature_name]
            if object_or_image_name is not None:
                parts.append(object_or_image_name)
            if not scale is None:
                parts.append(scale)
            value = '_'.join(parts)
        return str(value)
        
    def get_category_choices(self, pipeline, object_name=None):
        '''Find the categories of measurements available from the object '''
        if object_name is None:
            object_name = self.__object_fn()
        categories = set()
        for module in pipeline.modules():
            if self.key() in [x.key() for x in module.settings()]:
                break
            categories.update(module.get_categories(pipeline, object_name))
        result = list(categories)
        result.sort()
        return result
    
    def get_category(self, pipeline, object_name = None):
        '''Return the currently chosen category'''
        categories = self.get_category_choices(pipeline, object_name)
        for category in categories:
            if (self.value.startswith(category+'_') or
                self.value == category):
                return category
        return None
    
    def get_feature_name_choices(self, pipeline,
                                 object_name = None,
                                 category = None):
        '''Find the feature name choices available for the chosen category'''
        if object_name is None:
            object_name = self.__object_fn()
        if category is None:
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
    
    def get_feature_name(self, pipeline,
                         object_name = None,
                         category = None):
        '''Return the currently selected feature name'''
        if category is None:
            category = self.get_category(pipeline, object_name)
        if category is None:
            return None
        feature_names = self.get_feature_name_choices(pipeline, 
                                                      object_name,
                                                      category)
        for feature_name in feature_names:
            head = '_'.join((category, feature_name))
            if (self.value.startswith(head+'_') or
                self.value == head):
                return feature_name
        return None
    
    def get_image_name_choices(self, pipeline, 
                               object_name = None,
                               category = None,
                               feature_name = None):
        '''Find the secondary image name choices available for a feature
        
        A measurement can still be valid, even if there are no available
        image name choices. The UI should not offer image name choices
        if no choices are returned.
        '''
        if object_name is None:
            object_name = self.__object_fn()
        if category is None:
            category = self.get_category(pipeline, object_name)
        if feature_name is None:
            feature_name = self.get_feature_name(pipeline, object_name, 
                                                 category)
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
    
    def get_image_name(self, pipeline,
                       object_name = None,
                       category = None,
                       feature_name = None):
        '''Return the currently chosen image name'''
        if object_name is None:
            object_name = self.__object_fn()
        if category is None:
            category = self.get_category(pipeline, object_name)
        if category is None:
            return None
        if feature_name is None:
            feature_name = self.get_feature_name(pipeline, object_name,
                                                 category)
        if feature_name is None:
            return None
        image_names = self.get_image_name_choices(pipeline,
                                                  object_name, category,
                                                  feature_name)
        for image_name in image_names:
            head = '_'.join((category, feature_name, image_name))
            if (self.value.startswith(head+'_') or
                self.value == head):
                return image_name
        return None
    
    def get_scale_choices(self, pipeline, 
                          object_name = None,
                          category = None,
                          feature_name = None,
                          image_name = None):
        '''Return the measured scales for the currently chosen measurement
        
        The setting may still be valid, even though there are no scale choices.
        In this case, the UI should not offer the user a scale choice.
        '''
        if object_name is None:
            object_name = self.__object_fn()
        if category is None:
            category = self.get_category(pipeline, object_name)
        if feature_name is None:
            feature_name = self.get_feature_name(pipeline, object_name,
                                                 category)
        if image_name is None:
            image_name = self.get_image_name(pipeline, object_name, category,
                                             feature_name)
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
        
    def get_scale(self, pipeline, 
                  object_name = None,
                  category = None,
                  feature_name = None,
                  image_name = None):
        '''Return the currently chosen scale'''
        if object_name is None:
            object_name = self.__object_fn()
        if category is None:
            category = self.get_category(pipeline, object_name)
        if feature_name is None:
            feature_name = self.get_feature_name(pipeline, object_name,
                                                 category)
        if image_name is None:
            image_name = self.get_image_name(pipeline, object_name,
                                             category, feature_name)
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
    
    def get_object_name_choices(self, pipeline, 
                                object_name = None,
                                category = None,
                                feature_name = None):
        '''Return a list of objects for a particular feature
        
        Typically these are image features measured on the objects in the image
        '''
        if object_name is None:
            object_name = self.__object_fn()
        if category is None:
            category = self.get_category(pipeline, object_name)
        if feature_name is None:
            feature_name = self.get_feature_name(pipeline, object_name, category)
        if any([x is None for x in (object_name, category, feature_name)]):
            return []
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
    
    def get_object_name(self, pipeline, object_name = None,
                        category = None,
                        feature_name = None):
        '''Return the currently chosen image name'''
        if object_name is None:
            object_name = self.__object_fn()
        if category is None:
            category = self.get_category(pipeline, object_name)
        if category is None:
            return None
        if feature_name is None:
            feature_name = self.get_feature_name(pipeline, object_name, 
                                                 category)
        if feature_name is None:
            return None
        object_names = self.get_object_name_choices(pipeline, object_name,
                                                    category, feature_name)
        for object_name in object_names:
            head = '_'.join((category, feature_name, object_name))
            if (self.value.startswith(head+'_') or
                self.value == head):
                return object_name
        return None
        
        
    def test_valid(self, pipeline):
        obname = self.__object_fn()
        category = self.get_category(pipeline, obname)
        if category is None:
            raise ValidationError("%s has an unavailable measurement category" %
                                  self.value, self)
        feature_name = self.get_feature_name(pipeline, obname, category)
        if feature_name is None:
            raise ValidationError("%s has an unmeasured feature name" %
                                  self.value, self)
        #
        # If there are any image names or object names, then there must
        # be a valid image name or object name
        #
        image_name = self.get_image_name(pipeline, obname, category, 
                                         feature_name) 
        image_names = self.get_image_name_choices(pipeline, obname, 
                                                  category, feature_name)
        sub_object_name = self.get_object_name(pipeline, obname, category, 
                                               feature_name) 
        sub_object_names = self.get_object_name_choices(pipeline, obname, 
                                                        category, feature_name)
        if (len(sub_object_names) > 0  and image_name is None and 
            sub_object_name is None):
            raise ValidationError("%s has an unavailable object name" %
                                  self.value, self)
        if (len(image_names) > 0 and image_name is None and 
            sub_object_name is None):
            raise ValidationError("%s has an unavailable image name" %
                                  self.value, self)
        scale_choices = self.get_scale_choices(pipeline, obname, category,
                                               feature_name)
        if (self.get_scale(pipeline, obname, category, feature_name) 
            not in scale_choices and len(scale_choices) > 0):
            raise ValidationError("%s has an unavailable scale" %
                                  self.value, self)
        for module in pipeline.modules():
            if self.key() in [s.key() for s in module.visible_settings()]:
                break
        if (not any([column[0] == obname and column[1] == self.value
                     for column in pipeline.get_measurement_columns(module)])):
            raise ValidationError("%s is not measured for %s"%
                                  (self.value, obname), self)

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
        
class Color(Setting):
    '''Represents a choice of color
    
    These are coded in hex unless a valid HTML name is available.
    '''
    def __init(self, text, value="gray", *args, **kwargs):
        super(Color, self).__init(text, value, *args, **kwargs)
        

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
        
    def visible_settings(self):
        '''Return a list of the settings in the group, in the order
        they were added to the group.
        '''
        # return a copy
        return list(self.settings)
    
    def pipeline_settings(self):
        '''Return a list of the settings, filtering out UI tidbits'''
        return [setting for setting in self.settings
                if setting.save_to_pipeline]
        
        
class NumberConnector(object):
    '''This object connects a function to a number slot
    
    You can use this if you have a value that changes contextually
    depending on other settings. You pass in a function that, when evaluated,
    gives the current value for the number. You can then pass in a number
    connector instead of an explicit value for things like minima and maxima
    for numeric settings.
    '''
    def __init__(self, fn):
        self.__fn = fn
        
    def __int__(self):
        return int(self.__fn())
    
    def __long__(self):
        return long(self.__fn())
    
    def __float__(self):
        return float(self.__fn())
    
    def __cmp__(self, other):
        return cmp(self.__fn(), other)
    
    def __hash__(self):
        return self.__fn().__hash__()

    def __str__(self):
        return str(self.__fn())


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
