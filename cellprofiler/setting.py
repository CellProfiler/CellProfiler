""" Setting.py - represents a module setting
"""

import logging

logger = logging.getLogger(__name__)
import json
import matplotlib.cm
import numpy as np
import os
import sys
import re
import uuid

from cellprofiler.preferences import \
    DEFAULT_INPUT_FOLDER_NAME, DEFAULT_OUTPUT_FOLDER_NAME, \
    DEFAULT_INPUT_SUBFOLDER_NAME, DEFAULT_OUTPUT_SUBFOLDER_NAME, \
    ABSOLUTE_FOLDER_NAME, URL_FOLDER_NAME, NO_FOLDER_NAME, \
    get_default_image_directory, get_default_output_directory, \
    standardize_default_folder_names
import cellprofiler.measurement

from cellprofiler.utilities.utf16encode import utf16encode
import skimage.morphology

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

MEASUREMENTS_GROUP = 'measurementsgroup'

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

SUPPORT_URLS_SHOW_DIR = "show_directory"


class Setting(object):
    """A module setting which holds a single string value

    """
    #
    # This should be set to False for UI elements like buttons and dividers
    #
    save_to_pipeline = True

    def __init__(self, text, value, doc=None, reset_view=False):
        """Initialize a setting with the enclosing module and its string value

        text   - the explanatory text for the setting
        value  - the default or initial value for the setting
        doc - documentation for the setting
        reset_view - True if miniscule editing should re-evaluate the module view
        """
        self.__text = text
        self.__value = value
        self.doc = doc
        self.__key = uuid.uuid4()
        self.reset_view = reset_view

    def set_value(self, value):
        self.__value = value

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

    def __internal_set_value(self, value):
        self.set_value(value)

    value = property(__internal_get_value, __internal_set_value)

    def get_value_text(self):
        '''Get the underlying string value'''
        return self.__value

    def set_value_text(self, value):
        '''Set the underlying string value

        Can be overridden as long as the base class set_value_text is
        called with the target value. An example is to allow the user to
        enter an invalid text value, but still maintain the last valid value
        entered.
        '''
        self.__value = value

    def __internal_set_value_text(self, value):
        self.set_value_text(value)

    value_text = property(get_value_text, __internal_set_value_text)

    def __eq__(self, x):
        # we test explicitly for other Settings to prevent matching if
        # their .values are the same.
        if isinstance(x, Setting):
            return self.__key == x.__key
        return self.eq(x)

    def eq(self, x):
        '''The equality test for things other than settings

        x - the thing to be compared, for instance a string

        override this to do things like compare whether an integer
        setting's value matches a given number
        '''
        return self.value == unicode(x)

    def __ne__(self, x):
        return not self.__eq__(x)

    def get_is_yes(self):
        """Return true if the setting's value is "Yes" """
        return self.__value == YES

    def set_is_yes(self, is_yes):
        """Set the setting value to Yes if true, No if false"""
        self.__value = (is_yes and YES) or NO

    is_yes = property(get_is_yes, set_is_yes)

    def get_is_do_not_use(self):
        """Return true if the setting's value is Do not use"""
        return self.value == DO_NOT_USE

    is_do_not_use = property(get_is_do_not_use)

    def test_valid(self, pipeline):
        """Throw a ValidationError if the value of this setting is inappropriate for the context"""
        pass

    def test_setting_warnings(self, pipeline):
        """Throw a ValidationError to warn the user about a setting value issue

        A setting should raise ValidationError if a setting's value is
        likely to be in error, but could possibly be correct. An example is
        a field that can be left blank, but is filled in, except for rare
        cases.
        """
        pass

    def __str__(self):
        '''Return value as a string.

        NOTE: strings are deprecated, use unicode_value instead.
        '''
        if isinstance(self.__value, unicode):
            return str(utf16encode(self.__value))
        if not isinstance(self.__value, str):
            raise ValidationError("%s was not a string" % self.__value, self)
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

    def __init__(self, sequence, text="Hidden"):
        super(HiddenCount, self).__init__(text, str(len(sequence)))
        self.__sequence = sequence

    def set_value(self, value):
        if not value.isdigit():
            raise ValueError("The value must be an integer")
        count = int(value)
        if count == len(self.__sequence):
            # The value was "inadvertantly" set, but is correct
            return
        raise NotImplementedError(
                "The count should be inferred, not set  - actual: %d, set: %d" % (len(self.__sequence), count))

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
        super(Text, self).__init__(text, value, *args, **kwargs)


class RegexpText(Setting):
    """A setting with a regexp button on the side
    """
    GUESS_FILE = "file"
    GUESS_FOLDER = "folder"

    def __init__(self, text, value, *args, **kwargs):
        '''initialize the setting

        text   - the explanatory text for the setting
        value  - the default or initial value for the setting
        doc - documentation for the setting
        get_example_fn - a function that returns an example string for the
                         metadata editor
        guess - either GUESS_FILE to use potential file-name regular expressions
                when guessing in the regexp editor or GUESS_FOLDER to
                use folder-name guesses.
        '''
        kwargs = kwargs.copy()
        self.get_example_fn = kwargs.pop("get_example_fn", None)
        self.guess = kwargs.pop("guess", self.GUESS_FILE)
        super(RegexpText, self).__init__(text, value, *args, **kwargs)

    def test_valid(self, pipeline):
        try:
            # Convert Matlab to Python
            pattern = re.sub('(\\(\\?)([<][^)>]+?[>])', '\\1P\\2', self.value)
            re.search('(|(%s))' % pattern, '')
        except re.error, v:
            raise ValidationError("Invalid regexp: %s" % v, self)


class DirectoryPath(Text):
    """A setting that displays a filesystem path name
    """
    DIR_ALL = [ABSOLUTE_FOLDER_NAME,
               DEFAULT_INPUT_FOLDER_NAME, DEFAULT_OUTPUT_FOLDER_NAME,
               DEFAULT_INPUT_SUBFOLDER_NAME,
               DEFAULT_OUTPUT_SUBFOLDER_NAME]

    def __init__(self, text, value=None, dir_choices=None,
                 allow_metadata=True, support_urls=False,
                 *args, **kwargs):
        if dir_choices is None:
            dir_choices = DirectoryPath.DIR_ALL
        if support_urls and not (URL_FOLDER_NAME in dir_choices):
            dir_choices = dir_choices + [URL_FOLDER_NAME]
        if value is None:
            value = DirectoryPath.static_join_string(
                    dir_choices[0], "")
        self.dir_choices = dir_choices
        self.allow_metadata = allow_metadata
        self.support_urls = support_urls
        super(DirectoryPath, self).__init__(text, value, *args, **kwargs)

    def split_parts(self):
        '''Return the directory choice and custom path as a tuple'''
        result = tuple(self.value.split('|', 1))
        if len(result) == 1:
            result = (result[0], ".")
        return result

    @staticmethod
    def split_string(value):
        return tuple(value.split('|', 1))

    def join_parts(self, dir_choice=None, custom_path=None):
        '''Join the directory choice and custom path to form a value'''
        self.value = self.join_string(dir_choice, custom_path)

    def join_string(self, dir_choice=None, custom_path=None):
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
        self.join_parts(dir_choice=choice)

    dir_choice = property(get_dir_choice, set_dir_choice)

    def get_custom_path(self):
        '''The custom path relative to the directory selection method'''
        return self.split_parts()[1]

    def set_custom_path(self, custom_path):
        self.join_parts(custom_path=custom_path)

    custom_path = property(get_custom_path, set_custom_path)

    @property
    def is_custom_choice(self):
        '''True if the current dir_choice requires a custom path'''
        return self.dir_choice in [
            ABSOLUTE_FOLDER_NAME, DEFAULT_INPUT_SUBFOLDER_NAME,
            DEFAULT_OUTPUT_SUBFOLDER_NAME, URL_FOLDER_NAME]

    def get_absolute_path(self, measurements=None, image_set_number=None):
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
                    custom_path = custom_path[:md_start].replace("\\\\", "\\")
                    custom_path = os.path.split(custom_path)[0]
                else:
                    custom_path = custom_path.replace("\\\\", "\\")
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
            custom_path = path[len(img_dir) + 1:]
        elif (cmp_path.startswith(out_dir) and
                      cmp_path[len(out_dir)] in seps):
            dir_choice = DEFAULT_OUTPUT_SUBFOLDER_NAME
            custom_path = path[len(out_dir) + 1:]
        else:
            dir_choice = ABSOLUTE_FOLDER_NAME
            custom_path = path
        return dir_choice, custom_path

    def alter_for_create_batch_files(self, fn_alter_path):
        '''Call this to alter the setting appropriately for batch execution'''
        custom_path = self.custom_path
        if custom_path.startswith("\g<") and sys.platform.startswith("win"):
            # So ugly, the "\" sets us up for the root directory during
            # os.path.join, so we need r".\\" at start to fake everyone out
            custom_path = r".\\" + custom_path

        if self.dir_choice == DEFAULT_INPUT_FOLDER_NAME:
            pass
        elif self.dir_choice == DEFAULT_OUTPUT_FOLDER_NAME:
            pass
        elif self.dir_choice == ABSOLUTE_FOLDER_NAME:
            self.custom_path = fn_alter_path(
                    self.custom_path, regexp_substitution=self.allow_metadata)
        elif self.dir_choice == DEFAULT_INPUT_SUBFOLDER_NAME:
            self.custom_path = fn_alter_path(
                    self.custom_path, regexp_substitution=self.allow_metadata)
        elif self.dir_choice == DEFAULT_OUTPUT_SUBFOLDER_NAME:
            self.custom_path = fn_alter_path(
                    self.custom_path, regexp_substitution=self.allow_metadata)

    def test_valid(self, pipeline):
        if self.dir_choice not in self.dir_choices + [NO_FOLDER_NAME]:
            raise ValidationError("Unsupported directory choice: %s" %
                                  self.dir_choice, self)
        if (not self.allow_metadata and self.is_custom_choice and
                    self.custom_path.find(r"\g<") != -1):
            raise ValidationError("Metadata not supported for this setting",
                                  self)
        if self.dir_choice == ABSOLUTE_FOLDER_NAME and (
                    (self.custom_path is None) or (len(self.custom_path) == 0)):
            raise ValidationError("Please enter a valid path", self)


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
       mode - Controls whether a file-open or file-save dialog is displayed
              when the user browses.
              FilenameText.MODE_OPEN - open a file that must exist
              FilenameText.MODE_APPEND - open a file for modification or
                    create a new file (using the Save dialog)
              FilenameText.MODE_OVERWRITE - create a new file and warn the
                    user if the file exists and will be overwritten.
    """
    MODE_OPEN = "Open"
    MODE_APPEND = "Append"
    MODE_OVERWRITE = "Overwrite"

    def __init__(self, text, value, *args, **kwargs):
        kwargs = kwargs.copy()
        self.get_directory_fn = kwargs.pop("get_directory_fn", None)
        self.set_directory_fn = kwargs.pop("set_directory_fn", None)
        self.browse_msg = kwargs.pop("browse_msg", "Choose a file")
        self.exts = kwargs.pop("exts", None)
        self.mode = kwargs.pop("mode", self.MODE_OPEN)
        super(FilenameText, self).__init__(text, value, *args, **kwargs)
        self.browsable = True

    def set_browsable(self, val):
        self.browsable = val


class Pathname(Text):
    """A setting that displays a path name

    text - text to display to right
    value - initial value
    wildcard - wildcard to filter files in browse dialog
    """

    def __init__(self, text, value="", *args, **kwargs):
        kwargs = kwargs.copy()
        if kwargs.has_key("wildcard"):
            self.wildcard = kwargs["wildcard"]
            del kwargs["wildcard"]
        else:
            self.wildcard = "All files (*.*)|*.*"
        super(Pathname, self).__init__(text, value, *args, **kwargs)

    def test_valid(self, pipeline):
        if not os.path.isfile(self.value):
            raise ValidationError("Can't find file, %s" % self.value, self)

    def alter_for_create_batch(self, fn_alter):
        self.value = fn_alter(self.value)


class PathnameOrURL(Pathname):
    """A setting that displays a path name or URL

    """

    def is_url(self):
        return any([self.value_text.lower().startswith(scheme)
                    for scheme in ("http:", "https:", "ftp:")])

    def test_valid(self, pipeline):
        if not self.is_url():
            super(PathnameOrURL, self).test_valid(pipeline)


class ImagePlane(Setting):
    """A setting that specifies an image plane

    This setting lets the user pick an image plane. This might be an image
    file, a URL, or a plane from an image stack.

    The text setting has four fields, delimited by a space character (which
    is luckily disallowed in a URL:

    Field 1: URL

    Field 2: series (or None if a space is followed by another)

    Field 3: index (or None if blank)

    Field 4: channel (or None if blank)
    """

    def __init__(self, text, *args, **kwargs):
        '''Initialize the setting

        text - informative text to display to the left
        '''
        super(ImagePlane, self).__init__(
                text, ImagePlane.build(""), *args, **kwargs)

    @staticmethod
    def build(url, series=None, index=None, channel=None):
        '''Build the string representation of the setting

        url - the URL of the file containing the plane

        series - the series for a multi-series stack or None if the whole file

        index - the index of the frame for a multi-frame stack or None if
                the whole stack

        channel - the channel of an interlaced color image or None if all
                  channels
        '''
        if " " in url:
            # Spaces are not legal characters in URLs, nevertheless, I try
            # to accomodate
            logger.warn(
                    "URLs should not contain spaces. %s is the offending URL" % url)
            url = url.replace(" ", "%20")
        return " ".join([str(x) if x is not None else ""
                         for x in url, series, index, channel])

    def __get_field(self, index):
        f = self.value_text.split(" ")[index]
        if len(f) == 0:
            return None
        return f

    def __get_int_field(self, index):
        f = self.__get_field(index)
        if f is None:
            return f
        return int(f)

    @property
    def url(self):
        '''The URL portion of the image plane descriptor'''
        uurl = self.__get_field(0)
        if uurl is not None:
            uurl = uurl.encode("utf-8")
        return uurl

    @property
    def series(self):
        '''The series portion of the image plane descriptor'''
        return self.__get_int_field(1)

    @property
    def index(self):
        '''The index portion of the image plane descriptor'''
        return self.__get_int_field(2)

    @property
    def channel(self):
        '''The channel portion of the image plane descriptor'''
        return self.__get_int_field(3)

    def test_valid(self, pipeline):
        if self.url is None:
            raise ValidationError(
                    "This setting's URL is blank. Please select a valid image",
                    self)


class AlphanumericText(Text):
    '''A setting for entering text values limited to alphanumeric + _ values

    This can be used for measurement names, object names, etc.
    '''

    def __init__(self, text, value, *args, **kwargs):
        '''Initializer

        text - the explanatory text for the setting UI

        value - the default / initial value

        first_must_be_alpha - True if the first character of the value must
                              be a letter or underbar.
        '''
        kwargs = kwargs.copy()
        self.first_must_be_alpha = kwargs.pop("first_must_be_alpha", False)
        super(AlphanumericText, self).__init__(text, value, *args, **kwargs)

    def test_valid(self, pipeline):
        '''Restrict names to legal ascii C variables

        First letter = a-zA-Z and underbar, second is that + digit.
        '''
        self.validate_alphanumeric_text(self.value, self, self.first_must_be_alpha)

    @staticmethod
    def validate_alphanumeric_text(text, setting, first_must_be_alpha):
        '''Validate text as alphanumeric, throwing a validation error if not

        text - text to be validated

        setting - blame this setting on failure

        first_must_be_alpha - True if the first letter has to be alpha or underbar
        '''
        if first_must_be_alpha:
            pattern = "^[A-Za-z_][A-Za-z_0-9]*$"
            error = (
                'Names must start with an ASCII letter or underbar ("_")'
                ' optionally followed by ASCII letters, underbars or digits.')
        else:
            pattern = "^[A-Za-z_0-9]+$"
            error = ('Only ASCII letters, digits and underbars ("_") can be '
                     'used here')

        match = re.match(pattern, text)
        if match is None:
            raise ValidationError(error, setting)


class Number(Text):
    """A setting that allows only numeric input
    """

    def __init__(self, text, value=0, minval=None, maxval=None, *args,
                 **kwargs):
        if isinstance(value, basestring):
            text_value = value
            value = self.str_to_value(value)
        else:
            text_value = self.value_to_str(value)
        super(Number, self).__init__(text, text_value, *args, **kwargs)
        self.__default = self.str_to_value(text_value)
        self.__minval = minval
        self.__maxval = maxval

    def str_to_value(self, str_value):
        """Return the value of the string passed

        Override this in a derived class to parse the numeric text or
        raise an exception if badly formatted.
        """
        raise NotImplementedError("Please define str_to_value in a subclass")

    def value_to_str(self, value):
        """Return the string representation of the value passed

        Override this in a derived class to convert a numeric value into text
        """
        raise NotImplementedError("Please define value_to_str in a subclass")

    def set_value(self, value):
        """Convert integer to string
        """
        str_value = unicode(value) if isinstance(value, basestring) \
            else self.value_to_str(value)
        self.set_value_text(str_value)

    def get_value(self, reraise=False):
        """Return the value of the setting as a float
        """
        return self.__default

    def set_value_text(self, value_text):
        super(Number, self).set_value_text(value_text)
        try:
            self.test_valid(None)
            self.__default = self.str_to_value(value_text)
        except:
            logger.debug("Number set to illegal value: %s" % value_text)

    def set_min_value(self, minval):
        '''Programatically set the minimum value allowed'''
        self.__minval = minval

    def set_max_value(self, minval):
        '''Programatically set the maximum value allowed'''
        self.__maxval = maxval

    def get_min_value(self):
        '''The minimum value (inclusive) that can legally be entered'''
        return self.__minval

    def get_max_value(self):
        '''The maximum value (inclusive) that can legally be entered'''
        return self.__maxval

    min_value = property(get_min_value, set_min_value)
    max_value = property(get_max_value, set_max_value)

    def test_valid(self, pipeline):
        """Return true only if the text value is float
        """
        try:
            value = self.str_to_value(self.value_text)
        except ValueError:
            raise ValidationError('Value not in decimal format', self)
        if self.__minval is not None and self.__minval > value:
            raise ValidationError(
                    'Must be at least %s, was %s' %
                    (self.value_to_str(self.__minval), self.value_text), self)
        if self.__maxval is not None and self.__maxval < value:
            raise ValidationError(
                    'Must be at most %s, was %s' %
                    (self.value_to_str(self.__maxval), self.value_text), self)

    def eq(self, x):
        '''Equal if our value equals the operand'''
        return self.value == x


class Integer(Number):
    """A setting that allows only integer input

    Initializer:
    text - explanatory text for setting
    value - default value
    minval - minimum allowed value defaults to no minimum
    maxval - maximum allowed value defaults to no maximum
    """

    def str_to_value(self, str_value):
        return int(str_value)

    def value_to_str(self, value):
        return u"%d" % value


class OddInteger(Integer):
    def test_valid(self, pipeline):
        super(self.__class__, self).test_valid(pipeline)

        value = self.str_to_value(self.value_text)

        if value % 2 == 0:
            raise ValidationError("Must be odd, was even", self)


class Range(Setting):
    """A setting representing a range between two values"""

    valid_format_text = '"%s" is formatted incorrectly'

    def __init__(self, text, value, minval=None, maxval=None, *args, **kwargs):
        '''Initialize a range

        text - helpful text to be displayed to the user

        value - default value as a string, should be in the form <min>,<max>

        minval - the minimum value for the range (or None if none)

        maxval - the maximum value of the range (or None if none)
        '''
        super(Range, self).__init__(text, value, *args, **kwargs)
        self._minval = minval
        self._maxval = maxval
        self.__default_min = self.min
        self.__default_max = self.max

    def str_to_value(self, value_str):
        '''Convert a min/max value as a string to the native type'''
        raise NotImplementedError("str_to_value must be implemented in derived class")

    def value_to_str(self, value):
        '''Convert a string to a min/max value in the native type'''
        raise NotImplementedError("value_to_str must be implemented in derived class")

    def get_value(self):
        '''Return the value of this range as a min/max tuple'''
        return self.min, self.max

    def set_value(self, value):
        '''Set the value of this range using either a string or a two-tuple'''
        if isinstance(value, basestring):
            self.set_value_text(value)
        elif hasattr(value, "__getitem__") and len(value) == 2:
            self.set_value_text(",".join([self.value_to_str(v) for v in value]))
        else:
            raise ValueError("Value for range must be a string or two-tuple")

    def get_min_text(self):
        """Get the minimum of the range as a text value"""
        return self.get_value_text().split(",")[0]

    def get_min(self):
        """Get the minimum of the range as a number"""
        try:
            value = self.str_to_value(self.get_min_text())
            if self._minval is not None and value < self._minval:
                return self._minval
            return value
        except:
            return self.__default_min

    def get_max_text(self):
        """Get the maximum of the range as a text value"""
        vv = self.get_value_text().split(",")
        if len(vv) < 2:
            return ""
        return vv[1]

    def get_max(self):
        """Get the maximum of the range as a number"""
        try:
            value = self.str_to_value(self.get_max_text())
            if self._maxval is not None and value > self._maxval:
                return self._maxval
            return value
        except:
            return self.__default_max

    def compose_min_text(self, value):
        """Return the text value that would set the minimum to the proposed value

        value - the proposed minimum value as text
        """
        return ",".join((value, self.get_max_text()))

    def set_min(self, value):
        """Set the minimum part of the value, given the minimum as a #"""
        self.set_value_text(self.compose_min_text(self.value_to_str(value)))

    def compose_max_text(self, value):
        """Return the text value that would set the maximum to the proposed value

        value - the proposed maximum value as text
        """
        return ",".join((self.get_min_text(), value))

    def set_max(self, value):
        """Set the maximum part of the value, given the maximum as a #"""
        self.set_value_text(self.compose_max_text(self.value_to_str(value)))

    min = property(get_min, set_min)
    min_text = property(get_min_text)
    max = property(get_max, set_max)
    max_text = property(get_max_text)

    def set_value_text(self, value):
        super(Range, self).set_value_text(value)
        try:
            self.test_valid(None)
            self.__default_min = self.min
            self.__default_max = self.max
        except:
            logger.debug("Illegal value in range setting: %s" % value)

    def test_valid(self, pipeline):
        values = self.value_text.split(',')
        if len(values) < 2:
            raise ValidationError("Minimum and maximum values must be separated by a comma", self)
        if len(values) > 2:
            raise ValidationError("Only two values allowed", self)
        for value in values:
            try:
                self.str_to_value(value)
            except:
                raise ValidationError(self.valid_format_text % value, self)
        v_min, v_max = [self.str_to_value(value) for value in values]
        if self._minval is not None and self._minval > v_min:
            raise ValidationError("%s can't be less than %s" % (
                self.min_text, self.value_to_str(self._minval)), self)
        if self._maxval is not None and self._maxval < v_max:
            raise ValidationError("%s can't be greater than %s" % (
                self.max_text, self.value_to_str(self._maxval)), self)
        if v_min > v_max:
            raise ValidationError("%s is greater than %s" %
                                  (self.min_text, self.max_text), self)

    def eq(self, x):
        '''If the operand is a sequence, true if it matches the min and max'''
        if hasattr(x, "__getitem__") and len(x) == 2:
            return x[0] == self.min and x[1] == self.max
        return False


class IntegerRange(Range):
    """A setting that allows only integer input between two constrained values
    """
    valid_format_text = "%s must be all digits"

    def __init__(self, text, value=(0, 1), minval=None, maxval=None, *args,
                 **kwargs):
        """Initialize an integer range
        text  - helpful text to be displayed to the user
        value - initial default value, a two-tuple as minimum and maximum
        minval - the minimum acceptable value of either
        maxval - the maximum acceptable value of either
        """
        super(IntegerRange, self).__init__(
                text, "%d,%d" % value, minval, maxval, *args, **kwargs)

    def str_to_value(self, value_str):
        return int(value_str)

    def value_to_str(self, value):
        return "%d" % value


class Coordinates(Setting):
    """A setting representing X and Y coordinates on an image
    """

    def __init__(self, text, value=(0, 0), *args, **kwargs):
        """Initialize an integer range
        text  - helpful text to be displayed to the user
        value - initial default value, a two-tuple as x and y
        """
        super(Coordinates, self).__init__(text, "%d,%d" % value, *args, **kwargs)

    def set_value(self, value):
        """Convert integer tuples to string
        """
        try:
            if len(value) == 2:
                super(Coordinates, self).set_value("%d,%d" % (value[0], value[1]))
                return
        except:
            pass
        super(Coordinates, self).set_value(value)

    def get_value(self):
        """Convert the underlying string to a two-tuple"""
        return self.get_x(), self.get_y()

    def get_x_text(self):
        """Get the x coordinate as text"""
        return self.get_value_text().split(",")[0]

    def get_x(self):
        """The x coordinate"""
        return int(self.get_x_text())

    x = property(get_x)

    def get_y_text(self):
        vv = self.get_value_text().split(",")
        if len(vv) < 2:
            return ""
        return vv[1]

    def get_y(self):
        """The y coordinate"""
        return int(self.get_y_text())

    y = property(get_y)

    def test_valid(self, pipeline):
        values = self.value_text.split(',')
        if len(values) < 2:
            raise ValidationError("X and Y values must be separated by a comma", self)
        if len(values) > 2:
            raise ValidationError("Only two values allowed", self)
        for value in values:
            if not value.isdigit():
                raise ValidationError("%s is not an integer" % value, self)


BEGIN = "begin"
END = "end"


class IntegerOrUnboundedRange(IntegerRange):
    """A setting that specifies an integer range where the minimum and maximum
    can be set to unbounded by the user.

    The maximum value can be relative to the far side in which case a negative
    number is returned for slicing.
    """

    def __init__(self, text, value=(0, END), minval=None, maxval=None,
                 *args, **kwargs):
        """Initialize an integer range
        text  - helpful text to be displayed to the user
        value - initial default value, a two-tuple as minimum and maximum
        minval - the minimum acceptable value of either
        maxval - the maximum acceptable value of either
        """
        Range.__init__(self, text, "%s,%s" % (str(value[0]), str(value[1])),
                       *args, **kwargs)

    def str_to_value(self, str_value):
        if str_value == BEGIN:
            return 0
        elif (self.is_abs() and str_value == END) or \
                (len(str_value) > 0 and str_value[1:] == END):
            return END
        return super(IntegerOrUnboundedRange, self).str_to_value(str_value)

    def value_to_str(self, value):
        if value in (BEGIN, END):
            return value
        return super(IntegerOrUnboundedRange, self).value_to_str(value)

    def get_unbounded_min(self):
        """True if there is no minimum"""
        return self.get_min() == 0

    unbounded_min = property(get_unbounded_min)

    def get_display_min(self):
        """What to display for the minimum"""
        return self.get_min_text()

    display_min = property(get_display_min)

    def get_unbounded_max(self):
        """True if there is no maximum"""
        return self.get_max_text() == END

    unbounded_max = property(get_unbounded_max)

    def get_display_max(self):
        """What to display for the maximum"""
        #
        # Remove the minus sign
        #
        mt = self.get_max_text()
        if self.is_abs():
            return mt
        return mt[1:]

    display_max = property(get_display_max)

    def compose_display_max_text(self, dm_value):
        """Compose a value_text value for the setting given a max text value

        dm_value - the displayed text for the maximum of the range

        Returns a text value suitable for this setting that sets the
        maximum while keeping the minimum and abs/rel the same
        """
        if self.is_abs():
            return self.compose_max_text(dm_value)
        else:
            return self.compose_max_text("-" + dm_value)

    def is_abs(self):
        """Return True if the maximum is an absolute # of pixels

        Returns False if the # of pixels is relative to the right edge.
        """
        mt = self.get_max_text()
        return len(mt) == 0 or mt[0] != "-"

    def compose_abs(self):
        """Compose a text value that uses absolute upper bounds coordinates

        Return a text value for IntegerOrUnboundedRange that keeps the min
        and the max the same, but states that the max is the distance in pixels
        from the origin.
        """
        return self.compose_max_text(self.get_display_max())

    def compose_rel(self):
        """Compose a text value that uses relative upper bounds coordinates

        Return a text value for IntegerOrUnboundedRange that keeps the min
        and the max the same, but states that the max is the distance in pixels
        from the side of the image opposite the origin.
        """
        return self.compose_max_text("-" + self.get_display_max())

    def test_valid(self, pipeline):
        values = self.value_text.split(',')
        if len(values) < 2:
            raise ValidationError("Minimum and maximum values must be separated by a comma", self)
        if len(values) > 2:
            raise ValidationError("Only two values allowed", self)
        if (not values[0].isdigit()) and values[0] != BEGIN:
            raise ValidationError("%s is not an integer" % (values[0]), self)
        if len(values[1]) == 0:
            raise ValidationError("The end value is blank", self)
        if not (values[1] == END or
                    values[1].isdigit() or
                    (values[1][0] == '-' and
                         (values[1][1:].isdigit() or values[1][1:] == END))):
            raise ValidationError("%s is not an integer or %s" % (values[1], END), self)
        if ((not self.unbounded_min) and
                self._minval and
                    self._minval > self.min):
            raise ValidationError(
                    "%s can't be less than %d" % (self.min_text, self._minval), self)
        if ((not self.unbounded_max) and
                self._maxval and
                    self._maxval < self.max):
            raise ValidationError("%d can't be greater than %d" %
                                  (self.max, self._maxval), self)
        if ((not self.unbounded_min) and (not self.unbounded_max) and
                    self.min > self.max and self.max > 0):
            raise ValidationError("%d is greater than %d" % (self.min, self.max), self)


class Float(Number):
    '''A class that only allows floating point input'''

    def str_to_value(self, str_value):
        return float(str_value)

    def value_to_str(self, value):
        text_value = (u"%f" % value).rstrip("0")
        if text_value.endswith("."):
            text_value += "0"
        return text_value


class FloatRange(Range):
    """A setting that allows only floating point input between two constrained values
    """
    valid_format_text = "%s must be a floating-point number"

    def __init__(self, text, value=(0, 1), *args,
                 **kwargs):
        """Initialize an integer range
        text  - helpful text to be displayed to the user
        value - initial default value, a two-tuple as minimum and maximum
        minval - the minimum acceptable value of either
        maxval - the maximum acceptable value of either
        """
        smin, smax = [(u"%f" % v).rstrip("0") for v in value]
        text_value = ",".join([x + "0" if x.endswith(".") else x for x in smin, smax])
        super(FloatRange, self).__init__(text, text_value, *args, **kwargs)

    def str_to_value(self, value_str):
        return float(value_str)

    def value_to_str(self, value):
        return "%f" % value


class BinaryMatrix(Setting):
    """A setting that allows editing of a 2D matrix of binary values
    """

    def __init__(self, text,
                 default_value=True,
                 default_width=5,
                 default_height=5, **kwargs):
        initial_value_text = self.to_value(
                [[default_value] * default_width] * default_height)
        Setting.__init__(self, text, initial_value_text, **kwargs)

    @staticmethod
    def to_value(matrix):
        '''Convert a matrix to a pickled form

        format is <row-count>,<column-count>,<0 or 1>*row-count*column-count

        e.g. [[True, False, True], [True, True, True]] -> "2,3,101111"
        '''
        h = len(matrix)
        w = 0 if h == 0 else len(matrix[0])
        return ",".join((
            str(h), str(w),
            "".join(["".join(["1" if v else "0" for v in row])
                     for row in matrix])))

    def get_matrix(self):
        '''Return the setting's matrix'''
        hs, ws, datas = self.value_text.split(",")
        h, w = int(hs), int(ws)
        return [[datas[i * w + j] == "1" for j in range(w)] for i in range(h)]

    def get_size(self):
        '''Return the size of the matrix

        returns a tuple of height, width
        '''
        hs, ws, datas = self.value_text.split(",")
        return int(hs), int(ws)


PROVIDED_ATTRIBUTES = "provided_attributes"


class NameProvider(AlphanumericText):
    """A setting that provides a named object
    """

    def __init__(self, text, group, value=DO_NOT_USE, *args, **kwargs):
        self.__provided_attributes = {"group": group}
        kwargs = kwargs.copy()
        if kwargs.has_key("provided_attributes"):
            self.__provided_attributes.update(kwargs["provided_attributes"])
            del kwargs[PROVIDED_ATTRIBUTES]
        kwargs["first_must_be_alpha"] = True
        super(NameProvider, self).__init__(text, value, *args, **kwargs)

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
        super(ImageNameProvider, self).__init__(text, IMAGE_GROUP, value,
                                                *args, **kwargs)


class FileImageNameProvider(ImageNameProvider):
    """A setting that provides an image name where the image has an associated file"""

    def __init__(self, text, value=DO_NOT_USE, *args, **kwargs):
        kwargs = kwargs.copy()
        if not kwargs.has_key(PROVIDED_ATTRIBUTES):
            kwargs[PROVIDED_ATTRIBUTES] = {}
        kwargs[PROVIDED_ATTRIBUTES][FILE_IMAGE_ATTRIBUTE] = True
        super(FileImageNameProvider, self).__init__(text, value, *args,
                                                    **kwargs)


class ExternalImageNameProvider(ImageNameProvider):
    """A setting that provides an image name where the image is loaded
    externally. (eg: from Java)"""

    def __init__(self, text, value=DO_NOT_USE, *args, **kwargs):
        kwargs = kwargs.copy()
        if not kwargs.has_key(PROVIDED_ATTRIBUTES):
            kwargs[PROVIDED_ATTRIBUTES] = {}
        kwargs[PROVIDED_ATTRIBUTES][EXTERNAL_IMAGE_ATTRIBUTE] = True
        super(ExternalImageNameProvider, self).__init__(text, value, *args,
                                                        **kwargs)


class CroppingNameProvider(ImageNameProvider):
    """A setting that provides an image name where the image has a cropping mask"""

    def __init__(self, text, value=DO_NOT_USE, *args, **kwargs):
        kwargs = kwargs.copy()
        if not kwargs.has_key(PROVIDED_ATTRIBUTES):
            kwargs[PROVIDED_ATTRIBUTES] = {}
        kwargs[PROVIDED_ATTRIBUTES][CROPPING_ATTRIBUTE] = True
        super(CroppingNameProvider, self).__init__(text, value, *args, **kwargs)


class ObjectNameProvider(NameProvider):
    """A setting that provides an image name
    """

    def __init__(self, text, value=DO_NOT_USE, *args, **kwargs):
        super(ObjectNameProvider, self).__init__(text, OBJECT_GROUP, value,
                                                 *args, **kwargs)

    def test_valid(self, pipeline):
        if self.value_text in cellprofiler.measurement.disallowed_object_names:
            raise ValidationError(
                    "Object names may not be any of %s" % (
                        ", ".join(cellprofiler.measurement.disallowed_object_names)),
                    self)
        super(ObjectNameProvider, self).test_valid(pipeline)


class OutlineNameProvider(ImageNameProvider):
    '''A setting that provides an object outline name
    '''

    def __init__(self, text, value=DO_NOT_USE, *args, **kwargs):
        super(OutlineNameProvider, self).__init__(text, value,
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
        if value is None:
            value = (can_be_blank and blank_text) or "None"
        self.__required_attributes = {"group": group}
        if kwargs.has_key(REQUIRED_ATTRIBUTES):
            self.__required_attributes.update(kwargs[REQUIRED_ATTRIBUTES])
            kwargs = kwargs.copy()
            del kwargs[REQUIRED_ATTRIBUTES]
        self.__can_be_blank = can_be_blank
        self.__blank_text = blank_text
        super(NameSubscriber, self).__init__(text, value, *args, **kwargs)

    def get_group(self):
        """This setting provides a name to this group

        Returns a group name, e.g. imagegroup or objectgroup
        """
        return self.__required_attributes["group"]

    group = property(get_group)

    def get_choices(self, pipeline):
        choices = []
        if self.__can_be_blank:
            choices.append((self.__blank_text, "", 0, False))
        return choices + sorted(get_name_provider_choices(pipeline, self, self.group))

    def get_is_blank(self):
        """True if the selected choice is the blank one"""
        return self.__can_be_blank and self.value == self.__blank_text

    is_blank = property(get_is_blank)

    def matches(self, setting):
        """Return true if this subscriber matches the category of the provider"""
        return all([setting.provided_attributes.get(key, None) ==
                    self.__required_attributes[key]
                    for key in self.__required_attributes.keys()])

    def test_valid(self, pipeline):
        choices = self.get_choices(pipeline)
        if len(choices) == 0:
            raise ValidationError("No prior instances of %s were defined" % self.group, self)
        if self.value not in [c[0] for c in choices]:
            raise ValidationError("%s not in %s" % (self.value, ", ".join(c[0] for c in self.get_choices(pipeline))),
                                  self)


def filter_duplicate_names(name_list):
    '''remove any repeated names from a list of (name, ...) keeping the last occurrence.'''
    name_dict = dict(zip((n[0] for n in name_list), name_list))
    return [name_dict[n[0]] for n in name_list]


def get_name_provider_choices(pipeline, last_setting, group):
    '''Scan the pipeline to find name providers for the given group

    pipeline - pipeline to scan
    last_setting - scan the modules in order until you arrive at this setting
    group - the name of the group of providers to scan
    returns a list of tuples, each with (provider name, module name, module number)
    '''
    choices = []
    for module in pipeline.modules(False):
        module_choices = [
            (other_name, module.module_name, module.module_num,
             module.is_input_module())
            for other_name in module.other_providers(group)]
        for setting in module.visible_settings():
            if setting.key() == last_setting.key():
                return filter_duplicate_names(choices)
            if (isinstance(setting, NameProvider) and
                    module.enabled and
                        setting != DO_NOT_USE and
                    last_setting.matches(setting)):
                module_choices.append((
                    setting.value, module.module_name, module.module_num,
                    module.is_input_module()))
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
    for module in pipeline.modules(False):
        module_choices = []
        for setting in module.visible_settings():
            if setting.key() == last_setting.key():
                return choices
            if (isinstance(setting, NameProvider) and
                        setting != DO_NOT_USE and
                    module.enabled and
                    last_setting.matches(setting) and
                        setting.value == last_setting.value):
                module_choices.append(setting)
        choices += module_choices
    assert False, "Setting not among visible settings in pipeline"


class ImageNameSubscriber(NameSubscriber):
    """A setting that provides an image name
    """

    def __init__(self, text, value=None, can_be_blank=False,
                 blank_text=LEAVE_BLANK, *args, **kwargs):
        super(ImageNameSubscriber, self).__init__(text, IMAGE_GROUP, value,
                                                  can_be_blank, blank_text,
                                                  *args, **kwargs)


class FileImageNameSubscriber(ImageNameSubscriber):
    """A setting that provides image names loaded from files"""

    def __init__(self, text, value=DO_NOT_USE, can_be_blank=False,
                 blank_text=LEAVE_BLANK, *args, **kwargs):
        kwargs = kwargs.copy()
        if not kwargs.has_key(REQUIRED_ATTRIBUTES):
            kwargs[REQUIRED_ATTRIBUTES] = {}
        kwargs[REQUIRED_ATTRIBUTES][FILE_IMAGE_ATTRIBUTE] = True
        super(FileImageNameSubscriber, self).__init__(text, value, can_be_blank,
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
        super(CroppingNameSubscriber, self).__init__(text, value, can_be_blank,
                                                     blank_text, *args,
                                                     **kwargs)


class ExternalImageNameSubscriber(ImageNameSubscriber):
    """A setting that provides image names loaded externally (eg: from Java)"""

    def __init__(self, text, value=DO_NOT_USE, can_be_blank=False,
                 blank_text=LEAVE_BLANK, *args, **kwargs):
        super(ExternalImageNameSubscriber, self).__init__(text, value, can_be_blank,
                                                          blank_text, *args,
                                                          **kwargs)


class ObjectNameSubscriber(NameSubscriber):
    """A setting that subscribes to the list of available object names
    """

    def __init__(self, text, value=DO_NOT_USE, can_be_blank=False,
                 blank_text=LEAVE_BLANK, *args, **kwargs):
        super(ObjectNameSubscriber, self).__init__(text, OBJECT_GROUP, value,
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

    def __init(self, text, value=DO_NOT_USE, *args, **kwargs):
        super(Setting, self).__init(text, value, *args, **kwargs)

    def get_choices(self, pipeline):
        choices = []
        for module in pipeline.modules():
            for setting in module.visible_settings():
                if setting.key() == self.key():
                    return choices
            choices.append("%d: %s" % (module.module_num, module.module_name))
        assert False, "Setting not among visible settings in pipeline"


class GridNameSubscriber(NameSubscriber):
    """A setting that subscribes to grid information providers
    """

    def __init__(self, text, value=DO_NOT_USE, can_be_blank=False,
                 blank_text=LEAVE_BLANK, *args, **kwargs):
        super(GridNameSubscriber, self).__init__(text, GRID_GROUP, value,
                                                 can_be_blank, blank_text,
                                                 *args, **kwargs)


class Binary(Setting):
    """A setting that is represented as either true or false
    The underlying value stored in the settings slot is "Yes" or "No"
    for historical reasons.
    """

    def __init__(self, text, value, callback=None, *args, **kwargs):
        """Initialize the binary setting with the module, explanatory
        text and value. The value for a binary setting is True or
        False.
        """
        str_value = (value and YES) or NO
        super(Binary, self).__init__(text, str_value, *args, **kwargs)
        self.__callback = callback

    def set_value(self, value):
        """When setting, translate true and false into yes and no"""
        if value == YES or value == NO or \
                isinstance(value, str) or isinstance(value, unicode):
            super(Binary, self).set_value(value)
        else:
            str_value = (value and YES) or NO
            super(Binary, self).set_value(str_value)

    def get_value(self):
        """Get the value of a binary setting as a truth value
        """
        return super(Binary, self).get_value() == YES

    def eq(self, x):
        if x == NO:
            x = False
        return (self.value and x) or (not self.value and not x)

    def __nonzero__(self):
        '''Return the value when testing for True / False'''
        return self.value

    def on_event_fired(self, selection):
        if self.__callback is not None:
            self.__callback(selection)


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
        super(Choice, self).__init__(text, value or choices[0], *args, **kwargs)
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

    def test_valid(self, pipeline):
        """Check to make sure that the value is among the choices"""
        if self.__choices_fn is not None:
            self.__choices = self.__choices_fn(pipeline)
        if self.value not in self.choices:
            raise ValidationError(
                    "%s is not one of %s" %
                    (self.value, ",".join(self.choices)), self)


class StructuringElement(Setting):
    def __init__(self, text="Structuring element", value="disk,1", doc=None):
        super(StructuringElement, self).__init__(text, value, doc=doc)

    @staticmethod
    def get_choices():
        return [
            "ball",
            "cube",
            "diamond",
            "disk",
            "octahedron",
            "square",
            "star"
        ]

    def get_value(self):
        return getattr(skimage.morphology, self.shape)(self.size)

    def set_value(self, value):
        self.value_text = value

    @property
    def shape(self):
        return str(self.value_text.split(",")[0])

    @shape.setter
    def shape(self, value):
        self.value_text = ",".join((value, str(self.size)))

    @property
    def size(self):
        return int(self.value_text.split(",")[1])

    @size.setter
    def size(self, value):
        self.value_text = ",".join((self.shape, str(value)))


class CustomChoice(Choice):
    def __init__(self, text, choices, value=None, *args, **kwargs):
        """Initializer
        text - the explanatory text for the setting
        choices - a sequence of string choices to be displayed in the drop-down
        value - the default choice or None to choose the first of the choices.
        """
        super(CustomChoice, self).__init__(text, choices, value, *args,
                                           **kwargs)

    def get_choices(self):
        """Put the custom choice at the top"""
        choices = list(super(CustomChoice, self).get_choices())
        if self.value not in choices:
            choices.insert(0, self.value)
        return choices

    def set_value(self, value):
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
        super(MultiChoice, self).__init__(text,
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
        raise ValueError("Unexpected value type: %s" % type(value))

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

    choices = property(__internal_get_choices, __internal_set_choices)

    def set_value(self, value):
        '''Set the value of a multi-choice setting

        value is either a single string, a comma-separated string of
        multiple choices or a list of strings
        '''
        super(MultiChoice, self).set_value(self.parse_value(value))

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
                elif len(self.choices) > 25:
                    raise ValidationError(
                            "%s is not one of the choices" % selection, self)
                raise ValidationError("%s is not one of %s" %
                                      (selection,
                                       reduce(lambda x, y: "%s,%s" %
                                                           (x, y), self.choices)),
                                      self)


class SubscriberMultiChoice(MultiChoice):
    '''A multi-choice setting that gets its choices through providers

    This setting operates similarly to the name subscribers. It gets
    its choices from the name providers for the subscriber's group.
    It displays a list of choices and the user can select multiple
    choices.
    '''

    def __init__(self, text, group, value=None, *args, **kwargs):
        self.__required_attributes = {"group": group}
        if kwargs.has_key(REQUIRED_ATTRIBUTES):
            self.__required_attributes.update(kwargs[REQUIRED_ATTRIBUTES])
            kwargs = kwargs.copy()
            del kwargs[REQUIRED_ATTRIBUTES]
        super(SubscriberMultiChoice, self).__init__(text, [], value,
                                                    *args, **kwargs)

    def load_choices(self, pipeline):
        '''Get the choice list from name providers'''
        self.choices = sorted([
                                  c[0] for c in get_name_provider_choices(pipeline, self, self.group)])

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
        return object_name.replace('|', '||')

    def decode_object_name(self, object_name):
        '''Decode the escaped object name'''
        return object_name.replace('||', '|')

    def split_choice(self, choice):
        '''Split object and feature within a choice'''
        subst_choice = choice.replace('||', '++')
        loc = subst_choice.find('|')
        if loc == -1:
            return subst_choice, "Invalid"
        return choice[:loc], choice[(loc + 1):]

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
        self.populate_choices(pipeline)
        super(MeasurementMultiChoice, self).test_valid(pipeline)

    def populate_choices(self, pipeline):
        #
        # Find our module
        #
        for module in pipeline.modules():
            for setting in module.visible_settings():
                if id(setting) == id(self):
                    break
        columns = pipeline.get_measurement_columns(module)

        def valid_mc(c):
            '''Disallow any measurement column with "," or "|" in its names'''
            return not any([any([bad in f for f in c[:2]]) for bad in ",", "|"])

        self.set_choices([self.make_measurement_choice(c[0], c[1])
                          for c in columns if valid_mc(c)])


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


class TreeChoice(Setting):
    '''A tree choice chooses one path to a leaf in a tree

    Trees are represented as collections of two-tuples. The first element is
    the name of the node and the second is either None if a leaf or
    a sub-collection of two-tuples. For instance:
    (("Foo", (("1", None),("2", None))), ("Bar", None))
    is a tree for selecting ("Foo", "1"), ("Foo", "2") or ("Bar",).

    A good UI choice would be a hierarchical menu.
    '''

    def __init__(self, text, value, tree, fn_is_leaf=None, **kwargs):
        '''Initializer

        text - informative label

        value - the text value, e.g. as encoded by encode_path_parts

        tree - the tree to chose from

        fn_is_leaf - if defined, a function that takes a tree node and
                     returns True if that node is a leaf (a node might
                     have subnodes, but also be a leaf)
        '''
        super(TreeChoice, self).__init__(text, value, **kwargs)
        self.__tree = tree
        self.fn_is_leaf = fn_is_leaf or self.default_fn_is_leaf

    @staticmethod
    def default_fn_is_leaf(node):
        return node[1] is None or len(node[1]) == 0

    def get_path_parts(self):
        '''Split at |, but || escapes to |'''
        result = re.split("(?<!\\|)\\|(?!\\|)", self.get_value_text())
        return [x.replace("||", "|") for x in result]

    @staticmethod
    def encode_path_parts(value):
        '''Return the setting value for a list of menu path parts'''
        return "|".join([x.replace("|", "||") for x in value])

    def get_leaves(self, path=[]):
        '''Get all leaf nodes of a given parent node

        path - the names of nodes traversing the path down the tree
        '''
        current = self.get_tree()
        while len(path) > 0:
            idx = current.index(path[0])
            if idx == -1 or current[idx][1] is None or len(current[idx][1]) == 0:
                return []
            current = current[idx][1]
            path = path[1:]
        return [x[0] for x in current if x[1] is None or len(x[1] == 0)]

    def get_subnodes(self, path=[]):
        '''Get all child nodes that are not leaves for a  given parent

        path - the names of nodes traversing the path down the tree
        '''
        current = self.get_tree()
        while len(path) > 0:
            idx = current.index(path[0])
            if idx == -1 or current[idx][1] is None:
                return []
            current = current[idx][1]
            path = path[1:]
        return [x[0] for x in current if x[1] is not None]

    def get_selected_leaf(self):
        '''Get the leaf node of the tree for the current setting value'''
        tree = self.get_tree()
        node = None
        for item in self.get_path_parts():
            nodes = [n for n in tree if n[0] == item]
            if len(nodes) != 1:
                raise ValidationError("Unable to find command " +
                                      ">".join(self.get_path_parts()), self)
            node = nodes[0]
            tree = node[1]
        return node

    def test_valid(self, pipeline):
        self.get_selected_leaf()

    def get_tree(self):
        if hasattr(self.__tree, "__call__"):
            return self.__tree()
        return self.__tree


class DoSomething(Setting):
    """Do something in response to a button press
    """
    save_to_pipeline = False

    def __init__(self, text, label, callback, *args, **kwargs):
        super(DoSomething, self).__init__(text, 'n/a', **kwargs)
        self.__label = label
        self.__callback = callback
        self.__args = args

    def get_label(self):
        """Return the text label for the button"""
        return self.__label

    def set_label(self, label):
        self.__label = label

    label = property(get_label, set_label)

    def on_event_fired(self):
        """Call the callback in response to the user's request to do something"""
        self.__callback(*self.__args)


class DoThings(Setting):
    """Do one of several things, depending on which button is pressed

    This setting consolidates several possible actions into one setting.
    Graphically, it displays as several buttons that are horizontally
    adjacent.
    """
    save_to_pipeline = False

    def __init__(self, text, labels_and_callbacks, *args, **kwargs):
        '''Initializer

        text - text to display to left of setting

        labels_and_callbacks - a sequence of two tuples of button label
        and callback to be called

        All additional function arguments are passed to the callback.
        '''
        super(DoThings, self).__init__(text, "n/a", **kwargs)
        self.__args = tuple(args)
        self.__labels_and_callbacks = labels_and_callbacks

    @property
    def count(self):
        '''The number of things to do

        returns the number of buttons to display = number of actions
        that can be performed.
        '''
        return len(self.__labels_and_callbacks)

    def get_label(self, idx):
        '''Retrieve one of the actions' labels

        idx - the index of the action
        '''
        return self.__labels_and_callbacks[idx][0]

    def set_label(self, idx, label):
        '''Set the label for an action

        idx - the index of the action

        label - the label to display for that action
        '''
        self.__labels_and_callbacks[idx] = \
            (label, self.__labels_and_callbacks[idx][1])

    def on_event_fired(self, idx):
        '''Call the indexed action's callback

        idx - index of the action to fire
        '''
        self.__labels_and_callbacks[idx][1](*self.__args)


class RemoveSettingButton(DoSomething):
    '''A button whose only purpose is to remove something from a list.'''

    def __init__(self, text, label, list, entry, **kwargs):
        super(RemoveSettingButton, self).__init__(text, label,
                                                  lambda: list.remove(entry),
                                                  **kwargs)


class Divider(Setting):
    """The divider setting inserts a vertical space, possibly with a horizontal line, in the GUI"""
    save_to_pipeline = False

    def __init__(self, text="", line=True, doc=None):
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

    def __init__(self, text, object_fn, value="None", *args, **kwargs):
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
            value = 'None'
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

    def get_measurement_object(self):
        '''Return the primary object for the measurement

        This is either "Image" if an image measurement or the name
        of the objects for per-object measurements. Please pardon the
        confusion with get_object_name which is the secondary object
        name, for instance for a measurement Relate.'''
        return self.__object_fn()

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

    def get_category(self, pipeline, object_name=None):
        '''Return the currently chosen category'''
        categories = self.get_category_choices(pipeline, object_name)
        for category in categories:
            if (self.value.startswith(category + '_') or
                        self.value == category):
                return category
        return None

    def get_feature_name_choices(self, pipeline,
                                 object_name=None,
                                 category=None):
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
                         object_name=None,
                         category=None):
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
            if (self.value.startswith(head + '_') or
                        self.value == head):
                return feature_name
        return None

    def get_image_name_choices(self, pipeline,
                               object_name=None,
                               category=None,
                               feature_name=None):
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
                       object_name=None,
                       category=None,
                       feature_name=None):
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
        # 1st pass - accept only exact match
        # 2nd pass - accept part match.
        # This handles things like "OriginalBlue_Nuclei" vs "OriginalBlue"
        # in MeasureImageIntensity.
        #
        for full_match in True, False:
            for image_name in image_names:
                head = '_'.join((category, feature_name, image_name))
                if (not full_match) and self.value.startswith(head + '_'):
                    return image_name
                if self.value == head:
                    return image_name
        return None

    def get_scale_choices(self, pipeline,
                          object_name=None,
                          category=None,
                          feature_name=None,
                          image_name=None):
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
                  object_name=None,
                  category=None,
                  feature_name=None,
                  image_name=None):
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
                                object_name=None,
                                category=None,
                                feature_name=None):
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

    def get_object_name(self, pipeline, object_name=None,
                        category=None,
                        feature_name=None):
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
            if (self.value.startswith(head + '_') or
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
        if (len(sub_object_names) > 0 and image_name is None and
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
            raise ValidationError("%s is not measured for %s" %
                                  (self.value, obname), self)


class Colormap(Choice):
    '''Represents the choice of a colormap'''

    def __init__(self, text, value=DEFAULT, *args, **kwargs):
        try:
            names = list(matplotlib.cm.cmapnames)
        except AttributeError:
            # matplotlib 99 does not have cmapnames
            names = ['Spectral', 'copper', 'RdYlGn', 'Set2', 'summer', 'spring', 'Accent', 'OrRd', 'RdBu', 'autumn',
                     'Set1', 'PuBu', 'Set3', 'gist_rainbow', 'pink', 'binary', 'winter', 'jet', 'BuPu', 'Dark2',
                     'prism', 'Oranges', 'gist_yarg', 'BuGn', 'hot', 'PiYG', 'YlOrBr', 'Reds', 'spectral', 'RdPu',
                     'Greens', 'gist_ncar', 'PRGn', 'gist_heat', 'YlGnBu', 'RdYlBu', 'Paired', 'flag', 'hsv', 'BrBG',
                     'Purples', 'cool', 'Pastel2', 'gray', 'Pastel1', 'gist_stern', 'GnBu', 'YlGn', 'Greys', 'RdGy',
                     'YlOrRd', 'PuOr', 'PuRd', 'gist_gray', 'Blues', 'PuBuGn', 'gist_earth', 'bone']
        names.sort()
        choices = [DEFAULT] + names
        super(Colormap, self).__init__(text, choices, value, *args, **kwargs)


class Color(Setting):
    '''Represents a choice of color

    These are coded in hex unless a valid HTML name is available.
    '''

    def __init(self, text, value="gray", *args, **kwargs):
        super(Color, self).__init(text, value, *args, **kwargs)

    def to_rgb(self):
        value = self.value.replace(" ", "")
        if value.startswith("#") and len(value) >= 7:
            return (int(value[1:3], 16),
                    int(value[3:5], 16),
                    int(value[5:7], 16))
        elif self.colortable.has_key(value.lower()):
            return self.colortable[value.lower()]
        else:
            raise ValueError("Unknown color: " + self.value)

    '''The HTML color table taken from the W3C CSS Color Module Level 3 spec

    http://www.w3.org/TR/2011/REC-css3-color-20110607
    '''
    colortable = {
        "aliceblue": (240, 248, 255),
        "antiquewhite": (250, 235, 215),
        "aqua": (0, 255, 255),
        "aquamarine": (127, 255, 212),
        "azure": (240, 255, 255),
        "beige": (245, 245, 220),
        "bisque": (255, 228, 196),
        "black": (0, 0, 0),
        "blanchedalmond": (255, 235, 205),
        "blue": (0, 0, 255),
        "blueviolet": (138, 43, 226),
        "brown": (165, 42, 42),
        "burlywood": (222, 184, 135),
        "cadetblue": (95, 158, 160),
        "chartreuse": (127, 255, 0),
        "chocolate": (210, 105, 30),
        "coral": (255, 127, 80),
        "cornflowerblue": (100, 149, 237),
        "cornsilk": (255, 248, 220),
        "crimson": (220, 20, 60),
        "cyan": (0, 255, 255),
        "darkblue": (0, 0, 139),
        "darkcyan": (0, 139, 139),
        "darkgoldenrod": (184, 134, 11),
        "darkgray": (169, 169, 169),
        "darkgreen": (0, 100, 0),
        "darkgrey": (169, 169, 169),
        "darkkhaki": (189, 183, 107),
        "darkmagenta": (139, 0, 139),
        "darkolivegreen": (85, 107, 47),
        "darkorange": (255, 140, 0),
        "darkorchid": (153, 50, 204),
        "darkred": (139, 0, 0),
        "darksalmon": (233, 150, 122),
        "darkseagreen": (143, 188, 143),
        "darkslateblue": (72, 61, 139),
        "darkslategray": (47, 79, 79),
        "darkslategrey": (47, 79, 79),
        "darkturquoise": (0, 206, 209),
        "darkviolet": (148, 0, 211),
        "deeppink": (255, 20, 147),
        "deepskyblue": (0, 191, 255),
        "dimgray": (105, 105, 105),
        "dimgrey": (105, 105, 105),
        "dodgerblue": (30, 144, 255),
        "firebrick": (178, 34, 34),
        "floralwhite": (255, 250, 240),
        "forestgreen": (34, 139, 34),
        "fuchsia": (255, 0, 255),
        "gainsboro": (220, 220, 220),
        "ghostwhite": (248, 248, 255),
        "gold": (255, 215, 0),
        "goldenrod": (218, 165, 32),
        "gray": (128, 128, 128),
        "green": (0, 128, 0),
        "greenyellow": (173, 255, 47),
        "grey": (128, 128, 128),
        "honeydew": (240, 255, 240),
        "hotpink": (255, 105, 180),
        "indianred": (205, 92, 92),
        "indigo": (75, 0, 130),
        "ivory": (255, 255, 240),
        "khaki": (240, 230, 140),
        "lavender": (230, 230, 250),
        "lavenderblush": (255, 240, 245),
        "lawngreen": (124, 252, 0),
        "lemonchiffon": (255, 250, 205),
        "lightblue": (173, 216, 230),
        "lightcoral": (240, 128, 128),
        "lightcyan": (224, 255, 255),
        "lightgoldenrodyellow": (250, 250, 210),
        "lightgray": (211, 211, 211),
        "lightgreen": (144, 238, 144),
        "lightgrey": (211, 211, 211),
        "lightpink": (255, 182, 193),
        "lightsalmon": (255, 160, 122),
        "lightseagreen": (32, 178, 170),
        "lightskyblue": (135, 206, 250),
        "lightslategray": (119, 136, 153),
        "lightslategrey": (119, 136, 153),
        "lightsteelblue": (176, 196, 222),
        "lightyellow": (255, 255, 224),
        "lime": (0, 255, 0),
        "limegreen": (50, 205, 50),
        "linen": (250, 240, 230),
        "magenta": (255, 0, 255),
        "maroon": (128, 0, 0),
        "mediumaquamarine": (102, 205, 170),
        "mediumblue": (0, 0, 205),
        "mediumorchid": (186, 85, 211),
        "mediumpurple": (147, 112, 219),
        "mediumseagreen": (60, 179, 113),
        "mediumslateblue": (123, 104, 238),
        "mediumspringgreen": (0, 250, 154),
        "mediumturquoise": (72, 209, 204),
        "mediumvioletred": (199, 21, 133),
        "midnightblue": (25, 25, 112),
        "mintcream": (245, 255, 250),
        "mistyrose": (255, 228, 225),
        "moccasin": (255, 228, 181),
        "navajowhite": (255, 222, 173),
        "navy": (0, 0, 128),
        "oldlace": (253, 245, 230),
        "olive": (128, 128, 0),
        "olivedrab": (107, 142, 35),
        "orange": (255, 165, 0),
        "orangered": (255, 69, 0),
        "orchid": (218, 112, 214),
        "palegoldenrod": (238, 232, 170),
        "palegreen": (152, 251, 152),
        "paleturquoise": (175, 238, 238),
        "palevioletred": (219, 112, 147),
        "papayawhip": (255, 239, 213),
        "peachpuff": (255, 218, 185),
        "peru": (205, 133, 63),
        "pink": (255, 192, 203),
        "plum": (221, 160, 221),
        "powderblue": (176, 224, 230),
        "purple": (128, 0, 128),
        "red": (255, 0, 0),
        "rosybrown": (188, 143, 143),
        "royalblue": (65, 105, 225),
        "saddlebrown": (139, 69, 19),
        "salmon": (250, 128, 114),
        "sandybrown": (244, 164, 96),
        "seagreen": (46, 139, 87),
        "seashell": (255, 245, 238),
        "sienna": (160, 82, 45),
        "silver": (192, 192, 192),
        "skyblue": (135, 206, 235),
        "slateblue": (106, 90, 205),
        "slategray": (112, 128, 144),
        "slategrey": (112, 128, 144),
        "snow": (255, 250, 250),
        "springgreen": (0, 255, 127),
        "steelblue": (70, 130, 180),
        "tan": (210, 180, 140),
        "teal": (0, 128, 128),
        "thistle": (216, 191, 216),
        "tomato": (255, 99, 71),
        "turquoise": (64, 224, 208),
        "violet": (238, 130, 238),
        "wheat": (245, 222, 179),
        "white": (255, 255, 255),
        "whitesmoke": (245, 245, 245),
        "yellow": (255, 255, 0),
        "yellowgreen": (154, 205, 50),
        #
        # Colors defined in wxPython-src-2.8.12.1/src/common/gdicmn.cpp
        # that are not in the spec.
        #
        "lightmagenta": (255, 0, 255),
        "mediumgrey": (100, 100, 100),
        "mediumforestgreen": (107, 142, 35),
        "mediumgoldenrod": (234, 234, 173)
    }


class Filter(Setting):
    '''A filter that can be applied to an object

    A filter returns a value when applied to an object such as a string
    which is evaluated as either True (accept it) or False (reject it).

    The setting value is composed of tokens with a scheme-like syntax:

    (and (filename contains "_w1_") (extension is "tif"))

    Each predicate has a symbolic name which is used to find it. The predicate
    has an evaluation function and a display name. Predicates also have lists
    of the predicates that they operate on. The leftmost predicate takes two
    arguments. Other predicates, it is up to the developer. Predicates
    are called with the object of interest as the first argument and the
    evaluation value of the predicate to the right as the second argument.

    For something like "filename contains "foo"", "contains" returns a function
    that returns true if the first argument is "foo" and "filename" parses
    the first of its arguments to get the filename and returns the result of
    applying the result of "contains" to the filename.

    There are three special predicates:
    "and", "or" and "literal".
    '''

    class FilterPredicate(object):
        def __init__(self, symbol, display_name, function, subpredicates,
                     doc=None):
            self.symbol = symbol
            self.display_name = display_name
            self.function = function
            self.subpredicates = subpredicates
            self.doc = doc

        def __call__(self, *args, **kwargs):
            return self.function(*args, **kwargs)

        def test_valid(self, pipeline, *args):
            '''Try running the filter on a test string'''
            self("", *args)

        @classmethod
        def encode_symbol(cls, symbol):
            '''Escape encode an abritrary symbol name

            The parser needs to have special characters escaped. These are
            backslash, open and close parentheses, space and double quote.
            '''
            return re.escape(symbol)

        @classmethod
        def decode_symbol(cls, symbol):
            '''Decode an escape-encoded symbol'''
            s = ''
            in_escape = False
            for c in symbol:
                if in_escape:
                    in_escape = False
                    s += c
                elif c == '\\':
                    in_escape = True
                else:
                    s += c
            return s

    class CompoundFilterPredicate(FilterPredicate):
        def test_valid(self, pipeline, *args):
            for subexp in args:
                subexp[0].test_valid(pipeline, *subexp[1:])

    @classmethod
    def eval_list(cls, fn, x, *args):
        results = [v for v in [arg[0](x, *arg[1:]) for arg in args]
                   if v is not None]
        if len(results) == 0:
            return None
        return fn(results)

    AND_PREDICATE = CompoundFilterPredicate(
            "and", "All",
            lambda x, *l: Filter.eval_list(all, x, *l), list,
            doc="All subordinate rules must be satisfied")
    OR_PREDICATE = CompoundFilterPredicate(
            "or", "Any",
            lambda x, *l: Filter.eval_list(any, x, *l), list,
            doc="Any one of the subordinate rules must be satisfied")
    LITERAL_PREDICATE = FilterPredicate(
            "literal", "Custom value", None, [],
            doc="Enter the rule's text")
    CONTAINS_PREDICATE = FilterPredicate(
            "contain", "Contain",
            lambda x, y: x.find(y) >= 0, [LITERAL_PREDICATE],
            doc="The element must contain the text that you enter to the right")
    STARTS_WITH_PREDICATE = FilterPredicate(
            "startwith", "Start with",
            lambda x, y: x.startswith(y), [LITERAL_PREDICATE],
            doc="The element must start with the text that you enter to the right")
    ENDSWITH_PREDICATE = FilterPredicate(
            "endwith", "End with",
            lambda x, y: x.endswith(y), [LITERAL_PREDICATE],
            doc="The element must end with the text that you enter to the right")

    class RegexpFilterPredicate(FilterPredicate):
        def __init__(self, display_name, subpredicates):
            super(self.__class__, self).__init__(
                    "containregexp", display_name, self.regexp_fn, subpredicates,
                    doc="The element must contain a match for the regular expression that you enter to the right")

        def regexp_fn(self, x, y):
            try:
                pattern = re.compile(y)
            except:
                raise ValueError("Badly formatted regular expression: %s" % y)
            return pattern.search(x) is not None

    CONTAINS_REGEXP_PREDICATE = RegexpFilterPredicate(
            "Contain regular expression", [LITERAL_PREDICATE])
    EQ_PREDICATE = FilterPredicate(
            "eq", "Exactly match", lambda x, y: x == y, [LITERAL_PREDICATE],
            doc="Must exactly match the text that you enter to the right")

    class DoesPredicate(FilterPredicate):
        '''Pass the arguments through (no-op)'''
        SYMBOL = "does"

        def __init__(self, subpredicates, text="Does",
                     doc="The rule passes if the condition to the right holds"):
            super(self.__class__, self).__init__(
                    self.SYMBOL, text,
                    lambda x, f, *l: f(x, *l), subpredicates,
                    doc=doc)

    class DoesNotPredicate(FilterPredicate):
        '''Negate the result of the arguments'''
        SYMBOL = "doesnot"

        def __init__(self, subpredicates, text="Does not",
                     doc="The rule fails if the condition to the right holds"):
            super(self.__class__, self).__init__(
                    self.SYMBOL, text,
                    lambda x, f, *l: not f(x, *l), subpredicates,
                    doc=doc)

    def __init__(self, text, predicates, value="", **kwargs):
        super(self.__class__, self).__init__(text, value, **kwargs)
        self.predicates = predicates
        self.cached_token_string = None
        self.cached_tokens = None

    def evaluate(self, x):
        '''Evaluate the value passed using the predicates'''
        try:
            tokens = self.parse()
            return tokens[0](x, *tokens[1:])
        except:
            return False

    def parse(self):
        '''Parse the value into filter predicates, literals and lists

        Returns the value of the text as a list.
        '''
        s = self.value_text
        if s == self.cached_token_string:
            return self.cached_tokens
        tokens = []
        predicates = self.predicates
        while len(s) > 0:
            token, s, predicates = self.parse_token(s, predicates)
            tokens.append(token)
        self.cached_tokens = list(tokens)
        self.cached_token_string = self.value_text
        return tokens

    def default(self, predicates=None):
        '''A default list of tokens to use if things go horribly wrong

        We need to be able to generate a default list of tokens if the
        pipeline has been corrupted and the text can't be parsed.
        '''
        tokens = []
        if predicates is None:
            predicates = self.predicates
        while len(predicates) > 0:
            token = predicates[0]
            if token is self.LITERAL_PREDICATE:
                tokens.append("")
                predicates = self.LITERAL_PREDICATE.subpredicates
            else:
                tokens.append(token)
                predicates = token.subpredicates
        return tokens

    @classmethod
    def parse_token(cls, s, predicates):
        '''Parse a token out of the front of the string

        Returns the next token in the string, the rest of the string
        and the acceptable tokens for the rest of the string.
        '''
        orig_predicates = predicates
        if list in predicates:
            needs_list = True
            predicates = list(predicates)
            predicates.remove(list)
        else:
            needs_list = False
        if s[0] == "(":
            if not needs_list:
                raise ValueError("List not allowed in current context")
            s = s[1:]
            result = []
            while s[0] != ")":
                token, s, predicates = cls.parse_token(s, predicates)
                result.append(token)
            if len(s) > 1 and s[1] == ' ':
                return result, s[2:], orig_predicates
            return result, s[1:], orig_predicates
        elif needs_list:
            raise ValueError("List required in current context")
        if s[0] == "\"":
            if cls.LITERAL_PREDICATE not in predicates:
                raise ValueError("Literal not allowed in current context")
            escape_next = False
            result = ""
            for i in range(1, len(s)):
                if escape_next:
                    result += s[i]
                    escape_next = False
                elif s[i] == "\\":
                    escape_next = True
                elif s[i] == "\"":
                    return result, s[(i + 1):], []
                else:
                    result += s[i]
            raise ValueError("Unterminated literal")
        #
        # (?:\\.|[^ )]) matches either backslash-anything or anything but
        # space and parentheses. So you can have variable names with spaces
        # and that's needed for arbitrary metadata names
        #
        match = re.match(r"^((?:\\.|[^ )])+) ?(.*)$", s)
        if match is None:
            kwd = s
            rest = ""
        else:
            kwd, rest = match.groups()
        kwd = Filter.FilterPredicate.decode_symbol(kwd)
        if kwd == cls.AND_PREDICATE.symbol:
            match = cls.AND_PREDICATE
        elif kwd == cls.OR_PREDICATE.symbol:
            match = cls.OR_PREDICATE
        else:
            matches = [x for x in predicates
                       if x is not list and x.symbol == kwd]
            if len(matches) == 0:
                raise ValueError('The filter predicate, "%s", was not in the list of allowed predicates ("%s")' %
                                 (kwd, '","'.join([x.symbol for x in predicates])))
            match = matches[0]
        if match.subpredicates is list:
            predicates = [list] + predicates
        elif match.subpredicates is not None:
            predicates = match.subpredicates
        return match, rest, predicates

    @classmethod
    def encode_literal(cls, literal):
        '''Encode a literal value with backslash escapes'''
        return literal.replace("\\", "\\\\").replace('"', '\\"')

    def build(self, structure):
        '''Build the textual representation of a filter from its structure

        structure: the processing structure, represented using a nested list.

        The top layer of the list corresponds to the tokens in the value
        string. For instance, a list of [foo, bar, baz] where foo, bar and baz
        are filter predicates that have symbolic names of "foo", "bar" and "baz"
        will yield the string, "foo bar baz". The list [foo, bar, "baz"] will
        treat "baz" as a literal and yield the string, 'foo bar "baz"'.

        Nesting can be done using nested lists. For instance,

        [or [eq "Hello"] [eq "World"]]

        becomes

        "or (eq "Hello")(eq "World")"

        The function sets the filter's value using the generated string.
        '''
        self.value = self.build_string(structure)

    @classmethod
    def build_string(cls, structure):
        '''Return the text representation of structure

        This is a helper function for self.build. See self.build's
        documentation.
        '''
        s = []
        for element in structure:
            if isinstance(element, Filter.FilterPredicate):
                s.append(
                        cls.FilterPredicate.encode_symbol(unicode(element.symbol)))
            elif isinstance(element, basestring):
                s.append(u'"' + cls.encode_literal(element) + u'"')
            else:
                s.append(u"(" + cls.build_string(element) + ")")
        return u" ".join(s)

    def test_valid(self, pipeline):
        try:
            import javabridge as J
            J.run_script("""
            importPackage(Packages.org.cellprofiler.imageset.filter);
            new Filter(expr, klass);
            """, dict(expr=self.value_text,
                      klass=J.class_for_name(
                              "org.cellprofiler.imageset.ImagePlaneDetailsStack")))
        except Exception, e:
            raise ValidationError(str(e), self)

    def test_setting_warnings(self, pipeline):
        '''Warn on empty literal token
        '''
        super(Filter, self).test_setting_warnings(pipeline)
        self.__warn_if_blank(self.parse())

    def __warn_if_blank(self, l):
        for x in l:
            if isinstance(x, (list, tuple)):
                self.__warn_if_blank(x)
            elif x == "":
                raise ValidationError(
                        "The text entry for an expression in this filter is blank",
                        self)


class FileCollectionDisplay(Setting):
    '''A setting to be used to display directories and their files

    The FileCollectionDisplay shows directory trees with mechanisms to
    communicate directory additions and deletions to its parent module.

    The central data structure is the dictionary, "self.file_tree". The keys
    for the top-level of the dictionary are the directories managed by the
    setting. If a key represents a directory, its value is another directory.
    If a key represents a file, its value is either True (the file is included
    in the collection) or False (the file is filtered out of the collection).

    Directory dictionaries can be filtered: this is done by setting the
    special key, "None" to either True or False.

    The FileCollectionDisplay manages the tree and it should be treated as
    read-only by callers. Callers can request that nodes be added, removed,
    filtered or not filtered by calling the appropriate notification function
    with a nested collection of two-tuples and strings (modpaths). Two-tuples
    represent directories whose subdirectories or files are being operated on.
    Strings represent directories or files that are being operated on. The first
    element of the two-tuple is the directory name and the second is a
    sub-collection of two-tuples. For instance, to operate on foo/bar, send:

    ("foo", ("bar", ))

    The FileCollectionDisplay communicates events on individual files or
    directories by specifying a path as a collection of path parts. These
    can be any sort of object and it is the caller's job to maintain the
    display names of each of them and their node categories (used for
    icon display).
    '''
    ADD = "ADD"
    REMOVE = "REMOVE"
    METADATA = "METADATA"
    NODE_DIRECTORY = "directory"
    NODE_COMPOSITE_IMAGE = "compositeimage"
    NODE_COLOR_IMAGE = "colorimage"
    NODE_MONOCHROME_IMAGE = "monochromeimage"
    NODE_IMAGE_PLANE = "imageplane"
    NODE_MOVIE = "movie"
    NODE_FILE = "file"
    NODE_CSV = "csv"
    BKGND_PAUSE = "pause"
    BKGND_RESUME = "resume"
    BKGND_STOP = "stop"
    BKGND_GET_STATE = "getstate"

    class DeleteMenuItem(object):
        '''A placeholder in the context menu for the delete command

        The DeleteMenuItem can be placed in the context menu returned
        by fn_get_path_info so that the user can delete the selected items
        from the context menu.

        text - the text to display in the context menu
        '''

        def __init__(self, text):
            self.text = text

    def __init__(self, text, value,
                 fn_on_drop,
                 fn_on_remove,
                 fn_get_path_info,
                 fn_on_menu_command,
                 fn_on_bkgnd_control,
                 hide_text="Hide filtered files", **kwargs):
        '''Constructor

        text - the label to the left of the setting

        value - the value for the control. This is a serialization of
                the appearance (for instance, whether to show or hide
                filtered files).

        fn_on_drop - called when files are dropped. The signature is
                     fn_on_drop(pathnames, check_for_directories) The first
                     argument is a list of pathnames of the dropped files.
                     The second argument is True if the user has performed
                     a file name drop which might include directories and
                     False if the user has dropped text file names.

        fn_on_remove - called when the UI requests that files be removed. Has
                       one argument which is a collection of paths to remove.

        fn_get_path_info - called when the UI needs to know the display name,
                     icon type, context menu and tool tip for an item. These
                     are returned in a four-tuple by the callee, e.g:
                     [ "image.tif", NODE_MONOCHROME_IMAGE,
                       "image of well A01 on plate P-12345",
                       ( "Show image", "Show metadata", "Delete image")]

        fn_on_menu_command - called when the user selects a context menu
                     command. The argument is the text from the context menu or
                     None if the default command.

        fn_on_bkgnd_control - called when the UI wants to stop, pause or resume
                     all background processing. BKGND_PAUSE asks for the
                     caller to pause processing, BKGND_RESUME asks for the
                     caller to resume, BKGND_STOP asks for processing to be
                     aborted, BKGND_GET_STATE asks for the caller to
                     return its current state = BKGND_PAUSE if it is paused,
                     BKGND_RESUME if it is running or BKGND_STOP if it is
                     idle.

        hide_text - the text displayed next to the hide checkbox.
        '''
        super(self.__class__, self).__init__(text, value, **kwargs)
        self.fn_on_drop = fn_on_drop
        self.fn_on_remove = fn_on_remove
        self.fn_get_path_info = fn_get_path_info
        self.fn_on_menu_command = fn_on_menu_command
        self.fn_on_bkgnd_control = fn_on_bkgnd_control
        self.hide_text = hide_text
        self.fn_update = None
        self.file_tree = {}
        self.properties = {self.SHOW_FILTERED: True}
        try:
            properties = json.loads(value)
            if isinstance(properties, dict):
                self.properties.update(properties)
        except:
            pass

    SHOW_FILTERED = "ShowFiltered"

    def update_value(self):
        '''Update the setting value after changing a property'''
        self.value_text = json.dumps(self.properties)

    def update_ui(self, cmd=None, mods=None):
        if self.fn_update is not None:
            self.fn_update(cmd, mods)

    def set_update_function(self, fn_update=None):
        '''Set the function that will be called when the file_tree is updated'''
        self.fn_update = fn_update

    def initialize_tree(self, mods):
        '''Remove all nodes in the file tree'''
        self.file_tree = {}
        self.add_subtree(mods, self.file_tree)

    def add(self, mods):
        '''Add nodes to the file tree

        mods - modification structure. See class documentation for its form.
        '''
        self.add_subtree(mods, self.file_tree)
        self.update_ui(self.ADD, mods)

    def modify(self, mods):
        '''Indicate a minor modification such as metadtaa change

        mods - modification structure. See class documentation for its form.
        '''
        self.update_ui(self.METADATA, mods)

    @classmethod
    def is_leaf(cls, mod):
        '''True if the modification structure is the leaf of a tree

        The leaves are either strings representing the last part of a path
        or 3-tuples representing image planes within an image file. Branches
        are two-tuples composed of a path part and more branches / leaves
        '''
        return len(mod) != 2 or not isinstance(mod[0], basestring)

    def node_count(self, file_tree=None):
        '''Count the # of nodes (leaves + directories) in the tree'''
        if file_tree is None:
            file_tree = self.file_tree
        count = 0
        for key in file_tree.keys():
            if key is None:
                pass
            elif isinstance(file_tree[key], dict):
                count += 1 + self.node_count(file_tree[key])
            else:
                count += 1
        return count

    def get_tree_modpaths(self, path):
        '''Create a modpath containing the selected node and all children

        root - list of paths to the selected node

        returns a modpath (two-tuples where the first is the key and the second
        is a list of sub-modpaths)
        '''
        tree = self.file_tree
        root_modlist = sub_modlist = []
        while len(path) > 1:
            next_sub_modlist = []
            sub_modlist.append((path[0], next_sub_modlist))
            tree = tree[path[0]]
            path = path[1:]
            sub_modlist = next_sub_modlist
        if isinstance(tree[path[0]], dict):
            sub_modlist.append((path[0], self.get_all_modpaths(tree[path[0]])))
        else:
            sub_modlist.append(path[0])
        return root_modlist[0]

    def get_all_modpaths(self, tree):
        '''Get all sub-modpaths from the branches of the given tree'''
        result = []
        for key in tree.keys():
            if key is None:
                continue
            elif not isinstance(tree[key], dict):
                result.append(key)
            else:
                result.append((key, self.get_all_modpaths(tree[key])))
        return result

    def add_subtree(self, mods, tree):
        for mod in mods:
            if self.is_leaf(mod):
                if not tree.has_key(mod):
                    tree[mod] = True
            else:
                if tree.has_key(mod[0]) and isinstance(tree[mod[0]], dict):
                    subtree = tree[mod[0]]
                else:
                    subtree = tree[mod[0]] = {}
                subtree[None] = True
                self.add_subtree(mod[1], subtree)

    def on_remove(self, mods):
        '''Called when the UI wants to remove nodes

        mods - a modlist of nodes to remove
        '''
        self.fn_on_remove(mods)

    def remove(self, mods):
        '''Remove nodes from the file tree

        mods - modification structure. See class documentation for its form.
        '''
        for mod in mods:
            self.remove_subtree(mod, self.file_tree)
        self.update_ui(self.REMOVE, mods)

    def remove_subtree(self, mod, tree):
        if not (isinstance(mod, tuple) and len(mod) == 2):
            if tree.has_key(mod):
                subtree = tree[mod]
                if isinstance(subtree, dict):
                    #
                    # Remove whole tree
                    #
                    for key in subtree.keys():
                        if key is None:
                            continue
                        if isinstance(subtree[key], dict):
                            self.remove_subtree(key, subtree)
                del tree[mod]
        elif tree.has_key(mod[0]):
            root_mod = mod[0]
            subtree = tree[root_mod]
            if isinstance(subtree, dict):
                for submod in mod[1]:
                    self.remove_subtree(submod, subtree)
                #
                # Delete the subtree if the subtree is emptied
                #
                if len(subtree) == 0 or (
                                len(subtree) == 1 and subtree.has_key(None)):
                    del tree[root_mod]
            else:
                del tree[root_mod]

    def mark(self, mods, keep):
        '''Mark tree nodes as filtered in or out

        mods - modification structure. See class documentation for its form.

        keep - true to mark a node as in the set, false to filter it out.
        '''
        self.mark_subtree(mods, keep, self.file_tree)
        self.update_ui()

    def mark_subtree(self, mods, keep, tree):
        for mod in mods:
            if self.is_leaf(mod):
                if tree.has_key(mod):
                    if isinstance(tree[mod], dict):
                        tree[mod][None] = keep
                    else:
                        tree[mod] = keep
            else:
                if tree.has_key(mod[0]):
                    self.mark_subtree(mod[1], keep, tree[mod[0]])
        kept = [tree[k][None] if isinstance(tree[k], dict)
                else tree[k]
                for k in tree.keys() if k is not None]
        tree[None] = any(kept)

    def get_node_info(self, path):
        '''Get the display name, node type and tool tip for a node

        path - path to the image plane as a list of nodes

        returns a tuple of display name, node type and tool tip
        '''
        display_name, node_type, tool_tip, menu = self.fn_get_path_info(path)
        return display_name, node_type, tool_tip

    def get_context_menu(self, path):
        '''Get the context menu associated with a path

        path - path to the image plane

        returns a list of context menu items.
        '''
        display_name, node_type, tool_tip, menu = self.fn_get_path_info(path)
        return menu

    def get_show_filtered(self):
        return self.properties[self.SHOW_FILTERED]

    def set_show_filtered(self, show_state):
        '''Mark that we should show filtered files in the user interface

        show_state - true to show files / false to hide them
        '''
        self.properties[self.SHOW_FILTERED] = show_state
        self.update_value()
        self.update_ui()

    show_filtered = property(get_show_filtered, set_show_filtered)


class PathListDisplay(Setting):
    '''This setting's only purpose is to signal that the path list should be shown

    Set self.using_filter to True if the module knows that the path list will
    be filtered or if the module doesn't know. Set it to False if the module
    knows the path list won't be filtered.
    '''

    def __init__(self):
        super(self.__class__, self).__init__(
                "", value="")
        self.using_filter = True


class PathListRefreshButton(DoSomething):
    '''A setting that displays as a button which refreshes the path list'''

    def __init__(self, text, label, *args, **kwargs):
        DoSomething.__init__(self, text, label, self.fn_callback, *args, **kwargs)
        # callback set by module view
        self.callback = None

    def fn_callback(self, *args, **kwargs):
        if self.callback is not None:
            self.callback(*args, **kwargs)


class ImageSetDisplay(DoSomething):
    '''A button that refreshes the image set display when pressed

    '''

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(args[0], args[1], None, *args[:2],
                                             **kwargs)


class Table(Setting):
    '''The Table setting displays a table of values'''

    ATTR_ERROR = "Error"

    def __init__(self, text,
                 min_size=(400, 300),
                 max_field_size=30,
                 use_sash=False,
                 corner_button=None,
                 **kwargs):
        '''Constructor

        text - text label to display to the left of the table
        min_size - initial size of the table before user stretches it
        max_field_size - any field with more than this # of characters will
                         be truncated using an ellipsis.
        use_sash - if True, place the table in the bottom resizable sash.
                   if False, place the table inline
        corner_button - if defined, consists of keyword arguments for the corner
                        button mixin: dict(fn_clicked=<function>, label=<label>,
                        tooltip=<tooltip>)
        '''
        super(self.__class__, self).__init__(text, "", **kwargs)
        self.column_names = []
        self.data = []
        self.row_attributes = {}
        self.cell_attributes = {}
        self.min_size = min_size
        self.max_field_size = max_field_size
        self.use_sash = use_sash
        self.corner_button = corner_button

    def insert_column(self, index, column_name):
        '''Insert a column at the given index

        index - the zero-based index of the column's position

        column_name - the name of the column

        Adds the column to the table and sets the value for any existing
        rows to None.
        '''
        self.column_names.insert(index, column_name)
        for row in self.data:
            row.insert(index, None)

    def add_rows(self, columns, data):
        '''Add rows to the table

        columns - define the columns for each row of data

        data - rows of data to add. Each field in a row is placed
               at the column indicated by "columns"
        '''
        indices = [columns.index(c) if c in columns else None
                   for c in self.column_names]
        for row in data:
            self.data.append([None if index is None else row[index]
                              for index in indices])

    def sort_rows(self, columns):
        '''Sort rows based on values in columns'''
        indices = [self.column_names.index(c) for c in columns]

        def compare_fn(row1, row2):
            for index in indices:
                x = cmp(row1[index], row2[index])
                if x != 0:
                    return x
            return 0

        self.data.sort(compare_fn)

    def clear_rows(self):
        self.data = []
        self.row_attributes = {}
        self.cell_attributes = {}

    def clear_columns(self):
        self.column_names = []

    def get_data(self, row_index, columns):
        '''Get the column values for a given row or rows

        row_index - can either be the index of one row or can be a slice or list
                    of rows

        columns - the names of the columns to fetch, in the order they will
                  appear in the row
        '''
        column_indices = [self.column_names.index(c) for c in columns]
        if isinstance(row_index, int):
            row_index = slice(row_index, row_index + 1)
        return [[row[ci] for ci in column_indices] for row in self.data[row_index]]

    def set_row_attribute(self, row_index, attribute, set_attribute=True):
        '''Set an attribute on a row

        row_index - index of row in question

        attribute - one of the ATTR_ values, for instance ATTR_ERROR

        set_attribute - True to set, False to clear
        '''
        if set_attribute:
            if self.row_attributes.has_key(row_index):
                self.row_attributes[row_index].add(attribute)
            else:
                self.row_attributes[row_index] = set([attribute])
        else:
            if self.row_attributes.has_key(row_index):
                s = self.row_attributes[row_index]
                s.remove(attribute)
                if len(s) == 0:
                    del self.row_attributes[row_index]

    def get_row_attributes(self, row_index):
        '''Get the set of attributes on a row

        row_index - index of the row being queried

        returns None if no attributes or a set of attributes set on the row
        '''
        return self.row_attributes.get(row_index, None)

    def set_cell_attribute(self, row_index, column_name,
                           attribute, set_attribute=True):
        '''Set an attribute on a cell

        row_index - index of row in question

        column_name - name of the cell's column

        attribute - one of the ATTR_ values, for instance ATTR_ERROR

        set_attribute - True to set, False to clear
        '''
        key = (row_index, self.column_names.index(column_name))
        if set_attribute:
            if self.cell_attributes.has_key(key):
                self.cell_attributes[key].add(attribute)
            else:
                self.cell_attributes[key] = set([attribute])
        else:
            if self.cell_attributes.has_key(key):
                s = self.cell_attributes[key]
                s.remove(attribute)
                if len(s) == 0:
                    del self.cell_attributes[key]

    def get_cell_attributes(self, row_index, column_name):
        '''Get the set of attributes on a row

        row_index - index of the row being queried

        returns None if no attributes or a set of attributes set on the row
        '''
        key = (row_index, self.column_names.index(column_name))
        return self.cell_attributes.get(key, None)


class HTMLText(Setting):
    '''The HTMLText setting displays a HTML control with content

    '''

    def __init__(self, text, content="", size=None, **kwargs):
        '''Initialize with the html content

        text - the text to the right of the setting

        content - the HTML to display

        size - a (x,y) tuple of the minimum window size in units of
               wx.SYS_CAPTION_Y (the height of the window caption).
        '''
        super(self.__class__, self).__init__(text, "", **kwargs)
        self.content = content
        self.size = size


class Joiner(Setting):
    '''The joiner setting defines a joining condition between conceptual tables

    You might want to join several tables by specifying the columns that match
    each other or might want to join images in an image set by matching
    their metadata. The joiner takes a dictionary of lists of column names
    or metadata keys where the dictionary key holds the table or image name
    and the list of values holds the names of table columns or metadata keys.

    The joiner's value is, conceptually, a list of dictionaries where each
    dictionary in the list documents how to join one column or metadata key
    in one of the tables or images to the others.

    The conceptual value is a list of dictionaries of unicode string keys
    and values (or value = None). This can be encoded using str() and
    can be decoded using eval.
    '''

    def __init__(self, text, value="[]", allow_none=True, **kwargs):
        '''Initialize the joiner

        text - label to the left of the joiner

        value - "repr" done on the joiner's underlying structure which is
                a list of dictionaries

        allow_none - True (by default) to allow one of the entities to have
                     None for a join, indicating that it matches against
                     everything
        '''
        super(self.__class__, self).__init__(text, value, **kwargs)
        self.entities = {}
        self.allow_none = allow_none

    def parse(self):
        '''Parse the value into a list of dictionaries

        return a list of dictionaries where the key is the table or image name
        and the value is the column or metadata
        '''
        return eval(self.value_text, {"__builtins__": None}, {})

    def default(self):
        '''Concoct a default join as a guess if setting is uninitialized'''
        all_names = {}
        best_name = None
        best_count = 0
        for value_list in self.entities.values():
            for value in value_list:
                if all_names.has_key(value):
                    all_names[value] += 1
                else:
                    all_names[value] = 1
                if best_count < all_names[value]:
                    best_count = all_names[value]
                    best_name = value
        if best_count == 0:
            return []
        else:
            return [dict([(k, best_name if best_name in self.entities[k]
            else None) for k in self.entities.keys()])]

    def build(self, dictionary_list):
        '''Build a value from a list of dictionaries'''
        self.value = self.build_string(dictionary_list)

    @classmethod
    def build_string(cls, dictionary_list):
        return str(dictionary_list)

    def test_valid(self, pipeline):
        '''Test the joiner setting to ensure that the join is supported

        '''
        join = self.parse()
        if len(join) == 0:
            raise ValidationError(
                    "This setting needs to be initialized by choosing items from each column",
                    self)
        for d in join:
            for column_name, value in d.items():
                if column_name in self.entities and \
                        (value not in self.entities[column_name] and
                                 value is not None):
                    raise ValidationError(
                            "%s is not a valid choice for %s" %
                            (value, column_name), self)


class DataTypes(Setting):
    '''The DataTypes setting assigns data types to measurement names

    Imported or extracted metadata might be textual or numeric and
    that interpretation should be up to the user. This setting lets
    the user pick the data type for their metadata.
    '''
    DT_TEXT = "text"
    DT_INTEGER = "integer"
    DT_FLOAT = "float"
    DT_NONE = "none"

    def __init__(self, text, value="{}", name_fn=None, *args, **kwargs):
        '''Initializer

        text - description of the setting

        value - initial value (a json-encodable key/value dictionary)

        name_fn - a function that returns the current list of feature names
        '''
        super(DataTypes, self).__init__(text, value, *args, **kwargs)

        self.__name_fn = name_fn

    def get_data_types(self):
        '''Get a dictionary of the data type for every name

        Using the name function, if present, create a dictionary of name
        to data type (DT_TEXT / INTEGER / FLOAT / NONE)
        '''
        result = json.loads(self.value_text)
        if self.__name_fn is not None:
            for name in self.__name_fn():
                if name not in result:
                    result[name] = self.DT_TEXT
        return result

    @staticmethod
    def decode_data_types(s):
        return json.loads(s)

    @staticmethod
    def encode_data_types(d):
        '''Encode a data type dictionary as a potential value for this setting'''
        return json.dumps(d)


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
        assert name not in self.__dict__, "%s already in SettingsGroup (previous setting or built in attribute)" % (
            name)
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

    def __init__(self, old_value, new_value):
        self.__old_value = old_value
        self.__new_value = new_value

    def get_old_value(self):
        return self.__old_value

    old_value = property(get_old_value)

    def get_new_value(self):
        return self.__new_value

    new_value = property(get_new_value)


class BeforeChangeSettingEvent(ChangeSettingEvent):
    """Indicates that a setting is about to change, allows a listener to cancel the change

    """

    def __init__(self, old_value, new_value):
        ChangeSettingEvent.__init__(self, old_value, new_value)
        self.__allow_change = True
        self.__cancel_reason = None

    def cancel_change(self, reason=None):
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

    def __init__(self, old_value, new_value):
        ChangeSettingEvent.__init__(self, old_value, new_value)


class DeleteSettingEvent:
    def __init__(self):
        pass


class ValidationError(ValueError):
    """An exception indicating that a setting's value prevents the pipeline from running
    """

    def __init__(self, message, setting):
        """Initialize with an explanatory message and the setting that caused the problem
        """
        super(ValidationError, self).__init__(message)
        self.__setting = setting

    def get_setting(self):
        """The setting responsible for the problem

        This might be one of several settings partially responsible
        for the problem.
        """
        return self.__setting

    setting = property(get_setting)
