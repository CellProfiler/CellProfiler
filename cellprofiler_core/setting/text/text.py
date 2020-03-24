import logging
import os
import re
import sys

import six

import cellprofiler_core.measurement
import cellprofiler_core.preferences
import cellprofiler_core.setting
import cellprofiler_core.utilities.legacy
from cellprofiler_core.setting._setting import Setting
from cellprofiler_core.setting._validation_error import ValidationError

logger = logging.getLogger(__name__)


class Text(Setting):
    """A setting that displays as an edit box, accepting a string

    """

    def __init__(self, text, value, *args, **kwargs):
        kwargs = kwargs.copy()
        self.multiline_display = kwargs.pop("multiline", False)
        self.metadata_display = kwargs.pop("metadata", False)
        super(Text, self).__init__(text, value, *args, **kwargs)


class DirectoryPath(Text):
    """A setting that displays a filesystem path name
    """

    DIR_ALL = [
        cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME,
        cellprofiler_core.preferences.DEFAULT_INPUT_FOLDER_NAME,
        cellprofiler_core.preferences.DEFAULT_OUTPUT_FOLDER_NAME,
        cellprofiler_core.preferences.DEFAULT_INPUT_SUBFOLDER_NAME,
        cellprofiler_core.preferences.DEFAULT_OUTPUT_SUBFOLDER_NAME,
    ]

    def __init__(
        self,
        text,
        value=None,
        dir_choices=None,
        allow_metadata=True,
        support_urls=False,
        *args,
        **kwargs,
    ):
        if dir_choices is None:
            dir_choices = DirectoryPath.DIR_ALL
        if support_urls and not (cellprofiler_core.preferences.URL_FOLDER_NAME in dir_choices):
            dir_choices = dir_choices + [cellprofiler_core.preferences.URL_FOLDER_NAME]
        if value is None:
            value = DirectoryPath.static_join_string(dir_choices[0], "")
        self.dir_choices = dir_choices
        self.allow_metadata = allow_metadata
        self.support_urls = support_urls
        super(DirectoryPath, self).__init__(text, value, *args, **kwargs)

    def split_parts(self):
        """Return the directory choice and custom path as a tuple"""
        result = tuple(self.value.split("|", 1))
        if len(result) == 1:
            result = (result[0], ".")
        return result

    @staticmethod
    def split_string(value):
        return tuple(value.split("|", 1))

    def join_parts(self, dir_choice=None, custom_path=None):
        """Join the directory choice and custom path to form a value"""
        self.value = self.join_string(dir_choice, custom_path)

    def join_string(self, dir_choice=None, custom_path=None):
        """Return the value string composed of a directory choice & path"""
        return self.static_join_string(
            dir_choice if dir_choice is not None else self.dir_choice,
            custom_path if custom_path is not None else self.custom_path,
        )

    @staticmethod
    def static_join_string(dir_choice, custom_path):
        return "|".join((dir_choice, custom_path))

    @staticmethod
    def upgrade_setting(value):
        dir_choice, custom_path = DirectoryPath.split_string(value)
        dir_choice = cellprofiler_core.preferences.standardize_default_folder_names(
            [dir_choice], 0
        )[0]
        return DirectoryPath.static_join_string(dir_choice, custom_path)

    def get_dir_choice(self):
        """The directory selection method"""
        return self.split_parts()[0]

    def set_dir_choice(self, choice):
        self.join_parts(dir_choice=choice)

    dir_choice = property(get_dir_choice, set_dir_choice)

    def get_custom_path(self):
        """The custom path relative to the directory selection method"""
        return self.split_parts()[1]

    def set_custom_path(self, custom_path):
        self.join_parts(custom_path=custom_path)

    custom_path = property(get_custom_path, set_custom_path)

    @property
    def is_custom_choice(self):
        """True if the current dir_choice requires a custom path"""
        return self.dir_choice in [
            cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME,
            cellprofiler_core.preferences.DEFAULT_INPUT_SUBFOLDER_NAME,
            cellprofiler_core.preferences.DEFAULT_OUTPUT_SUBFOLDER_NAME,
            cellprofiler_core.preferences.URL_FOLDER_NAME,
        ]

    def is_url(self):
        return self.dir_choice == cellprofiler_core.preferences.URL_FOLDER_NAME

    def get_absolute_path(self, measurements=None, image_set_number=None):
        """Return the absolute path specified by the setting

        Concoct an absolute path based on the directory choice,
        the custom path and metadata taken from the measurements.
        """
        if self.dir_choice == cellprofiler_core.preferences.DEFAULT_INPUT_FOLDER_NAME:
            return cellprofiler_core.preferences.get_default_image_directory()
        if self.dir_choice == cellprofiler_core.preferences.DEFAULT_OUTPUT_FOLDER_NAME:
            return cellprofiler_core.preferences.get_default_output_directory()
        if self.dir_choice == cellprofiler_core.preferences.DEFAULT_INPUT_SUBFOLDER_NAME:
            root_directory = cellprofiler_core.preferences.get_default_image_directory()
        elif self.dir_choice == cellprofiler_core.preferences.DEFAULT_OUTPUT_SUBFOLDER_NAME:
            root_directory = cellprofiler_core.preferences.get_default_output_directory()
        elif self.dir_choice == cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME:
            root_directory = os.curdir
        elif self.dir_choice == cellprofiler_core.preferences.URL_FOLDER_NAME:
            root_directory = ""
        elif self.dir_choice == cellprofiler_core.preferences.NO_FOLDER_NAME:
            return ""
        else:
            raise ValueError("Unknown directory choice: %s" % self.dir_choice)
        if self.allow_metadata:
            if measurements is not None:
                custom_path = measurements.apply_metadata(
                    self.custom_path, image_set_number
                )
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
        if self.dir_choice == cellprofiler_core.preferences.URL_FOLDER_NAME:
            return custom_path
        path = os.path.join(root_directory, custom_path)
        return os.path.abspath(path)

    def get_parts_from_path(self, path):
        """Figure out how to set up dir_choice and custom path given a path"""
        path = os.path.abspath(path)
        custom_path = self.custom_path
        img_dir = cellprofiler_core.preferences.get_default_image_directory()
        out_dir = cellprofiler_core.preferences.get_default_output_directory()
        if sys.platform.startswith("win"):
            # set to lower-case for comparisons
            cmp_path = path.lower()
            img_dir = img_dir.lower()
            out_dir = out_dir.lower()
        else:
            cmp_path = path
        seps = [os.path.sep]
        if hasattr(os, "altsep"):
            seps += [os.altsep]
        if cmp_path == img_dir:
            dir_choice = cellprofiler_core.preferences.DEFAULT_INPUT_FOLDER_NAME
        elif cmp_path == out_dir:
            dir_choice = cellprofiler_core.preferences.DEFAULT_OUTPUT_FOLDER_NAME
        elif cmp_path.startswith(img_dir) and cmp_path[len(img_dir)] in seps:
            dir_choice = cellprofiler_core.preferences.DEFAULT_INPUT_SUBFOLDER_NAME
            custom_path = path[len(img_dir) + 1 :]
        elif cmp_path.startswith(out_dir) and cmp_path[len(out_dir)] in seps:
            dir_choice = cellprofiler_core.preferences.DEFAULT_OUTPUT_SUBFOLDER_NAME
            custom_path = path[len(out_dir) + 1 :]
        else:
            dir_choice = cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
            custom_path = path
        return dir_choice, custom_path

    def alter_for_create_batch_files(self, fn_alter_path):
        """Call this to alter the setting appropriately for batch execution"""
        custom_path = self.custom_path
        if custom_path.startswith("\g<") and sys.platform.startswith("win"):
            # So ugly, the "\" sets us up for the root directory during
            # os.path.join, so we need r".\\" at start to fake everyone out
            custom_path = r".\\" + custom_path

        if self.dir_choice == cellprofiler_core.preferences.DEFAULT_INPUT_FOLDER_NAME:
            pass
        elif self.dir_choice == cellprofiler_core.preferences.DEFAULT_OUTPUT_FOLDER_NAME:
            pass
        elif self.dir_choice == cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME:
            self.custom_path = fn_alter_path(
                self.custom_path, regexp_substitution=self.allow_metadata
            )
        elif self.dir_choice == cellprofiler_core.preferences.DEFAULT_INPUT_SUBFOLDER_NAME:
            self.custom_path = fn_alter_path(
                self.custom_path, regexp_substitution=self.allow_metadata
            )
        elif self.dir_choice == cellprofiler_core.preferences.DEFAULT_OUTPUT_SUBFOLDER_NAME:
            self.custom_path = fn_alter_path(
                self.custom_path, regexp_substitution=self.allow_metadata
            )

    def test_valid(self, pipeline):
        if self.dir_choice not in self.dir_choices + [
            cellprofiler_core.preferences.NO_FOLDER_NAME
        ]:
            raise ValidationError(
                "Unsupported directory choice: %s" % self.dir_choice, self
            )
        if (
            not self.allow_metadata
            and self.is_custom_choice
            and self.custom_path.find(r"\g<") != -1
        ):
            raise ValidationError("Metadata not supported for this setting", self)
        if self.dir_choice == cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME and (
            (self.custom_path is None) or (len(self.custom_path) == 0)
        ):
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
        if "wildcard" in kwargs:
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
        return any(
            [
                self.value_text.lower().startswith(scheme)
                for scheme in ("http:", "https:", "ftp:")
            ]
        )

    def test_valid(self, pipeline):
        if not self.is_url():
            super(PathnameOrURL, self).test_valid(pipeline)


class AlphanumericText(Text):
    """A setting for entering text values limited to alphanumeric + _ values

    This can be used for measurement names, object names, etc.
    """

    def __init__(self, text, value, *args, **kwargs):
        """Initializer

        text - the explanatory text for the setting UI

        value - the default / initial value

        first_must_be_alpha - True if the first character of the value must
                              be a letter or underbar.
        """
        kwargs = kwargs.copy()
        self.first_must_be_alpha = kwargs.pop("first_must_be_alpha", False)
        super(AlphanumericText, self).__init__(text, value, *args, **kwargs)

    def test_valid(self, pipeline):
        """Restrict names to legal ascii C variables

        First letter = a-zA-Z and underbar, second is that + digit.
        """
        self.validate_alphanumeric_text(self.value, self, self.first_must_be_alpha)

    @staticmethod
    def validate_alphanumeric_text(text, setting, first_must_be_alpha):
        """Validate text as alphanumeric, throwing a validation error if not

        text - text to be validated

        setting - blame this setting on failure

        first_must_be_alpha - True if the first letter has to be alpha or underbar
        """
        if first_must_be_alpha:
            pattern = "^[A-Za-z_][A-Za-z_0-9]*$"
            error = (
                'Names must start with an ASCII letter or underbar ("_")'
                " optionally followed by ASCII letters, underbars or digits."
            )
        else:
            pattern = "^[A-Za-z_0-9]+$"
            error = 'Only ASCII letters, digits and underbars ("_") can be ' "used here"

        match = re.match(pattern, text)
        if match is None:
            raise ValidationError(error, setting)


class Number(Text):
    """A setting that allows only numeric input
    """

    def __init__(self, text, value=0, minval=None, maxval=None, *args, **kwargs):
        if isinstance(value, six.string_types):
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
        str_value = (
            six.text_type(value)
            if isinstance(value, six.string_types)
            else self.value_to_str(value)
        )
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
        """Programatically set the minimum value allowed"""
        self.__minval = minval

    def set_max_value(self, maxval):
        """Programatically set the maximum value allowed"""
        self.__maxval = maxval

    def get_min_value(self):
        """The minimum value (inclusive) that can legally be entered"""
        return self.__minval

    def get_max_value(self):
        """The maximum value (inclusive) that can legally be entered"""
        return self.__maxval

    min_value = property(get_min_value, set_min_value)
    max_value = property(get_max_value, set_max_value)

    def test_valid(self, pipeline):
        """Return true only if the text value is float
        """
        try:
            value = self.str_to_value(self.value_text)
        except ValueError:
            raise ValidationError("Value not in decimal format", self)
        if self.__minval is not None and self.__minval > value:
            raise ValidationError(
                "Must be at least %s, was %s"
                % (self.value_to_str(self.__minval), self.value_text),
                self,
            )
        if self.__maxval is not None and self.__maxval < value:
            raise ValidationError(
                "Must be at most %s, was %s"
                % (self.value_to_str(self.__maxval), self.value_text),
                self,
            )

    def eq(self, x):
        """Equal if our value equals the operand"""
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
        return "%d" % value


class Float(Number):
    """A class that only allows floating point input"""

    def str_to_value(self, str_value):
        return float(str_value)

    def value_to_str(self, value):
        text_value = ("%f" % value).rstrip("0")
        if text_value.endswith("."):
            text_value += "0"
        return text_value


class NameProvider(AlphanumericText):
    """A setting that provides a named object
    """

    def __init__(self, text, group, value="Do not use", *args, **kwargs):
        self.__provided_attributes = {"group": group}
        kwargs = kwargs.copy()
        if "provided_attributes" in kwargs:
            self.__provided_attributes.update(kwargs["provided_attributes"])
            del kwargs["provided_attributes"]
        kwargs["first_must_be_alpha"] = True
        super(NameProvider, self).__init__(text, value, *args, **kwargs)

    def get_group(self):
        """This setting provides a name to this group

        Returns a group name, e.g., imagegroup or objectgroup
        """
        return self.__provided_attributes["group"]

    group = property(get_group)

    @property
    def provided_attributes(self):
        """Return the dictionary of attributes of this provider

        These are things like the group ("objectgroup" for instance) and
        hints about the thing itself, such as that it is an image
        that was loaded from  a file.
        """
        return self.__provided_attributes


class ImageNameProvider(NameProvider):
    """A setting that provides an image name
    """

    def __init__(self, text, value="Do not use", *args, **kwargs):
        super(ImageNameProvider, self).__init__(
            text, "imagegroup", value, *args, **kwargs
        )


class ObjectNameProvider(NameProvider):
    """A setting that provides an image name
    """

    def __init__(self, text, value="Do not use", *args, **kwargs):
        super(ObjectNameProvider, self).__init__(
            text, "objectgroup", value, *args, **kwargs
        )

    def test_valid(self, pipeline):
        if self.value_text in cellprofiler_core.measurement.disallowed_object_names:
            raise ValidationError(
                "Object names may not be any of %s"
                % (", ".join(cellprofiler_core.measurement.disallowed_object_names)),
                self,
            )
        super(ObjectNameProvider, self).test_valid(pipeline)


class FileImageNameProvider(ImageNameProvider):
    """A setting that provides an image name where the image has an associated file"""

    def __init__(self, text, value="Do not use", *args, **kwargs):
        kwargs = kwargs.copy()
        if "provided_attributes" not in kwargs:
            kwargs["provided_attributes"] = {}
        kwargs["provided_attributes"]["file_image"] = True
        super(FileImageNameProvider, self).__init__(text, value, *args, **kwargs)


class ExternalImageNameProvider(ImageNameProvider):
    """A setting that provides an image name where the image is loaded
    externally. (eg: from Java)"""

    def __init__(self, text, value="Do not use", *args, **kwargs):
        kwargs = kwargs.copy()
        if "provided_attributes" not in kwargs:
            kwargs["provided_attributes"] = {}
        kwargs["provided_attributes"]["external_image"] = True
        super(ExternalImageNameProvider, self).__init__(text, value, *args, **kwargs)


class CroppingNameProvider(ImageNameProvider):
    """A setting that provides an image name where the image has a cropping mask"""

    def __init__(self, text, value="Do not use", *args, **kwargs):
        kwargs = kwargs.copy()
        if "provided_attributes" not in kwargs:
            kwargs["provided_attributes"] = {}
        kwargs["provided_attributes"]["cropping_image"] = True
        super(CroppingNameProvider, self).__init__(text, value, *args, **kwargs)


class OutlineNameProvider(ImageNameProvider):
    """A setting that provides an object outline name
    """

    def __init__(self, text, value="Do not use", *args, **kwargs):
        super(OutlineNameProvider, self).__init__(text, value, *args, **kwargs)


class GridNameProvider(NameProvider):
    """A setting that provides a GridInfo object
    """

    def __init__(self, text, value="Grid", *args, **kwargs):
        super(GridNameProvider, self).__init__(
            text, "gridgroup", value, *args, **kwargs
        )


class OddInteger(Integer):
    def test_valid(self, pipeline):
        super(self.__class__, self).test_valid(pipeline)

        value = self.str_to_value(self.value_text)

        if value % 2 == 0:
            raise ValidationError("Must be odd, was even", self)


def filter_duplicate_names(name_list):
    """remove any repeated names from a list of (name, ...) keeping the last occurrence."""
    name_dict = dict(list(zip((n[0] for n in name_list), name_list)))
    return [name_dict[n[0]] for n in name_list]


def get_name_provider_choices(pipeline, last_setting, group):
    """Scan the pipeline to find name providers for the given group

    pipeline - pipeline to scan
    last_setting - scan the modules in order until you arrive at this setting
    group - the name of the group of providers to scan
    returns a list of tuples, each with (provider name, module name, module number)
    """
    choices = []
    for module in pipeline.modules(False):
        module_choices = [
            (
                other_name,
                module.module_name,
                module.module_num,
                module.is_input_module(),
            )
            for other_name in module.other_providers(group)
        ]
        for setting in module.visible_settings():
            if setting.key() == last_setting.key():
                return filter_duplicate_names(choices)
            if (
                isinstance(setting, NameProvider)
                and module.enabled
                and setting != "Do not use"
                and last_setting.matches(setting)
            ):
                module_choices.append(
                    (
                        setting.value,
                        module.module_name,
                        module.module_num,
                        module.is_input_module(),
                    )
                )
        choices += module_choices
    assert False, "Setting not among visible settings in pipeline"
