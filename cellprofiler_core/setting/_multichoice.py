import functools
import os

import six

from cellprofiler_core.setting.text import get_name_provider_choices, DirectoryPath
from . import _setting
from ._validation_error import ValidationError


class MultiChoice(_setting.Setting):
    """A setting that represents selection of multiple choices from a list"""

    def __init__(self, text, choices, value=None, *args, **kwargs):
        """Initializer

        text - the explanatory text for the setting
        choices - a sequence of string choices to be selected
        value - a list of selected choices or a comma-separated string list
        """
        super(MultiChoice, self).__init__(
            text, self.parse_value(value), *args, **kwargs
        )
        self.__choices = choices

    @staticmethod
    def parse_value(value):
        if value is None:
            return ""
        elif isinstance(value, six.string_types):
            return value
        elif hasattr(value, "__getitem__"):
            return ",".join(value)
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
        """Set the value of a multi-choice setting

        value is either a single string, a comma-separated string of
        multiple choices or a list of strings
        """
        super(MultiChoice, self).set_value(self.parse_value(value))

    def get_selections(self):
        """Return the currently selected values"""
        value = self.get_value()
        if len(value) == 0:
            return ()
        return value.split(",")

    selections = property(get_selections)

    def test_valid(self, pipeline):
        """Ensure that the selections are among the choices"""
        for selection in self.get_selections():
            if selection not in self.choices:
                if len(self.choices) == 0:
                    raise ValidationError("No available choices", self)
                elif len(self.choices) > 25:
                    raise ValidationError(
                        "%s is not one of the choices" % selection, self
                    )
                raise ValidationError(
                    "%s is not one of %s"
                    % (
                        selection,
                        functools.reduce(lambda x, y: "%s,%s" % (x, y), self.choices),
                    ),
                    self,
                )


class SubscriberMultiChoice(MultiChoice):
    """A multi-choice setting that gets its choices through providers

    This setting operates similarly to the name subscribers. It gets
    its choices from the name providers for the subscriber's group.
    It displays a list of choices and the user can select multiple
    choices.
    """

    def __init__(self, text, group, value=None, *args, **kwargs):
        self.__required_attributes = {"group": group}
        if "required_attributes" in kwargs:
            self.__required_attributes.update(kwargs["required_attributes"])
            kwargs = kwargs.copy()
            del kwargs["required_attributes"]
        super(SubscriberMultiChoice, self).__init__(text, [], value, *args, **kwargs)

    def load_choices(self, pipeline):
        """Get the choice list from name providers"""
        self.choices = sorted(
            [c[0] for c in get_name_provider_choices(pipeline, self, self.group)]
        )

    @property
    def group(self):
        return self.__required_attributes["group"]

    def matches(self, provider):
        """Return true if the provider is compatible with this subscriber

        This method can be used to be more particular about the providers
        that are selected. For instance, if you want a list of only
        FileImageNameProviders (images loaded from files), you can
        check that here.
        """
        return all(
            [
                provider.provided_attributes.get(key, None)
                == self.__required_attributes[key]
                for key in self.__required_attributes
            ]
        )

    def test_valid(self, pipeline):
        self.load_choices(pipeline)
        super(SubscriberMultiChoice, self).test_valid(pipeline)


class ObjectSubscriberMultiChoice(SubscriberMultiChoice):
    """A multi-choice setting that displays objects

    This setting displays a list of objects taken from ObjectNameProviders.
    """

    def __init__(self, text, value=None, *args, **kwargs):
        super(ObjectSubscriberMultiChoice, self).__init__(
            text, "objectgroup", value, *args, **kwargs
        )


class ImageNameSubscriberMultiChoice(SubscriberMultiChoice):
    """A multi-choice setting that displays images

    This setting displays a list of images taken from ImageNameProviders.
    """

    def __init__(self, text, value=None, *args, **kwargs):
        super(ImageNameSubscriberMultiChoice, self).__init__(
            text, "imagegroup", value, *args, **kwargs
        )


class MeasurementMultiChoice(MultiChoice):
    """A multi-choice setting for selecting multiple measurements"""

    def __init__(self, text, value="", *args, **kwargs):
        """Initialize the measurement multi-choice

        At initialization, the choices are empty because the measurements
        can't be fetched here. It's done (bit of a hack) in test_valid.
        """
        super(MeasurementMultiChoice, self).__init__(text, [], value, *args, **kwargs)

    @staticmethod
    def encode_object_name(object_name):
        """Encode object name, escaping |"""
        return object_name.replace("|", "||")

    @staticmethod
    def decode_object_name(object_name):
        """Decode the escaped object name"""
        return object_name.replace("||", "|")

    @staticmethod
    def split_choice(choice):
        """Split object and feature within a choice"""
        subst_choice = choice.replace("||", "++")
        loc = subst_choice.find("|")
        if loc == -1:
            return subst_choice, "Invalid"
        return choice[:loc], choice[(loc + 1) :]

    def get_measurement_object(self, choice):
        return self.decode_object_name(self.split_choice(choice)[0])

    def get_measurement_feature(self, choice):
        return self.split_choice(choice)[1]

    def make_measurement_choice(self, object_name, feature):
        return self.encode_object_name(object_name) + "|" + feature

    @staticmethod
    def get_value_string(choices):
        """Return the string value representing the choices made

        choices - a collection of choices as returned by make_measurement_choice
        """
        return ",".join(choices)

    def test_valid(self, pipeline):
        """Get the choices here and call the superclass validator"""
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
            """Disallow any measurement column with "," or "|" in its names"""
            return not any([any([bad in f for f in c[:2]]) for bad in (",", "|")])

        self.set_choices(
            [self.make_measurement_choice(c[0], c[1]) for c in columns if valid_mc(c)]
        )


class SubdirectoryFilter(MultiChoice):
    """A setting that indicates which subdirectories should be excluded from an operation

    The subdirectory filter holds a collection of subdirectories that
    should be excluded from a file discovery operation that scans
    subdirectories.
    """

    def __init__(self, text, value="", directory_path=None, **kwargs):
        """Initialize the setting

        text - a tag for the setting that briefly indicates its purpose

        value - the value for the setting, as saved in the pipeline

        directory_path - an optional DirectoryPath setting that can be used
                         to find the root of the subdirectory tree.
        """
        super(SubdirectoryFilter, self).__init__(text, value, **kwargs)
        assert (directory_path is None) or isinstance(directory_path, DirectoryPath)
        self.directory_path = directory_path

    @staticmethod
    def get_value_string(choices):
        """Return the string value representing the choices made

        choices - a collection of choices as returned by make_measurement_choice
        """
        return ",".join(choices)

    def alter_for_create_batch_files(self, fn_alter_path):
        selections = [fn_alter_path(selection) for selection in self.get_selections()]
        self.value = self.get_value_string(selections)

    def test_valid(self, pipeline):
        if self.directory_path is not None:
            root = self.directory_path.get_absolute_path()
            for subdirectory in self.get_selections():
                path = os.path.join(root, subdirectory)
                if not os.path.isdir(path):
                    raise ValidationError("%s is not a valid directory" % path, self)
