import os
import sys

from ._text import Text
from .._validation_error import ValidationError
from ...preferences import ABSOLUTE_FOLDER_NAME
from ...preferences import DEFAULT_INPUT_FOLDER_NAME
from ...preferences import DEFAULT_INPUT_SUBFOLDER_NAME
from ...preferences import DEFAULT_OUTPUT_FOLDER_NAME
from ...preferences import DEFAULT_OUTPUT_SUBFOLDER_NAME
from ...preferences import NO_FOLDER_NAME
from ...preferences import URL_FOLDER_NAME
from ...preferences import get_default_image_directory
from ...preferences import get_default_output_directory
from ...preferences import standardize_default_folder_names


class Directory(Text):
    """A setting that displays a filesystem path name
    """

    DIR_ALL = [
        ABSOLUTE_FOLDER_NAME,
        DEFAULT_INPUT_FOLDER_NAME,
        DEFAULT_OUTPUT_FOLDER_NAME,
        DEFAULT_INPUT_SUBFOLDER_NAME,
        DEFAULT_OUTPUT_SUBFOLDER_NAME,
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
            dir_choices = Directory.DIR_ALL
        if support_urls and not (URL_FOLDER_NAME in dir_choices):
            dir_choices = dir_choices + [URL_FOLDER_NAME]
        if value is None:
            value = Directory.static_join_string(dir_choices[0], "")
        self.dir_choices = dir_choices
        self.allow_metadata = allow_metadata
        self.support_urls = support_urls
        super(Directory, self).__init__(text, value, *args, **kwargs)

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
        if custom_path is not None and "\\\\" not in custom_path:
            custom_path = custom_path.replace("\\", "\\\\")
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
        dir_choice, custom_path = Directory.split_string(value)
        dir_choice = standardize_default_folder_names([dir_choice], 0)[0]
        return Directory.static_join_string(dir_choice, custom_path)

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
            ABSOLUTE_FOLDER_NAME,
            DEFAULT_INPUT_SUBFOLDER_NAME,
            DEFAULT_OUTPUT_SUBFOLDER_NAME,
            URL_FOLDER_NAME,
        ]

    def is_url(self):
        return self.dir_choice == URL_FOLDER_NAME

    def get_absolute_path(self, measurements=None, image_set_number=None):
        """Return the absolute path specified by the setting

        Concoct an absolute path based on the directory choice,
        the custom path and metadata taken from the measurements.
        """
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
            root_directory = ""
        elif self.dir_choice == NO_FOLDER_NAME:
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
        if self.dir_choice == URL_FOLDER_NAME:
            return custom_path
        path = os.path.join(root_directory, custom_path)
        return os.path.abspath(path)

    def get_parts_from_path(self, path):
        """Figure out how to set up dir_choice and custom path given a path"""
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
        if hasattr(os, "altsep"):
            seps += [os.altsep]
        if cmp_path == img_dir:
            dir_choice = DEFAULT_INPUT_FOLDER_NAME
        elif cmp_path == out_dir:
            dir_choice = DEFAULT_OUTPUT_FOLDER_NAME
        elif cmp_path.startswith(img_dir) and cmp_path[len(img_dir)] in seps:
            dir_choice = DEFAULT_INPUT_SUBFOLDER_NAME
            custom_path = path[len(img_dir) + 1 :]
        elif cmp_path.startswith(out_dir) and cmp_path[len(out_dir)] in seps:
            dir_choice = DEFAULT_OUTPUT_SUBFOLDER_NAME
            custom_path = path[len(out_dir) + 1 :]
        else:
            dir_choice = ABSOLUTE_FOLDER_NAME
            custom_path = path
        return dir_choice, custom_path

    def alter_for_create_batch_files(self, fn_alter_path):
        """Call this to alter the setting appropriately for batch execution"""
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
                self.custom_path, regexp_substitution=self.allow_metadata
            )
        elif self.dir_choice == DEFAULT_INPUT_SUBFOLDER_NAME:
            self.custom_path = fn_alter_path(
                self.custom_path, regexp_substitution=self.allow_metadata
            )
        elif self.dir_choice == DEFAULT_OUTPUT_SUBFOLDER_NAME:
            self.custom_path = fn_alter_path(
                self.custom_path, regexp_substitution=self.allow_metadata
            )

    def test_valid(self, pipeline):
        if self.dir_choice not in self.dir_choices + [NO_FOLDER_NAME]:
            raise ValidationError(
                "Unsupported directory choice: %s" % self.dir_choice, self
            )
        if (
            not self.allow_metadata
            and self.is_custom_choice
            and self.custom_path.find(r"\g<") != -1
        ):
            raise ValidationError("Metadata not supported for this setting", self)
        if self.dir_choice == ABSOLUTE_FOLDER_NAME and (
            (self.custom_path is None) or (len(self.custom_path) == 0)
        ):
            raise ValidationError("Please enter a valid path", self)
