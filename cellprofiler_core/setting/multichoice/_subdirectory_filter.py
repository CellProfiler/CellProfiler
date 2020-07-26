import os

from ._multichoice import MultiChoice
from .._validation_error import ValidationError
from ..text import Directory


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
        assert (directory_path is None) or isinstance(directory_path, Directory)
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
