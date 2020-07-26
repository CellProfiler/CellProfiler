import re

from ._setting import Setting
from ._validation_error import ValidationError


class RegexpText(Setting):
    """A setting with a regexp button on the side
    """

    GUESS_FILE = "file"
    GUESS_FOLDER = "folder"

    def __init__(self, text, value, *args, **kwargs):
        """initialize the setting

        text   - the explanatory text for the setting
        value  - the default or initial value for the setting
        doc - documentation for the setting
        get_example_fn - a function that returns an example string for the
                         metadata editor
        guess - either GUESS_FILE to use potential file-name regular expressions
                when guessing in the regexp editor or GUESS_FOLDER to
                use folder-name guesses.
        """
        kwargs = kwargs.copy()
        self.get_example_fn = kwargs.pop("get_example_fn", None)
        self.guess = kwargs.pop("guess", self.GUESS_FILE)
        super(RegexpText, self).__init__(text, value, *args, **kwargs)

    def test_valid(self, pipeline):
        try:
            # Convert Matlab to Python
            pattern = re.sub("(\\(\\?)([<][^)>]+?[>])", "\\1P\\2", self.value)
            re.search("(|(%s))" % pattern, "")
        except re.error as v:
            raise ValidationError("Invalid regexp: %s" % v, self)
