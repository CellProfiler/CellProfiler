import os.path

from .._text import Text
from ..._validation_error import ValidationError


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
