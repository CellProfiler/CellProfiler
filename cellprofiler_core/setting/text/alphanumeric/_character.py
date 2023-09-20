from ._alphanumeric import Alphanumeric
from ..._validation_error import ValidationError


class Character(Alphanumeric):
    """ A Setting for text entries of size one
    """
    def __init__(self, text, value, *args, **kwargs):
        super().__init__(text, value, *args, **kwargs)

    def test_valid(self, pipeline):
        """Restrict value to single character
        """
        super().test_valid(pipeline)
        if len(self.value) > 1:
            raise ValidationError("Only single characters can be used.", self)

