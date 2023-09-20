from .._subscriber import Subscriber
from ..._validation_error import ValidationError


class ListSubscriber(Subscriber):
    """
    Stores name provider names as a list
    """

    def __init__(
        self,
        text,
        group,
        value=None,
        can_be_blank=False,
        blank_text="Leave blank",
        *args,
        **kwargs,
    ):
        if value is None:
            value = ""
        super(ListSubscriber, self).__init__(
            text, group, value, can_be_blank, blank_text, *args, **kwargs
        )
        self.value = value

    def get_value_text(self):
        """Convert the underlying list to a string"""
        return ", ".join(map(str, self._Setting__value))

    def __internal_set_value_text(self, value):
        self.set_value_text(value)

    def set_value_text(self, value):
        self._Setting__value = value

    value_text = property(get_value_text, __internal_set_value_text)

    def __internal_set_value(self, value):
        """Convert a saved string into a list"""
        if len(value) == 0:
            value = []
        else:
            value = value.split(", ")
        self._Setting__value = value

    def __internal_get_value(self):
        return self.get_value()

    value = property(__internal_get_value, __internal_set_value)

    def test_valid(self, pipeline):
        choices = self.get_choices(pipeline)
        if len(choices) == 0:
            raise ValidationError(
                "No prior instances of %s were defined" % self.group, self
            )
        for name in self.value:
            if name not in [c[0] for c in choices]:
                raise ValidationError(
                    "%s not in %s"
                    % (name, ", ".join(c[0] for c in self.get_choices(pipeline))),
                    self,
                )
