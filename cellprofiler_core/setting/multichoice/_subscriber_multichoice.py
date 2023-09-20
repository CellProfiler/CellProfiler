from ._multichoice import MultiChoice
from ...constants.setting import get_name_provider_choices


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
