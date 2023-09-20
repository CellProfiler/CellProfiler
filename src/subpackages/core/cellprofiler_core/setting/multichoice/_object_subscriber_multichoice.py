from ._subscriber_multichoice import SubscriberMultiChoice


class ObjectSubscriberMultiChoice(SubscriberMultiChoice):
    """A multi-choice setting that displays objects

    This setting displays a list of objects taken from ObjectNameProviders.
    """

    def __init__(self, text, value=None, *args, **kwargs):
        super(ObjectSubscriberMultiChoice, self).__init__(
            text, "objectgroup", value, *args, **kwargs
        )
