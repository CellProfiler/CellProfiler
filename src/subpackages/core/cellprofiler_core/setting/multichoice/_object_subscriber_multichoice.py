from ._subscriber_multichoice import SubscriberMultiChoice
from ...constants.pipeline import OBJECT_GROUP


class ObjectSubscriberMultiChoice(SubscriberMultiChoice):
    """A multi-choice setting that displays objects

    This setting displays a list of objects taken from ObjectNameProviders.
    """

    def __init__(self, text, value=None, *args, **kwargs):
        super(ObjectSubscriberMultiChoice, self).__init__(
            text, OBJECT_GROUP, value, *args, **kwargs
        )
