from ._subscriber import Subscriber
from ...constants.pipeline import OBJECT_GROUP


class LabelSubscriber(Subscriber):
    """
    A setting that subscribes to the list of available object names
    """

    def __init__(
        self,
        text,
        value="Do not use",
        can_be_blank=False,
        blank_text="Leave blank",
        *args,
        **kwargs,
    ):
        super(LabelSubscriber, self).__init__(
            text, OBJECT_GROUP, value, can_be_blank, blank_text, *args, **kwargs
        )
