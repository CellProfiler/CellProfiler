from ._list_subscriber import ListSubscriber
from ....constants.pipeline import OBJECT_GROUP


class LabelListSubscriber(ListSubscriber):
    """
    A setting that provides an image name
    """

    def __init__(
        self,
        text,
        value=None,
        can_be_blank=False,
        blank_text="Leave blank",
        *args,
        **kwargs,
    ):
        super(LabelListSubscriber, self).__init__(
            text, OBJECT_GROUP, value, can_be_blank, blank_text, *args, **kwargs
        )
