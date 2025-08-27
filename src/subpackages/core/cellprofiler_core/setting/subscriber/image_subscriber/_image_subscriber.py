from .._subscriber import Subscriber
from ....constants.pipeline import IMAGE_GROUP


class ImageSubscriber(Subscriber):
    """A setting that provides an image name
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
        super(ImageSubscriber, self).__init__(
            text, IMAGE_GROUP, value, can_be_blank, blank_text, *args, **kwargs
        )
