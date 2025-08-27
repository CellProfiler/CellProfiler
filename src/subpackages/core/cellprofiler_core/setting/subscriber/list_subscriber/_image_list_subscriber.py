from ._list_subscriber import ListSubscriber
from ....constants.pipeline import IMAGE_GROUP


class ImageListSubscriber(ListSubscriber):
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
        super(ImageListSubscriber, self).__init__(
            text, IMAGE_GROUP, value, can_be_blank, blank_text, *args, **kwargs
        )
