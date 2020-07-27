from ._image_subscriber import ImageSubscriber


class CropImageSubscriber(ImageSubscriber):
    """A setting that provides image names that have cropping masks"""

    def __init__(
        self,
        text,
        value="Do not use",
        can_be_blank=False,
        blank_text="Leave blank",
        *args,
        **kwargs,
    ):
        kwargs = kwargs.copy()
        if "required_attributes" not in kwargs:
            kwargs["required_attributes"] = {}
        kwargs["required_attributes"]["cropping_image"] = True
        super(CropImageSubscriber, self).__init__(
            text, value, can_be_blank, blank_text, *args, **kwargs
        )
