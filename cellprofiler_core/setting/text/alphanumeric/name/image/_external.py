from ._image import Image


class External(Image):
    """
    A setting that provides an image name where the image is loaded externally. (eg: from Java)
    """

    def __init__(self, text, value="Do not use", *args, **kwargs):
        kwargs = kwargs.copy()

        if "provided_attributes" not in kwargs:
            kwargs["provided_attributes"] = {}

        kwargs["provided_attributes"]["external_image"] = True

        super(External, self).__init__(text, value, *args, **kwargs)
