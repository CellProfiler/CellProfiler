from ._image_name import ImageName


class OutlineImageName(ImageName):
    """
    A setting that provides an object outline name
    """

    def __init__(self, text, value="Do not use", *args, **kwargs):
        super(OutlineImageName, self).__init__(text, value, *args, **kwargs)
