from .._name import Name


class ImageName(Name):
    """
    A setting that provides an image name
    """

    def __init__(self, text, value="Do not use", *args, **kwargs):
        super(ImageName, self).__init__(text, "imagegroup", value, *args, **kwargs)
