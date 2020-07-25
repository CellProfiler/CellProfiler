from ._image import Image


class Outline(Image):
    """
    A setting that provides an object outline name
    """
    def __init__(self, text, value="Do not use", *args, **kwargs):
        super(Outline, self).__init__(text, value, *args, **kwargs)
