from .._name import Name
from ......constants.pipeline import IMAGE_GROUP


class ImageName(Name):
    """
    A setting that provides an image name
    """

    def __init__(self, text, value="Do not use", *args, **kwargs):
        super(ImageName, self).__init__(text, IMAGE_GROUP, value, *args, **kwargs)
