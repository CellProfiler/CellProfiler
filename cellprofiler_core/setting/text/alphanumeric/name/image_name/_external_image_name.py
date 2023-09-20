from ._image_name import ImageName


class ExternalImageName(ImageName):
    """
    A setting that provides an image name where the image is loaded externally. (eg: from Java)
    """

    def __init__(self, text, value="Do not use", *args, **kwargs):
        kwargs = kwargs.copy()

        if "provided_attributes" not in kwargs:
            kwargs["provided_attributes"] = {}

        kwargs["provided_attributes"]["external_image"] = True

        super(ExternalImageName, self).__init__(text, value, *args, **kwargs)
