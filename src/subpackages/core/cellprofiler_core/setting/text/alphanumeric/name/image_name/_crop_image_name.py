from ._image_name import ImageName


class CropImageName(ImageName):
    """
    A setting that provides an image name where the image has a cropping mask
    """

    def __init__(self, text, value="Do not use", *args, **kwargs):
        kwargs = kwargs.copy()

        if "provided_attributes" not in kwargs:
            kwargs["provided_attributes"] = {}

        kwargs["provided_attributes"]["cropping_image"] = True

        super(CropImageName, self).__init__(text, value, *args, **kwargs)
