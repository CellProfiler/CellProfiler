from ._image_name import ImageName


class FileImageName(ImageName):
    """
    A setting that provides an image name where the image has an associated file
    """

    def __init__(self, text, value="Do not use", *args, **kwargs):
        kwargs = kwargs.copy()

        if "provided_attributes" not in kwargs:
            kwargs["provided_attributes"] = {}

        kwargs["provided_attributes"]["file_image"] = True

        super(FileImageName, self).__init__(text, value, *args, **kwargs)
