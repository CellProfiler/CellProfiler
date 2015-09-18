class ImageNameProvider(NameProvider):
    """A setting that provides an image name
    """

    def __init__(self, text, value=DO_NOT_USE, *args, **kwargs):
        super(ImageNameProvider, self).__init__(text, IMAGE_GROUP, value,
                                                *args, **kwargs)
