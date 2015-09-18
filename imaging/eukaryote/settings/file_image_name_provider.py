class FileImageNameProvider(ImageNameProvider):
    """A setting that provides an image name where the image has an associated file"""

    def __init__(self, text, value=DO_NOT_USE, *args, **kwargs):
        kwargs = kwargs.copy()
        if not kwargs.has_key(PROVIDED_ATTRIBUTES):
            kwargs[PROVIDED_ATTRIBUTES] = {}
        kwargs[PROVIDED_ATTRIBUTES][FILE_IMAGE_ATTRIBUTE] = True
        super(FileImageNameProvider, self).__init__(text, value, *args,
                                                    **kwargs)
