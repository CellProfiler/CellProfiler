class ExternalImageNameProvider(ImageNameProvider):
    """A setting that provides an image name where the image is loaded
    externally. (eg: from Java)"""

    def __init__(self, text, value=DO_NOT_USE, *args, **kwargs):
        kwargs = kwargs.copy()
        if not kwargs.has_key(PROVIDED_ATTRIBUTES):
            kwargs[PROVIDED_ATTRIBUTES] = {}
        kwargs[PROVIDED_ATTRIBUTES][EXTERNAL_IMAGE_ATTRIBUTE] = True
        super(ExternalImageNameProvider, self).__init__(text, value, *args,
                                                        **kwargs)
