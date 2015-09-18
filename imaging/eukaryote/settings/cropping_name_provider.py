class CroppingNameProvider(ImageNameProvider):
    """A setting that provides an image name where the image has a cropping mask"""

    def __init__(self, text, value=DO_NOT_USE, *args, **kwargs):
        kwargs = kwargs.copy()
        if not kwargs.has_key(PROVIDED_ATTRIBUTES):
            kwargs[PROVIDED_ATTRIBUTES] = {}
        kwargs[PROVIDED_ATTRIBUTES][CROPPING_ATTRIBUTE] = True
        super(CroppingNameProvider, self).__init__(text, value, *args, **kwargs)
