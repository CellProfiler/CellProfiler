class ImageNameSubscriberMultiChoice(SubscriberMultiChoice):
    '''A multi-choice setting that displays images

    This setting displays a list of images taken from ImageNameProviders.
    '''

    def __init__(self, text, value=None, *args, **kwargs):
        super(ImageNameSubscriberMultiChoice, self).__init__(text, IMAGE_GROUP,
                                                             value, *args, **kwargs)
