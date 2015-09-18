class OutlineNameProvider(ImageNameProvider):
    '''A setting that provides an object outline name
    '''

    def __init__(self, text, value=DO_NOT_USE, *args, **kwargs):
        super(OutlineNameProvider, self).__init__(text, value,
                                                  *args, **kwargs)
