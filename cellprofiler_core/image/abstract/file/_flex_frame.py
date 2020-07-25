from ._file import File


class FlexFrame(File):
    """Provide an image by filename:frame, loading the file as it is requested
    """

    def __init__(self, name, pathname, filename, series, index, rescale):
        super(FlexFrame, self).__init__(
            name, pathname, filename, rescale=rescale, series=series, index=index
        )
