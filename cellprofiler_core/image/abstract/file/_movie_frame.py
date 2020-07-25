from ._file import File


class MovieFrame(File):
    """Provide an image by filename:frame, loading the file as it is requested
    """

    def __init__(self, name, pathname, filename, frame, rescale):
        super(MovieFrame, self).__init__(name, pathname, filename, rescale, index=frame)
