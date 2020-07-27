from ._file_image import FileImage


class MovieFrameImage(FileImage):
    """Provide an image by filename:frame, loading the file as it is requested
    """

    def __init__(self, name, pathname, filename, frame, rescale):
        super(MovieFrameImage, self).__init__(
            name, pathname, filename, rescale, index=frame
        )
