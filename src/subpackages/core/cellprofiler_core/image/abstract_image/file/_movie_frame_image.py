from ._file_image import FileImage


class MovieFrameImage(FileImage):
    """Provide an image by filename:frame, loading the file as it is requested
    """

    def __init__(self, name, pathname, filename, frame, rescale_range=None):
        super(MovieFrameImage, self).__init__(
            name,
            pathname,
            filename,
            rescale_range=rescale_range,
            metadata_rescale=False,
            index=frame
        )
