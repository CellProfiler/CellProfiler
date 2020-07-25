from ._file import File


class STKFrame(File):
    """Provide an image by filename:frame from an STK file"""

    def __init__(self, name, pathname, filename, frame, rescale):
        """Initialize the provider

        name - name of the provider for access from image set
        pathname - path to the file
        filename - name of the file
        frame - # of the frame to provide
        """
        super(STKFrame, self).__init__(
            name, pathname, filename, rescale=rescale, index=frame
        )
