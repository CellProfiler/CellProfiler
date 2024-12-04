from ._file_image import FileImage


class FlexFrameImage(FileImage):
    """Provide an image by filename:frame, loading the file as it is requested
    """

    def __init__(self,
                 name,
                 pathname,
                 filename,
                 series,
                 index,
                 rescale_range=None,
                 metadata_rescale=False
                ):
        super(FlexFrameImage, self).__init__(
            name,
            pathname,
            filename,
            rescale_range=rescale_range,
            metadata_rescale=metadata_rescale,
            series=series,
            index=index
        )
