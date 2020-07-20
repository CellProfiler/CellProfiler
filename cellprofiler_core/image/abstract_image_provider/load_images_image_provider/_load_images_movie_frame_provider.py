from cellprofiler_core.image.abstract_image_provider.load_images_image_provider._load_images_image_provider import \
    LoadImagesImageProvider


class LoadImagesMovieFrameProvider(LoadImagesImageProvider):
    """Provide an image by filename:frame, loading the file as it is requested
    """

    def __init__(self, name, pathname, filename, frame, rescale):
        super(LoadImagesMovieFrameProvider, self).__init__(
            name, pathname, filename, rescale, index=frame
        )