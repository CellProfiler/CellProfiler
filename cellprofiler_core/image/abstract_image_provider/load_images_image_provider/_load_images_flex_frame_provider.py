from cellprofiler_core.image.abstract_image_provider.load_images_image_provider._load_images_image_provider import (
    LoadImagesImageProvider,
)


class LoadImagesFlexFrameProvider(LoadImagesImageProvider):
    """Provide an image by filename:frame, loading the file as it is requested
    """

    def __init__(self, name, pathname, filename, series, index, rescale):
        super(LoadImagesFlexFrameProvider, self).__init__(
            name, pathname, filename, rescale=rescale, series=series, index=index
        )
