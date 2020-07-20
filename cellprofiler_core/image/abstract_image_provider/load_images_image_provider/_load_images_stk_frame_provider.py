from cellprofiler_core.image.abstract_image_provider.load_images_image_provider._load_images_image_provider import (
    LoadImagesImageProvider,
)


class LoadImagesSTKFrameProvider(LoadImagesImageProvider):
    """Provide an image by filename:frame from an STK file"""

    def __init__(self, name, pathname, filename, frame, rescale):
        """Initialize the provider

        name - name of the provider for access from image set
        pathname - path to the file
        filename - name of the file
        frame - # of the frame to provide
        """
        super(LoadImagesSTKFrameProvider, self).__init__(
            name, pathname, filename, rescale=rescale, index=frame
        )
