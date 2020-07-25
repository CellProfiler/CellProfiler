from cellprofiler_core.image.abstract_image_provider.load_images_image_provider.load_images_image_provider_url._monochrome_image_provider import (
    MonochromeImageProvider,
)


class MaskImageProvider(MonochromeImageProvider):
    """Provide a boolean image, converting nonzero to True, zero to False if needed"""

    def __init__(self, name, url, series, index, channel, volume=False, spacing=None):
        MonochromeImageProvider.__init__(
            self,
            name,
            url,
            rescale=True,
            series=series,
            index=index,
            channel=channel,
            volume=volume,
            spacing=spacing,
        )

    def provide_image(self, image_set):
        image = MonochromeImageProvider.provide_image(self, image_set)
        if image.pixel_data.dtype.kind != "b":
            image.pixel_data = image.pixel_data != 0
        return image
