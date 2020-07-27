from ._monochrome_image import MonochromeImage


class MaskImage(MonochromeImage):
    """Provide a boolean image, converting nonzero to True, zero to False if needed"""

    def __init__(self, name, url, series, index, channel, volume=False, spacing=None):
        MonochromeImage.__init__(
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
        image = MonochromeImage.provide_image(self, image_set)
        if image.pixel_data.dtype.kind != "b":
            image.pixel_data = image.pixel_data != 0
        return image
