from ._monochrome import Monochrome


class Mask(Monochrome):
    """Provide a boolean image, converting nonzero to True, zero to False if needed"""

    def __init__(self, name, url, series, index, channel, volume=False, spacing=None):
        Monochrome.__init__(
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
        image = Monochrome.provide_image(self, image_set)
        if image.pixel_data.dtype.kind != "b":
            image.pixel_data = image.pixel_data != 0
        return image
