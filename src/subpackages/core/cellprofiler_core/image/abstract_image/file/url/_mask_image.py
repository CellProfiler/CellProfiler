from ._monochrome_image import MonochromeImage


class MaskImage(MonochromeImage):
    """Provide a boolean image, converting nonzero to True, zero to False if needed"""

    def __init__(self,
                 name,
                 url,
                 series,
                 index,
                 channel,
                 volume=False,
                 spacing=None,
                 z=None,
                 t=None
                ):
        MonochromeImage.__init__(
            self,
            name,
            url,
            rescale_range=None,
            metadata_rescale=False,
            series=series,
            index=index,
            channel=channel,
            volume=volume,
            spacing=spacing,
            z=z,
            t=t
        )

    def provide_image(self, image_set):
        image = MonochromeImage.provide_image(self, image_set)
        if image.pixel_data.dtype.kind != "b":
            image.pixel_data = image.pixel_data != 0
        return image
