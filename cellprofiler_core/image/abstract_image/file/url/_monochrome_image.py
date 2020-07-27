import skimage.color

from ._url_image import URLImage


class MonochromeImage(URLImage):
    """Provide a monochrome image, combining RGB if needed"""

    def __init__(
        self,
        name,
        url,
        series,
        index,
        channel,
        rescale=True,
        volume=False,
        spacing=None,
    ):
        URLImage.__init__(
            self,
            name,
            url,
            rescale=rescale,
            series=series,
            index=index,
            channel=channel,
            volume=volume,
            spacing=spacing,
        )

    def provide_image(self, image_set):
        image = URLImage.provide_image(self, image_set)

        if image.pixel_data.ndim == image.dimensions + 1:
            image.pixel_data = skimage.color.rgb2gray(image.pixel_data)

        return image
