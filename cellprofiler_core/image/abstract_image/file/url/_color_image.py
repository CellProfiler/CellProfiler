import skimage.color

from ._url_image import URLImage


class ColorImage(URLImage):
    """Provide a color image, tripling a monochrome plane if needed"""

    def __init__(
        self, name, url, series, index, rescale=True, volume=False, spacing=None
    ):
        URLImage.__init__(
            self,
            name,
            url,
            rescale=rescale,
            series=series,
            index=index,
            volume=volume,
            spacing=spacing,
        )

    def provide_image(self, image_set):
        image = URLImage.provide_image(self, image_set)

        if image.pixel_data.ndim == image.dimensions:
            image.pixel_data = skimage.color.gray2rgb(image.pixel_data)

        return image
