import skimage.color

from cellprofiler_core.image import URL


class Color(URL):
    """Provide a color image, tripling a monochrome plane if needed"""

    def __init__(
        self, name, url, series, index, rescale=True, volume=False, spacing=None
    ):
        URL.__init__(
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
        image = URL.provide_image(self, image_set)

        if image.pixel_data.ndim == image.dimensions:
            image.pixel_data = skimage.color.gray2rgb(image.pixel_data, alpha=False)

        return image
