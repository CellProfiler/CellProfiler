import skimage.color

import cellprofiler_core.image


class MonochromeImageProvider(cellprofiler_core.image.LoadImagesImageProviderURL):
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
        cellprofiler_core.image.LoadImagesImageProviderURL.__init__(
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
        image = cellprofiler_core.image.LoadImagesImageProviderURL.provide_image(self, image_set)

        if image.pixel_data.ndim == image.dimensions + 1:
            image.pixel_data = skimage.color.rgb2gray(image.pixel_data)

        return image
