import os

from cellprofiler_core.image.abstract_image_provider.load_images_image_provider._load_images_image_provider import \
    LoadImagesImageProvider
from cellprofiler_core.utilities.pathname import url2pathname


class LoadImagesImageProviderURL(LoadImagesImageProvider):
    """Reference an image via a URL"""

    def __init__(
        self,
        name,
        url,
        rescale=True,
        series=None,
        index=None,
        channel=None,
        volume=False,
        spacing=None,
    ):
        if url.lower().startswith("file:"):
            path = url2pathname(url)
            pathname, filename = os.path.split(path)
        else:
            pathname = ""
            filename = url
        super(LoadImagesImageProviderURL, self).__init__(
            name, pathname, filename, rescale, series, index, channel, volume, spacing
        )
        self.url = url

    def get_url(self):
        if self.cache_file():
            return super(LoadImagesImageProviderURL, self).get_url()
        return self.url