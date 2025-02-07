import os
from .....utilities.pathname import url2pathname

from .._file_image import FileImage


class URLImage(FileImage):
    """Reference an image via a URL"""

    def __init__(
        self,
        name,
        url,
        rescale_range=None,
        metadata_rescale=False,
        series=None,
        index=None,
        channel=None,
        volume=False,
        spacing=None,
        z=None,
        t=None
    ):
        if url.lower().startswith("file:"):
            path = url2pathname(url)
            pathname, filename = os.path.split(path)
        else:
            pathname = ""
            filename = url
        super(URLImage, self).__init__(
            name,
            pathname,
            filename,
            rescale_range=rescale_range,
            metadata_rescale=metadata_rescale,
            series=series,
            index=index,
            channel=channel,
            volume=volume,
            spacing=spacing,
            z=z,
            t=t
        )
        self.url = url

    def get_url(self):
        if self.cache_file():
            return super(URLImage, self).get_url()
        return self.url
