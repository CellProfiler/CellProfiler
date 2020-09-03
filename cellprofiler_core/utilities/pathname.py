import urllib.request

from cellprofiler_core.utilities.image import is_file_url
from cellprofiler_core.constants.image import FILE_SCHEME, PASSTHROUGH_SCHEMES


def pathname2url(path):
    """Convert the unicode path to a file: url"""
    utf8_path = str(path)
    if any([utf8_path.lower().startswith(x) for x in PASSTHROUGH_SCHEMES]):
        return utf8_path
    return FILE_SCHEME + urllib.request.pathname2url(utf8_path)


def url2pathname(url):
    if isinstance(url, str):
        url = url
    if any([url.lower().startswith(x) for x in PASSTHROUGH_SCHEMES]):
        return url
    if is_file_url(url):
        utf8_url = urllib.request.url2pathname(url[len(FILE_SCHEME) :])
        return utf8_url
    else:
        return url
