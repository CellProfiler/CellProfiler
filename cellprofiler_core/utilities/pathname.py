import urllib.request

from cellprofiler_core.utilities.image import is_file_url
from cellprofiler_core.constants.image import FILE_SCHEME, PASSTHROUGH_SCHEMES


def pathname2url(path):
    """Convert the unicode path to a file: url"""
    lower_path = path.lower()
    if any((lower_path.startswith(x) for x in PASSTHROUGH_SCHEMES)):
        return path
    return FILE_SCHEME + urllib.request.pathname2url(path)


def url2pathname(url):
    lower_url = url.lower()
    if any((lower_url.startswith(x) for x in PASSTHROUGH_SCHEMES)):
        return url
    if is_file_url(url):
        return urllib.request.url2pathname(url[len(FILE_SCHEME):])
    return url
