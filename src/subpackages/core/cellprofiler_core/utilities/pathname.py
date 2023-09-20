import pathlib
import urllib
import logging

from cellprofiler_core.constants.image import FILE_SCHEME, PASSTHROUGH_SCHEMES
from cellprofiler_core.utilities.image import is_file_url

LOGGER = logging.getLogger(__name__)

def pathname2url(path):
    """Convert the unicode path to a file: url"""
    lower_path = path.lower()
    if any((lower_path.startswith(x) for x in PASSTHROUGH_SCHEMES)):
        return path
    path_object = pathlib.Path(path)
    if path_object.is_absolute():
        # Produces a valid URI regardless of platform.
        return path_object.as_uri()
    # Produces CellProfiler's interpretation of a relative path URI.
    return FILE_SCHEME + urllib.request.pathname2url(path)

def url2pathname(url):
    lower_url = url.lower()

    if any((lower_url.startswith(x) for x in PASSTHROUGH_SCHEMES)):
        return url

    if is_file_url(url):
        return urllib.request.url2pathname(url[len(FILE_SCHEME):])
