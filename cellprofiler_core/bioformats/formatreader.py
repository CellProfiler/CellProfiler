import logging
import scyjava

from ..utilities.image import is_file_url
from ..utilities.pathname import url2pathname

logger = logging.getLogger(__name__)

K_OMERO_SERVER = None
K_OMERO_PORT = None
K_OMERO_USER = None
K_OMERO_SESSION_ID = None
K_OMERO_CONFIG_FILE = None

scyjava.config.endpoints.append("ome:formats-gpl")

def get_image_reader(key, path=None, url=None):
    '''Make or find an image reader appropriate for the given path

    path - pathname to the reader on disk.

    key - use this key to keep only a single cache member associated with
          that key open at a time.
    '''
    # E.g.: 6335094544, None, file:///Users/curtis/data/c-elegans-cell-fusion.tif
    logger.info("Getting image reader for: %s, %s, %s" % (key, path, url))
    ImageReader = scyjava.jimport("loci.formats.ImageReader")
    reader = ImageReader()
    if not is_file_url(url):
        raise ValueError("Bio-Formats only supports file URLs for the moment")
    reader.setId(url2pathname(url))
    return reader

def release_image_reader():
    raise RuntimeError("unimplemented")

def clear_image_reader_cache():
    pass
    #$raise RuntimeError("unimplemented")

def set_omero_login_hook(omero_login):
    pass
    #raise RuntimeError("unimplemented")

def get_omero_credentials():
    raise RuntimeError("unimplemented")

def use_omero_credentials(credentials):
    raise RuntimeError("unimplemented")
