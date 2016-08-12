import os
import urllib
import urlparse

import cellprofiler.utilities.hdf5_dict

FILE_SCHEME = "file:"
PASSTHROUGH_SCHEMES = ("http", "https", "ftp", "omero")


def pathname2url(path):
    """Convert the unicode path to a file: url"""
    utf8_path = path.encode('utf-8')
    if any([utf8_path.lower().startswith(x) for x in PASSTHROUGH_SCHEMES]):
        return utf8_path
    return FILE_SCHEME + urllib.pathname2url(utf8_path)


def is_file_url(url):
    return url.lower().startswith(FILE_SCHEME)


def url2pathname(url):
    if isinstance(url, unicode):
        url = url.encode("utf-8")
    if any([url.lower().startswith(x) for x in PASSTHROUGH_SCHEMES]):
        return url
    assert is_file_url(url)
    utf8_url = urllib.url2pathname(url[len(FILE_SCHEME):])
    return unicode(utf8_url, 'utf-8')


def urlfilename(url):
    """Return just the file part of a URL

    For instance http://cellprofiler.org/linked_files/file%20has%20spaces.txt
    has a file part of "file has spaces.txt"
    """
    if is_file_url(url):
        return os.path.split(url2pathname(url))[1]
    path = urlparse.urlparse(url)[2]
    if "/" in path:
        return urllib.unquote(path.rsplit("/", 1)[1])
    else:
        return urllib.unquote(path)


def urlpathname(url):
    """Return the path part of a URL

    For instance, http://cellprofiler.org/Comma%2Cseparated/foo.txt
    has a path of http://cellprofiler.org/Comma,separated

    A file url has the normal sort of path that you'd expect.
    """
    if is_file_url(url):
        return os.path.split(url2pathname(url))[0]
    scheme, netloc, path = urlparse.urlparse(url)[:3]
    path = urlparse.urlunparse([scheme, netloc, path, "", "", ""])
    if "/" in path:
        return urllib.unquote(path.rsplit("/", 1)[0])
    else:
        return urllib.unquote(path)


def modpath_to_url(modpath):
    if modpath[0] in ("http", "https", "ftp"):
        if len(modpath) == 1:
            return modpath[0] + ":"
        elif len(modpath) == 2:
            return modpath[0] + ":" + modpath[1]
        else:
            return modpath[0] + ":" + modpath[1] + "/" + "/".join(
                [urllib.quote(part) for part in modpath[2:]])
    path = os.path.join(*modpath)
    return pathname2url(path)


def url_to_modpath(url):
    if not url.lower().startswith("file:"):
        schema, rest = cellprofiler.utilities.hdf5_dict.HDF5FileList.split_url(url)
        return [schema] + rest[0:1] + [urllib.unquote(part) for part in rest[1:]]
    path = urllib.url2pathname(url[5:])
    parts = []
    while True:
        new_path, part = os.path.split(path)
        if len(new_path) == 0 or len(part) == 0:
            parts.insert(0, path)
            break
        parts.insert(0, part)
        path = new_path
    return parts
