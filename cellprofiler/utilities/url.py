import os
import urllib
import urlparse

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
