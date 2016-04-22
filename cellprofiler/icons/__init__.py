import logging

logger = logging.getLogger(__package__)
import os.path
import glob
import weakref
import sys

if hasattr(sys, 'frozen'):
    path = os.path.split(os.path.abspath(sys.argv[0]))[0]
    path = os.path.join(path, 'artwork')
else:
    path = os.path.join(os.path.dirname(os.path.dirname(__path__[0])),
                        'artwork')

image_cache = weakref.WeakValueDictionary()


def get_builtin_image(name):
    import wx
    try:
        return image_cache[name]
    except KeyError:
        image_cache[name] = im = wx.Image(os.path.join(path, name + '.png'))
        return im


def get_builtin_images_path():
    return os.path.join(path, '')


def get_icon_copyrights():
    icpath = os.path.join(path, "icon_copyrights.txt")
    try:
        with open(icpath, "r") as fd:
            return fd.read()
    except:
        logger.warning('Could not find the icon copyrights file, "%s".' % icpath)
        return None
