import os.path
import weakref

import pkg_resources

images = os.path.join("data", "images")

resources = pkg_resources.resource_filename("cellprofiler", images)

image_cache = weakref.WeakValueDictionary()


def get_builtin_image(name):
    import wx

    try:
        return image_cache[name]
    except KeyError:
        pathname = os.path.join(resources, name + ".png")

        image_cache[name] = image = wx.Image(pathname)

        return image


def get_builtin_images_path():
    return os.path.join(resources, "")
