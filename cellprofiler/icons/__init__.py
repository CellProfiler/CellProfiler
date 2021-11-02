import os.path
import weakref

import pkg_resources

images = os.path.join("data", "images")

resources = pkg_resources.resource_filename("cellprofiler", images)

image_cache = weakref.WeakValueDictionary()

app = None

def get_builtin_image(name):
    import wx

    # We can't make wx.Image objects without a wx.App object existing somewhere. There should always be a GUI to get
    # here, but the build system needs to scan CellProfiler without one and it can't if there's no app.
    # So here we generate a "fake" app that the build scripts can use.
    global app
    if not wx.App.Get() and app is None:
        app = wx.App()

    try:
        return image_cache[name]
    except KeyError:
        pathname = os.path.join(resources, name + ".png")

        image_cache[name] = image = wx.Image(pathname)

        return image


def get_builtin_images_path():
    return os.path.join(resources, "")
