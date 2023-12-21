import os.path
import weakref

import importlib.resources

images = os.path.join("data", "images")

resources = str(importlib.resources.files("cellprofiler").joinpath(images))

image_cache = weakref.WeakValueDictionary()

app = None

def get_builtin_image(name):
    import wx

    # If we're not running in a GUI environment, we can't load images.
    # This happens when running a non-windowed build of python (i.e. non-framework build)
    # Which can occur during a build system scan via e.g. pyinstaller.
    if not wx.App.IsDisplayAvailable():
        return b''

    # We can't make wx.Image objects without a wx.App object existing somewhere. There should always be a GUI to get
    # here, but if there is not (e.g. a build system scanning CellProfiler using a window-capable build of python),
    # we generate a "fake" app that the build scripts can use.
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
