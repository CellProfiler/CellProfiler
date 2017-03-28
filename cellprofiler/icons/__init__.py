import os.path
import pkg_resources

directory = pkg_resources.resource_filename("cellprofiler", "data")


def icon(name):
    import wx

    filename = "{}.png".format(name)

    pathname = os.path.join(directory, filename)

    return wx.Image(pathname)
