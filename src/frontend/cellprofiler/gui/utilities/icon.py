import sys

import importlib.resources

import cellprofiler.icons

cp_image = None


def get_cp_image():
    """The CellProfiler icon as a wx.Image"""
    global cp_image

    if cp_image is None:
        import wx

        try:
            cp_image = cellprofiler.icons.image_cache["CellProfiler"]
        except KeyError:
            pathname = str(importlib.resources.files("cellprofiler").joinpath(
                "data", "icons", "CellProfiler.png"
            ))

            cellprofiler.icons.image_cache["CellProfiler"] = cp_image = wx.Image(
                pathname
            )

    return cp_image


def get_cp_bitmap(size=None):
    """The CellProfiler icon as a wx.Bitmap"""
    import wx

    img = get_cp_image()
    if size is not None:
        img.Rescale(size, size, wx.IMAGE_QUALITY_HIGH)
    return wx.Bitmap(img)


def get_cp_icon(size=None):
    """The CellProfiler icon as a wx.Icon"""
    import wx

    if sys.platform.startswith("win"):
        path = str(importlib.resources.files("cellprofiler").joinpath(
            "data", "icons", "CellProfiler.ico"
        ))
        icon = wx.Icon()
        icon.LoadFile(path, wx.BITMAP_TYPE_ICO)
        return icon
    icon = wx.Icon()
    icon.CopyFromBitmap(get_cp_bitmap(size))
    return icon
