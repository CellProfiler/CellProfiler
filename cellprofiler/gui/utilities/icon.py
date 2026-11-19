import datetime
import logging
import os
import sys

import pkg_resources

import cellprofiler.icons

LOGGER = logging.getLogger(__name__)

DEFAULT_ICON_NAME = "CellProfiler"

SEASONAL_ICON_NAMES = {
    1: "CellebrateProfiler",
    2: "CellentinesProfiler",
    10: "SpellProfiler",
    11: "CellProfaller",
    12: "JingleCellProfiler",
}

ICON_MONTH_ENV = "CELLPROFILER_ICON_MONTH"

cp_image = None


def _icon_month(today=None):
    override = os.environ.get(ICON_MONTH_ENV)
    if override:
        try:
            month = int(override)
            if 1 <= month <= 12:
                return month
        except ValueError:
            pass
        LOGGER.warning(f"Ignoring invalid {ICON_MONTH_ENV}={override!r}")
    if today is None:
        today = datetime.date.today()
    return today.month


def seasonal_icon_name(today=None):
    """The basename of the CellProfiler icon for the given (or current) date"""
    return SEASONAL_ICON_NAMES.get(_icon_month(today), DEFAULT_ICON_NAME)


def _icon_path(extension):
    for name in (seasonal_icon_name(), DEFAULT_ICON_NAME):
        path = pkg_resources.resource_filename(
            "cellprofiler", os.path.join("data", "icons", name + extension)
        )
        if os.path.isfile(path):
            return path
        LOGGER.debug(f"Icon not found: {path}")
    return path


def get_cp_image():
    """The CellProfiler icon as a wx.Image"""
    global cp_image

    if cp_image is None:
        import wx

        path = _icon_path(".png")
        try:
            cp_image = cellprofiler.icons.image_cache[path]
        except KeyError:
            cellprofiler.icons.image_cache[path] = cp_image = wx.Image(path)

    return cp_image


def get_cp_bitmap(size=None):
    """The CellProfiler icon as a wx.Bitmap"""
    import wx

    img = get_cp_image()
    if size is not None:
        img = img.Scale(size, size, wx.IMAGE_QUALITY_HIGH)
    return wx.Bitmap(img)


def get_cp_icon(size=None):
    """The CellProfiler icon as a wx.Icon"""
    import wx

    if sys.platform.startswith("win"):
        icon = wx.Icon()
        icon.LoadFile(_icon_path(".ico"), wx.BITMAP_TYPE_ICO)
        return icon
    icon = wx.Icon()
    icon.CopyFromBitmap(get_cp_bitmap(size))
    return icon


def get_cp_icon_bundle():
    """The CellProfiler icon at multiple sizes as a wx.IconBundle"""
    import wx

    if sys.platform.startswith("win"):
        return wx.IconBundle(_icon_path(".ico"), wx.BITMAP_TYPE_ICO)
    bundle = wx.IconBundle()
    for size in (16, 24, 32, 48, 64, 128, 256):
        bundle.AddIcon(get_cp_icon(size))
    return bundle
