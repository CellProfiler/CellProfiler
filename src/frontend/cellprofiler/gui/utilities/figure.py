import functools
import logging
import os
import sys

import numpy
import wx
from cellprofiler_core.image import FileImage
from cellprofiler_core.preferences import reset_cpfigure_position
from cellprofiler_core.preferences import get_interpolation_mode
from cellprofiler_core.preferences import IM_NEAREST
from cellprofiler_core.preferences import IM_BILINEAR
from cellprofiler_core.preferences import IM_BICUBIC

from .. import errordialog

LOGGER = logging.getLogger(__name__)

CROSSHAIR_CURSOR = None


def wraparound(sequence):
    while True:
        for l in sequence:
            yield l


def match_rgbmask_to_image(rgb_mask, image):
    rgb_mask = list(rgb_mask)  # copy
    nchannels = image.shape[-1]
    del rgb_mask[nchannels:]
    if len(rgb_mask) < nchannels:
        rgb_mask = rgb_mask + [1] * (nchannels - len(rgb_mask))
    return rgb_mask


def window_name(module):
    """Return a module's figure window name"""
    return "CellProfiler:%s:%s" % (module.module_name, module.module_num)


def find_fig(parent=None, title="", name=wx.FrameNameStr, subplots=None):
    """Find a figure frame window. Returns the window or None"""
    for w in wx.GetTopLevelWindows():
        if w.GetName() == name:
            return w


def create_or_find(
    parent=None,
    identifier=-1,
    title="",
    pos=wx.DefaultPosition,
    size=wx.DefaultSize,
    style=wx.DEFAULT_FRAME_STYLE,
    name=wx.FrameNameStr,
    subplots=None,
    on_close=None,
):
    """Create or find a figure frame window"""
    from ..figure import Figure

    win = find_fig(parent, title, name, subplots)
    return win or Figure(
        parent, identifier, title, pos, size, style, name, subplots, on_close
    )


def close_all(parent):
    from ..figure import Figure

    windows = [x for x in parent.GetChildren() if isinstance(x, wx.Frame)]

    for window in windows:
        if isinstance(window, Figure):
            window.on_close(None)
        else:
            window.Close()

    reset_cpfigure_position()


def allow_sharexy(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if "sharexy" in kwargs:
            assert ("sharex" not in kwargs) and (
                "sharey" not in kwargs
            ), "Cannot specify sharexy with sharex or sharey"
            kwargs["sharex"] = kwargs["sharey"] = kwargs.pop("sharexy")
        return fn(*args, **kwargs)

    if wrapper.__doc__ is not None:
        wrapper.__doc__ += (
            "\n        sharexy=ax can be used to specify sharex=ax, sharey=ax"
        )
    return wrapper


def get_menu_id(d, idx):
    if idx not in d:
        d[idx] = wx.NewId()
    return d[idx]


def format_plate_data_as_array(plate_dict, plate_type):
    """ Returns an array shaped like the given plate type with the values from
    plate_dict stored in it.  Wells without data will be set to np.NaN
    plate_dict  -  dict mapping well names to data. eg: d["A01"] --> data
                   data values must be of numerical or string types
    plate_type  - '96' (return 8x12 array) or '384' (return 16x24 array)
    """
    if plate_type == "96":
        plate_shape = (8, 12)
    elif plate_type == "384":
        plate_shape = (16, 24)
    alphabet = "ABCDEFGHIJKLMNOP"
    data = numpy.zeros(plate_shape)
    data[:] = numpy.nan
    display_error = True
    for well, val in list(plate_dict.items()):
        r = alphabet.index(well[0].upper())
        c = int(well[1:]) - 1
        if r >= data.shape[0] or c >= data.shape[1]:
            if display_error:
                LOGGER.warning(
                    "A well value (%s) does not fit in the given plate type.\n" % well
                )
                display_error = False
            continue
        data[r, c] = val
    return data


def show_image(url, parent=None, needs_raise_after=True, dimensions=2, series=None):
    from ..figure import Figure

    filename = url[(url.rfind("/") + 1) :]

    try:
        provider = FileImage(
            filename=filename,
            name=os.path.splitext(filename)[0],
            pathname=os.path.dirname(url),
            volume=True if dimensions == 3 else False,
            series=series,
        )
        image = provider.provide_image(None).pixel_data
    except IOError:
        wx.MessageBox(
            'Failed to open file, "{}"'.format(filename), caption="File open error"
        )
        return
    except Exception as e:
        errordialog.display_error_dialog(
            None, e, None, "Failed to load {}".format(url), continue_only=True
        )
        return

    frame = Figure(parent=parent, title=filename)
    frame.set_subplots(dimensions=dimensions, subplots=(1, 1))

    if dimensions == 2 and image.ndim == 3:  # multichannel images
        frame.subplot_imshow_color(0, 0, image[:, :, :3], title=filename, normalize=True)
    else:  # grayscale image or volume
        frame.subplot_imshow_grayscale(0, 0, image, title=filename, normalize=True)

    frame.panel.draw()

    if needs_raise_after:
        # %$@ hack hack hack
        wx.CallAfter(lambda: frame.Raise())

    return True


def get_matplotlib_interpolation_preference():
    interpolation = get_interpolation_mode()
    if interpolation == IM_NEAREST:
        return "nearest"
    elif interpolation == IM_BILINEAR:
        return "bilinear"
    elif interpolation == IM_BICUBIC:
        return "bicubic"
    return "nearest"


def get_crosshair_cursor():
    global CROSSHAIR_CURSOR
    if CROSSHAIR_CURSOR is None:
        if sys.platform.lower().startswith("win"):
            #
            # Build the crosshair cursor image as a numpy array.
            #
            buf = numpy.ones((16, 16, 3), dtype="uint8") * 255
            buf[7, 1:-1, :] = buf[1:-1, 7, :] = 0
            abuf = numpy.ones((16, 16), dtype="uint8") * 255
            abuf[:6, :6] = abuf[9:, :6] = abuf[9:, 9:] = abuf[:6, 9:] = 0
            image = wx.ImageFromBuffer(16, 16, buf.tostring(), abuf.tostring())
            image.SetOption(wx.IMAGE_OPTION_CUR_HOTSPOT_X, 7)
            image.SetOption(wx.IMAGE_OPTION_CUR_HOTSPOT_Y, 7)
            CROSSHAIR_CURSOR = wx.Cursor(image)
        else:
            CROSSHAIR_CURSOR = wx.CROSS_CURSOR
    return CROSSHAIR_CURSOR
