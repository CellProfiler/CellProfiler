"""
Readers are modular classes designed to read in image data. Like modules, readers can also be added as plugins
using this template class. Place readers into the plugins directory to load them on startup.

This abstract class extends Reader for large image specific reading. Large images are tiled and or pyramidal.

A LargeImageReader is (more) stateful compared to Reader. While Reader *may* keep state, such as the file handler,
LargeImageReader keeps track of the current tile and pyramid level.
"""
from typing import TypedDict, Optional, Union, Tuple

from ._reader import Reader

class ReadTracker(TypedDict):
    # pyramid level
    level: Optional[int]
    # tile number
    nth: Optional[int]
    # following may contain a single single idx, or a list of idxs
    c: Optional[Union[int, list[int]]]
    z: Optional[Union[int, list[int]]]
    t: Optional[Union[int, list[int]]]

class LargeImageReader(Reader):
    def __init__(self, image_file):
        super().__init__(image_file)

        self._read_tracker: ReadTracker = {
            "level": None,
            "nth": None,
            "c": None,
            "z": None,
            "t": None,
        }

    def read_tiled(self,
                   wants_metadata_rescale=False,
                   # TODO: LIS - support c,z,t,xywh
                   c=None,
                   z=None,
                   t=None,
                   xywh=None,
                   channel_names=None,
                   ):
        """Read from a tiled, pyramdial image file.
        :param wants_metadata_rescale: if `True`, return a tuple of image and a
               tuple of (min, max) for range values of image dtype gathered from
               file metadata; if `False`, returns only the image
        :param c: read from this channel. `None` = read color image if multichannel
            or interleaved RGB.
        :param z: z-stack index
        :param t: time index
        n.b. either z or t should be "None" to specify which channel to read across.
        :param xywh: a (x, y, w, h) tuple
        :param channel_names: provide the channel names for the OME metadata

        Should return a data array with channel order [Z, ]Y, X[, C]
        """
        raise NotImplementedError(f"This reader ({self.reader_name}) does not support tiled reads")

    def get_level(self):
        return self._read_tracker["level"]

    def set_level(self, level: int):
        self._read_tracker["level"] = level

    def del_level(self):
        self._read_tracker["level"] = None

    level = property(get_level, set_level, del_level, "pyramid level")

    def get_nth(self):
        return self._read_tracker["nth"]

    def set_nth(self, nth: int):
        self._read_tracker["nth"] = nth

    def del_nth(self):
        self._read_tracker["nth"] = None

    nth = property(get_nth, set_nth, del_nth, "tile number")

    def get_channel(self):
        return self._read_tracker["c"]

    def set_channel(self, channel: Union[int, list[int]]):
        self._read_tracker["c"] = channel

    def del_channel(self):
        self._read_tracker["c"] = None

    channel = property(get_channel, set_channel, del_channel, "channel number(s)")

    def get_plane(self):
        return self._read_tracker["z"]

    def set_plane(self, plane: Union[int, list[int]]):
        self._read_tracker["z"] = plane

    def del_plane(self):
        self._read_tracker["z"] = None

    plane = property(get_plane, set_plane, del_plane, "plane number(s)")

    def get_frame(self):
        return self._read_tracker["t"]

    def set_frame(self, frame: Union[int, list[int]]):
        self._read_tracker["t"] = frame

    def del_frame(self):
        self._read_tracker["t"] = None

    frame = property(get_frame, set_frame, del_frame, "time frame number(s)")

    def go_tile_left(self):
        raise NotImplementedError(f"This reader ({self.reader_name}) does not support tiled reads")

    def go_tile_right(self):
        raise NotImplementedError(f"This reader ({self.reader_name}) does not support tiled reads")

    def go_tile_up(self):
        raise NotImplementedError(f"This reader ({self.reader_name}) does not support tiled reads")

    def go_tile_down(self):
        raise NotImplementedError(f"This reader ({self.reader_name}) does not support tiled reads")

    #  down the inverted pyramid (downscale)
    def go_level_up(self):
        raise NotImplementedError(f"This reader ({self.reader_name}) does not support pyramdial reads")

    # up the inverted pyramid (upscale)
    def go_level_down(self):
        raise NotImplementedError(f"This reader ({self.reader_name}) does not support pyramidal reads")
