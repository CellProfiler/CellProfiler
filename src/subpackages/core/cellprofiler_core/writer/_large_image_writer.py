from typing import TypedDict, Optional, Union

from ._writer import Writer

class WriteTracker(TypedDict):
    # pyramid level
    level: Optional[int]
    # tile number
    nth: Optional[int]
    # following may contain a single single idx, or a list of idxs
    c: Optional[Union[int, list[int]]]
    z: Optional[Union[int, list[int]]]
    t: Optional[Union[int, list[int]]]

class LargeImageWriter(Writer):
    def __init_values(self):
        self._write_tracker: WriteTracker = {
            "level": None,
            "nth": None,
            "c": None,
            "z": None,
            "t": None,
        }

    def __init__(self, file_path):
        super().__init__(file_path)
        self.__init_values()

    def write_tiled(self,
                   series=None,
                   c=None,
                   z=None,
                   t=None,
                   xywh=None,
                   channel_names=None,
                   ):
        """Write a series of planes from the image file. Mimics the Bioformats API
        :param series: series (pyramid level)
        :param c: write from this channel. `None` = write color image if multichannel
            or interleaved RGB.
        :param z: z-stack index
        :param t: time index
        n.b. either z or t should be "None" to specify which channel to write across.
        :param xywh: a (x, y, w, h) tuple
        :param channel_names: provide the channel names for the OME metadata
        """
        raise NotImplementedError(f"This writer ({self.writer_name}) does not support tiled writing.")

    def close(self):
        super().close()
        self.__init_values()

    def get_level(self):
        return self._write_tracker["level"]

    def set_level(self, level: int):
        self._write_tracker["level"] = level

    def del_level(self):
        self._write_tracker["level"] = None

    level = property(get_level, set_level, del_level, "pyramid level")

    def get_nth(self):
        return self._write_tracker["nth"]

    def set_nth(self, nth: int):
        self._write_tracker["nth"] = nth

    def del_nth(self):
        self._write_tracker["nth"] = None

    nth = property(get_nth, set_nth, del_nth, "tile number")

    def get_channel(self):
        return self._write_tracker["c"]

    def set_channel(self, channel: Union[int, list[int]]):
        self._write_tracker["c"] = channel

    def del_channel(self):
        self._write_tracker["c"] = None

    channel = property(get_channel, set_channel, del_channel, "channel number(s)")

    def get_plane(self):
        return self._write_tracker["z"]

    def set_plane(self, plane: Union[int, list[int]]):
        self._write_tracker["z"] = plane

    def del_plane(self):
        self._write_tracker["z"] = None

    plane = property(get_plane, set_plane, del_plane, "plane number(s)")

    def get_frame(self):
        return self._write_tracker["t"]

    def set_frame(self, frame: Union[int, list[int]]):
        self._write_tracker["t"] = frame

    def del_frame(self):
        self._write_tracker["t"] = None

    frame = property(get_frame, set_frame, del_frame, "time frame number(s)")
