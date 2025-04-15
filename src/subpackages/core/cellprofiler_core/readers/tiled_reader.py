import tifffile
import zarr
import numpy
import dask.array
from dask.array.core import Array as daskArray
import xmltodict

from math import ceil
from typing import TypedDict, Literal, Optional
from collections import defaultdict

from ..constants.image import MD_SIZE_S, MD_SIZE_C, MD_SIZE_Z, MD_SIZE_T, MD_SIZE_Y, MD_SIZE_X, MD_SERIES_NAME
from ..reader import Reader

class Resolution(TypedDict):
    shape: tuple[int, ...]
    dims: tuple[str, ...]
    height: int
    width: int
    channels: int
    max_tile_height: int
    max_tile_width: int
    n_tiles_y: int
    n_tiles_x: int


class StandardMetadata(TypedDict):
    endiness: Literal["<", ">"]
    dim_order: str
    y_idx: int
    x_idx: int
    c_idx: int
    z_idx: int
    t_idx: int
    y_size: int
    x_size: int
    z_size: int
    c_size: int
    t_size: int
    y_mag: float
    y_mag_unit: str
    x_mag: float
    x_mag_unit: str
    z_mag: float
    z_mag_unit: str
    shape: tuple[int, ...]
    dtype: str
    channel_names: tuple[str, ...]
    tile_height: int
    tile_width: int
    resolutions: dict[int, Resolution]

SUPPORTED_EXTENSIONS = {'.ome.tif', '.ome.tiff'}
SUPPORTED_SCHEMES = {'file'}

class TiledImageReader(Reader):
    """
    Reads tiled/pyramidal ome-tiff images
    """

    reader_name = "TiledImage"
    variable_revision_number = 1
    supported_filetypes = SUPPORTED_EXTENSIONS
    supported_schemes = SUPPORTED_SCHEMES

    def __init__(self, image_file):
        self.variable_revision_number = 1
        self.__data = None
        self.__zarr_data = None
        self.__store = None
        self.__lru_cache = None
        self.__reader = None
        self.__path = None
        self.__cached_meta = None
        self.__cached_full_meta = None
        self.__read_tracker = {
                "level": None,
                "frame": None,
                "nth": None,
                "frame_idx": 0,
                "row_idx": 1,
                "col_idx": 2,
        }
        super().__init__(image_file)

    def __del__(self):
        self.close()

    def __get_reader(self):
        if self.__reader is None:
            self.__path = self.file.path 
            self.__cached_meta = None
            self.__cached_full_meta = None
            self.__read_tracker = {
                    "level": None,
                    "frame": None,
                    "nth": None,
                    "frame_idx": 0,
                    "row_idx": 1,
                    "col_idx": 2,
            }

            self.__store = tifffile.imread(self.__path, aszarr=True)
            self.__lru_cache = zarr.LRUStoreCache(self.__store, max_size=2**29)
            self.__reader = zarr.open(self.__lru_cache, mode='r')
        return self.__reader

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
        def order_dims(dask_array: dask.array.Array, level: int):
            row_idx = self._res[level]["dims"].index("height")
            col_idx = self._res[level]["dims"].index("width")
            frame_idx = self._res[level]["dims"].index("sequence")

            standard_idxs = (0, 1, 2)
            assert row_idx != col_idx
            assert row_idx != frame_idx
            assert col_idx != frame_idx
            assert row_idx in standard_idxs
            assert col_idx in standard_idxs
            assert frame_idx in standard_idxs

            idxs: dict[int, slice] = dict()
            self.__read_tracker["row_idx"] = row_idx
            self.__read_tracker["col_idx"] = col_idx
            self.__read_tracker["frame_idx"] = frame_idx

            if dask_array.ndim == 3:
                return dask_array.transpose([row_idx, col_idx, frame_idx])
            if dask_array.ndim == 4:
                # TODO: LIS - implement volumetric
                raise NotImplementedError("Not yet implemented")
            return dask_array

        path = self.file.path
        if not self.__data or path is None or path != self.__path:
            reader = self.__get_reader()
            self.__zarr_data: list[zarr.Array] = [
                reader[int(dataset["path"])]
                for dataset in reader.attrs["multiscales"][0]["datasets"]
            ]
            self.__data: list[daskArray] = [
                order_dims(
                    dask.array.from_zarr(z),
                    len(self.__zarr_data)-1
                ) for z in self.__zarr_data
            ]

        if channel_names is not None:
            channel_names.extend(self._meta["channel_names"])

        self.__read_tracker["level"] = len(self.__data) - 1
        self.__read_tracker["frame"] = self._set_frame(start=0, stop=3, lvl=self.__read_tracker["level"])
        self.__read_tracker["nth"] = 0

        if wants_metadata_rescale:
            dtype = self._meta["dtype"]
            if numpy.issubdtype(dtype, numpy.integer):
                info = numpy.iinfo(dtype)
            elif numpy.issubdtype(dtype, numpy.floating):
                info = numpy.finfo(dtype)
            else:
                raise TypeError(f"Unsupported data type: {dtype}")

            return self.__data[self.__read_tracker["level"]], (float(info.min), float(info.max))

        # TODO: - LIS: Not sure what the best thing is to return here yet
        # right now just the lowest resolution in the pyramid
        return self.__data[self.__read_tracker["level"]]

    def get_level(self):
        return self.__read_tracker["level"]

    def get_nth(self):
        return self.__read_tracker["nth"]

    def get_frame(self):
        return self.__read_tracker["frame"]

    def _tracked_tile(self):
        nth = self.__read_tracker["nth"]
        level = self.__read_tracker["level"]
        frame = self.__read_tracker["frame"]

        assert nth >= 0
        assert nth <= self._nn(level), f"only {self._nn(level)} tiles at level {level}, got {nth}"

        return self._tile_n(nth=nth, frame=frame, level=level)

    def go_tile_left(self):
        nth = self.__read_tracker["nth"]
        level = self.__read_tracker["level"]
        curr_x = nth % self._nx(level)
        if curr_x > 0:
            self.__read_tracker["nth"] = nth - 1
        return self._tracked_tile()

    def go_tile_right(self):
        nth = self.__read_tracker["nth"]
        level = self.__read_tracker["level"]
        curr_x = nth % self._nx(level)
        if curr_x < (self._nx(level) - 1):
            self.__read_tracker["nth"] = nth + 1
        return self._tracked_tile()

    def go_tile_up(self):
        nth = self.__read_tracker["nth"]
        level = self.__read_tracker["level"]
        new_nth = nth - self._nx(level)
        if new_nth >= 0:
            self.__read_tracker["nth"] = new_nth
        return self._tracked_tile()

    # up the inverted pyramid (upscale)
    def go_tile_down(self):
        nth = self.__read_tracker["nth"]
        level = self.__read_tracker["level"]
        new_nth = nth + self._nx(level)
        if new_nth < self._nn(level):
            self.__read_tracker["nth"] = new_nth
        return self._tracked_tile()

    #  down the inverted pyramid (downscale)
    def go_level_up(self):
        level = self.__read_tracker["level"]
        nth = self.__read_tracker["nth"]
        if level < (len(self._res) - 1):
            new_iy = self._iy(level, nth) // 2
            new_ix = self._ix(level, nth) // 2

            level += 1

            new_nx = self._nx(level)

            self.__read_tracker["level"] = level
            self.__read_tracker["nth"] = new_iy * new_nx + new_ix
        return self._tracked_tile()

    def go_level_down(self):
        level = self.__read_tracker["level"]
        nth = self.__read_tracker["nth"]
        if level > 0:
            new_iy = self._iy(level, nth) * 2
            new_ix = self._ix(level, nth) * 2

            level = max(0, level - 1)

            new_nx = self._nx(level)

            self.__read_tracker["level"] = level
            self.__read_tracker["nth"] = new_iy * new_nx + new_ix
        return self._tracked_tile()

    @classmethod
    def supports_format(cls, image_file, allow_open=False, volume=False, tiled=False):
        """This function needs to evaluate whether a given ImageFile object
        can be read by this reader class.

        Return value should be an integer representing suitability:
        -1 - 'I can't read this at all'
        1 - 'I am the one true reader for this format, don't even bother checking any others'
        2 - 'I am well-suited to this format'
        3 - 'I can read this format, but I might not be the best',
        4 - 'I can give it a go, if you must'

        The allow_open parameter dictates whether the reader is permitted to read the file when
        making this decision. If False the decision should be made using file extension only.
        Any opened files should be closed before returning.

        The volume parameter specifies whether the reader will need to return a 3D array.
        ."""
        if not tiled:
            return -1
        if image_file.scheme not in SUPPORTED_SCHEMES:
            return -1
        if image_file.safe_full_extension in SUPPORTED_EXTENSIONS:
            return 1
        return -1

    def close(self):
        # If your reader opens a file, this needs to release any active lock,


        if self.__lru_cache:
            self.__lru_cache.invalidate()
            self.__lru_cache.close()
        if self.__store:
            self.__store.close()

        self.__data = None
        self.__zarr_data = None
        self.__store = None
        self.__lru_cache = None
        self.__reader = None
        self.__path = None
        self.__cached_meta = None
        self.__cached_full_meta = None
        self.__read_tracker = {
                "level": None,
                "frame": None,
                "nth": None,
                "frame_idx": 0,
                "row_idx": 1,
                "col_idx": 2,
        }

    def get_series_metadata(self):
        """Should return a dictionary with the following keys:
        Key names are in cellprofiler_core.constants.image
        MD_SIZE_S - int reflecting the number of series
        MD_SIZE_Y - list of Y dimension sizes, one element per series.
        MD_SIZE_X - list of X dimension sizes, one element per series.
        MD_SIZE_Z - list of Z dimension sizes, one element per series.
        MD_SIZE_C - list of C dimension sizes, one element per series.
        MD_SIZE_T - list of T dimension sizes, one element per series.
        MD_SERIES_NAME - list of series names, one element per series.
        """
        full_meta = self._full_meta
        standard_meta = self._meta
        meta_dict = defaultdict(list)

        meta_dict[MD_SIZE_S] = full_meta["num_series"]

        reader = self.__get_reader()
        series_count = reader.get_length()
        meta_dict[MD_SIZE_S] = series_count
        for i in range(series_count):
            meta_series = full_meta["series"][i]
            meta_series_shape = meta_series["shape"]

            meta_dict[MD_SIZE_Z].append(standard_meta["size_z"])
            meta_dict[MD_SIZE_T].append(standard_meta["size_t"])
            # TODO: LIS - don't hardcode 0,1,2
            meta_dict[MD_SIZE_C].append(meta_series_shape[0])
            meta_dict[MD_SIZE_Y].append(meta_series_shape[1])
            meta_dict[MD_SIZE_X].append(meta_series_shape[2])
            meta_dict[MD_SERIES_NAME].append(meta_series["name"] or "<no_name>")
        return meta_dict

    def _tile_n(self, nth: int, frame: slice = slice(0,1,1), level: int = 0) -> daskArray:
        assert self.__data, "No data read yet (read_tile failed or was never called)"
        assert len(self.__data) > level
        assert level >= 0

        _res = self._res
        row_slice, col_slice = self._n_slices(nth, level)

        tile = self.__data[level][row_slice, col_slice, frame]

        assert 0 not in tile.shape, f"invalid shape {tile.shape}, from idxs {idxs}"

        return tile

    def _decrement_frame(self, curr_frame: slice, level: Optional[int]) -> slice:
        return self._set_frame(start = curr_frame.start - 1, stop = curr_frame.stop - 1, step = curr_frame.step, level = level)

    def _increment_frame(self, curr_frame: slice, level: Optional[int]) -> slice:
        return self._set_frame(start = curr_frame.start + 1, stop = curr_frame.stop + 1, step = curr_frame.step, level = level)

    def _set_frame(self, start: int, stop: Optional[int] = None, step: Optional[int] = None, lvl: Optional[int] = None) -> slice:
        if lvl:
            max_frame = self._res[lvl]["channels"] - 1
        else:
            max_frame = self._meta["c_size"] - 1

        # start can't surpass max_frame, can't go below 0
        start = max(0, min(start, max_frame))

        if stop is None:
            stop = start + 1
        else:
            # end must be at least one greater than start
            stop = max(start + 1, stop)

        if step is None:
            step = 1
        # step is allowed to exceed max_frame, as long as stop is set properly

        return slice(start, stop, step)

    def _n_slices(self, n: int, lvl: int = 0) -> tuple[slice, slice]:
        """0-indexed, assumes row major"""
        n_tiles_x = self._res[lvl]["n_tiles_x"]
        tile_row = int(n // n_tiles_x)
        tile_col = int(n % n_tiles_x)

        assert n_tiles_x > 0

        row_start = int(tile_row * self._res[lvl]["max_tile_height"])
        row_end = int(row_start + self._res[lvl]["max_tile_height"])

        col_start = int(tile_col * self._res[lvl]["max_tile_width"])
        col_end = int(col_start + self._res[lvl]["max_tile_width"])

        assert row_end > row_start
        assert col_end > col_start
        
        return (slice(row_start, row_end, 1), slice(col_start, col_end, 1))

    def _nn(self, lvl: int):
        """num of nth values"""
        return self._nx(lvl) * self._ny(lvl)

    def _iy(self, lvl: int, n: int):
        """idx of tile in the y direction"""
        _tile_width = self._tile_width(lvl)
        _img_width = self._res[lvl]["width"]

        n_tile_cols = ceil(_img_width / _tile_width)

        return n // n_tile_cols

    def _ny(self, lvl: int):
        """num tiles in y direction"""
        return self._res[lvl]["n_tiles_y"]

    def _ix(self, lvl: int, n: int):
        """idx of tile in the x direction"""
        _tile_width = self._tile_width(lvl)
        _img_width = self._res[lvl]["width"]

        n_tile_cols = ceil(_img_width / _tile_width)

        return n % n_tile_cols

    def _nx(self, lvl: int):
        """num tiles in the x direction"""
        return self._res[lvl]["n_tiles_x"]

    def _tile_height(self, lvl: int):
        return self._res[lvl]["max_tile_height"]

    def _tile_width(self, lvl: int):
        return self._res[lvl]["max_tile_width"]

    @property
    def _res(self):
        meta = self._meta
        return meta["resolutions"]

    @property
    def _meta(self):
        if not self.__cached_meta:
            self.__cached_meta = self.__extract_standard_metadata()
        return self.__cached_meta

    @property
    def _full_meta(self):
        if not self.__cached_full_meta:
            self.__cached_full_meta = self.__extract_metadata(max_pages=None, include_tags=True)
        return self.__cached_full_meta

    # TODO: LIS - Clean this up

    def __extract_standard_metadata(self) -> StandardMetadata:
        full_meta = self.__extract_metadata(max_pages=None, include_tags=False)
        pixels_meta = full_meta["metadatas"]["ome"]["OME"]["Image"]["Pixels"]

        endiness = "<" if pixels_meta["@BigEndian"] == "false" else ">"
        dim_order = str(pixels_meta["@DimensionOrder"])

        y_idx = dim_order.index("Y")
        x_idx = dim_order.index("X")
        c_idx = dim_order.index("C")
        z_idx = dim_order.index("Z")
        t_idx = dim_order.index("T")

        y_size = int(pixels_meta["@SizeY"])
        x_size = int(pixels_meta["@SizeX"])
        z_size = int(pixels_meta["@SizeZ"])
        c_size = int(pixels_meta["@SizeC"])
        t_size = int(pixels_meta["@SizeT"])

        y_mag = float(pixels_meta["@PhysicalSizeY"])
        y_mag_unit = str(pixels_meta["@PhysicalSizeYUnit"])
        x_mag = float(pixels_meta["@PhysicalSizeX"])
        x_mag_unit = str(pixels_meta["@PhysicalSizeXUnit"])
        z_mag = float(pixels_meta["@PhysicalSizeZ"])
        z_mag_unit = str(pixels_meta["@PhysicalSizeZUnit"])

        shape = [1] * 5
        shape[y_idx] = y_size
        shape[x_idx] = x_size
        shape[c_idx] = c_size
        shape[z_idx] = z_size
        shape[t_idx] = t_size
        shape = tuple(shape)

        dtype = str(pixels_meta["@Type"])

        channel_names = tuple(map(lambda channel_info: str(channel_info["@Name"]), pixels_meta["Channel"]))

        tile_height = int(full_meta["pages"][0]["tilelength"])
        tile_width = int(full_meta["pages"][0]["tilewidth"])

        resolutions: dict[int, Resolution] = dict()

        levels_dim_order = tuple(map(lambda d: str(d), full_meta["series"][0]["dims"]))
        levels_height_idx = levels_dim_order.index("height")
        levels_width_idx = levels_dim_order.index("width")
        levels_seq_idx = levels_dim_order.index("sequence")

        levels = full_meta["series"][0]["pyramid"]["levels"]
        for i, level in enumerate(levels):
            level_shape = tuple(map(lambda v: int(v), level["shape"]))

            level_dims = levels_dim_order
            level_height = level_shape[levels_height_idx]
            level_width = level_shape[levels_width_idx]
            level_channels = level_shape[levels_seq_idx]
            level_max_tile_height = min(tile_height, level_height)
            level_max_tile_width = min(tile_width, level_width)
            level_n_tiles_y = ceil(level_height / level_max_tile_height)
            level_n_tiles_x = ceil(level_width / level_max_tile_width)

            resolutions[i] = {
                "shape": level_shape,
                "dims": level_dims,
                "height": level_height,
                "width": level_width,
                "channels": level_channels,
                "max_tile_height": level_max_tile_height,
                "max_tile_width": level_max_tile_width,
                "n_tiles_y": level_n_tiles_y,
                "n_tiles_x": level_n_tiles_x,
            }

        return {
            "endiness": endiness,
            "dim_order": dim_order,
            "y_idx": y_idx,
            "x_idx": x_idx,
            "c_idx": c_idx,
            "z_idx": z_idx,
            "t_idx": t_idx,
            "y_size": y_size,
            "x_size": x_size,
            "z_size": z_size,
            "c_size": c_size,
            "t_size": t_size,
            "y_mag": y_mag,
            "y_mag_unit": y_mag_unit,
            "x_mag": x_mag,
            "x_mag_unit": x_mag_unit,
            "z_mag": z_mag,
            "z_mag_unit": z_mag_unit,
            "shape": shape,
            "dtype": dtype,
            "channel_names": channel_names,
            "tile_height": tile_height,
            "tile_width": tile_width,
            "resolutions": resolutions,
        }

    def __extract_metadata(self, max_pages: Optional[int] = None, include_tags: bool = False):
        def sp(val): return f"{val:_}" if type(val) is type(
            1) or type(val) is type(1.1) else val

        def spmap(val): return tuple(
            map(lambda x: sp(x), val)
        )\
            if type(val) is type((1, 1))\
            else list(
            map(lambda x: sp(x), val)
        )\
            if type(val) is type([1, 1])\
            else val

        metadata = dict()

        with tifffile.TiffFile(self.file.path) as tif:
            metadata['byteorder'] = tif.byteorder
            metadata['num_pages'] = len(tif.pages)
            metadata['num_series'] = len(tif.series)

            is_thing_list = [x for x in dir(tif) if x.startswith('is_')]
            has_metadata_list = [x for x in dir(tif) if x.endswith('_metadata')]
            metadata['kinds'] = list()
            metadata['metadatas'] = dict()
            for is_thing in is_thing_list:
                is_thing_val = getattr(tif, is_thing)
                if is_thing_val:
                    metadata['kinds'].append(is_thing[3:])
            for md in has_metadata_list:
                md_val = getattr(tif, md)
                if md_val:
                    if md == "ome_metadata":
                        metadata['metadatas'][md[:-9]] = xmltodict.parse(md_val)
                    else:
                        metadata['metadatas'][md[:-9]] = md_val

            if metadata['num_series'] > 0:
                metadata['series'] = list()
                for i, _ in enumerate(tif.series):
                    metadata['series'].append(dict())
                    metadata['series'][i]['axes'] = tif.series[i].axes
                    if hasattr(tif.series[i], '_axes_expanded'):
                        metadata['series'][i]['axes_expanded'] = tif.series[i]._axes_expanded  # type: ignore
                    if hasattr(tif.series[i], 'axes_expanded'):
                        metadata['series'][i]['axes_expanded'] = tif.series[i]._axes_expanded  # type: ignore
                    if hasattr(tif.series[i], 'dims'):
                        metadata['series'][i]['dims'] = tif.series[i].dims
                    metadata['series'][i]['dtype'] = str(tif.series[i].dtype)
                    metadata['series'][i]['is_multifile'] = tif.series[i].is_multifile
                    metadata['series'][i]['kind'] = tif.series[i].kind
                    metadata['series'][i]['name'] = tif.series[i].name
                    metadata['series'][i]['shape'] = spmap(tif.series[i].shape)
                    if hasattr(tif.series[i], '_shape_expanded'):
                        metadata['series'][i]['shape_expanded'] = spmap(
                            tif.series[i]._shape_expanded)  # type: ignore
                    metadata['series'][i]['size'] = sp(tif.series[i].size)
                    metadata['series'][i]['sizes'] = tif.series[i].sizes
                    metadata['series'][i]['is_pyramidal'] = tif.series[i].is_pyramidal
                    if metadata['series'][i]['is_pyramidal']:
                        metadata['series'][i]['pyramid'] = dict()
                        metadata['series'][i]['pyramid']['num_levels'] = len(tif.series[i].levels)
                        metadata['series'][i]['pyramid']['levels'] = list()
                        for ii, level in enumerate(tif.series[i].levels):
                            metadata['series'][i]['pyramid']['levels'].append(dict())
                            metadata['series'][i]['pyramid']['levels'][ii]['dims'] = spmap(level.dims)
                            metadata['series'][i]['pyramid']['levels'][ii]['shape'] = spmap(level.shape)
                            metadata['series'][i]['pyramid']['levels'][ii]['size'] = sp(level.size)
                            metadata['series'][i]['pyramid']['levels'][ii]['sizes'] = level.sizes

            metadata['pages'] = list()
            for i, _ in enumerate(tif.pages):
                if max_pages is not None and i >= max_pages:
                    break
                metadata['pages'].append(dict())
                metadata['pages'][i]['axes'] = hasattr(
                    tif.pages[i], 'axes') and tif.pages[i].axes or 'undefined'
                metadata['pages'][i]['chunked'] = hasattr(
                    tif.pages[i], 'chunked') and spmap(tif.pages[i].chunked) or 'undefined'
                metadata['pages'][i]['chunks'] = hasattr(
                    tif.pages[i], 'chunks') and spmap(tif.pages[i].chunks) or 'undefined'
                metadata['pages'][i]['compression'] = hasattr(
                    tif.pages[i], 'compression') and str(tif.pages[i].compression) or 'undefined'
                if hasattr(tif.pages[i], 'colormap'):
                    metadata['pages'][i]['colormap'] = dict()
                    if hasattr(tif.pages[i].colormap, 'ndim'):  # type: ignore
                        metadata['pages'][i]['colormap']['ndim'] = tif.pages[i].colormap.ndim  # type: ignore
                    else:
                        metadata['pages'][i]['colormap']['ndim'] = 'undefined'
                    if hasattr(tif.pages[i].colormap, 'nbytes'):  # type: ignore
                        metadata['pages'][i]['colormap']['nbytes'] = tif.pages[i].colormap.nbytes  # type: ignore
                    else:
                        metadata['pages'][i]['colormap']['nbytes'] = 'undefined'
                    if hasattr(tif.pages[i].colormap, 'shape'):  # type: ignore
                        metadata['pages'][i]['shape'] = spmap(
                            tif.pages[i].colormap.shape)  # type: ignore
                    else:
                        metadata['pages'][i]['shape'] = 'undefined'
                metadata['pages'][i]['dtype'] = hasattr(
                    tif.pages[i], 'dtype') and str(tif.pages[i].dtype) or 'undefined'
                metadata['pages'][i]['imagedepth'] = hasattr(
                    tif.pages[i], 'imagedepth') and tif.pages[i].imagedepth or 'undefined'  # type: ignore
                metadata['pages'][i]['imagelength'] = hasattr(
                    tif.pages[i], 'imagelength') and sp(tif.pages[i].imagelength) or 'undefined'  # type: ignore
                metadata['pages'][i]['imagewidth'] = hasattr(
                    tif.pages[i], 'imagewidth') and sp(tif.pages[i].imagewidth) or 'undefined'  # type: ignore
                metadata['pages'][i]['nbytes'] = hasattr(
                    tif.pages[i], 'nbytes') and sp(tif.pages[i].nbytes) or 'undefined'
                metadata['pages'][i]['shape'] = hasattr(
                    tif.pages[i], 'shape') and spmap(tif.pages[i].shape) or 'undefined'
                metadata['pages'][i]['size'] = hasattr(tif.pages[i].size, "size") and sp(tif.pages[i].size) or "n/a"
                if hasattr(tif.pages[i], 'resolution'):
                    metadata['pages'][i]['resolution'] = spmap(tif.pages[i].resolution)  # type: ignore
                if hasattr(tif.pages[i], 'resolutionunit'):
                    metadata['pages'][i]['resolutionunit_name'] = tif.pages[i].resolutionunit.name  # type: ignore
                    metadata['pages'][i]['resolutionunit_value'] = sp(tif.pages[i].resolutionunit.value)  # type: ignore
                if hasattr(tif.pages[i], 'software'):
                    metadata['pages'][i]['software'] = tif.pages[i].software  # type: ignore
                if hasattr(tif.pages[i], 'software'):
                    metadata['pages'][i]['tile'] = spmap(tif.pages[i].tile)  # type: ignore
                if hasattr(tif.pages[i], 'tile'):
                    metadata['pages'][i]['tilewidth'] = sp(tif.pages[i].tilewidth)  # type: ignore
                if hasattr(tif.pages[i], 'tilewidth'):
                    metadata['pages'][i]['tilelength'] = sp(tif.pages[i].tilelength)  # type: ignore

                if include_tags:
                    tag_keys = hasattr(
                        tif.pages[i], 'tags') and tif.pages[i].tags.keys() or []  # type: ignore
                    metadata['pages'][i]['tags'] = dict()
                    for key in tag_keys:
                        metadata['pages'][i]['tags'][key] = dict()  # type: ignore
                        metadata['pages'][i]['tags'][key]['code'] = tif.pages[i].tags[key].code  # type: ignore
                        # type: ignore
                        metadata['pages'][i]['tags'][key]['count'] = tif.pages[i].tags[key].count  # type: ignore
                        metadata['pages'][i]['tags'][key]['dtype_name'] = tif.pages[i].tags[key].dtype_name  # type: ignore
                        metadata['pages'][i]['tags'][key]['name'] = tif.pages[i].tags[key].name  # type: ignore
                        metadata['pages'][i]['tags'][key]['dataformat'] = tif.pages[i].tags[key].dataformat  # type: ignore
                        metadata['pages'][i]['tags'][key]['valuebytecount'] = tif.pages[i].tags[key].valuebytecount  # type: ignore
                        val = tif.pages[i].tags[key].value  # type: ignore
                        if tif.pages[i].tags[key].valuebytecount < 100:  # type: ignore
                            if type(val) is type(1):
                                metadata['pages'][i]['tags'][key]['value'] = val
                            elif type(val) is type(1.0):
                                metadata['pages'][i]['tags'][key]['value'] = val
                            else:
                                metadata['pages'][i]['tags'][key]['value'] = str(val)
                        else:
                            metadata['pages'][i]['tags'][key]['value'] = "LOTS OF STUFF"
        del tif

        return metadata

