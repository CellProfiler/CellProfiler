import logging
import os
import re
from urllib.parse import unquote, urlsplit
from urllib.request import url2pathname

import fsspec
import numpy
import zarr
from fsspec.core import url_to_fs
from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper
from lxml import etree
from zarr.storage import FsspecStore, LocalStore

from ..constants.image import MD_SIZE_S, MD_SIZE_C, MD_SIZE_Z, MD_SIZE_T, MD_SIZE_Y, MD_SIZE_X, MD_SERIES_NAME
from ..constants.reader import ZARR_FILETYPE

from ..reader import Reader

LOGGER = logging.getLogger(__name__)

FORMAT_TESTER = re.compile(r"\.zarr([\\/]|$)", flags=re.IGNORECASE)
SUPPORTED_SCHEMES = {'file', 's3'}


class NGFFReader(Reader):
    """
    A reader for OME-NGFF files with the .zarr extension. Supports both 'normal' and
    'HCS' storage modes.

    OME-NGFF files below spec version 0.4 which are stored on S3/object storage
    should have the zarr.consolidate_metadata() function run on them
    to provide optimal performance over a network.

    Other non-OME .zarr format images may be readable, but are not strictly supported.
    """

    reader_name = "OME-NGFF"
    variable_revision_number = 1
    supported_filetypes = {'.zarr', '.ome.zarr'}
    supported_schemes = SUPPORTED_SCHEMES

    ZARR_READER_CACHE = {}

    def __init__(self, image_file):
        super().__init__(image_file)

        self._reader = None
        # List of tuples denoting (array_path, array_name)
        self._series_map = None
        self.root, self.group = ZARR_FILETYPE.split(self.file.url, maxsplit=1)
        self.group = unquote(self.group).strip("/\\").replace("\\", "/")
        parsed = urlsplit(self.root)
        if parsed.scheme.lower() == "file":
            self._local_path = url2pathname(parsed.path)
            if parsed.netloc and parsed.netloc != "localhost":
                self._local_path = f"//{parsed.netloc}{self._local_path}"
        elif not parsed.scheme or (os.name == "nt" and len(parsed.scheme) == 1):
            self._local_path = self.root
        else:
            self._local_path = None
        self.planes_extracted = False

    def __del__(self):
        self.close()

    def get_reader(self):
        cached = self.ZARR_READER_CACHE.get(self.root)
        if cached is None:
            if self._local_path is not None:
                if not os.path.isdir(self._local_path):
                    raise FileNotFoundError(f'The file, "{self.root}", does not exist.')
                store = LocalStore(self._local_path, read_only=True)
            else:
                filesystem, path = url_to_fs(self.root)
                store = FsspecStore(
                    AsyncFileSystemWrapper(
                        fs=filesystem, asynchronous=True, skip_instance_cache=True
                    ),
                    path=path,
                    read_only=True,
                )
            try:
                root_reader = zarr.open(store=store, mode="r")
            except Exception:
                store.close()
                raise
            if self.root.lower().startswith("s3"):
                LOGGER.info("Zarr is stored on S3, will try to read directly.")
                if (
                    isinstance(root_reader, zarr.Group)
                    and root_reader.metadata.consolidated_metadata is None
                ):
                    LOGGER.warning(
                        "Image is on S3 but lacks consolidated metadata. "
                        "This may degrade reading performance. URL: %s", self.root
                    )
            cached = root_reader, {}
            self.ZARR_READER_CACHE[self.root] = cached
        root_reader, series_maps = cached
        if self._reader is None or self._reader.store is not root_reader.store:
            self._reader = root_reader[self.group] if self.group else root_reader
            self._series_map = series_maps.get(self.group)
        return self._reader

    @classmethod
    def clear_cached_readers(cls):
        for reader, _ in cls.ZARR_READER_CACHE.values():
            reader.store.close()
        cls.ZARR_READER_CACHE.clear()

    def read(self,
             wants_metadata_rescale=False,
             series=None,
             index=None,
             c=None,
             z=None,
             t=None,
             xywh=None,
             channel_names=None,
             ):
        """Read a single plane from the image file.
        :param wants_metadata_rescale: if `True`, return a tuple of image and a
               tuple of (min, max) for range values of image dtype gathered from
               file metadata; if `False`, returns only the image
        :param c: read from this channel. `None` = read color image if multichannel
            or interleaved RGB.
        :param z: z-stack index
        :param t: time index
        :param series: series for ``.flex`` and similar multi-stack formats
        :param index: if `None`, fall back to ``zct``, otherwise load the indexed frame
        :param xywh: a (x, y, w, h) tuple
        :param channel_names: provide the channel names for the OME metadata
        """
        LOGGER.debug(f"Reading {c=}, {z=}, {t=}, {series=}, {index=}, {xywh=}")
        c2 = None if c is None else c + 1
        z2 = None if z is None else z + 1
        t2 = None if t is None else t + 1
        if xywh is not None:
            x, y, w, h = xywh
            x = round(x)
            y = round(y)
            x2 = x + w
            y2 = y + h
        else:
            y, y2, x, x2 = None, None, None, None
        reader = self.get_reader()
        if series is None:
            series = 0
        if self._series_map is None or (
            series >= len(self._series_map) and not self.planes_extracted
        ):
            first_only = not self.planes_extracted and series == 0
            arrays = self.find_arrays(reader, multiscales=False, first_only=first_only)
            self.build_series_map(arrays, complete=not first_only)
        if not self._series_map:
            raise IOError("Zarr appears to contain no valid series. Unable to load data.")
        series_path, series_name = self._series_map[series]
        series_reader = reader if isinstance(reader, zarr.Array) else reader[series_path]
        num_dimensions = len(series_reader.shape)
        if num_dimensions == 5:
            image = series_reader[t:t2, c:c2, z:z2, y:y2, x:x2]
        elif num_dimensions == 4:
            image = series_reader[c:c2, z:z2, y:y2, x:x2]
        elif num_dimensions == 3:
            image = series_reader[z:z2, y:y2, x:x2]
        elif num_dimensions == 2:
            image = series_reader[y:y2, x:x2]
        else:
            raise NotImplementedError(f"Unsupported dimensionality: {num_dimensions}")
        # Remove redundant axes
        image = numpy.squeeze(image)
        # C needs to be the last axis, but z should be first. Thank you CellProfiler.
        if len(image.shape) > 2 and z is not None:
            image = numpy.moveaxis(image, 0, -1)
        elif len(image.shape) > 3:
            image = numpy.moveaxis(image, 0, -1)

        if wants_metadata_rescale:
            scale = self.get_max_sample_value()
            return image, (0, scale) if scale is not None else None
        return image

    def read_volume(self,
                    wants_metadata_rescale=False,
                    series=None,
                    c=None,
                    z=None,
                    t=None,
                    xywh=None,
                    channel_names=None,
                    ):
        return self.read(
            wants_metadata_rescale=wants_metadata_rescale,
            series=series,
            c=c,
            z=None,
            t=t,
            xywh=xywh,
            channel_names=channel_names
        )

    @classmethod
    def supports_format(cls, image_file, allow_open=False, volume=False):
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
        if FORMAT_TESTER.search(image_file.url) is not None:
            head, tail = os.path.splitext(image_file.path)
            if tail.lower() not in ('', '.zarr'):
                # An especially mean user may be pointing at a non-zarr file within the zarr?
                return 2
            return 1
        return -1

    def close(self):
        # If your reader opens a file, this needs to release any active lock,
        self._reader = None
        self._series_map = None

    def build_series_map(self, arrays, complete=False):
        prefix = self._reader.path + "/" if self._reader.path else ""
        self._series_map = [
            (array.path.removeprefix(prefix), name) for array, name in arrays
        ]
        if complete:
            self.ZARR_READER_CACHE[self.root][1][self.group] = self._series_map

    def get_series_metadata(self):
        """Should return a dictionary with the following keys:
        Key names are in cellprofiler_core.constants.image
        MD_SIZE_S - int reflecting the number of series
        MD_SIZE_X - list of X dimension sizes, one element per series.
        MD_SIZE_Y - list of Y dimension sizes, one element per series.
        MD_SIZE_Z - list of Z dimension sizes, one element per series.
        MD_SIZE_C - list of C dimension sizes, one element per series.
        MD_SIZE_T - list of T dimension sizes, one element per series.
        MD_SERIES_NAME - list of series names, one element per series.
        """
        self.planes_extracted = True
        meta_dict = {
            MD_SIZE_S: 0,
            MD_SIZE_X: [],
            MD_SIZE_Y: [],
            MD_SIZE_Z: [],
            MD_SIZE_C: [],
            MD_SIZE_T: [],
            MD_SERIES_NAME: [],
        }
        reader = self.get_reader()
        arrays = self.find_arrays(reader, multiscales=False, first_only=False)
        self.build_series_map(arrays, complete=True)
        series_count = len(arrays)
        meta_dict[MD_SIZE_S] = series_count
        for idx, (array, series_name) in enumerate(arrays):
            dims = list(array.shape)
            meta_dict[MD_SIZE_X].append(dims.pop())
            meta_dict[MD_SIZE_Y].append(dims.pop())
            meta_dict[MD_SIZE_Z].append(dims.pop() if dims else 1)
            meta_dict[MD_SIZE_C].append(dims.pop() if dims else 1)
            meta_dict[MD_SIZE_T].append(dims.pop() if dims else 1)
            meta_dict[MD_SERIES_NAME].append(series_name)
        return meta_dict

    def _read_ome_xml(self):
        root = self._local_path if self._local_path is not None else self.root
        try:
            with fsspec.open(root.rstrip("/\\") + "/OME/METADATA.ome.xml", "rb") as stream:
                return etree.fromstring(stream.read())
        except FileNotFoundError:
            return None

    def get_max_sample_value(self):
        root = self._read_ome_xml()
        if root is None:
            return None

        namespaces = {
            "ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"
        }

        # Find the MaxSampleValue field
        # only present in transitional bioformats2raw.layout
        # https://ngff.openmicroscopy.org/0.5/index.html#bf2raw
        max_sample_value = root.xpath(
            "//ome:XMLAnnotation[ome:Value/ome:OriginalMetadata/ome:Key='MaxSampleValue']"
            "/ome:Value/ome:OriginalMetadata/ome:Value",
            namespaces=namespaces
        )

        # Retrieve the text content of MaxSampleValue, if found
        if max_sample_value:
            try:
                max_sample_value = int(max_sample_value[0].text)
            except (TypeError, ValueError):
                max_sample_value = None
            return max_sample_value
        else:
            return None

    @staticmethod
    def _metadata(zarr_obj):
        return zarr_obj.attrs.get("ome", zarr_obj.attrs)

    def iterate_groups(self, zarr_obj):
        metadata = self._metadata(zarr_obj)
        if 'plate' in metadata and 'wells' in metadata['plate']:
            # This is an HCS zarr with plate metadata. Use that to find the series.
            plate = metadata['plate']
            column_names = plate['columns']
            row_names = plate['rows']
            for attribs in plate['wells']:
                well = zarr_obj[attribs['path']]
                well_metadata = self._metadata(well).get("well", {})
                fields = (
                    well_metadata.get("images")
                    if "ome" in zarr_obj.attrs or 'field_count' not in plate
                    else None
                )
                if fields is None:
                    fields = [{"path": str(index)} for index in range(plate['field_count'])]
                row_index = attribs.get("rowIndex", attribs.get("row_index"))
                column_index = attribs.get("columnIndex", attribs.get("column_index"))
                for field in fields:
                    yield [well[field['path']],
                           f"r{row_names[row_index]['name']}"
                           f"c{column_names[column_index]['name']}"
                           f"f{field['path']}"]
        elif 'multiscales' in metadata:
            # User has pointed directly to a zarr series, we can assume no child groups.
            yield [zarr_obj, ""]
        else:
            # This is a simple or older zarr. Look for an ome.xml metadata file and read that.
            root = self._read_ome_xml()
            if root is not None:
                ns = '{http://www.openmicroscopy.org/Schemas/OME/2016-06}'
                if ns not in root.tag:
                    raise ValueError(f"Expected {ns} in XML file")
                for image in root.findall(f"{ns}Image"):
                    series_path = image.attrib["ID"].split(":")[1]
                    series_name = image.attrib.get("Name", series_path)
                    if series_path in zarr_obj:
                        yield [zarr_obj[series_path], series_name]
                    else:
                        LOGGER.warning(f"Series specified in OME-XML "
                                       f"was absent from the "
                                       f"zarr - {series_path}")
            elif (
                "ome" in zarr_obj.attrs
                and "OME" in zarr_obj
                and "series" in self._metadata(zarr_obj["OME"])
            ):
                for series_path in self._metadata(zarr_obj["OME"])["series"]:
                    yield [zarr_obj[series_path], ""]
            else:
                # No XML or zattrs = some other software made this zarr.
                # We can try scanning for groups, but this is slow.
                # Scans will return nothing if the storage backend is object-based (e.g. S3)
                LOGGER.error("Unable to find built-in or XML metadata, "
                             "will try a file scan")
                for key in sorted(zarr_obj.group_keys()):
                    yield [zarr_obj[key], key]

    def find_arrays(self, zarr_obj, multiscales=False, first_only=False):
        """
        Searches a zarr object and produces a list of arrays within that object.
        :param zarr_obj: zarr.Array or zarr.Group to scan.
        :param multiscales: True = Consider all resolutions as arrays. False = Only return resolution 0.
        :param first_only: True = only grab the first array, don't bother scanning the whole file.
        :return: List of tuples in format (zarr_array, name). Empty string denotes no special name.
        """
        if isinstance(zarr_obj, zarr.Array):
            # We're already pointed directly at an array object. No scan needed.
            return [(zarr_obj, "")]
        if not multiscales and zarr_obj is self._reader:
            complete_map = self.ZARR_READER_CACHE.get(self.root, (None, {}))[1].get(
                self.group
            )
            if complete_map is not None:
                return [(zarr_obj[path], name) for path, name in complete_map]
        # Groups should be a list of groups which contain arrays as children.
        result = []

        for group_obj, group_name in self.iterate_groups(zarr_obj):
            if isinstance(group_obj, zarr.Array):
                result.append((group_obj, group_name))
            elif 'multiscales' in self._metadata(group_obj):
                # The main reason we do this - do we want just resolution 0 or all resolutions?
                scales = self._metadata(group_obj)['multiscales']
                if multiscales:
                    for params in scales[0]['datasets']:
                        key = params['path']
                        result.append((group_obj[key], group_name))
                else:
                    key = scales[0]['datasets'][0]['path']
                    result.append((group_obj[key], group_name))
            else:
                LOGGER.error(f"Subgroup has no contents? {group_obj.path}")
            if first_only:
                break
        return result

