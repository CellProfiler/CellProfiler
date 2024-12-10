import numpy
import zarr

from ..constants.image import MD_SIZE_S, MD_SIZE_C, MD_SIZE_Z, MD_SIZE_T, MD_SIZE_Y, MD_SIZE_X, MD_SERIES_NAME
from ..constants.reader import ZARR_FILETYPE

from ..reader import Reader

import os

import re
import logging
from lxml import etree

LOGGER = logging.getLogger(__name__)

FORMAT_TESTER = re.compile(r"\.zarr([\\/]|$)", flags=re.IGNORECASE)
SUPPORTED_SCHEMES = {'file', 's3'}

class NGFFReader(Reader):
    """
    A reader for OME-NGFF files with the .zarr extension. Supports both 'normal' and
    'HCS' storage modes.

    OME-NGFF files below spec version 0.4 which are stored on S3/object storage
    should have the zarr.convenience.consolidate_metadata() function run on them
    to provide optimal performance over a network.

    Other non-OME .zarr format images may be readable, but are not strictly supported.
    """

    reader_name = "OME-NGFF"
    variable_revision_number = 1
    supported_filetypes = {'.zarr', '.ome.zarr'}
    supported_schemes = SUPPORTED_SCHEMES

    # Reader cache maps a path to a tuple of (zarr_root_group, series_map).
    ZARR_READER_CACHE = {}

    def __init__(self, image_file):
        super().__init__(image_file)

        self._reader = None
        # List of tuples denoting (array_path, array_name)
        self._series_map = None
        self.root, self.group = ZARR_FILETYPE.split(self.file.url, maxsplit=1)
        self.planes_extracted = False

    def __del__(self):
        self.close()

    def get_reader(self):
        if self._reader is not None:
            return self._reader
        elif self.root in NGFFReader.ZARR_READER_CACHE:
            self._reader, self._series_map = NGFFReader.ZARR_READER_CACHE[self.root]
        else:
            store = zarr.storage.FSStore(self.root)
            if self.root.lower().startswith('s3'):
                LOGGER.info("Zarr is stored on S3, will try to read directly.")
                if '.zmetadata' in store:
                    # Zarr has consolidated metadata.
                    self._reader = zarr.convenience.open_consolidated(store, mode='r')
                else:
                    LOGGER.warning(f"Image is on S3 but lacks consolidated metadata. "
                                    f"This may degrade reading performance. URL: {self.root}")
                    self._reader = zarr.open(store, mode='r')
            elif not os.path.isdir(store.path):
                raise IOError("The file, \"%s\", does not exist." % self.root)
            else:
                self._reader = zarr.open(store, mode='r')
            NGFFReader.ZARR_READER_CACHE[self.root] = self._reader, None
        if self.group:
            LOGGER.warning(f"Reader had a group? {self.group}")
            return self._reader[self.group]
        return self._reader

    @classmethod
    def clear_cached_readers(cls):
        NGFFReader.ZARR_READER_CACHE.clear()

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
        if self._series_map is None:
            arrays = self.find_arrays(reader, multiscales=False, first_only=not self.planes_extracted)
            self.build_series_map(arrays)
        if not self._series_map:
            raise IOError("Zarr appears to contain no valid series. Unable to load data.")
        series_path, series_name = self._series_map[series]
        series_reader = reader[series_path]
        num_dimensions = len(series_reader.shape)
        # Todo: Handle V4 Axis object in .zattrs once it releases
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
            scale = self.get_max_sample_value(reader)
            return image, scale
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

    def build_series_map(self, arrays):
        if self.root in NGFFReader.ZARR_READER_CACHE:
            _, series_map = NGFFReader.ZARR_READER_CACHE[self.root]
            if series_map is not None:
                self._series_map = series_map
                return
        self._series_map = [(array.path, name) for array, name in arrays]
        if self.planes_extracted:
            NGFFReader.ZARR_READER_CACHE[self.root] = self._reader, self._series_map

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
        self.build_series_map(arrays)
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

    @staticmethod
    def get_max_sample_value(zarr_obj):
        xml_path = (zarr_obj.store.path + '/OME/METADATA.ome.xml')
        if not os.path.exists(xml_path):
            return None

        root = etree.parse(xml_path)

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
            except:
                max_sample_value = None
            return max_sample_value
        else:
            return None

    @staticmethod
    def iterate_groups(zarr_obj):
        if 'plate' in zarr_obj.attrs and 'wells' in zarr_obj.attrs['plate']:
            # This is an HCS zarr with plate metadata. Use that to find the series.
            num_fields = zarr_obj.attrs['plate']['field_count']
            column_names = zarr_obj.attrs['plate']['columns']
            row_names = zarr_obj.attrs['plate']['rows']
            for attribs in zarr_obj.attrs['plate']['wells']:
                for field_idx in range(num_fields):
                    yield [zarr_obj[f"{attribs['path']}/{field_idx}"],
                           f"r{row_names[attribs['row_index']]['name']}"
                           f"c{column_names[attribs['column_index']]['name']}"
                           f"f{field_idx}"]
        elif 'XXXOME' in zarr_obj:
            # This is a zarr with OME metadata stored as a group. Grab a series list from this.
            ome_group = zarr_obj['OME']
            if 'series' in ome_group.attrs:
                series_paths = ome_group.attrs['series']
                for idx, series_path in enumerate(series_paths):
                    target = zarr_obj[series_path]
                    yield [target, ""]
        elif 'multiscales' in zarr_obj.attrs:
            # User has pointed directly to a zarr series, we can assume no child groups.
            yield [zarr_obj, ""]
        else:
            # This is a simple or older zarr. Look for an ome.xml metadata file and read that.
            xml_path = (zarr_obj.store.path + '/OME/METADATA.ome.xml')
            if os.path.exists(xml_path):
                ns = '{http://www.openmicroscopy.org/Schemas/OME/2016-06}'
                tree = etree.parse(xml_path)
                root = tree.getroot()
                if ns not in root.tag:
                    raise Exception(f"Expected {ns} in XML file")
                for image in root.findall(f"{ns}Image"):
                    series_path = image.attrib["ID"].split(":")[1]
                    series_name = image.attrib.get("Name", series_path)
                    if series_path in zarr_obj:
                        yield [zarr_obj[series_path], series_name]
                    else:
                        LOGGER.warning(f"Series specified in OME-XML "
                                       f"was absent from the "
                                       f"zarr - {series_path}")
            else:
                # No XML or zattrs = some other software made this zarr.
                # We can try scanning for groups, but this is slow.
                # Scans will return nothing if the storage backend is object-based (e.g. S3)
                LOGGER.error("Unable to find built-in or XML metadata, "
                             "will try a file scan")
                for key in zarr_obj.group_keys():
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
        if self._series_map is not None:
            return [(zarr_obj[path], name) for path, name in self._series_map]
        # Groups should be a list of groups which contain arrays as children.
        result = []

        for group_obj, group_name in self.iterate_groups(zarr_obj):
            if isinstance(group_obj, zarr.Array):
                result.append((group_obj, group_name))
            elif 'multiscales' in group_obj.attrs:
                # The main reason we do this - do we want just resolution 0 or all resolutions?
                scales = group_obj.attrs['multiscales']
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

