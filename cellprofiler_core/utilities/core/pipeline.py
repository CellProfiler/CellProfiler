import bisect
import re

import numpy

from ...constants.pipeline import make_cell_struct_dtype
from ...constants.pipeline import PIPELINE
from ...constants.pipeline import IMAGE_PLANE_DESCRIPTOR_VERSION
from ...constants.pipeline import H_PLANE_COUNT
from ...constants.pipeline import H_URL
from ...constants.pipeline import H_SERIES
from ...constants.pipeline import H_INDEX
from ...constants.pipeline import H_CHANNEL
from ...constants.pipeline import RF_STATE_PREQUOTE
from ...constants.pipeline import RF_STATE_FIELD
from ...constants.pipeline import RF_STATE_BACKSLASH_ESCAPE
from ...constants.pipeline import RF_STATE_SEPARATOR
from ..legacy import cmp


def add_all_images(handles, image_set, object_set):
    """ Add all images to the handles structure passed

    Add images to the handles structure, for example in the Python sandwich.
    """
    images = {}
    for provider in image_set.providers:
        name = provider.name()
        image = image_set.get_image(name)
        images[name] = image.image
        if image.has_mask:
            images["CropMask" + name] = image.mask

    for object_name in object_set.object_names:
        objects = object_set.get_objects(object_name)
        images["Segmented" + object_name] = objects.segmented
        if objects.has_unedited_segmented():
            images["UneditedSegmented" + object_name] = objects.unedited_segmented
        if objects.has_small_removed_segmented():
            images[
                "SmallRemovedSegmented" + object_name
            ] = objects.small_removed_segmented

    npy_images = numpy.ndarray(
        (1, 1), dtype=make_cell_struct_dtype(list(images.keys()))
    )
    for key, image in list(images.items()):
        npy_images[key][0, 0] = image
    handles[PIPELINE] = npy_images


def map_feature_names(feature_names, max_size=63):
    """Map feature names to legal Matlab field names

    returns a dictionary where the key is the field name and
    the value is the feature name.
    """
    mapping = {}
    seeded = False

    # def shortest_first(a, b):
    #     return (
    #         -1
    #         if len(a) < len(b)
    #         else 1
    #         if len(b) < len(a)
    #         else cellprofiler_core.utilities.legacy.cmp(a, b)
    #     )

    for feature_name in feature_names:
        # if len(feature_name) > max_size:
        #     name = feature_name
        #     to_remove = len(feature_name) - max_size
        #     remove_count = 0
        #     for to_drop in (
        #         ("a", "e", "i", "o", "u"),
        #         (
        #             "b",
        #             "c",
        #             "d",
        #             "f",
        #             "g",
        #             "h",
        #             "j",
        #             "k",
        #             "l",
        #             "m",
        #             "n",
        #             "p",
        #             "q",
        #             "r",
        #             "s",
        #             "t",
        #             "v",
        #             "w",
        #             "x",
        #             "y",
        #             "z",
        #         ),
        #         (
        #             "A",
        #             "B",
        #             "C",
        #             "D",
        #             "E",
        #             "F",
        #             "G",
        #             "H",
        #             "I",
        #             "J",
        #             "K",
        #             "L",
        #             "M",
        #             "N",
        #             "O",
        #             "P",
        #             "Q",
        #             "R",
        #             "S",
        #             "T",
        #             "U",
        #             "V",
        #             "W",
        #             "X",
        #             "Y",
        #             "Z",
        #         ),
        #     ):
        #         for index in range(len(name) - 1, -1, -1):
        #             if name[index] in to_drop:
        #                 name = name[:index] + name[index + 1 :]
        #                 remove_count += 1
        #                 if remove_count == to_remove:
        #                     break
        #         if remove_count == to_remove:
        #             break
        #     if name in list(mapping.keys()) or len(name) > max_size:
        #         # Panic mode - a duplication
        #         if not seeded:
        #             numpy.random.seed(0)
        #             seeded = True
        #         while True:
        #             npname = numpy.fromstring(feature_name, "|S1")
        #             indices = numpy.random.permutation(len(name))[:max_size]
        #             indices.sort()
        #             name = npname[indices]
        #             name = name.tostring()
        #             if not name in list(mapping.keys()):
        #                 break
        # else:
        #     name = feature_name
        mapping[feature_name] = feature_name
    return mapping


def read_file_list(file_or_fd):
    """Read a file list from a file or file object

    file_or_fd - either a path string or a file-like object

    Returns a collection of urls

    The unicode text for each field is utf-8 encoded, then string_escape encoded.
    All fields are delimited by quotes and separated by commas. A "None" value is
    represented by two consecutive commas. The series, index, and channel
    are no longer set, so the URL is followed by three commas.

    There are two header lines. The first header line has key/value pairs.
    Required key/value pairs are "Version" and "PlaneCount". "Version" is
    the format version. "PlaneCount" is the number of image planes to be
    read or saved.

    The second header line is legacy at this point - it used to specify
    a particular image plane in the file, but doesn't do that any longer.

    A minimal example:

    "Version":"1","PlaneCount":"1"
    "URL","Series","Index","Channel"
    "file:///imaging/analysis/singleplane.tif",,,
    """

    if isinstance(file_or_fd, str):
        needs_close = True
        fd = open(file_or_fd, "r")
    else:
        needs_close = False
        fd = file_or_fd
    try:
        line = next(fd)
        properties = dict(read_fields(line))
        if "Version" not in properties:
            raise ValueError("Image plane details header is missing its version #")
        version = int(properties["Version"])
        if version != IMAGE_PLANE_DESCRIPTOR_VERSION:
            raise ValueError(
                "Unable to read image plane details version # %d" % version
            )
        plane_count = int(properties[H_PLANE_COUNT])
        header = read_fields(next(fd))
        result = []
        pattern = r'(?:"((?:[^\\]|\\.)+?)")?(?:,|\s+)'
        for i in range(plane_count):
            fields = [x.groups()[0] for x in re.finditer(pattern, next(fd))]
            fields = [None if x is None else x for x in fields]
            url = fields[0]
            result.append(url)
        return result
    finally:
        if needs_close:
            fd.close()


def write_file_list(file_or_fd, file_list):
    """Write the file list out to a file.

    See read_image_plane_details for the file format.

    file_or_fd - a path or a file like object

    file_list - collection of URLs to be output

    """
    if isinstance(file_or_fd, str):
        fd = open(file_or_fd, "w")
        needs_close = True
    else:
        fd = file_or_fd
        needs_close = False
    try:
        fd.write(
            '"%s":"%d","%s":"%d"\n'
            % ("Version", IMAGE_PLANE_DESCRIPTOR_VERSION, H_PLANE_COUNT, len(file_list))
        )
        fd.write('"' + '","'.join([H_URL, H_SERIES, H_INDEX, H_CHANNEL]) + '"\n')
        for url in file_list:
            if isinstance(url, str):
                url = url
            # url = url.encode("string_escape").replace('"', r"\"")
            line = '"%s",,,\n' % url
            fd.write(line)
    finally:
        if needs_close:
            fd.close()


def read_fields(line):
    state = RF_STATE_PREQUOTE
    kv = False
    result = []
    field = ""
    for c in line:
        if state == RF_STATE_PREQUOTE:
            if c == '"':
                state = RF_STATE_FIELD
                field = ""
            elif c == ":":
                key = None
                kv = True
            elif c in ",\n":
                if kv:
                    result.append((key, None))
                else:
                    result.append(None)
                kv = False
        elif state == RF_STATE_BACKSLASH_ESCAPE:
            field += c
            state = RF_STATE_FIELD
        elif state == RF_STATE_SEPARATOR:
            if c == ":":
                key = field
                kv = True
                state = RF_STATE_PREQUOTE
            elif c in ",\n":
                field = field
                if kv:
                    result.append((key, field))
                else:
                    result.append(field)
                kv = False
                state = RF_STATE_PREQUOTE
        elif c == "\\":
            field += c
            state = RF_STATE_BACKSLASH_ESCAPE
        elif c == '"':
            state = RF_STATE_SEPARATOR
        else:
            field += c
    return result


def find_image_plane_details(exemplar, ipds):
    """Find the ImagePlaneDetails instance matching the exemplar

    The point of this function is to retrieve the ImagePlaneDetails from
    the list provided and, in doing so, get the attached metadata and the
    Java IPD object as well.

    exemplar - an IPD with the URL, series, index and channel filled in

    ipds - an ordered list of ImagePlaneDetails instances

    Returns the match or None if not found
    """
    pos = bisect.bisect_left(ipds, exemplar)
    if pos == len(ipds) or cmp(ipds[pos], exemplar):
        return None
    return ipds[pos]


def new_string_cell_array(shape):
    """Return a numpy.ndarray that looks like {NxM cell} to Matlab

    Return a numpy.ndarray that looks like {NxM cell} to Matlab.
    Each of the cells looks empty.
    shape - the shape of the array that's generated, e.g., (5,19) for a 5x19 cell array.
            Currently, this must be a 2-d shape.
    The object returned is a numpy.ndarray with dtype=dtype('object') and the given shape
    with each cell in the array filled with a numpy.ndarray with shape = (1,0)
    and dtype=dtype('float64'). This appears to be the form that's created in matlab
    for this sort of object.
    """
    result = numpy.ndarray(shape, dtype=numpy.dtype("object"))
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            result[i, j] = numpy.empty((0, 0))
    return result


def encapsulate_strings_in_arrays(handles):
    """Recursively descend through the handles structure, replacing strings as arrays packed with strings

    This function makes the handles structure loaded through the sandwich compatible with loadmat. It operates on the array in-place.
    """
    if handles.dtype.kind == "O":
        # cells - descend recursively
        flat = handles.flat
        for i in range(0, len(flat)):
            if isinstance(flat[i], str):
                flat[i] = encapsulate_string(flat[i])
            elif isinstance(flat[i], numpy.ndarray):
                encapsulate_strings_in_arrays(flat[i])
    elif handles.dtype.fields:
        # A structure: iterate over all structure elements.
        for field in list(handles.dtype.fields.keys()):
            if isinstance(handles[field], str):
                handles[field] = encapsulate_string(handles[field])
            elif isinstance(handles[field], numpy.ndarray):
                encapsulate_strings_in_arrays(handles[field])


def encapsulate_string(s):
    """Encapsulate a string in an array of shape 1 of the length of the string
    """
    result = numpy.ndarray((1,), "<U%d" % (len(s)))

    result[0] = s

    return result
