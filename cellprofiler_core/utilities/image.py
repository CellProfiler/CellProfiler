import os
import re
import shutil
import sys
import tempfile
import urllib.request

import boto3
import javabridge
import numpy
import pkg_resources
import scipy.io

from .measurement import is_well_row_token
from .measurement import is_well_column_token
from ..constants.image import SUPPORTED_IMAGE_EXTENSIONS
from ..constants.image import SUPPORTED_MOVIE_EXTENSIONS
from ..constants.image import PASSTHROUGH_SCHEMES
from ..constants.image import FILE_SCHEME
from ..constants.measurement import FTR_WELL


def convert_image_to_objects(image):
    """Interpret an image as object indices

    image - a grayscale or color image, assumes zero == background

    returns - a similarly shaped integer array with zero representing background
              and other values representing the indices of the associated object.
    """
    assert isinstance(image, numpy.ndarray)
    if image.ndim == 2:
        unique_indices = numpy.unique(image.ravel())
        if len(unique_indices) * 2 > max(numpy.max(unique_indices), 254) and numpy.all(
            numpy.abs(numpy.round(unique_indices, 1) - unique_indices)
            <= numpy.finfo(float).eps
        ):
            # Heuristic: reinterpret only if sparse and roughly integer
            return numpy.round(image).astype(int)

        def sorting(x):
            return [x]

        def comparison(i0, i1):
            return image.ravel()[i0] != image.ravel()[i1]

    else:
        i, j = numpy.mgrid[0 : image.shape[0], 0 : image.shape[1]]

        def sorting(x):
            return [x[:, :, 2], x[:, :, 1], x[:, :, 0]]

        def comparison(i0, i1):
            return numpy.any(
                image[i.ravel()[i0], j.ravel()[i0], :]
                != image[i.ravel()[i1], j.ravel()[i1], :],
                1,
            )

    order = numpy.lexsort([x.ravel() for x in sorting(image)])
    different = numpy.hstack([[False], comparison(order[:-1], order[1:])])
    index = numpy.cumsum(different)
    image = numpy.zeros(image.shape[:2], index.dtype)
    image.ravel()[order] = index
    return image


def default_cpimage_name(index):
    # the usual suspects
    names = ["DNA", "Actin", "Protein"]
    if index < len(names):
        return names[index]
    return "Channel%d" % (index + 1)


def well_metadata_tokens(tokens):
    """Return the well row and well column tokens out of a set of metadata tokens"""

    well_row_token = None
    well_column_token = None
    for token in tokens:
        if is_well_row_token(token):
            well_row_token = token
        if is_well_column_token(token):
            well_column_token = token
    return well_row_token, well_column_token


def needs_well_metadata(tokens):
    """Return true if, based on a set of metadata tokens, we need a well token

    Check for a row and column token and the absence of the well token.
    """
    if FTR_WELL.lower() in [x.lower() for x in tokens]:
        return False
    well_row_token, well_column_token = well_metadata_tokens(tokens)
    return (well_row_token is not None) and (well_column_token is not None)


def is_image(filename):
    """Determine if a filename is a potential image file based on extension"""
    ext = os.path.splitext(filename)[1].lower()
    return ext in SUPPORTED_IMAGE_EXTENSIONS


def is_movie(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in SUPPORTED_MOVIE_EXTENSIONS


def is_numpy_file(filename):
    return os.path.splitext(filename)[-1].lower() == ".npy"


def is_matlab_file(filename):
    return os.path.splitext(filename)[-1].lower() == ".mat"


def loadmat(path):
    imgdata = scipy.io.matlab.mio.loadmat(path, struct_as_record=True)
    img = imgdata["Image"]

    return img


def load_data_file(pathname_or_url, load_fn):
    ext = os.path.splitext(pathname_or_url)[-1].lower()

    if any([pathname_or_url.startswith(scheme) for scheme in PASSTHROUGH_SCHEMES]):
        url = generate_presigned_url(pathname_or_url)

        try:
            src = urllib.request.urlopen(url)
            fd, path = tempfile.mkstemp(suffix=ext)
            with os.fdopen(fd, mode="wb") as dest:
                shutil.copyfileobj(src, dest)
            img = load_fn(path)
        finally:
            try:
                src.close()
                os.remove(path)
            except NameError:
                pass

        return img

    return load_fn(pathname_or_url)


def is_file_url(url):
    return url.lower().startswith(FILE_SCHEME)


def is_image_extension(suffix):
    """Return True if the extension is one of those recongized by bioformats"""
    extensions = javabridge.get_collection_wrapper(
        javabridge.static_call(
            "org/cellprofiler/imageset/filter/IsImagePredicate",
            "getImageSuffixes",
            "()Ljava/util/Set;",
        )
    )
    return extensions.contains(suffix.lower())


def crop_image(image, crop_mask, crop_internal=False):
    """Crop an image to the size of the nonzero portion of a crop mask"""
    i_histogram = crop_mask.sum(axis=1)
    i_cumsum = numpy.cumsum(i_histogram != 0)
    j_histogram = crop_mask.sum(axis=0)
    j_cumsum = numpy.cumsum(j_histogram != 0)
    if i_cumsum[-1] == 0:
        # The whole image is cropped away
        return numpy.zeros((0, 0), dtype=image.dtype)
    if crop_internal:
        #
        # Make up sequences of rows and columns to keep
        #
        i_keep = numpy.argwhere(i_histogram > 0)
        j_keep = numpy.argwhere(j_histogram > 0)
        #
        # Then slice the array by I, then by J to get what's not blank
        #
        return image[i_keep.flatten(), :][:, j_keep.flatten()].copy()
    else:
        #
        # The first non-blank row and column are where the cumsum is 1
        # The last are at the first where the cumsum is it's max (meaning
        # what came after was all zeros and added nothing)
        #
        i_first = numpy.argwhere(i_cumsum == 1)[0]
        i_last = numpy.argwhere(i_cumsum == i_cumsum.max())[0]
        i_end = i_last + 1
        j_first = numpy.argwhere(j_cumsum == 1)[0]
        j_last = numpy.argwhere(j_cumsum == j_cumsum.max())[0]
        j_end = j_last + 1

        if image.ndim == 3:
            return image[i_first[0] : i_end[0], j_first[0] : j_end[0], :].copy()

        return image[i_first[0] : i_end[0], j_first[0] : j_end[0]].copy()


def make_dictionary_key(key):
    """Make a dictionary into a stable key for another dictionary"""
    return ", ".join([":".join([str(y) for y in x]) for x in sorted(key.items())])


def image_resource(filename):
    try:
        if os.path.join("CellProfiler", "docs") in os.path.abspath(os.curdir):
            # We're probably trying to buld the manual
            thepath = os.path.relpath(
                os.path.abspath(
                    pkg_resources.resource_filename(
                        "cellprofiler", os.path.join("..", "images", filename)
                    )
                ),
                os.path.abspath(os.curdir),
            )
        else:
            if hasattr(sys, "frozen"):
                # Use relative paths if we're frozen.
                thepath = os.path.relpath(
                    pkg_resources.resource_filename(
                        "cellprofiler", os.path.join("data", "images", filename)
                    )
                )
            else:
                thepath = os.path.abspath(
                    pkg_resources.resource_filename(
                        "cellprofiler", os.path.join("data", "images", filename)
                    )
                )
        return thepath.replace("\\", "/")
    except ModuleNotFoundError:
        # CellProfiler is not installed so the assets are missing.
        # In theory an icon should never be called without the GUI anyway
        print("CellProfiler image assets were not found")
    return ""


def generate_presigned_url(url):
    """
    Generate a presigned URL, if necessary (e.g., s3).

    :param url: An unsigned URL.
    :return: The presigned URL.
    """
    if url.startswith("s3"):
        client = boto3.client("s3")

        bucket_name, filename = (
            re.compile("s3://([\w\d\-.]+)/(.*)").search(url).groups()
        )

        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": filename.replace("+", " ")},
        )

    return url


def url_to_modpath(url):
    import urllib.parse

    from .hdf5_dict import HDF5FileList

    if not url.lower().startswith("file:"):
        schema, rest = HDF5FileList.split_url(url)
        return [schema] + rest[0:1] + [urllib.parse.unquote(part) for part in rest[1:]]
    path = urllib.request.url2pathname(url[5:])
    parts = []
    while True:
        new_path, part = os.path.split(path)
        if len(new_path) == 0 or len(part) == 0:
            parts.insert(0, path)
            break
        parts.insert(0, part)
        path = new_path
    return parts
