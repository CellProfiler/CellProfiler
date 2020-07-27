import os
import shutil
import tempfile
import urllib.request

import javabridge
import numpy
import scipy.io

import cellprofiler_core.constants.measurement
import cellprofiler_core.measurement
import cellprofiler_core.utilities.measurement
from cellprofiler_core.utilities import generate_presigned_url

UIC1_TAG = 33628
UIC2_TAG = 33629
UIC3_TAG = 33630
UIC4_TAG = 33631
C_MD5_DIGEST = "MD5Digest"
C_SCALING = "Scaling"
C_HEIGHT = "Height"
C_WIDTH = "Width"
MS_EXACT_MATCH = "Text-Exact match"
MS_REGEXP = "Text-Regular expressions"
MS_ORDER = "Order"
FF_INDIVIDUAL_IMAGES = "individual images"
FF_STK_MOVIES = "stk movies"
FF_AVI_MOVIES = "avi,mov movies"
FF_AVI_MOVIES_OLD = ["avi movies"]
FF_OTHER_MOVIES = "tif,tiff,flex,zvi movies"
FF_OTHER_MOVIES_OLD = ["tif,tiff,flex movies", "tif,tiff,flex movies, zvi movies"]
IO_IMAGES = "Images"
IO_OBJECTS = "Objects"
IO_ALL = (IO_IMAGES, IO_OBJECTS)
IMAGE_FOR_OBJECTS_F = "IMAGE_FOR_%s"
SUPPORTED_IMAGE_EXTENSIONS = {
    ".ppm",
    ".grib",
    ".im",
    ".rgba",
    ".rgb",
    ".pcd",
    ".h5",
    ".jpe",
    ".jfif",
    ".jpg",
    ".fli",
    ".sgi",
    ".gbr",
    ".pcx",
    ".mpeg",
    ".jpeg",
    ".ps",
    ".flc",
    ".tif",
    ".hdf",
    ".icns",
    ".gif",
    ".palm",
    ".mpg",
    ".fits",
    ".pgm",
    ".mic",
    ".fit",
    ".xbm",
    ".eps",
    ".emf",
    ".dcx",
    ".bmp",
    ".bw",
    ".pbm",
    ".dib",
    ".ras",
    ".cur",
    ".fpx",
    ".png",
    ".msp",
    ".iim",
    ".wmf",
    ".tga",
    ".bufr",
    ".ico",
    ".psd",
    ".xpm",
    ".arg",
    ".pdf",
    ".tiff",
}
SUPPORTED_MOVIE_EXTENSIONS = {
    ".avi",
    ".mpeg",
    ".stk",
    ".flex",
    ".mov",
    ".tif",
    ".tiff",
    ".zvi",
}
FF = [FF_INDIVIDUAL_IMAGES, FF_STK_MOVIES, FF_AVI_MOVIES, FF_OTHER_MOVIES]
M_NONE = "None"
M_FILE_NAME = "File name"
M_PATH = "Path"
M_BOTH = "Both"
M_Z = "Z"
M_T = "T"
C_SERIES = "Series"
C_FRAME = "Frame"
P_IMAGES = "LoadImagesImageProvider"
V_IMAGES = 1
P_MOVIES = "LoadImagesMovieProvider"
V_MOVIES = 2
P_FLEX = "LoadImagesFlexFrameProvider"
V_FLEX = 1
I_INTERLEAVED = "Interleaved"
I_SEPARATED = "Separated"
SUB_NONE = "None"
SUB_ALL = "All"
SUB_SOME = "Some"
FILE_SCHEME = "file:"
PASSTHROUGH_SCHEMES = ("http", "https", "ftp", "omero", "s3")


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
        if cellprofiler_core.utilities.measurement.is_well_row_token(token):
            well_row_token = token
        if cellprofiler_core.utilities.measurement.is_well_column_token(token):
            well_column_token = token
    return well_row_token, well_column_token


def needs_well_metadata(tokens):
    """Return true if, based on a set of metadata tokens, we need a well token

    Check for a row and column token and the absence of the well token.
    """
    if cellprofiler_core.constants.measurement.FTR_WELL.lower() in [x.lower() for x in tokens]:
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
