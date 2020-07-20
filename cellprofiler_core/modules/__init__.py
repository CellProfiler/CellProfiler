# coding=utf-8
import shutil
import tempfile

import glob
import logging
import os
import os.path
import re
import sys
import types

import six

import cellprofiler_core.measurement
import cellprofiler_core.module
import cellprofiler_core.preferences
from cellprofiler_core.utilities import generate_presigned_url
from cellprofiler_core.utilities.pathname import url2pathname

logger = logging.getLogger(__name__)


def plugin_list(plugin_dir):
    if plugin_dir is not None and os.path.isdir(plugin_dir):
        file_list = glob.glob(os.path.join(plugin_dir, "*.py"))
        return [
            os.path.basename(f)[:-3]
            for f in file_list
            if not f.endswith(("__init__.py", "_help.py"))
        ]
    return []


class PluginImporter(object):
    def find_module(self, fullname, path=None):
        if not fullname.startswith("cellprofiler.modules.plugins"):
            return None
        prefix, modname = fullname.rsplit(".", 1)
        if prefix != "cellprofiler.modules.plugins":
            return None
        if os.path.exists(
            os.path.join(
                cellprofiler_core.preferences.get_plugin_directory(), modname + ".py"
            )
        ):
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]

        prefix, modname = fullname.rsplit(".", 1)
        assert prefix == "cellprofiler_core.modules.plugins"

        try:
            mod = types.ModuleType(fullname)
            sys.modules[fullname] = mod
            mod.__loader__ = self
            mod.__file__ = os.path.join(
                cellprofiler_core.preferences.get_plugin_directory(), modname + ".py"
            )

            contents = open(mod.__file__, "r").read()
            exec(compile(contents, mod.__file__, "exec"), mod.__dict__)
            return mod
        except:
            if fullname in sys.modules:
                del sys.modules[fullname]


sys.meta_path.append(PluginImporter())

# python modules and their corresponding cellprofiler.module classes
pymodule_to_cpmodule = {
    "align": "Align",
    "images": "Images",
    "loadimages": "LoadImages",
    "measurementfixture": "MeasurementFixture",
}

# the builtin CP modules that will be loaded from the cellprofiler_core.modules directory
builtin_modules = [
    "align",
    "groups",
    "images",
    "loaddata",
    "metadata",
    "namesandtypes",
]

all_modules = {}
svn_revisions = {}
pymodules = []
badmodules = []

do_not_override = ["set_settings", "create_from_handles", "test_valid", "module_class"]
should_override = ["create_settings", "settings", "run"]


def check_module(module, name):
    if hasattr(module, "do_not_check"):
        return
    assert (
        name == module.module_name
    ), "Module %s should have module_name %s (is %s)" % (name, name, module.module_name)
    for method_name in do_not_override:
        assert getattr(module, method_name) == getattr(
            cellprofiler_core.module.Module, method_name
        ), "Module %s should not override method %s" % (name, method_name)
    for method_name in should_override:
        assert getattr(module, method_name) != getattr(
            cellprofiler_core.module.Module, method_name
        ), "Module %s should override method %s" % (name, method_name)


def find_cpmodule(m):
    """Returns the CPModule from within the loaded Python module

    m - an imported module

    returns the CPModule class
    """
    for v, val in list(m.__dict__.items()):
        if isinstance(val, type) and issubclass(val, cellprofiler_core.module.Module):
            return val
    raise ValueError(
        "Could not find cellprofiler_core.module.Module class in %s" % m.__file__
    )


def fill_modules():
    del pymodules[:]
    del badmodules[:]
    all_modules.clear()
    svn_revisions.clear()

    def add_module(mod, check_svn):
        try:
            m = __import__(mod, globals(), locals(), ["__all__"], 0)
            cp_module = find_cpmodule(m)
            name = cp_module.module_name
        except Exception as e:
            logger.warning("Could not load %s", mod, exc_info=True)
            badmodules.append((mod, e))
            return

        try:
            pymodules.append(m)
            if name in all_modules:
                logger.warning(
                    "Multiple definitions of module %s\n\told in %s\n\tnew in %s",
                    name,
                    sys.modules[all_modules[name].__module__].__file__,
                    m.__file__,
                )
            all_modules[name] = cp_module
            check_module(cp_module, name)
            # attempt to instantiate
            if not hasattr(cp_module, "do_not_check"):
                cp_module()
            if check_svn and hasattr(m, "__version__"):
                match = re.match("^\$Revision: ([0-9]+) \$$", m.__version__)
                if match is not None:
                    svn_revisions[name] = match.groups()[0]
        except Exception as e:
            logger.warning("Failed to load %s", name, exc_info=True)
            badmodules.append((mod, e))
            if name in all_modules:
                del all_modules[name]
                del pymodules[-1]

    # Import core modules
    for mod in builtin_modules:
        add_module("cellprofiler_core.modules." + mod, True)

    # Import CellProfiler modules if CellProfiler is installed
    cpinstalled = False
    try:
        import cellprofiler.modules

        cpinstalled = True
    except ImportError:
        print("No CellProfiler installation detected, only base modules will be loaded")
    if cpinstalled:
        for mod in cellprofiler.modules.builtin_modules:
            add_module("cellprofiler.modules." + mod, True)

    # Find and import plugins
    plugin_directory = cellprofiler_core.preferences.get_plugin_directory()
    if plugin_directory is not None:
        old_path = sys.path
        sys.path.insert(0, plugin_directory)
        try:
            for mod in plugin_list(plugin_directory):
                add_module(mod, False)
        finally:
            sys.path = old_path

    if len(badmodules) > 0:
        logger.warning(
            "could not load these modules: %s", ",".join([x[0] for x in badmodules])
        )


def add_module_for_tst(module_class):
    all_modules[module_class.module_name] = module_class


fill_modules()

__all__ = [
    "instantiate_module",
    "get_module_names",
    "reload_modules",
    "add_module_for_tst",
    "builtin_modules",
]

replaced_modules = {
    "LoadImageDirectory": ["LoadData"],
    "GroupMovieFrames": ["LoadData"],
    "IdentifyPrimLoG": ["IdentifyPrimaryObjects"],
    "FileNameMetadata": ["LoadData"],
    "LoadSingleImage": ["LoadData"],
    "LoadImages": ["LoadData"],
}
depricated_modules = ["CorrectIllumination_Calculate_kate", "SubtractBackground"]
unimplemented_modules = ["LabelImages", "Restart", "SplitOrSpliceMovie"]


def get_module_class(module_name):
    module_class = module_name.split(".")[-1]
    if module_class not in all_modules:
        if module_class in unimplemented_modules:
            raise ValueError(
                (
                    "The %s module has not yet been implemented. "
                    "It will be available in a later version "
                    "of CellProfiler."
                )
                % module_class
            )
        if module_class in depricated_modules:
            raise ValueError(
                (
                    "The %s module has been deprecated and will "
                    "not be implemented in CellProfiler 4.0."
                )
                % module_class
            )
        if module_class in replaced_modules:
            raise ValueError(
                (
                    "The %s module no longer exists. You can find "
                    "similar functionality in: %s"
                )
                % (module_class, ", ".join(replaced_modules[module_class]))
            )
        raise ValueError("Could not find the %s module" % module_class)
    return all_modules[module_class]


def instantiate_module(module_name):
    module = get_module_class(module_name)()
    if module_name in svn_revisions:
        module.svn_version = svn_revisions[module_name]
    return module


def get_module_names():
    names = list(all_modules.keys())
    names.sort()
    return names


def reload_modules():
    for m in pymodules:
        try:
            del sys.modules[m.__name__]
        except:
            pass
    fill_modules()


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
        if cellprofiler_core.measurement.is_well_row_token(token):
            well_row_token = token
        if cellprofiler_core.measurement.is_well_column_token(token):
            well_column_token = token
    return well_row_token, well_column_token


def needs_well_metadata(tokens):
    """Return true if, based on a set of metadata tokens, we need a well token

    Check for a row and column token and the absence of the well token.
    """
    if cellprofiler_core.measurement.FTR_WELL.lower() in [x.lower() for x in tokens]:
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
            src = six.moves.urllib.urlopen(url)
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


FILE_SCHEME = "file:"
PASSTHROUGH_SCHEMES = ("http", "https", "ftp", "omero", "s3")


def is_file_url(url):
    return url.lower().startswith(FILE_SCHEME)


def urlfilename(url):
    """Return just the file part of a URL

    For instance http://nucleus.org/linked_files/file%20has%20spaces.txt
    has a file part of "file has spaces.txt"
    """
    if is_file_url(url):
        return os.path.split(url2pathname(url))[1]
    path = six.moves.urllib.parse.urlparse(url)[2]
    if "/" in path:
        return six.moves.urllib.unquote(path.rsplit("/", 1)[1])
    else:
        return six.moves.urllib.unquote(path)


def urlpathname(url):
    """Return the path part of a URL

    For instance, http://nucleus.org/Comma%2Cseparated/foo.txt
    has a path of http://nucleus.org/Comma,separated

    A file url has the normal sort of path that you'd expect.
    """
    if is_file_url(url):
        return os.path.split(url2pathname(url))[0]
    scheme, netloc, path = six.moves.urllib.parse.urlparse(url)[:3]
    path = six.moves.urllib.parse.urlunparse([scheme, netloc, path, "", "", ""])
    if "/" in path:
        return six.moves.urllib.unquote(path.rsplit("/", 1)[0])
    else:
        return six.moves.urllib.unquote(path)
