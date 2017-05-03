"""Pipeline.py - an ordered set of modules to be executed
"""
from __future__ import with_statement

import bisect
import csv
import hashlib
import json
import logging
import uuid

import numpy as np
import scipy
import scipy.io.matlab

try:
    # implemented in scipy.io.matlab.miobase.py@5582
    from scipy.io.matlab.miobase import MatReadError

    has_mat_read_error = True
except:
    has_mat_read_error = False

import os
import StringIO  # XXX - replace with cStringIO?
import sys
import tempfile
import traceback
import datetime
import traceback
import threading
import urlparse
import urllib
import urllib2
import re
import numpy

logger = logging.getLogger(__name__)
pipeline_stats_logger = logging.getLogger("PipelineStatistics")
import cellprofiler
import cellprofiler.preferences as cpprefs
import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.workspace as cpw
import cellprofiler.setting as cps
from cellprofiler.utilities.utf16encode import utf16encode, utf16decode
from bioformats.omexml import OMEXML
from bioformats.formatreader import clear_image_reader_cache
import javabridge as J

'''The measurement name of the image number'''
IMAGE_NUMBER = cpmeas.IMAGE_NUMBER
GROUP_NUMBER = cpmeas.GROUP_NUMBER
GROUP_INDEX = cpmeas.GROUP_INDEX
CURRENT = 'Current'
NUMBER_OF_IMAGE_SETS = 'NumberOfImageSets'
NUMBER_OF_MODULES = 'NumberOfModules'
SET_BEING_ANALYZED = 'SetBeingAnalyzed'
SAVE_OUTPUT_HOW_OFTEN = 'SaveOutputHowOften'
TIME_STARTED = 'TimeStarted'
STARTING_IMAGE_SET = 'StartingImageSet'
STARTUP_DIRECTORY = 'StartupDirectory'
DEFAULT_MODULE_DIRECTORY = 'DefaultModuleDirectory'
DEFAULT_IMAGE_DIRECTORY = 'DefaultImageDirectory'
DEFAULT_OUTPUT_DIRECTORY = 'DefaultOutputDirectory'
IMAGE_TOOLS_FILENAMES = 'ImageToolsFilenames'
IMAGE_TOOL_HELP = 'ImageToolHelp'
PREFERENCES = 'Preferences'
PIXEL_SIZE = 'PixelSize'
SKIP_ERRORS = 'SkipErrors'
INTENSITY_COLOR_MAP = 'IntensityColorMap'
LABEL_COLOR_MAP = 'LabelColorMap'
STRIP_PIPELINE = 'StripPipeline'
DISPLAY_MODE_VALUE = 'DisplayModeValue'
DISPLAY_WINDOWS = 'DisplayWindows'
FONT_SIZE = 'FontSize'
IMAGES = 'Images'
MEASUREMENTS = 'Measurements'
PIPELINE = 'Pipeline'
SETTINGS = 'Settings'
VARIABLE_VALUES = 'VariableValues'
VARIABLE_INFO_TYPES = 'VariableInfoTypes'
MODULE_NAMES = 'ModuleNames'
PIXEL_SIZE = 'PixelSize'
NUMBERS_OF_VARIABLES = 'NumbersOfVariables'
VARIABLE_REVISION_NUMBERS = 'VariableRevisionNumbers'
MODULE_REVISION_NUMBERS = 'ModuleRevisionNumbers'
MODULE_NOTES = 'ModuleNotes'
CURRENT_MODULE_NUMBER = 'CurrentModuleNumber'
SHOW_WINDOW = 'ShowFrame'
BATCH_STATE = 'BatchState'
EXIT_STATUS = 'Exit_Status'
SETTINGS_DTYPE = np.dtype([(VARIABLE_VALUES, '|O4'),
                           (VARIABLE_INFO_TYPES, '|O4'),
                           (MODULE_NAMES, '|O4'),
                           (NUMBERS_OF_VARIABLES, '|O4'),
                           (PIXEL_SIZE, '|O4'),
                           (VARIABLE_REVISION_NUMBERS, '|O4'),
                           (MODULE_REVISION_NUMBERS, '|O4'),
                           (MODULE_NOTES, '|O4'),
                           (SHOW_WINDOW, '|O4'),
                           (BATCH_STATE, '|O4')])


def make_cell_struct_dtype(fields):
    """Makes the dtype of a struct composed of cells

    fields - the names of the fields in the struct
    """
    return numpy.dtype([(str(x), '|O4') for x in fields])


CURRENT_DTYPE = make_cell_struct_dtype([NUMBER_OF_IMAGE_SETS,
                                        SET_BEING_ANALYZED, NUMBER_OF_MODULES,
                                        SAVE_OUTPUT_HOW_OFTEN, TIME_STARTED,
                                        STARTING_IMAGE_SET,
                                        STARTUP_DIRECTORY,
                                        DEFAULT_OUTPUT_DIRECTORY,
                                        DEFAULT_IMAGE_DIRECTORY,
                                        IMAGE_TOOLS_FILENAMES,
                                        IMAGE_TOOL_HELP])
PREFERENCES_DTYPE = make_cell_struct_dtype([PIXEL_SIZE,
                                            DEFAULT_MODULE_DIRECTORY,
                                            DEFAULT_OUTPUT_DIRECTORY,
                                            DEFAULT_IMAGE_DIRECTORY,
                                            INTENSITY_COLOR_MAP,
                                            LABEL_COLOR_MAP,
                                            STRIP_PIPELINE, SKIP_ERRORS,
                                            DISPLAY_MODE_VALUE, FONT_SIZE,
                                            DISPLAY_WINDOWS])

'''Save pipeline in Matlab format'''
FMT_MATLAB = "Matlab"

'''Save pipeline in native format'''
FMT_NATIVE = "Native"

'''The current pipeline file format version'''
NATIVE_VERSION = 3

'''The version of the image plane descriptor section'''
IMAGE_PLANE_DESCRIPTOR_VERSION = 1

H_VERSION = 'Version'
H_SVN_REVISION = 'SVNRevision'
H_DATE_REVISION = 'DateRevision'
'''A pipeline file header variable for faking a matlab pipeline file'''
H_FROM_MATLAB = 'FromMatlab'
'''The GIT hash of the revision'''
H_GIT_HASH = 'GitHash'

'''The number of image planes in the file'''
H_PLANE_COUNT = "PlaneCount"

'''URL column header'''
H_URL = "URL"

'''Series column header'''
H_SERIES = "Series"

'''Index column header'''
H_INDEX = "Index"

'''Channel column header'''
H_CHANNEL = "Channel"

'''The number of modules in the pipeline'''
H_MODULE_COUNT = "ModuleCount"

'''Indicates whether the pipeline has an image plane details section'''
H_HAS_IMAGE_PLANE_DETAILS = "HasImagePlaneDetails"

'''A message for a user, to be displayed when pipeline is loaded'''
H_MESSAGE_FOR_USER = "MessageForUser"

'''The cookie that identifies a file as a CellProfiler pipeline'''
COOKIE = "CellProfiler Pipeline: http://www.cellprofiler.org"

'''Sad proofpoint cookie: see issue #1318'''
SAD_PROOFPOINT_COOKIE = r"CellProfiler Pipeline: https?://\S+.proofpoint.com.+http-3A__www.cellprofiler\.org"

'''HDF5 file header according to the specification

see http://www.hdfgroup.org/HDF5/doc/H5.format.html#FileMetaData
'''
HDF5_HEADER = (chr(137) + chr(72) + chr(68) + chr(70) + chr(13) + chr(10) +
               chr(26) + chr(10))
C_PIPELINE = "Pipeline"
C_CELLPROFILER = "CellProfiler"
F_PIPELINE = "Pipeline"
F_USER_PIPELINE = "UserPipeline"
M_PIPELINE = "_".join((C_PIPELINE, F_PIPELINE))
M_USER_PIPELINE = "_".join((C_PIPELINE, F_USER_PIPELINE))
F_VERSION = "Version"
M_VERSION = "_".join((C_CELLPROFILER, F_VERSION))
C_RUN = "Run"
C_MODIFICATION = "Modification"
F_TIMESTAMP = "Timestamp"
M_TIMESTAMP = "_".join((C_RUN, F_TIMESTAMP))
M_MODIFICATION_TIMESTAMP = "_".join((C_MODIFICATION, F_TIMESTAMP))

"""Default input folder measurement"""
M_DEFAULT_INPUT_FOLDER = "Default_InputFolder"

"""Default output folder measurement"""
M_DEFAULT_OUTPUT_FOLDER = "Default_OutputFolder"


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
            images['CropMask' + name] = image.mask

    for object_name in object_set.object_names:
        objects = object_set.get_objects(object_name)
        images['Segmented' + object_name] = objects.segmented
        if objects.has_unedited_segmented():
            images['UneditedSegmented' + object_name] = objects.unedited_segmented
        if objects.has_small_removed_segmented():
            images['SmallRemovedSegmented' + object_name] = objects.small_removed_segmented

    npy_images = np.ndarray((1, 1), dtype=make_cell_struct_dtype(images.keys()))
    for key, image in images.iteritems():
        npy_images[key][0, 0] = image
    handles[PIPELINE] = npy_images


def map_feature_names(feature_names, max_size=63):
    '''Map feature names to legal Matlab field names

    returns a dictionary where the key is the field name and
    the value is the feature name.
    '''
    mapping = {}
    seeded = False

    def shortest_first(a, b):
        return -1 if len(a) < len(b) else 1 if len(b) < len(a) else cmp(a, b)

    for feature_name in sorted(feature_names, shortest_first):
        if len(feature_name) > max_size:
            name = feature_name
            to_remove = len(feature_name) - max_size
            remove_count = 0
            for to_drop in (('a', 'e', 'i', 'o', 'u'),
                            ('b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n',
                             'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z'),
                            ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                             'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                             'W', 'X', 'Y', 'Z')):
                for index in range(len(name) - 1, -1, -1):
                    if name[index] in to_drop:
                        name = name[:index] + name[index + 1:]
                        remove_count += 1
                        if remove_count == to_remove:
                            break
                if remove_count == to_remove:
                    break
            if name in mapping.keys() or len(name) > max_size:
                # Panic mode - a duplication
                if not seeded:
                    np.random.seed(0)
                    seeded = True
                while True:
                    npname = np.fromstring(feature_name, '|S1')
                    indices = np.random.permutation(len(name))[:max_size]
                    indices.sort()
                    name = npname[indices]
                    name = name.tostring()
                    if not name in mapping.keys():
                        break
        else:
            name = feature_name
        mapping[name] = feature_name
    return mapping


def add_all_measurements(handles, measurements):
    """Add all measurements from our measurements object into the numpy structure passed

    """
    object_names = [name for name in measurements.get_object_names()
                    if len(measurements.get_feature_names(name)) > 0]
    measurements_dtype = make_cell_struct_dtype(object_names)
    npy_measurements = np.ndarray((1, 1), dtype=measurements_dtype)
    handles[MEASUREMENTS] = npy_measurements
    image_numbers = measurements.get_image_numbers()
    max_image_number = np.max(image_numbers)
    has_image_number = np.zeros(max_image_number + 1, bool)
    has_image_number[image_numbers] = True
    for object_name in object_names:
        if object_name == cpmeas.EXPERIMENT:
            continue
        mapping = map_feature_names(measurements.get_feature_names(object_name))
        object_dtype = make_cell_struct_dtype(mapping.keys())
        object_measurements = np.ndarray((1, 1), dtype=object_dtype)
        npy_measurements[object_name][0, 0] = object_measurements
        for field, feature_name in mapping.iteritems():
            feature_measurements = np.ndarray((1, max_image_number),
                                              dtype='object')
            object_measurements[field][0, 0] = feature_measurements
            for i in np.argwhere(~ has_image_number[1:]).flatten():
                feature_measurements[0, i] = np.zeros(0)
            dddata = measurements[object_name, feature_name, image_numbers]
            for i, ddata in zip(image_numbers, dddata):
                if np.isscalar(ddata) and np.isreal(ddata):
                    feature_measurements[0, i - 1] = np.array([ddata])
                elif ddata is not None:
                    feature_measurements[0, i - 1] = ddata
                else:
                    feature_measurements[0, i - 1] = np.zeros(0)
    if cpmeas.EXPERIMENT in measurements.object_names:
        mapping = map_feature_names(measurements.get_feature_names(cpmeas.EXPERIMENT))
        object_dtype = make_cell_struct_dtype(mapping.keys())
        experiment_measurements = np.ndarray((1, 1), dtype=object_dtype)
        npy_measurements[cpmeas.EXPERIMENT][0, 0] = experiment_measurements
        for field, feature_name in mapping.iteritems():
            feature_measurements = np.ndarray((1, 1), dtype='object')
            feature_measurements[0, 0] = measurements.get_experiment_measurement(feature_name)
            experiment_measurements[field][0, 0] = feature_measurements


class ImagePlaneDetails(object):
    '''This class represents the location and metadata for a 2-d image plane

    You need four pieces of information to reference an image plane:

    * The URL

    * The series number

    * The index

    * The channel # (to reference a monochrome plane in an interleaved image)

    In addition, image planes have associated metadata which is represented
    as a dictionary of keys and values.
    '''
    MD_COLOR_FORMAT = "ColorFormat"
    MD_MONOCHROME = "monochrome"
    MD_RGB = "RGB"
    MD_PLANAR = "Planar"
    MD_SIZE_C = "SizeC"
    MD_SIZE_Z = "SizeZ"
    MD_SIZE_T = "SizeT"
    MD_SIZE_X = "SizeX"
    MD_SIZE_Y = "SizeY"
    MD_C = "C"
    MD_Z = "Z"
    MD_T = "T"
    MD_CHANNEL_NAME = "ChannelName"

    def __init__(self, jipd):
        self.jipd = jipd

    @property
    def path(self):
        '''The file path if a file: URL, otherwise the URL'''
        if self.url.startswith("file:"):
            return urllib.url2pathname(self.url[5:]).decode('utf8')
        return self.url

    @property
    def url(self):
        return J.run_script(
                "o.getImagePlane().getImageFile().getURI().toString()",
                dict(o=self.jipd)).encode("utf-8")

    @property
    def series(self):
        return J.run_script(
                "o.getImagePlane().getSeries().getSeries()",
                dict(o=self.jipd))

    @property
    def index(self):
        return J.run_script("o.getImagePlane().getIndex()",
                            dict(o=self.jipd))

    @property
    def channel(self):
        return J.run_script("o.getImagePlane().getChannel()",
                            dict(o=self.jipd))

    @property
    def metadata(self):
        return json.loads(
                J.call(self.jipd, "jsonSerialize", "()Ljava/lang/String;"))


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

    if isinstance(file_or_fd, basestring):
        needs_close = True
        fd = open(file_or_fd, "r")
    else:
        needs_close = False
        fd = file_or_fd
    try:
        line = fd.next()
        properties = dict(read_fields(line))
        if not properties.has_key(H_VERSION):
            raise ValueError("Image plane details header is missing its version #")
        version = int(properties[H_VERSION])
        if version != IMAGE_PLANE_DESCRIPTOR_VERSION:
            raise ValueError("Unable to read image plane details version # %d" % version)
        plane_count = int(properties[H_PLANE_COUNT])
        header = read_fields(fd.next())
        result = []
        pattern = r'(?:"((?:[^\\]|\\.)+?)")?(?:,|\s+)'
        for i in range(plane_count):
            fields = [x.groups()[0] for x in re.finditer(pattern, fd.next())]
            fields = [None if x is None else x.decode('string-escape')
                      for x in fields]
            url = fields[0]
            result.append(url)
        return result
    finally:
        if needs_close:
            fd.close()


def write_file_list(file_or_fd, file_list):
    '''Write the file list out to a file.

    See read_image_plane_details for the file format.

    file_or_fd - a path or a file like object

    file_list - collection of URLs to be output

    '''
    if isinstance(file_or_fd, basestring):
        fd = open(file_or_fd, "w")
        needs_close = True
    else:
        fd = file_or_fd
        needs_close = False
    try:
        fd.write('"%s":"%d","%s":"%d"\n' % (
            H_VERSION, IMAGE_PLANE_DESCRIPTOR_VERSION, H_PLANE_COUNT,
            len(file_list)))
        fd.write('"' + '","'.join([H_URL, H_SERIES, H_INDEX, H_CHANNEL]) + '"\n')
        for url in file_list:
            if isinstance(url, unicode):
                url = url.encode("utf-8")
            url = url.encode("string_escape").replace('"', r'\"')
            line = "\"%s\",,,\n" % url
            fd.write(line)
    finally:
        if needs_close:
            fd.close()


RF_STATE_PREQUOTE = 0
RF_STATE_FIELD = 1
RF_STATE_BACKSLASH_ESCAPE = 2
RF_STATE_SEPARATOR = 3


def read_fields(line):
    state = RF_STATE_PREQUOTE
    kv = False
    result = []
    field = ""
    for c in line:
        if state == RF_STATE_PREQUOTE:
            if c == "\"":
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
                key = field.decode("string_escape").decode("utf-8")
                kv = True
                state = RF_STATE_PREQUOTE
            elif c in ",\n":
                field = field.decode("string_escape").decode("utf-8")
                if kv:
                    result.append((key, field))
                else:
                    result.append(field)
                kv = False
                state = RF_STATE_PREQUOTE
        elif c == "\\":
            field += c
            state = RF_STATE_BACKSLASH_ESCAPE
        elif c == "\"":
            state = RF_STATE_SEPARATOR
        else:
            field += c
    return result


class Pipeline(object):
    """A pipeline represents the modules that a user has put together
    to analyze their images.

    """

    def __init__(self):
        self.__modules = []
        self.__listeners = []
        self.__measurement_columns = {}
        self.__measurement_column_hash = None
        self.__test_mode = False
        self.message_for_user = None
        self.__settings = []
        self.__undo_stack = []
        self.__undo_start = None
        # The file list is the list of URLs operated on by the
        # input modules
        self.__file_list = []
        #
        # A cookie that's shared between the workspace and pipeline
        # and is used to figure out whether the two are synchronized
        #
        self.__file_list_generation = None
        #
        # The filtered file list is the list of URLS after filtering using
        # the Images module. The images settings are used to determine
        # whether the cache is valid
        #
        self.__filtered_file_list = []
        self.__filtered_file_list_images_settings = tuple()
        #
        # The image plane details are generated by the metadata module
        # from the file list
        #
        self.__image_plane_details = []
        self.__image_plane_details_metadata_settings = tuple()

        self.__undo_stack = []
        self.__volumetric = False

    def set_volumetric(self, value):
        self.__volumetric = value

    def volumetric(self):
        return self.__volumetric

    def copy(self, save_image_plane_details=True):
        '''Create a copy of the pipeline modules and settings'''
        fd = StringIO.StringIO()
        self.save(fd, save_image_plane_details=save_image_plane_details)
        pipeline = Pipeline()
        fd.seek(0)
        pipeline.load(fd)
        return pipeline

    def settings_hash(self, until_module=None, as_string=False):
        '''Return a hash of the module settings

        This function can be used to invalidate a cached calculation
        that's based on pipeline settings - if the settings change, the
        hash changes and the calculation must be performed again.

        We use secure hashing functions which are really good at avoiding
        collisions for small changes in data.
        '''
        h = hashlib.md5()
        for module in self.modules():
            h.update(module.module_name)
            for setting in module.settings():
                h.update(setting.unicode_value.encode('utf-8'))
            if module.module_name == until_module:
                break
        if as_string:
            return h.hexdigest()
        return h.digest()

    def create_from_handles(self, handles):
        """Read a pipeline's modules out of the handles structure

        """
        self.__modules = []
        try:
            settings = handles[SETTINGS][0, 0]
            module_names = settings[MODULE_NAMES]
        except Exception, instance:
            logger.error("Failed to load pipeline", exc_info=True)
            e = LoadExceptionEvent(instance, None)
            self.notify_listeners(e)
            return
        module_count = module_names.shape[1]
        real_module_num = 1
        for module_num in range(1, module_count + 1):
            idx = module_num - 1
            module_name = module_names[0, idx][0]
            module = None
            try:
                module = self.instantiate_module(module_name)
                module.create_from_handles(handles, module_num)
                module.module_num = real_module_num
            except Exception, instance:
                logger.error("Failed to load pipeline", exc_info=True)
                number_of_variables = settings[NUMBERS_OF_VARIABLES][0, idx]
                module_settings = [settings[VARIABLE_VALUES][idx, i]
                                   for i in range(number_of_variables)]
                module_settings = [('' if np.product(x.shape) == 0
                                    else str(x[0])) if isinstance(x, np.ndarray)
                                   else str(x)
                                   for x in module_settings]

                event = LoadExceptionEvent(instance, module, module_name,
                                           module_settings)
                self.notify_listeners(event)
                if event.cancel_run:
                    # The pipeline is somewhat loaded at this point
                    # so we break the loop and clean up as well as we can
                    break
            if module is not None:
                self.__modules.append(module)
                real_module_num += 1
        for module in self.__modules:
            module.post_pipeline_load(self)

        self.notify_listeners(PipelineLoadedEvent())

    def instantiate_module(self, module_name):
        import cellprofiler.modules
        return cellprofiler.modules.instantiate_module(module_name)

    def reload_modules(self):
        """Reload modules from source, and attempt to update pipeline to new versions.
        Returns True if pipeline was successfully updated.

        """
        # clear previously seen errors on reload
        import cellprofiler.modules
        reload(cellprofiler.modules)
        cellprofiler.modules.reload_modules()
        # attempt to reinstantiate pipeline with new modules
        try:
            self.copy()  # if this fails, we probably can't reload
            fd = StringIO.StringIO()
            self.save(fd)
            fd.seek(0)
            self.loadtxt(fd, raise_on_error=True)
            return True
        except Exception, e:
            logging.warning("Modules reloaded, but could not reinstantiate pipeline with new versions.", exc_info=True)
            return False

    def save_to_handles(self):
        """Create a numpy array representing this pipeline

        """
        settings = np.ndarray(shape=[1, 1], dtype=SETTINGS_DTYPE)
        handles = {SETTINGS: settings}
        setting = settings[0, 0]
        # The variables are a (modules,max # of variables) array of cells (objects)
        # where an empty cell is a (1,0) array of float64

        try:
            variable_count = max([len(module.settings()) for module in self.modules(False)])
        except:
            for module in self.modules(False):
                if not isinstance(module.settings(), list):
                    raise ValueError('Module %s.settings() did not return a list\n value: %s' % (
                        module.module_name, module.settings()))
                raise

        module_count = len(self.modules(False))
        setting[VARIABLE_VALUES] = new_string_cell_array((module_count, variable_count))
        # The variable info types are similarly shaped
        setting[VARIABLE_INFO_TYPES] = new_string_cell_array((module_count, variable_count))
        setting[MODULE_NAMES] = new_string_cell_array((1, module_count))
        setting[NUMBERS_OF_VARIABLES] = np.ndarray((1, module_count),
                                                   dtype=np.dtype('uint8'))
        setting[PIXEL_SIZE] = cpprefs.get_pixel_size()
        setting[VARIABLE_REVISION_NUMBERS] = np.ndarray((1, module_count),
                                                        dtype=np.dtype('uint8'))
        setting[MODULE_REVISION_NUMBERS] = np.ndarray((1, module_count),
                                                      dtype=np.dtype('uint16'))
        setting[MODULE_NOTES] = new_string_cell_array((1, module_count))
        setting[SHOW_WINDOW] = np.ndarray((1, module_count),
                                          dtype=np.dtype('uint8'))
        setting[BATCH_STATE] = np.ndarray((1, module_count),
                                          dtype=np.dtype('object'))
        for i in range(module_count):
            setting[BATCH_STATE][0, i] = np.zeros((0,), np.uint8)

        for module in self.modules(False):
            module.save_to_handles(handles)
        return handles

    @staticmethod
    def is_pipeline_txt_file(filename):
        '''Test a file to see if it can be loaded by Pipeline.loadtxt

        filename - path to the file

        returns True if the file starts with the CellProfiler cookie.
        '''
        with open(filename, "rb") as fd:
            return Pipeline.is_pipeline_txt_fd(fd)

    @staticmethod
    def is_pipeline_txt_fd(fd):
        header = fd.read(1024)
        fd.seek(0)
        if header.startswith(COOKIE):
            return True
        if re.search(SAD_PROOFPOINT_COOKIE, header):
            logger.info(
                    "print_emoji(\":cat_crying_because_of_proofpoint:\")")
            return True
        return False

    def load(self, fd_or_filename):
        """Load the pipeline from a file

        fd_or_filename - either the name of a file or a file-like object
        """
        self.__modules = []
        self.__undo_stack = []
        self.__undo_start = None
        filename = None
        if hasattr(fd_or_filename, 'seek') and hasattr(fd_or_filename, 'read'):
            fd = fd_or_filename
            needs_close = False
        elif hasattr(fd_or_filename, 'read') and hasattr(fd_or_filename, 'url'):
            # This is a URL file descriptor. Read into a StringIO so that
            # seek is available.
            fd = StringIO.StringIO()
            while True:
                text = fd_or_filename.read()
                if len(text) == 0:
                    break
                fd.write(text)
            fd.seek(0)
            needs_close = False
        elif os.path.exists(fd_or_filename):
            fd = open(fd_or_filename, 'rb')
            needs_close = True
            filename = fd_or_filename
        else:
            # Assume is string URL
            parsed_path = urlparse.urlparse(fd_or_filename)
            if len(parsed_path.scheme) < 2:
                raise IOError("Could not find file, " + fd_or_filename)
            fd = urllib2.urlopen(fd_or_filename)
            return self.load(fd)
        if Pipeline.is_pipeline_txt_fd(fd):
            self.loadtxt(fd)
            return
        header = fd.read(len(HDF5_HEADER))
        if needs_close:
            fd.close()
        else:
            fd.seek(0)
        if header[:8] == HDF5_HEADER:
            if filename is None:
                fid, filename = tempfile.mkstemp(".h5")
                fd_out = os.fdopen(fid, "wb")
                fd_out.write(fd.read())
                fd_out.close()
                self.load(filename)
                os.unlink(filename)
                return
            else:
                m = cpmeas.load_measurements(filename)
                pipeline_text = m.get_experiment_measurement(M_PIPELINE)
                pipeline_text = pipeline_text.encode('us-ascii')
                self.load(StringIO.StringIO(pipeline_text))
                return

        if has_mat_read_error:
            try:
                handles = scipy.io.matlab.mio.loadmat(fd_or_filename,
                                                      struct_as_record=True)
            except MatReadError:
                logging.error("Caught exception in Matlab reader\n", exc_info=True)
                e = MatReadError(
                        "%s is an unsupported .MAT file, most likely a measurements file.\nYou can load this as a pipeline if you load it as a pipeline using CellProfiler 1.0 and then save it to a different file.\n" %
                        fd_or_filename)
                self.notify_listeners(LoadExceptionEvent(e, None))
                return
            except Exception, e:
                logging.error("Tried to load corrupted .MAT file: %s\n" % fd_or_filename,
                              exc_info=True)
                self.notify_listeners(LoadExceptionEvent(e, None))
                return
        else:
            handles = scipy.io.matlab.mio.loadmat(fd_or_filename,
                                                  struct_as_record=True)

        if handles.has_key("handles"):
            #
            # From measurements...
            #
            handles = handles["handles"][0, 0]
        self.create_from_handles(handles)
        self.__settings = [self.capture_module_settings(module)
                           for module in self.modules(False)]
        self.__undo_stack = []

    def respond_to_version_mismatch_error(self, message):
        logging.warning(message)

    def loadtxt(self, fd_or_filename, raise_on_error=False):
        '''Load a pipeline from a text file

        fd_or_filename - either a path to a file or a file-descriptor-like
                         object.
        raise_on_error - if there is an error loading the pipeline, raise an
                         exception rather than generating a LoadException event.

        See savetxt for more comprehensive documentation.
        '''
        self.__modules = []
        self.caption_for_user = None
        self.message_for_user = None
        module_count = sys.maxint
        if hasattr(fd_or_filename, 'seek') and hasattr(fd_or_filename, 'read'):
            fd = fd_or_filename
        else:
            fd = open(fd_or_filename, 'r')

        def rl():
            '''Read a line from fd'''
            try:
                line = fd.next()
                if line is None:
                    return None
                line = line.strip("\r\n")
                return line
            except StopIteration:
                return None

        header = rl()
        if not self.is_pipeline_txt_fd(StringIO.StringIO(header)):
            raise NotImplementedError('Invalid header: "%s"' % header)
        version = NATIVE_VERSION
        from_matlab = False
        do_utf16_decode = False
        has_image_plane_details = False
        git_hash = None
        pipeline_version = cellprofiler.__version__
        CURRENT_VERSION = None
        while True:
            line = rl()
            if line is None:
                if module_count == 0:
                    break
                raise ValueError("Pipeline file unexpectedly truncated before module section")
            elif len(line.strip()) == 0:
                break
            kwd, value = line.split(':')
            if kwd == H_VERSION:
                version = int(value)
                if version > NATIVE_VERSION:
                    raise ValueError("Pipeline file version is {}.\nCellProfiler can only read version {} or less.\nPlease upgrade to the latest version of CellProfiler.".format(version, NATIVE_VERSION))
                elif version > 1:
                    do_utf16_decode = True
            elif kwd in (H_SVN_REVISION, H_DATE_REVISION):
                pipeline_version = int(value)
                CURRENT_VERSION = int(re.sub(r"\.|rc\d{1}", "", cellprofiler.__version__))
            elif kwd == H_FROM_MATLAB:
                from_matlab = (value == "True")
            elif kwd == H_MODULE_COUNT:
                module_count = int(value)
            elif kwd == H_HAS_IMAGE_PLANE_DETAILS:
                has_image_plane_details = (value == "True")
            elif kwd == H_MESSAGE_FOR_USER:
                value = value.decode("string_escape")
                self.caption_for_user, self.message_for_user = value.split("|", 1)
            elif kwd == H_GIT_HASH:
                git_hash = value
            else:
                print line

        if pipeline_version > 20080101000000 and\
           pipeline_version < 30080101000000:
            # being optomistic... a millenium should be OK, no?
            second, minute, hour, day, month = [
                int(pipeline_version / (100 ** i)) % 100
                for i in range(5)]
            year = int(pipeline_version / (100 ** 5))
            pipeline_date = datetime.datetime(
                year, month, day, hour, minute, second).strftime(" @ %c")
        else:
            pipeline_date = ""

        if CURRENT_VERSION is None:
            pass
        if pipeline_version > CURRENT_VERSION:
            message = "Your pipeline version is {} but you are running CellProfiler version {}. Loading this pipeline may fail or have unpredictable results.".format(pipeline_version, CURRENT_VERSION)

            self.respond_to_version_mismatch_error(message)
        else:
            if (not cpprefs.get_headless()) and pipeline_version < CURRENT_VERSION:
                if git_hash is not None:
                    message = (
                        "Your pipeline was saved using an old version\n"
                        "of CellProfiler (rev {}{}).\n"
                        "The current version of CellProfiler can load\n"
                        "and run this pipeline, but if you make changes\n"
                        "to it and save, the older version of CellProfiler\n"
                        "(perhaps the version your collaborator has?) may\n"
                        "not be able to load it.\n\n"
                        "You can ignore this warning if you do not plan to save\n"
                        "this pipeline or if you will only use it with this or\n"
                        "later versions of CellProfiler."
                    ).format(git_hash, pipeline_date)
                    logging.warning(message)
                else:
                    message = (
                        "Your pipeline was saved using an old version\n"
                        "of CellProfiler (version {}). The current version\n"
                        "of CellProfiler can load and run this pipeline, but\n"
                        "if you make changes to it and save, the older version\n"
                        "of CellProfiler (perhaps the version your collaborator\n"
                        "has?) may not be able to load it.\n\n"
                        "You can ignore this warning if you do not plan to save\n"
                        "this pipeline or if you will only use it with this or\n"
                        "later versions of CellProfiler."
                    ).format(pipeline_version)
                    logging.warning(message)

        #
        # The module section
        #
        new_modules = []
        module_number = 1
        skip_attributes = ['svn_version', 'module_num']
        for i in xrange(module_count):
            line = rl()
            if line is None:
                break
            settings = []
            try:
                module = None
                module_name = None
                split_loc = line.find(':')
                if split_loc == -1:
                    raise ValueError("Invalid format for module header: %s" % line)
                module_name = line[:split_loc].strip()
                attribute_string = line[(split_loc + 1):]
                #
                # Decode the settings
                #
                last_module = False
                while True:
                    line = rl()
                    if line is None:
                        last_module = True
                        break
                    if len(line.strip()) == 0:
                        break
                    if len(line.split(':')) != 2:
                        raise ValueError("Invalid format for setting: %s" % line)
                    text, setting = line.split(':')
                    setting = setting.decode('string_escape')
                    if do_utf16_decode:
                        setting = utf16decode(setting)
                    settings.append(setting)
                #
                # Set up the module
                #
                module_name = module_name.decode('string_escape')
                module = self.instantiate_module(module_name)
                module.module_num = module_number
                #
                # Decode the attributes. These are turned into strings using
                # repr, so True -> 'True', etc. They are then encoded using
                # Pipeline.encode_txt.
                #
                if (len(attribute_string) < 2 or attribute_string[0] != '[' or
                            attribute_string[-1] != ']'):
                    raise ValueError("Invalid format for attributes: %s" %
                                     attribute_string)
                attribute_strings = attribute_string[1:-1].split('|')
                variable_revision_number = None
                # make batch_state decodable from text pipelines
                array = np.array
                uint8 = np.uint8
                for a in attribute_strings:
                    if len(a.split(':')) != 2:
                        raise ValueError("Invalid attribute string: %s" % a)
                    attribute, value = a.split(':')
                    value = value.decode('string_escape')
                    value = eval(value)
                    if attribute == 'variable_revision_number':
                        variable_revision_number = value
                    elif attribute in skip_attributes:
                        pass
                    else:
                        setattr(module, attribute, value)
                if variable_revision_number is None:
                    raise ValueError("Module %s did not have a variable revision # attribute" % module_name)
                module.set_settings_from_values(settings,
                                                variable_revision_number,
                                                module_name, from_matlab)
            except Exception, instance:
                if raise_on_error:
                    raise
                logging.error("Failed to load pipeline", exc_info=True)
                event = LoadExceptionEvent(instance, module, module_name,
                                           settings)
                self.notify_listeners(event)
                if event.cancel_run:
                    break
            if module is not None:
                new_modules.append(module)
                module_number += 1
        if has_image_plane_details:
            self.clear_urls(add_undo=False)
            self.__file_list = read_file_list(fd)
            self.__filtered_file_list_images_settings = None
            self.__filtered_image_plane_details_metadata_settings = None

        self.__modules = new_modules
        self.__settings = [self.capture_module_settings(module)
                           for module in self.modules(False)]
        for module in self.modules(False):
            module.post_pipeline_load(self)
        self.notify_listeners(PipelineLoadedEvent())
        if has_image_plane_details:
            self.notify_listeners(URLsAddedEvent(
                    self.__file_list))
        self.__undo_stack = []
        return pipeline_version, git_hash

    def save(self, fd_or_filename,
             format=FMT_NATIVE,
             save_image_plane_details=True):
        """Save the pipeline to a file

        fd_or_filename - either a file descriptor or the name of the file
        """
        if format == FMT_MATLAB:
            handles = self.save_to_handles()
            self.savemat(fd_or_filename, handles)
        elif format == FMT_NATIVE:
            self.savetxt(fd_or_filename,
                         save_image_plane_details=save_image_plane_details)
        else:
            raise NotImplementedError("Unknown pipeline file format: %s" %
                                      format)

    def encode_txt(self, s):
        '''Encode a string for saving in the text format

        s - input string
        Encode for automatic decoding using the 'string_escape' decoder.
        We encode the special characters, '[', ':', '|' and ']' using the '\\x'
        syntax.
        '''
        s = s.encode('string_escape')
        s = s.replace(':', '\\x3A')
        s = s.replace('|', '\\x7C')
        s = s.replace('[', '\\x5B').replace(']', '\\x5D')
        return s

    def savetxt(self, fd_or_filename,
                modules_to_save=None,
                save_image_plane_details=True):
        '''Save the pipeline in a text format

        fd_or_filename - can be either a "file descriptor" with a "write"
                         attribute or the path to the file to write.

        modules_to_save - if present, the module numbers of the modules to save

        save_image_plane_details - True to save the image plane details (image
                          URL, series, index, channel and metadata)

        The format of the file is the following:
        Strings are encoded using a backslash escape sequence. The colon
        character is encoded as \\x3A if it should happen to appear in a string
        and any non-printing character is encoded using the \\x## convention.

        Line 1: The cookie, identifying this as a CellProfiler pipeline file.
        The header, i
        Line 2: "Version:#" the file format version #
        Line 3: "DateRevision:#" the version # of the CellProfiler
                that wrote this file (date encoded as int, see cp.utitlities.version)
        Line 4: "ModuleCount:#" the number of modules saved in the file
        Line 5: "HasImagePlaneDetails:True/False" has the list of image plane info after the settings
        Line 5: blank

        The module list follows. Each module has a header composed of
        the module name, followed by attributes to be set on the module
        using setattr (the string following the attribute is first evaluated
        using eval()). For instance:
        Align:[show_window:True|notes='Align my image']

        The settings follow. Each setting has text and a value. For instance,
        Enter object name:Nuclei

        The image plane details can be saved along with the pipeline. These
        are a collection of images and their metadata.
        See read_image_plane_details for the file format
        '''
        if hasattr(fd_or_filename, "write"):
            fd = fd_or_filename
            needs_close = False
        else:
            fd = open(fd_or_filename, "wt")
            needs_close = True

        # Don't write image plane details if we don't have any
        if len(self.__file_list) == 0:
            save_image_plane_details = False

        fd.write("%s\n" % COOKIE)
        fd.write("%s:%d\n" % (H_VERSION, NATIVE_VERSION))
        fd.write("%s:%d\n" % (H_DATE_REVISION, int(re.sub(r"\.|rc\d{1}", "", cellprofiler.__version__))))
        fd.write("%s:%s\n" % (H_GIT_HASH, ""))
        fd.write("%s:%d\n" % (H_MODULE_COUNT, len(self.__modules)))
        fd.write("%s:%s\n" % (H_HAS_IMAGE_PLANE_DETAILS, str(save_image_plane_details)))
        attributes = (
            'module_num', 'svn_version', 'variable_revision_number',
            'show_window', 'notes', 'batch_state', 'enabled', 'wants_pause')
        notes_idx = 4
        for module in self.__modules:
            if ((modules_to_save is not None) and
                        module.module_num not in modules_to_save):
                continue
            fd.write("\n")
            attribute_values = [repr(getattr(module, attribute))
                                for attribute in attributes]
            attribute_values = [self.encode_txt(v) for v in attribute_values]
            attribute_strings = [attribute + ':' + value
                                 for attribute, value
                                 in zip(attributes, attribute_values)]
            attribute_string = '[%s]' % ('|'.join(attribute_strings))
            fd.write('%s:%s\n' % (self.encode_txt(module.module_name),
                                  attribute_string))
            for setting in module.settings():
                setting_text = setting.text
                if isinstance(setting_text, unicode):
                    setting_text = setting_text.encode('utf-8')
                else:
                    setting_text = str(setting_text)
                fd.write('    %s:%s\n' % (
                    self.encode_txt(setting_text),
                    self.encode_txt(utf16encode(setting.unicode_value))))
        if save_image_plane_details:
            fd.write("\n")
            write_file_list(fd, self.__file_list)
        if needs_close:
            fd.close()

    def save_pipeline_notes(self, fd, indent=2):
        '''Save pipeline notes to a text file

        fd - file descriptor of the file.

        indent - indent of the notes relative to module header.
        '''
        lines = []
        for module in self.modules(exclude_disabled=False):
            if module.enabled:
                fmt = "[%4.d] [%s]"
            else:
                fmt = "[%4.d] [%s] (disabled)"
            lines.append(fmt % (module.module_num, module.module_name))
            for note in module.notes:
                lines.append("%s%s" % ("".join([" "] * indent), note))
            lines.append("")
        fd.write("\n".join(lines))

    def save_measurements(self, filename, measurements):
        """Save the measurements and the pipeline settings in a Matlab file

        filename     - name of file to create, or a file-like object
        measurements - measurements structure that is the result of running the pipeline
        """
        handles = self.build_matlab_handles()
        add_all_measurements(handles, measurements)
        handles[CURRENT][NUMBER_OF_IMAGE_SETS][0, 0] = float(measurements.image_set_number + 1)
        handles[CURRENT][SET_BEING_ANALYZED][0, 0] = float(measurements.image_set_number + 1)
        #
        # For the output file, you have to bury it a little deeper - the root has to have
        # a single field named "handles"
        #
        root = {'handles': np.ndarray((1, 1), dtype=make_cell_struct_dtype(handles.keys()))}
        for key, value in handles.iteritems():
            root['handles'][key][0, 0] = value
        self.savemat(filename, root)

    def write_pipeline_measurement(self, m, user_pipeline=False):
        '''Write the pipeline experiment measurement to the measurements

        m - write into these measurements

        user_pipeline - if True, write the pipeline into M_USER_PIPELINE
                        M_USER_PIPELINE is the pipeline that should be loaded
                        by the UI for the user for cases like a pipeline
                        created by CreateBatchFiles.
        '''
        assert (isinstance(m, cpmeas.Measurements))
        fd = StringIO.StringIO()
        self.savetxt(fd, save_image_plane_details=False)
        m.add_measurement(cpmeas.EXPERIMENT,
                          M_USER_PIPELINE if user_pipeline else M_PIPELINE,
                          fd.getvalue(),
                          can_overwrite=True)

    def clear_measurements(self, m):
        '''Erase all measurements, but make sure to re-establish the pipeline one

        m - measurements to be cleared
        '''
        m.clear()
        self.write_experiment_measurements(m)

    def savemat(self, filename, root):
        '''Save a handles structure accounting for scipy version compatibility to a filename or file-like object'''
        sver = scipy.__version__.split('.')
        if (len(sver) >= 2 and sver[0].isdigit() and int(sver[0]) == 0 and
                sver[1].isdigit() and int(sver[1]) < 8):
            #
            # 1-d -> 2-d not done
            #
            scipy.io.matlab.mio.savemat(filename, root, format='5',
                                        long_field_names=True)
        else:
            scipy.io.matlab.mio.savemat(filename, root, format='5',
                                        long_field_names=True,
                                        oned_as='column')

    def build_matlab_handles(self, image_set=None, object_set=None, measurements=None):
        handles = self.save_to_handles()
        image_tools_dir = os.path.join(cpprefs.cell_profiler_root_directory(), 'ImageTools')
        if os.access(image_tools_dir, os.R_OK):
            image_tools = [str(os.path.split(os.path.splitext(filename)[0])[1])
                           for filename in os.listdir(image_tools_dir)
                           if os.path.splitext(filename)[1] == '.m']
        else:
            image_tools = []
        image_tools.insert(0, 'Image tools')
        npy_image_tools = np.ndarray((1, len(image_tools)), dtype=np.dtype('object'))
        for tool, idx in zip(image_tools, range(0, len(image_tools))):
            npy_image_tools[0, idx] = tool

        current = np.ndarray(shape=[1, 1], dtype=CURRENT_DTYPE)
        handles[CURRENT] = current
        current[NUMBER_OF_IMAGE_SETS][0, 0] = [(image_set is not None and image_set.legacy_fields.has_key(
                NUMBER_OF_IMAGE_SETS) and image_set.legacy_fields[NUMBER_OF_IMAGE_SETS]) or 1]
        current[SET_BEING_ANALYZED][0, 0] = [(measurements and measurements.image_set_number) or 1]
        current[NUMBER_OF_MODULES][0, 0] = [len(self.__modules)]
        current[SAVE_OUTPUT_HOW_OFTEN][0, 0] = [1]
        current[TIME_STARTED][0, 0] = str(datetime.datetime.now())
        current[STARTING_IMAGE_SET][0, 0] = [1]
        current[STARTUP_DIRECTORY][0, 0] = cpprefs.cell_profiler_root_directory()
        current[DEFAULT_OUTPUT_DIRECTORY][0, 0] = cpprefs.get_default_output_directory()
        current[DEFAULT_IMAGE_DIRECTORY][0, 0] = cpprefs.get_default_image_directory()
        current[IMAGE_TOOLS_FILENAMES][0, 0] = npy_image_tools
        current[IMAGE_TOOL_HELP][0, 0] = []

        preferences = np.ndarray(shape=(1, 1), dtype=PREFERENCES_DTYPE)
        handles[PREFERENCES] = preferences
        preferences[PIXEL_SIZE][0, 0] = cpprefs.get_pixel_size()
        preferences[DEFAULT_MODULE_DIRECTORY][0, 0] = cpprefs.module_directory()
        preferences[DEFAULT_OUTPUT_DIRECTORY][0, 0] = cpprefs.get_default_output_directory()
        preferences[DEFAULT_IMAGE_DIRECTORY][0, 0] = cpprefs.get_default_image_directory()
        preferences[INTENSITY_COLOR_MAP][0, 0] = 'gray'
        preferences[LABEL_COLOR_MAP][0, 0] = 'jet'
        preferences[STRIP_PIPELINE][0, 0] = 'Yes'  # TODO - get from preferences
        preferences[SKIP_ERRORS][0, 0] = 'No'  # TODO - get from preferences
        preferences[DISPLAY_MODE_VALUE][0, 0] = [1]  # TODO - get from preferences
        preferences[FONT_SIZE][0, 0] = [10]  # TODO - get from preferences
        preferences[DISPLAY_WINDOWS][0, 0] = [1 for module in
                                              self.__modules]  # TODO - UI allowing user to choose whether to display a window

        images = {}
        if image_set:
            for provider in image_set.providers:
                image = image_set.get_image(provider.name)
                if image.image is not None:
                    images[provider.name] = image.image
                if image.mask is not None:
                    images['CropMask' + provider.name] = image.mask
            for key, value in image_set.legacy_fields.iteritems():
                if key != NUMBER_OF_IMAGE_SETS:
                    images[key] = value

        if object_set:
            for name, objects in object_set.all_objects:
                images['Segmented' + name] = objects.segmented
                if objects.has_unedited_segmented():
                    images['UneditedSegmented' + name] = objects.unedited_segmented
                if objects.has_small_removed_segmented():
                    images['SmallRemovedSegmented' + name] = objects.small_removed_segmented

        if len(images):
            pipeline_dtype = make_cell_struct_dtype(images.keys())
            pipeline = np.ndarray((1, 1), dtype=pipeline_dtype)
            handles[PIPELINE] = pipeline
            for name, image in images.items():
                pipeline[name][0, 0] = images[name]

        no_measurements = (measurements is None or len(measurements.get_object_names()) == 0)
        if not no_measurements:
            measurements_dtype = make_cell_struct_dtype(measurements.get_object_names())
            npy_measurements = np.ndarray((1, 1), dtype=measurements_dtype)
            handles['Measurements'] = npy_measurements
            for object_name in measurements.get_object_names():
                object_dtype = make_cell_struct_dtype(measurements.get_feature_names(object_name))
                object_measurements = np.ndarray((1, 1), dtype=object_dtype)
                npy_measurements[object_name][0, 0] = object_measurements
                for feature_name in measurements.get_feature_names(object_name):
                    feature_measurements = np.ndarray((1, measurements.image_set_number), dtype='object')
                    object_measurements[feature_name][0, 0] = feature_measurements
                    data = measurements.get_current_measurement(object_name, feature_name)
                    feature_measurements.fill(np.ndarray((0,), dtype=np.float64))
                    if data is not None:
                        feature_measurements[0, measurements.image_set_number - 1] = data
        return handles

    def find_external_input_images(self):
        '''Find the names of the images that need to be supplied externally

        run_external needs a dictionary of name -> image pixels with
        one name entry for every external image that must be provided.
        This function returns a list of those names.
        '''
        result = []
        for module in self.modules():
            for setting in module.settings():
                if isinstance(setting, cps.ExternalImageNameProvider):
                    result.append(setting.value)
        return result

    def find_external_output_images(self):
        result = []
        for module in self.modules():
            for setting in module.settings():
                if isinstance(setting, cps.ExternalImageNameSubscriber):
                    result.append(setting.value)
        return result

    def can_convert_legacy_input_modules(self):
        '''Can legacy modules like LoadImages be converted to modern form?

        Returns True if all legacy input modules can be converted to
        Images / Metadata / NamesAndTypes / Groups.
        '''
        needs_conversion = False
        try:
            for module in self.__modules:
                if module.needs_conversion():
                    needs_conversion = True
            return needs_conversion
        except:
            return False

    def convert_legacy_input_modules(self):
        '''Convert a pipeline from legacy to using Images, NamesAndTypes etc'''
        if not self.can_convert_legacy_input_modules():
            return
        from cellprofiler.modules.images import Images, FILTER_CHOICE_NONE
        from cellprofiler.modules.metadata import Metadata
        from cellprofiler.modules.namesandtypes import NamesAndTypes
        from cellprofiler.modules.groups import Groups
        with self.undoable_action("Legacy modules converted"):
            images, metadata, namesandtypes, groups = (
                Images(), Metadata(), NamesAndTypes(), Groups())
            images.filter_choice.value = FILTER_CHOICE_NONE
            for i, module in enumerate((images, metadata, namesandtypes, groups)):
                module.set_module_num(i + 1)
                module.show_window = cpprefs.get_headless()
                if module.notes:
                    module.notes += ["---"]
                module.notes += ["Settings converted from legacy pipeline."]
                self.add_module(module)
            for module in list(self.__modules):
                if module.needs_conversion():
                    module.convert(self, metadata, namesandtypes, groups)
                    self.remove_module(module.module_num)
            self.notify_listeners(PipelineLoadedEvent())

    def convert_default_input_folder(self, path):
        '''Convert all references to the default input folder to abolute paths

        path - the path to use in place of the default input folder
        '''
        with self.undoable_action("Convert default input folder"):
            for module in self.modules(False):
                was_edited = False
                for setting in module.settings():
                    if isinstance(setting, cps.DirectoryPath):
                        if setting.dir_choice == cpprefs.DEFAULT_INPUT_FOLDER_NAME:
                            setting.dir_choice = cpprefs.ABSOLUTE_FOLDER_NAME
                            setting.custom_path = path
                            was_edited = True
                        elif setting.dir_choice == cpprefs.DEFAULT_INPUT_SUBFOLDER_NAME:
                            subpath = os.path.join(path, setting.custom_path)
                            setting.dir_choice = cpprefs.ABSOLUTE_FOLDER_NAME
                            setting.custom_path = subpath
                if was_edited:
                    self.edit_module(module.module_num, True)
            self.notify_listeners(PipelineLoadedEvent())

    def fix_legacy_pipeline(self):
        '''Perform inter-module fixes needed for some legacy pipelines'''
        from cellprofiler.modules.loadsingleimage import LoadSingleImage
        #
        # LoadSingleImage used to work if placed before LoadImages or
        # LoadData, but doesn't any more
        #
        while True:
            for i, module in enumerate(self.modules()):
                if isinstance(module, LoadSingleImage):
                    for other_module in self.modules()[(i + 1):]:
                        if other_module.is_load_module():
                            self.move_module(other_module.module_num,
                                             DIRECTION_UP)
                            break
                    else:
                        continue
                    break  # Rerun from start
            else:
                break

    def requires_aggregation(self):
        '''Return True if the pipeline requires aggregation across image sets

        If a pipeline has aggregation modules, the image sets in a group
        need to be run sequentially on the same worker.
        '''
        for module in self.modules():
            if module.is_aggregation_module():
                return True
        return False

    def obfuscate(self):
        '''Tell all modules in the pipeline to obfuscate any sensitive info

        This call is designed to erase any information that users might
        not like to see uploaded. You should copy a pipeline before obfuscating.
        '''
        for module in self.modules(False):
            module.obfuscate()

    def run_external(self, image_dict):
        """Runs a single iteration of the pipeline with the images provided in
        image_dict and returns a dictionary mapping from image names to images
        specified by ExternalImageNameSubscribers.

        image_dict - dictionary mapping image names to image pixel data in the
                     form of a numpy array.
        """
        import cellprofiler.setting as cps
        from cellprofiler import object as cpo

        output_image_names = self.find_external_output_images()
        input_image_names = self.find_external_input_images()

        # Check that the incoming dictionary matches the names expected by the
        # ExternalImageProviders
        for name in input_image_names:
            assert name in image_dict, 'Image named "%s" was not provided in the input dictionary' % name

        # Create image set from provided dict
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        for image_name in input_image_names:
            input_pixels = image_dict[image_name]
            image_set.add(image_name, cpi.Image(input_pixels))
        object_set = cpo.ObjectSet()
        measurements = cpmeas.Measurements()

        # Run the modules
        for module in self.modules():
            workspace = cpw.Workspace(self, module, image_set, object_set,
                                      measurements, image_set_list)
            self.run_module(module, workspace)

        # Populate a dictionary for output with the images to be exported
        output_dict = {}
        for name in output_image_names:
            output_dict[name] = image_set.get_image(name).pixel_data

        return output_dict

    def run(self,
            frame=None,
            image_set_start=1,
            image_set_end=None,
            grouping=None,
            measurements_filename=None,
            initial_measurements=None):
        """Run the pipeline

        Run the pipeline, returning the measurements made
        frame - the frame to be used when displaying graphics or None to
                run headless
        image_set_start - the image number of the first image to be run
        image_set_end - the index of the last image to be run + 1
        grouping - a dictionary that gives the keys and values in the
                   grouping to run or None to run all groupings
        measurements_filename - name of file to use for measurements
        """
        measurements = cpmeas.Measurements(
                image_set_start=image_set_start,
                filename=measurements_filename,
                copy=initial_measurements)
        if not self.in_batch_mode() and initial_measurements is not None:
            #
            # Need file list in order to call prepare_run
            #
            from cellprofiler.utilities.hdf5_dict import HDF5FileList
            src = initial_measurements.hdf5_dict.hdf5_file
            dest = measurements.hdf5_dict.hdf5_file
            if HDF5FileList.has_file_list(src):
                HDF5FileList.copy(src, dest)
                self.add_urls(HDF5FileList(dest).get_filelist())

        measurements.is_first_image = True
        for m in self.run_with_yield(frame, image_set_start, image_set_end,
                                     grouping,
                                     run_in_background=False,
                                     initial_measurements=measurements):
            measurements = m
        return measurements

    def group(self, grouping, image_set_start, image_set_end, initial_measurements, workspace):
        """Enumerate relevant image sets.  This function is side-effect free, so it can be called more than once."""

        keys, groupings = self.get_groupings(workspace)

        if grouping is not None and set(keys) != set(grouping.keys()):
            raise ValueError("The grouping keys specified on the command line (%s) must be the same as those defined by the modules in the pipeline (%s)" % (", ".join(grouping.keys()), ", ".join(keys)))

        for gn, (grouping_keys, image_numbers) in enumerate(groupings):
            if grouping is not None and grouping != grouping_keys:
                continue

            need_to_run_prepare_group = True

            for gi, image_number in enumerate(image_numbers):
                if image_number < image_set_start:
                    continue

                if image_set_end is not None and image_number > image_set_end:
                    continue

                if initial_measurements is not None and all([initial_measurements.has_feature(cpmeas.IMAGE, f) for f in GROUP_NUMBER, GROUP_INDEX]):
                    group_number, group_index = [initial_measurements[cpmeas.IMAGE, f, image_number] for f in GROUP_NUMBER, GROUP_INDEX]
                else:
                    group_number = gn + 1

                    group_index = gi + 1

                if need_to_run_prepare_group:
                    yield group_number, group_index, image_number, lambda: self.prepare_group(workspace, grouping_keys, image_numbers)
                else:
                    yield group_number, group_index, image_number, lambda: True

                need_to_run_prepare_group = False

            if not need_to_run_prepare_group:
                yield None, None, None, lambda workspace: self.post_group(workspace, grouping_keys)

    def run_with_yield(self, frame=None, image_set_start=1, image_set_end=None, grouping=None, run_in_background=True, status_callback=None, initial_measurements=None):
        """Run the pipeline, yielding periodically to keep the GUI alive.
        Yields the measurements made.

        Arguments:
           status_callback - None or a callable with arguments
                             (module, image_set) that will be called before
                             running each module.

        Run the pipeline, returning the measurements made
        """

        can_display = not cpprefs.get_headless()

        columns = self.get_measurement_columns()

        if image_set_start is not None:
            assert isinstance(image_set_start, int), "Image set start must be an integer"

        if image_set_end is not None:
            assert isinstance(image_set_end, int), "Image set end must be an integer"

        if initial_measurements is None:
            measurements = cpmeas.Measurements(image_set_start)
        else:
            measurements = initial_measurements

        image_set_list = cpi.ImageSetList()

        workspace = cpw.Workspace(self, None, None, None, measurements, image_set_list, frame)

        try:
            if not self.prepare_run(workspace):
                return

            #
            # Remove image sets outside of the requested ranges
            #
            image_numbers = measurements.get_image_numbers()

            to_remove = []

            if image_set_start is not None:
                to_remove += [x for x in image_numbers if x < image_set_start]

                image_numbers = [x for x in image_numbers if x >= image_set_start]

            if image_set_end is not None:
                to_remove += [x for x in image_numbers if x > image_set_end]

                image_numbers = [x for x in image_numbers if x <= image_set_end]

            if grouping is not None:
                keys, groupings = self.get_groupings(workspace)

                for grouping_keys, grouping_image_numbers in groupings:
                    if grouping_keys != grouping:
                        to_remove += list(grouping_image_numbers)

            if len(to_remove) > 0 and measurements.has_feature(cpmeas.IMAGE, cpmeas.IMAGE_NUMBER):
                for image_number in np.unique(to_remove):
                    measurements.remove_measurement(cpmeas.IMAGE, cpmeas.IMAGE_NUMBER, image_number)

            # Keep track of progress for the benefit of the progress window.
            num_image_sets = len(measurements.get_image_numbers())

            image_set_count = -1

            is_first_image_set = True

            last_image_number = None

            pipeline_stats_logger.info("Times reported are CPU times for each module, not wall-clock time")

            __group = self.group(grouping, image_set_start, image_set_end, initial_measurements, workspace)

            for group_number, group_index, image_number, closure in __group:
                if image_number is None:
                    if not closure(workspace):
                        measurements.add_experiment_measurement(EXIT_STATUS, "Failure")

                        return

                    continue

                image_set_count += 1

                if not closure():
                    return

                last_image_number = image_number

                measurements.clear_cache()

                for provider in measurements.providers:
                    provider.release_memory()

                measurements.next_image_set(image_number)

                if is_first_image_set:
                    measurements.image_set_start = image_number

                    measurements.is_first_image = True

                    is_first_image_set = False

                measurements.group_number = group_number

                measurements.group_index = group_index

                numberof_windows = 0

                slot_number = 0

                object_set = cpo.ObjectSet()

                image_set = measurements

                outlines = {}

                should_write_measurements = True

                grids = None

                for module in self.modules():
                    if module.should_stop_writing_measurements():
                        should_write_measurements = False
                    else:
                        module_error_measurement = ('ModuleError_%02d%s' % (module.module_num, module.module_name))

                        execution_time_measurement = ('ExecutionTime_%02d%s' % (module.module_num, module.module_name))

                    failure = 1

                    exception = None

                    tb = None

                    frame_if_shown = frame if module.show_window else None

                    workspace = cpw.Workspace(self, module, image_set, object_set, measurements, image_set_list, frame_if_shown, outlines=outlines)

                    grids = workspace.set_grids(grids)

                    if status_callback:
                        status_callback(module, len(self.modules()), image_set_count, num_image_sets)

                    start_time = datetime.datetime.now()

                    t0 = sum(os.times()[:-1])

                    try:
                        self.run_module(module, workspace)
                    except Exception as instance:
                        logger.error("Error detected during run of module %s", module.module_name, exc_info=True)

                        exception = instance

                        tb = sys.exc_info()[2]

                    yield measurements

                    t1 = sum(os.times()[:-1])

                    delta_sec = max(0, t1 - t0)

                    pipeline_stats_logger.info("%s: Image # %d, module %s # %d: %.2f sec" % (start_time.ctime(), image_number, module.module_name, module.module_num, delta_sec))

                    if module.show_window and can_display and (exception is None):
                        try:
                            fig = workspace.get_module_figure(module, image_number)

                            module.display(workspace, fig)

                            fig.Refresh()
                        except Exception as instance:
                            logger.error("Failed to display results for module %s", module.module_name, exc_info=True)

                            exception = instance

                            tb = sys.exc_info()[2]

                    workspace.refresh()

                    failure = 0

                    if exception is not None:
                        event = RunExceptionEvent(exception, module, tb)

                        self.notify_listeners(event)

                        if event.cancel_run:
                            return
                        elif event.skip_thisset:
                            # Skip this image, continue to others
                            workspace.set_disposition(cpw.DISPOSITION_SKIP)

                            should_write_measurements = False

                            measurements = None

                    # Paradox: ExportToDatabase must write these columns in order
                    #  to complete, but in order to do so, the module needs to
                    #  have already completed. So we don't report them for it.
                    if module.module_name != 'Restart' and should_write_measurements:
                        measurements.add_measurement('Image', module_error_measurement, np.array([failure]))

                        measurements.add_measurement('Image', execution_time_measurement, np.array([delta_sec]))

                    while workspace.disposition == cpw.DISPOSITION_PAUSE and frame is not None:
                        # try to leave measurements temporary file in a readable state
                        measurements.flush()

                        yield measurements

                    if workspace.disposition == cpw.DISPOSITION_SKIP:
                        break
                    elif workspace.disposition == cpw.DISPOSITION_CANCEL:
                        measurements.add_experiment_measurement(EXIT_STATUS, "Failure")

                        return
            # Close cached readers.
            # This may play a big role with cluster deployments or long standing jobs
            # by freeing up memory and resources.
            clear_image_reader_cache()
            if measurements is not None:
                workspace = cpw.Workspace(self, None, None, None, measurements, image_set_list, frame)

                exit_status = self.post_run(workspace)

                #
                # Record the status after post_run
                #
                measurements.add_experiment_measurement(EXIT_STATUS, exit_status)
        finally:
            if measurements is not None:
                # XXX - We want to force the measurements to update the
                # underlying file, or else we get partially written HDF5
                # files.  There must be a better way to do this.
                measurements.flush()

                del measurements

            self.end_run()

    def run_image_set(self, measurements, image_set_number, interaction_handler, display_handler, cancel_handler):
        """Run the pipeline for a single image set storing the results in measurements.

        Arguments:
             measurements - source of image information, destination for results.
             image_set_number - what image to analyze.
             interaction_handler - callback (to be set in workspace) for
                 interaction requests
             display_handler - callback for display requests

             self.prepare_run() and self.prepare_group() must have already been called.

        Returns a workspace suitable for use in self.post_group()
        """
        measurements.next_image_set(image_set_number)
        measurements.group_number = measurements[cpmeas.IMAGE, cpmeas.GROUP_NUMBER]
        measurements.group_index = measurements[cpmeas.IMAGE, cpmeas.GROUP_INDEX]
        object_set = cpo.ObjectSet()
        image_set = measurements
        measurements.clear_cache()
        for provider in measurements.providers:
            provider.release_memory()
        outlines = {}
        grids = None
        should_write_measurements = True
        for module in self.modules():
            print "Running module", module.module_name, module.module_num
            if module.should_stop_writing_measurements():
                should_write_measurements = False
            workspace = cpw.Workspace(self,
                                      module,
                                      image_set,
                                      object_set,
                                      measurements,
                                      None,
                                      outlines=outlines)
            workspace.interaction_handler = interaction_handler
            workspace.cancel_handler = cancel_handler

            grids = workspace.set_grids(grids)

            start_time = datetime.datetime.now()
            t0 = sum(os.times()[:-1])
            try:
                self.run_module(module, workspace)
                if module.show_window:
                    display_handler(module, workspace.display_data, image_set_number)
            except CancelledException:
                # Analysis worker interaction handler is telling us that
                # the UI has cancelled the run. Forward exception upward.
                raise
            except Exception, exception:
                logger.error("Error detected during run of module %s#%d",
                             module.module_name, module.module_num, exc_info=True)
                if should_write_measurements:
                    measurements[cpmeas.IMAGE,
                                 'ModuleError_%02d%s' % (module.module_num, module.module_name)] = 1
                evt = RunExceptionEvent(exception, module, sys.exc_info()[2])
                self.notify_listeners(evt)
                if evt.cancel_run or evt.skip_thisset:
                    # actual cancellation or skipping handled upstream.
                    return

            t1 = sum(os.times()[:-1])
            delta_secs = max(0, t1 - t0)
            pipeline_stats_logger.info("%s: Image # %d, module %s # %d: %.2f secs" %
                                       (start_time.ctime(), image_set_number,
                                        module.module_name, module.module_num,
                                        delta_secs))
            # Paradox: ExportToDatabase must write these columns in order
            #  to complete, but in order to do so, the module needs to
            #  have already completed. So we don't report them for it.
            if should_write_measurements:
                measurements[cpmeas.IMAGE,
                             'ModuleError_%02d%s' % (module.module_num, module.module_name)] = 0
                measurements[cpmeas.IMAGE,
                             'ExecutionTime_%02d%s' % (module.module_num, module.module_name)] = delta_secs

            measurements.flush()
            if workspace.disposition == cpw.DISPOSITION_SKIP:
                break
        return cpw.Workspace(self, None, measurements, object_set,
                             measurements, None, outlines=outlines)

    def end_run(self):
        '''Tell everyone that a run is ending'''
        self.notify_listeners(EndRunEvent())

    def run_group_with_yield(self, workspace, grouping, image_numbers, stop_module, title, message):
        '''Run the modules for the image_numbers in a group up to an agg module

        This method runs a pipeline up to an aggregation step on behalf of
        an aggregation module. At present, you can call this within
        prepare_group to collect the images needed for aggregation.

        workspace - workspace containing the pipeline, image_set_list and
        measurements.

        grouping - the grouping dictionary passed into prepare_group.

        image_numbers - the image numbers that comprise the group

        stop_module - the aggregation module to stop at.

        The function yields the current workspace at the end of processing
        each image set. The workspace has a valid image_set and the
        measurements' image_number is the current image number.
        '''
        m = workspace.measurements
        pipeline = workspace.pipeline
        image_set_list = workspace.image_set_list
        orig_image_number = m.image_set_number

        progress_dialog = self.create_progress_dialog(message, pipeline, title)

        try:
            for i, image_number in enumerate(image_numbers):
                m.image_set_number = image_number
                image_set = m
                object_set = cpo.ObjectSet()
                old_providers = list(image_set.providers)
                for module in pipeline.modules():
                    w = cpw.Workspace(self, module, image_set, object_set, m,
                                      image_set_list)
                    if module == stop_module:
                        yield w
                        # Reset state of image set
                        del image_set.providers[:]
                        image_set.providers.extend(old_providers)
                        break
                    else:
                        self.run_module(module, w)
                    if progress_dialog is not None:
                        should_continue, skip = progress_dialog.Update(i + 1)
                        if not should_continue:
                            progress_dialog.EndModal(0)
                            return
        finally:
            if progress_dialog is not None:
                progress_dialog.Destroy()
            m.image_set_number = orig_image_number

    def create_progress_dialog(self, message, pipeline, title):
        return None

    def run_module(self, module, workspace):
        '''Run one CellProfiler module

        Run the CellProfiler module with whatever preparation and cleanup
        needs to be done before and after.
        '''
        module.run(workspace)

    def write_experiment_measurements(self, m):
        '''Write the standard experiment measurments to the measurements file

        Write the pipeline, version # and timestamp.
        '''
        assert isinstance(m, cpmeas.Measurements)
        self.write_pipeline_measurement(m)
        m.add_experiment_measurement(M_VERSION, cellprofiler.__version__)
        m.add_experiment_measurement(M_TIMESTAMP,
                                     datetime.datetime.now().isoformat())
        m.flush()

    def prepare_run(self, workspace, end_module=None):
        """Do "prepare_run" on each module to initialize the image_set_list

        workspace - workspace for the run

             pipeline - this pipeline

             image_set_list - the image set list for the run. The modules
                              should set this up with the image sets they need.
                              The caller can set test mode and
                              "combine_path_and_file" on the image set before
                              the call.

             measurements - the modules should record URL information here

             frame - the CPFigureFrame if not headless

             Returns True if all modules succeeded, False if any module reported
             failure or threw an exception

        test_mode - None = use pipeline's test mode, True or False to set explicitly

        end_module - if present, terminate before executing this module
        """
        assert (isinstance(workspace, cpw.Workspace))
        m = workspace.measurements
        if self.has_legacy_loaders():
            # Legacy - there may be cached group number/group index
            #          image measurements which may be incorrect.
            m.remove_measurement(cpmeas.IMAGE, cpmeas.GROUP_INDEX)
            m.remove_measurement(cpmeas.IMAGE, cpmeas.GROUP_NUMBER)
        self.write_experiment_measurements(m)

        prepare_run_error_detected = [False]

        def on_pipeline_event(
                pipeline, event,
                prepare_run_error_detected=prepare_run_error_detected):
            if isinstance(event, PrepareRunErrorEvent):
                prepare_run_error_detected[0] = True

        had_image_sets = False
        with self.PipelineListener(self, on_pipeline_event):
            for module in self.modules():
                if module == end_module:
                    break
                try:
                    workspace.set_module(module)
                    workspace.show_frame(module.show_window)
                    if ((not module.prepare_run(workspace)) or
                            prepare_run_error_detected[0]):
                        if workspace.measurements.image_set_count > 0:
                            had_image_sets = True
                        self.clear_measurements(workspace.measurements)
                        break
                except Exception, instance:
                    logging.error("Failed to prepare run for module %s",
                                  module.module_name, exc_info=True)
                    event = PrepareRunExceptionEvent(instance, module, sys.exc_info()[2])
                    self.notify_listeners(event)
                    if event.cancel_run:
                        self.clear_measurements(workspace.measurements)
                        return False
        if workspace.measurements.image_set_count == 0:
            if not had_image_sets:
                self.report_prepare_run_error(
                        None,
                        "The pipeline did not identify any image sets.\n"
                        "Please correct any problems in your input module settings\n"
                        "and try again.")
            return False

        if not m.has_feature(cpmeas.IMAGE, cpmeas.GROUP_NUMBER):
            # Legacy pipelines don't populate group # or index
            key_names, groupings = self.get_groupings(workspace)
            image_numbers = m.get_image_numbers()
            indexes = np.zeros(np.max(image_numbers) + 1, int)
            indexes[image_numbers] = np.arange(len(image_numbers))
            group_numbers = np.zeros(len(image_numbers), int)
            group_indexes = np.zeros(len(image_numbers), int)
            for i, (key, group_image_numbers) in enumerate(groupings):
                iii = indexes[group_image_numbers]
                group_numbers[iii] = i + 1
                group_indexes[iii] = np.arange(
                        len(iii)) + 1
            m.add_all_measurements(
                    cpmeas.IMAGE, cpmeas.GROUP_NUMBER, group_numbers)
            m.add_all_measurements(
                    cpmeas.IMAGE, cpmeas.GROUP_INDEX, group_indexes)
            #
            # The grouping for legacy pipelines may not be monotonically
            # increasing by group number and index.
            # We reorder here.
            #
            order = np.lexsort((group_indexes, group_numbers))
            if np.any(order[1:] != order[:-1] + 1):
                new_image_numbers = np.zeros(max(image_numbers) + 1, int)
                new_image_numbers[image_numbers[order]] = \
                    np.arange(len(image_numbers)) + 1
                m.reorder_image_measurements(new_image_numbers)
        m.flush()

        if self.volumetric():
            unsupported = [module.module_name for module in self.__modules if not module.volumetric()]

            if len(unsupported) > 0:
                self.report_prepare_run_error(
                    None,
                    "Cannot run pipeline. "
                    "The pipeline is configured to process data as 3D. "
                    "The pipeline contains modules which do not support 3D processing:"
                    "\n\n{}".format(", ".join(unsupported))
                )

                return False


        return True

    def post_run(self, *args):
        """Do "post_run" on each module to perform aggregation tasks

        New interface:
        workspace - workspace with pipeline, module and measurements valid

        Old interface:

        measurements - the measurements for the run
        image_set_list - the image set list for the run
        frame - the topmost frame window or None if no GUI
        """
        from cellprofiler.module import Module
        if len(args) == 3:
            measurements, image_set_list, frame = args
            workspace = cpw.Workspace(self,
                                      module,
                                      None,
                                      None,
                                      measurements,
                                      image_set_list,
                                      frame)
        else:
            workspace = args[0]
        for module in self.modules():
            workspace.refresh()
            try:
                module.post_run(workspace)
            except Exception, instance:
                logging.error(
                        "Failed to complete post_run processing for module %s." %
                        module.module_name, exc_info=True)
                event = PostRunExceptionEvent(instance, module, sys.exc_info()[2])
                self.notify_listeners(event)
                if event.cancel_run:
                    return "Failure"
            if module.show_window and \
                            module.__class__.display_post_run != Module.display_post_run:
                try:
                    workspace.post_run_display(module)
                except Exception, instance:
                    # Warn about display failure but keep going.
                    logging.warn(
                            "Caught exception during post_run_display for module %s." %
                            module.module_name, exc_info=True)
        workspace.measurements.add_experiment_measurement(
                M_MODIFICATION_TIMESTAMP, datetime.datetime.now().isoformat())

        return "Complete"

    def prepare_to_create_batch(self, workspace, fn_alter_path):
        '''Prepare to create a batch file

        This function is called when CellProfiler is about to create a
        file for batch processing. It will pickle the image set list's
        "legacy_fields" dictionary. This callback lets a module prepare for
        saving.

        workspace - the workspace to be saved
        fn_alter_path - this is a function that takes a pathname on the local
                        host and returns a pathname on the remote host. It
                        handles issues such as replacing backslashes and
                        mapping mountpoints. It should be called for every
                        pathname stored in the settings or legacy fields.
        '''
        assert workspace.pipeline == self
        for module in self.modules():
            try:
                workspace.set_module(module)
                module.prepare_to_create_batch(workspace,
                                               fn_alter_path)
            except Exception, instance:
                logger.error("Failed to collect batch information for module %s",
                             module.module_name, exc_info=True)
                event = RunExceptionEvent(instance, module, sys.exc_info()[2])
                self.notify_listeners(event)
                if event.cancel_run:
                    return

    def get_groupings(self, workspace):
        '''Return the image groupings of the image sets in an image set list

        returns a tuple of key_names and group_list:
        key_names - the names of the keys that identify the groupings
        group_list - a sequence composed of two-tuples.
                     the first element of the tuple has the values for
                     the key_names for this group.
                     the second element of the tuple is a sequence of
                     image numbers comprising the image sets of the group
        For instance, an experiment might have key_names of 'Metadata_Row'
        and 'Metadata_Column' and a group_list of:
        [ (('A','01'), [0,96,192]),
          (('A','02'), [1,97,193]),... ]
        '''
        groupings = None
        grouping_module = None
        for module in self.modules():
            workspace.set_module(module)
            new_groupings = module.get_groupings(workspace)
            if new_groupings is None:
                continue
            if groupings is None:
                groupings = new_groupings
                grouping_module = module
            else:
                raise ValueError("The pipeline has two grouping modules: # %d "
                                 "(%s) and # %d (%s)" %
                                 (grouping_module.module_num,
                                  grouping_module.module_name,
                                  module.module_num,
                                  module.module_name))
        if groupings is None:
            return (), (((), workspace.measurements.get_image_numbers()),)
        return groupings

    def get_undefined_metadata_tags(self, pattern):
        """Find metadata tags not defined within the current measurements

        pattern - a regexp-like pattern that specifies how to insert
                  metadata into a string. Each token has the form:
                  "\(?<METADATA_TAG>\)" (matlab-style) or
                  "\g<METADATA_TAG>" (Python-style)
        """
        columns = self.get_measurement_columns()
        current_metadata = []
        for column in columns:
            object_name, feature, coltype = column[:3]
            if object_name == cpmeas.IMAGE and feature.startswith(cpmeas.C_METADATA):
                current_metadata.append(feature[(len(cpmeas.C_METADATA) + 1):])

        m = re.findall('\\(\\?[<](.+?)[>]\\)', pattern)
        if not m:
            m = re.findall('\\\\g[<](.+?)[>]', pattern)
        if m:
            m = filter((lambda x: not any(
                    [x.startswith(y) for y in cpmeas.C_SERIES, cpmeas.C_FRAME])), m)
            undefined_tags = list(set(m).difference(current_metadata))
            return undefined_tags
        else:
            return []

    def prepare_group(self, workspace, grouping, image_numbers):
        '''Prepare to start processing a new group

        workspace - the workspace containing the measurements and image set list
        grouping - a dictionary giving the keys and values for the group

        returns true if the group should be run
        '''
        for module in self.modules():
            try:
                module.prepare_group(workspace, grouping, image_numbers)
            except Exception, instance:
                logger.error("Failed to prepare group in module %s",
                             module.module_name, exc_info=True)
                event = RunExceptionEvent(instance, module, sys.exc_info()[2])
                self.notify_listeners(event)
                if event.cancel_run:
                    return False
        return True

    def post_group(self, workspace, grouping):
        '''Do post-processing after a group completes

        workspace - the last workspace run
        '''
        from cellprofiler.module import Module
        for module in self.modules():
            try:
                module.post_group(workspace, grouping)
            except Exception, instance:
                logging.error(
                        "Failed during post-group processing for module %s" %
                        module.module_name, exc_info=True)
                event = RunExceptionEvent(instance, module, sys.exc_info()[2])
                self.notify_listeners(event)
                if event.cancel_run:
                    return False
            if module.show_window and \
                            module.__class__.display_post_group != Module.display_post_group:
                try:
                    workspace.post_group_display(module)
                except:
                    logging.warn(
                            "Failed during post group display for module %s" %
                            module.module_name, exc_info=True)
        return True

    def has_create_batch_module(self):
        for module in self.modules():
            if module.is_create_batch_module():
                return True
        return False

    def in_batch_mode(self):
        '''Return True if the pipeline is in batch mode'''
        for module in self.modules():
            batch_mode = module.in_batch_mode()
            if batch_mode is not None:
                return batch_mode

    def turn_off_batch_mode(self):
        '''Reset the pipeline to an editable state if batch mode is on

        A module is allowed to create hidden information that it uses
        to turn batch mode on or to save state to be used in batch mode.
        This call signals that the pipeline has been opened for editing,
        even if it is a batch pipeline; all modules should be restored
        to a state that's appropriate for creating a batch file, not
        for running a batch file.
        '''
        for module in self.modules():
            module.turn_off_batch_mode()

    def get_test_mode(self):
        return self.__test_mode

    def set_test_mode(self, val):
        self.__test_mode = val

    test_mode = property(get_test_mode, set_test_mode)

    def clear(self):
        self.start_undoable_action()
        try:
            while len(self.__modules) > 0:
                self.remove_module(self.__modules[-1].module_num)
            self.notify_listeners(PipelineClearedEvent())
            self.init_modules()
        finally:
            self.stop_undoable_action()

    def init_modules(self):
        '''Initialize the module list

        Initialize the modules list to contain the four file modules.
        '''
        from cellprofiler.modules.images import Images
        from cellprofiler.modules.metadata import Metadata
        from cellprofiler.modules.namesandtypes import NamesAndTypes
        from cellprofiler.modules.groups import Groups
        for i, module in enumerate(
                (Images(), Metadata(), NamesAndTypes(), Groups())):
            module.set_module_num(i + 1)
            module.show_window = cpprefs.get_headless()
            self.add_module(module)

    def move_module(self, module_num, direction):
        """Move module # ModuleNum either DIRECTION_UP or DIRECTION_DOWN in the list

        Move the 1-indexed module either up one or down one in the list, displacing
        the other modules in the list
        """
        idx = module_num - 1
        if direction == DIRECTION_DOWN:
            if module_num >= len(self.__modules):
                raise ValueError(
                        '%(module_num)d is at or after the last module in the pipeline and can''t move down' % (
                            locals()))
            module = self.__modules[idx]
            new_module_num = module_num + 1
            module.set_module_num(module_num + 1)
            next_module = self.__modules[idx + 1]
            next_module.set_module_num(module_num)
            self.__modules[idx] = next_module
            self.__modules[idx + 1] = module
            next_settings = self.__settings[idx + 1]
            self.__settings[idx + 1] = self.__settings[idx]
            self.__settings[idx] = next_settings
        elif direction == DIRECTION_UP:
            if module_num <= 1:
                raise ValueError('The module is at the top of the pipeline and can''t move up')
            module = self.__modules[idx]
            prev_module = self.__modules[idx - 1]
            new_module_num = prev_module.module_num
            module.module_num = new_module_num
            prev_module.module_num = module_num
            self.__modules[idx] = self.__modules[idx - 1]
            self.__modules[idx - 1] = module
            prev_settings = self.__settings[idx - 1]
            self.__settings[idx - 1] = self.__settings[idx]
            self.__settings[idx] = prev_settings
        else:
            raise ValueError('Unknown direction: %s' % direction)
        self.notify_listeners(ModuleMovedPipelineEvent(new_module_num, direction, False))

        def undo():
            self.move_module(module.module_num,
                             DIRECTION_DOWN if direction == DIRECTION_UP
                             else DIRECTION_UP)

        message = "Move %s %s" % (module.module_name, direction)
        self.__undo_stack.append((undo, message))

    def enable_module(self, module):
        '''Enable a module = make it executable'''
        if module.enabled:
            logger.warn(
                    "Asked to enable module %s, but it was already enabled" %
                    module.module_name)
            return
        module.enabled = True
        self.notify_listeners(ModuleEnabledEvent(module))

        def undo():
            self.disable_module(module)

        message = "Enable %s" % module.module_name
        self.__undo_stack.append((undo, message))

    def disable_module(self, module):
        '''Disable a module = prevent it from being executed'''
        if not module.enabled:
            logger.warn(
                    "Asked to disable module %s, but it was already disabled" %
                    module.module_name)
        module.enabled = False
        self.notify_listeners(ModuleDisabledEvent(module))

        def undo():
            self.enable_module(module)

        message = "Disable %s" % module.module_name
        self.__undo_stack.append((undo, message))

    def show_module_window(self, module, state=True):
        '''Set the module's show_window state

        module - module to show or hide

        state - True to show, False to hide
        '''
        if state != module.show_window:
            module.show_window = state
            self.notify_listeners(ModuleShowWindowEvent(module))

            def undo():
                self.show_module_window(module, not state)

            message = "%s %s window" % (
                ("Show" if state else "Hide"), module.module_name)
            self.__undo_stack.append((undo, message))

    def add_urls(self, urls, add_undo=True):
        '''Add URLs to the file list

        urls - a collection of URLs
        add_undo - True to add the undo operation of this to the undo stack
        '''
        real_list = []
        urls = sorted(urls)
        start = 0
        uid = uuid.uuid4()
        n = len(urls)
        for i, url in enumerate(urls):
            if i % 100 == 0:
                path = urlparse.urlparse(url).path
                if "/" in path:
                    filename = path.rsplit("/", 1)[1]
                else:
                    filename = path
                filename = urllib.url2pathname(filename)
                cpprefs.report_progress(
                        uid, float(i) / n,
                             u"Adding %s" % filename)
            pos = bisect.bisect_left(self.__file_list, url, start)
            if (pos == len(self.file_list) or
                        self.__file_list[pos] != url):
                real_list.append(url)
                self.__file_list.insert(pos, url)
            start = pos
        if n > 0:
            cpprefs.report_progress(uid, 1, "Done")
        # Invalidate caches
        self.__file_list_generation = uid
        self.__filtered_file_list_images_settings = None
        self.__image_plane_details_metadata_settings = None
        self.notify_listeners(URLsAddedEvent(real_list))
        if add_undo:
            def undo():
                self.remove_urls(real_list)

            self.__undo_stack.append((undo, "Add images"))

    def remove_urls(self, urls):
        real_list = []
        urls = sorted(urls)
        start = 0
        for url in urls:
            pos = bisect.bisect_left(self.__file_list, url, start)
            if (pos != len(self.__file_list) and
                        self.__file_list[pos] == url):
                real_list.append(url)
                del self.__file_list[pos]
            start = pos
        if len(real_list):
            self.__filtered_file_list_images_settings = None
            self.__image_plane_details_metadata_settings = None
            self.__image_plane_details = []
            self.__file_list_generation = uuid.uuid4()
            self.notify_listeners(URLsRemovedEvent(real_list))

            def undo():
                self.add_urls(real_list, False)

            self.__undo_stack.append((undo, "Remove images"))

    def clear_urls(self, add_undo=True):
        '''Remove all URLs from the pipeline'''
        old_urls = list(self.__file_list)
        self.__file_list = []
        if len(old_urls):
            self.__filtered_file_list_images_settings = None
            self.__image_plane_details_metadata_settings = None
            self.__image_plane_details = []
            self.notify_listeners(URLsRemovedEvent(old_urls))
            if add_undo:
                def undo():
                    self.add_urls(old_urls, False)

                self.__undo_stack.append((undo, "Remove images"))

    def load_file_list(self, workspace):
        '''Load the pipeline's file_list from the workspace file list

        '''
        file_list = workspace.file_list
        if self.__file_list_generation == file_list.generation:
            return
        try:
            urls = file_list.get_filelist()
        except Exception, instance:
            logger.error("Failed to get file list from workspace", exc_info=True)
            x = IPDLoadExceptionEvent("Failed to get file list from workspace")
            self.notify_listeners(x)
            if x.cancel_run:
                raise instance
        self.start_undoable_action()
        self.clear_urls()
        self.add_urls(urls)
        self.stop_undoable_action(name="Load file list")
        self.__filtered_image_plane_details_images_settings = tuple()
        self.__filtered_image_plane_details_metadata_settings = tuple()
        self.__image_plane_details_generation = file_list.generation

    def read_file_list(self, path_or_fd, add_undo=True):
        '''Read a file of one file or URL per line into the file list

        path - a path to a file or a URL
        '''
        if isinstance(path_or_fd, basestring):
            from cellprofiler.modules.loadimages import \
                url2pathname, FILE_SCHEME, PASSTHROUGH_SCHEMES
            pathname = path_or_fd
            if pathname.startswith(FILE_SCHEME):
                pathname = url2pathname(pathname)
                with open(pathname, "r") as fd:
                    self.read_file_list(fd, add_undo=add_undo)
            elif any(pathname.startswith(_) for _ in PASSTHROUGH_SCHEMES):
                import urllib2
                try:
                    fd = urllib2.urlopen(pathname)
                    self.read_file_list(fd, add_undo=add_undo)
                finally:
                    fd.close()
            else:
                with open(pathname, "r") as fd:
                    self.read_file_list(fd, add_undo=add_undo)
            return
        self.add_pathnames_to_file_list(
                map((lambda x: x.strip()),
                    filter((lambda x: len(x) > 0), path_or_fd)),
                add_undo=add_undo)

    def add_pathnames_to_file_list(self, pathnames, add_undo=True):
        '''Add a sequence of paths or URLs to the file list'''
        from cellprofiler.modules.loadimages import pathname2url
        urls = []
        for pathname in pathnames:
            if len(pathname) == 0:
                continue
            if (pathname.startswith("http:") or
                    pathname.startswith("https:") or
                    pathname.startswith("ftp:") or
                    pathname.startswith("omero:") or
                    pathname.startswith("file:")):
                urls.append(pathname)
            else:
                urls.append(pathname2url(pathname))
        self.add_urls(urls, add_undo=add_undo)

    def get_module_state(self, module_name_or_module):
        '''Return an object representing the state of the named module

        module_name - the name of the module

        returns an object that represents the state of the first instance
        of the named module or None if not in pipeline
        '''
        if isinstance(module_name_or_module, basestring):
            modules = [module for module in self.modules()
                       if module.module_name == module_name_or_module]
            if len(modules) == 0:
                return None
            module = modules[0]
        else:
            module = module_name_or_module
        return tuple([s.unicode_value for s in module.settings()])

    def __prepare_run_module(self, module_name, workspace):
        '''Execute "prepare_run" on the first instance of the named module'''
        modules = [module for module in self.modules()
                   if module.module_name == module_name]
        if len(modules) == 0:
            return False
        return modules[0].prepare_run(workspace)

    def has_cached_filtered_file_list(self):
        '''True if the filtered file list is currently cached'''
        images_settings = self.get_module_state("Images")
        if images_settings is None:
            return False
        return self.__filtered_file_list_images_settings == images_settings

    def get_filtered_file_list(self, workspace):
        '''Return the file list as filtered by the Images module

        '''
        if not self.has_cached_filtered_file_list():
            self.__image_plane_details_metadata_settings = None
            self.__prepare_run_module("Images", workspace)
        return self.__filtered_file_list

    def has_cached_image_plane_details(self):
        '''Return True if we have up-to-date image plane details cached'''
        if not self.has_cached_filtered_file_list():
            return False
        metadata_settings = self.get_module_state("Metadata")
        if metadata_settings is None:
            return False
        return self.__image_plane_details_metadata_settings == metadata_settings

    def get_image_plane_details(self, workspace):
        '''Return the image plane details with metadata computed

        '''
        if self.has_cached_image_plane_details():
            return self.__image_plane_details
        self.__available_metadata_keys = set()
        self.__prepare_run_module("Metadata", workspace)
        return self.__image_plane_details

    def get_available_metadata_keys(self):
        '''Get the metadata keys from extraction and their types

        Returns a dictionary of metadata key to measurements COLTYPE
        '''
        modules = [module for module in self.modules()
                   if module.module_name == "Metadata"]
        if len(modules) == 0:
            return {}
        module = modules[0]
        return module.get_data_type(module.get_metadata_keys())

    def use_case_insensitive_metadata_matching(self, key):
        '''Return TRUE if metadata should be matched without regard to case'''
        modules = [module for module in self.modules()
                   if module.module_name == "Metadata"]
        if len(modules) == 0:
            return False
        return modules[0].wants_case_insensitive_matching(key)

    def set_filtered_file_list(self, file_list, module):
        '''The Images module calls this to report its list of filtered files'''
        self.__filtered_file_list = file_list
        self.__filtered_file_list_images_settings = \
            self.get_module_state(module)

    def set_image_plane_details(self, ipds, available_metadata_keys, module):
        '''The Metadata module calls this to report on the extracted IPDs

        ipds - the image plane details to be fed into NamesAndTypes
        available_metadata_keys - the metadata keys collected during IPD
                                  metadata extraction.
        module - the metadata module that made them (so we can cache based
                 on the module's settings.
        '''
        self.__image_plane_details = ipds
        self.__available_metadata_keys = available_metadata_keys
        self.__image_plane_details_metadata_settings = \
            self.get_module_state(module)

    class ImageSetChannelDescriptor(object):
        '''This class represents the metadata for one image set channel

        An image set has a collection of channels which are either planar
        images or objects. The ImageSetChannelDescriptor describes one
        of these:

        The channel's name

        The channel's type - grayscale image / color image / objects / mask
        or illumination function
        '''
        # Channel types
        CT_GRAYSCALE = "Grayscale"
        CT_COLOR = "Color"
        CT_MASK = "Mask"
        CT_OBJECTS = "Objects"
        CT_FUNCTION = "Function"

        def __init__(self, name, channel_type):
            self.name = name
            self.channel_type = channel_type

    LEGACY_LOAD_MODULES = ["LoadImages", "LoadData", "LoadSingleImage"]

    def has_legacy_loaders(self):
        return any(m.module_name in self.LEGACY_LOAD_MODULES for m in self.modules())

    def needs_default_image_folder(self):
        '''Return True if this pipeline makes use of the default image folder'''
        for module in self.modules():
            if module.needs_default_image_folder(self):
                return True
        return False

    def get_image_sets(self, workspace, end_module=None):
        '''Return the pipeline's image sets

        end_module - if present, build the image sets by scanning up to this module

        Return a three-tuple.

        The first element of the two-tuple is a list of
        ImageSetChannelDescriptors - the ordering in the list defines the
        order of ipds in the rows of each image set

        The second element of the two-tuple is a collection of metadata
        key names appropriate for display.

        The last element is a dictionary of lists where the dictionary keys
        are the metadata values for the image set (or image numbers if
        organized by number) and the values are lists of the IPDs for that
        image set.

        This function leaves out any image set that is ill-defined.
        '''

        pipeline = self.copy(save_image_plane_details=False)
        if end_module is not None:
            end_module_idx = self.modules().index(end_module)
            end_module = pipeline.modules()[end_module_idx]
        temp_measurements = cpmeas.Measurements(mode="memory")
        new_workspace = None
        try:
            new_workspace = cpw.Workspace(
                    pipeline, None, None, None,
                    temp_measurements, cpi.ImageSetList())
            new_workspace.set_file_list(workspace.file_list)
            pipeline.prepare_run(new_workspace, end_module)

            iscds = temp_measurements.get_channel_descriptors()
            metadata_key_names = temp_measurements.get_metadata_tags()

            d = {}
            all_image_numbers = temp_measurements.get_image_numbers()
            if len(all_image_numbers) == 0:
                return iscds, metadata_key_names, {}
            metadata_columns = [
                temp_measurements.get_measurement(
                        cpmeas.IMAGE, feature, all_image_numbers)
                for feature in metadata_key_names]

            def get_column(image_category, objects_category, iscd):
                if iscd.channel_type == iscd.CT_OBJECTS:
                    category = objects_category
                else:
                    category = image_category
                feature_name = "_".join((category, iscd.name))
                if feature_name in temp_measurements.get_feature_names(cpmeas.IMAGE):
                    return temp_measurements.get_measurement(
                            cpmeas.IMAGE, feature_name, all_image_numbers)
                else:
                    return [None] * len(all_image_numbers)

            url_columns = [get_column(cpmeas.C_URL, cpmeas.C_OBJECTS_URL, iscd)
                           for iscd in iscds]
            series_columns = [get_column(cpmeas.C_SERIES, cpmeas.C_OBJECTS_SERIES, iscd)
                              for iscd in iscds]
            index_columns = [get_column(cpmeas.C_FRAME, cpmeas.C_OBJECTS_FRAME, iscd)
                             for iscd in iscds]
            channel_columns = [get_column(cpmeas.C_CHANNEL, cpmeas.C_OBJECTS_CHANNEL, iscd)
                               for iscd in iscds]
            d = {}
            for idx in range(len(all_image_numbers)):
                key = tuple([mc[idx] for mc in metadata_columns])
                value = [
                    pipeline.find_image_plane_details(
                            ImagePlaneDetails(u[idx], s[idx], i[idx], c[idx]))
                    for u, s, i, c in zip(url_columns, series_columns,
                                          index_columns, channel_columns)]
                d[key] = value
            return iscds, metadata_key_names, d
        finally:
            if new_workspace is not None:
                new_workspace.set_file_list(None)
            temp_measurements.close()

    def has_undo(self):
        '''True if an undo action can be performed'''
        return len(self.__undo_stack)

    def undo(self):
        '''Undo the last action'''
        if len(self.__undo_stack):
            action = self.__undo_stack.pop()[0]
            real_undo_stack = self.__undo_stack
            self.__undo_stack = []
            try:
                action()
            finally:
                self.__undo_stack = real_undo_stack

    def undo_action(self):
        '''A user-interpretable string telling the user what the action was'''
        if len(self.__undo_stack) == 0:
            return "Nothing to undo"
        return self.__undo_stack[-1][1]

    def undoable_action(self, name="Composite edit"):
        '''Return an object that starts and stops an undoable action

        Use this with the "with" statement to create a scope where all
        actions are collected for undo:

        with pipeline.undoable_action():
            pipeline.add_module(module1)
            pipeline.add_module(module2)
        '''

        class UndoableAction:
            def __init__(self, pipeline, name):
                self.pipeline = pipeline
                self.name = name

            def __enter__(self):
                self.pipeline.start_undoable_action()

            def __exit__(self, ttype, value, traceback):
                self.pipeline.stop_undoable_action(name)

        return UndoableAction(self, name)

    def start_undoable_action(self):
        '''Start editing the pipeline

        This marks a start of a series of actions which will be undone
        all at once.
        '''
        self.__undo_start = len(self.__undo_stack)

    def stop_undoable_action(self, name="Composite edit"):
        '''Stop editing the pipeline, combining many actions into one'''
        if len(self.__undo_stack) > self.__undo_start + 1:
            # Only combine if two or more edits
            actions = self.__undo_stack[self.__undo_start:]
            del self.__undo_stack[self.__undo_start:]

            def undo():
                for action, message in reversed(actions):
                    action()

            self.__undo_stack.append((undo, name))

    def modules(self, exclude_disabled=True):
        '''Return the list of modules

        exclude_disabled - only return enabled modules if True (default)
        '''
        if exclude_disabled:
            return [m for m in self.__modules if m.enabled]
        else:
            return self.__modules

    def module(self, module_num):
        module = self.__modules[module_num - 1]
        assert module.module_num == module_num, 'Misnumbered module. Expected %d, got %d' % (
            module_num, module.module_num)
        return module

    @staticmethod
    def capture_module_settings(module):
        '''Capture a module's settings for later undo

        module - module in question

        Return a list of setting values that can be fed into the module's
        set_settings_from_values method to reconstruct the module in its original form.
        '''
        return [setting.get_unicode_value() for setting in module.settings()]

    def add_module(self, new_module):
        """Insert a module into the pipeline with the given module #

        Insert a module into the pipeline with the given module #.
        'file_name' - the path to the file containing the variables for the module.
        ModuleNum - the one-based index for the placement of the module in the pipeline
        """
        is_image_set_modification = new_module.is_load_module()
        module_num = new_module.module_num
        idx = module_num - 1
        self.__modules = self.__modules[:idx] + [new_module] + self.__modules[idx:]
        for module, mn in zip(self.__modules[idx + 1:], range(module_num + 1, len(self.__modules) + 1)):
            module.module_num = mn
        self.notify_listeners(ModuleAddedPipelineEvent(
                module_num, is_image_set_modification=is_image_set_modification))
        self.__settings.insert(idx, self.capture_module_settings(new_module))

        def undo():
            self.remove_module(new_module.module_num)

        self.__undo_stack.append((undo,
                                  "Add %s module" % new_module.module_name))

    def remove_module(self, module_num):
        """Remove a module from the pipeline

        Remove a module from the pipeline
        ModuleNum - the one-based index of the module
        """
        idx = module_num - 1
        removed_module = self.__modules[idx]
        is_image_set_modification = removed_module.is_load_module()
        self.__modules = self.__modules[:idx] + self.__modules[idx + 1:]
        for module in self.__modules[idx:]:
            module.module_num = module.module_num - 1
        self.notify_listeners(ModuleRemovedPipelineEvent(
                module_num, is_image_set_modification=is_image_set_modification))
        del self.__settings[idx]

        def undo():
            self.add_module(removed_module)

        self.__undo_stack.append((undo, "Remove %s module" %
                                  removed_module.module_name))

    def edit_module(self, module_num, is_image_set_modification):
        """Notify listeners of a module edit

        """
        idx = module_num - 1
        old_settings = self.__settings[idx]
        module = self.__modules[idx]
        new_settings = self.capture_module_settings(module)
        self.notify_listeners(ModuleEditedPipelineEvent(
                module_num, is_image_set_modification=is_image_set_modification))
        self.__settings[idx] = new_settings
        variable_revision_number = module.variable_revision_number
        module_name = module.module_name

        def undo():
            module = self.__modules[idx]
            module.set_settings_from_values(old_settings,
                                            variable_revision_number,
                                            module_name, False)
            self.notify_listeners(ModuleEditedPipelineEvent(module_num))
            self.__settings[idx] = old_settings

        self.__undo_stack.append((undo, "Edited %s" % module_name))

    @property
    def file_list(self):
        return self.__file_list

    @property
    def image_plane_details(self):
        return self.__image_plane_details

    def on_walk_completed(self):
        self.notify_listeners(FileWalkEndedEvent())

    def wp_add_files(self, dirpath, directories, filenames):
        ipds = []
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            url = "file:" + urllib.pathname2url(path)
            ipd = ImagePlaneDetails(url, None, None, None)
            ipds.append(ipd)
        self.add_image_plane_details(ipds)

    def wp_add_image_metadata(self, path, metadata):
        self.add_image_metadata("file:" + urllib.pathname2url(path), metadata)

    def add_image_metadata(self, url, metadata, ipd=None):
        if metadata.image_count == 1:
            m = {}
            pixels = metadata.image(0).Pixels
            m[ImagePlaneDetails.MD_SIZE_C] = str(pixels.SizeC)
            m[ImagePlaneDetails.MD_SIZE_Z] = str(pixels.SizeZ)
            m[ImagePlaneDetails.MD_SIZE_T] = str(pixels.SizeT)

            if pixels.SizeC == 1:
                #
                # Monochrome image
                #
                m[ImagePlaneDetails.MD_COLOR_FORMAT] = \
                    ImagePlaneDetails.MD_MONOCHROME
                channel = pixels.Channel(0)
                channel_name = channel.Name
                if channel_name is not None:
                    m[ImagePlaneDetails.MD_CHANNEL_NAME] = channel_name
            elif pixels.channel_count == 1:
                #
                # Oh contradictions! It's interleaved, really RGB or RGBA
                #
                m[ImagePlaneDetails.MD_COLOR_FORMAT] = \
                    ImagePlaneDetails.MD_RGB
            else:
                m[ImagePlaneDetails.MD_COLOR_FORMAT] = \
                    ImagePlaneDetails.MD_PLANAR
            exemplar = ImagePlaneDetails(url, None, None, None)
            if ipd is None:
                ipd = self.find_image_plane_details(exemplar)
            if ipd is not None:
                ipd.metadata.update(m)
                self.notify_listeners(ImagePlaneDetailsMetadataEvent(ipd))

        #
        # If there are planes, we create image plane descriptors for them
        #
        n_series = metadata.image_count
        to_add = []
        for series in range(n_series):
            pixels = metadata.image(series).Pixels
            if pixels.plane_count > 0:
                for index in range(pixels.plane_count):
                    addr = (series, index, None)
                    m = {}
                    plane = pixels.Plane(index)
                    c = plane.TheC
                    m[ImagePlaneDetails.MD_C] = plane.TheC
                    m[ImagePlaneDetails.MD_T] = plane.TheT
                    m[ImagePlaneDetails.MD_Z] = plane.TheZ
                    if pixels.channel_count > c:
                        channel = pixels.Channel(c)
                        channel_name = channel.Name
                        if channel_name is not None:
                            m[ImagePlaneDetails.MD_CHANNEL_NAME] = channel_name
                        if channel.SamplesPerPixel == 1:
                            m[ImagePlaneDetails.MD_COLOR_FORMAT] = \
                                ImagePlaneDetails.MD_MONOCHROME
                        else:
                            m[ImagePlaneDetails.MD_COLOR_FORMAT] = \
                                ImagePlaneDetails.MD_RGB
                    exemplar = ImagePlaneDetails(url, series, index, None)
                    ipd = self.find_image_plane_details(exemplar)
                    if ipd is None:
                        exemplar.metadata.update(m)
                        to_add.append(exemplar)
                    else:
                        ipd.metadata.update(m)
                        self.notify_listeners(ImagePlaneDetailsMetadataEvent(ipd))

            elif pixels.SizeZ > 1 or pixels.SizeT > 1:
                #
                # Movie metadata might not have planes
                #
                if pixels.SizeC == 1:
                    color_format = ImagePlaneDetails.MD_MONOCHROME
                    n_channels = 1
                elif pixels.channel_count == 1:
                    color_format = ImagePlaneDetails.MD_RGB
                    n_channels = 1
                else:
                    color_format = ImagePlaneDetails.MD_MONOCHROME
                    n_channels = pixels.SizeC
                n = 1
                dims = []
                for d in pixels.DimensionOrder[2:]:
                    if d == 'C':
                        dim = n_channels
                        c_idx = len(dims)
                    elif d == 'Z':
                        dim = pixels.SizeZ
                        z_idx = len(dims)
                    elif d == 'T':
                        dim = pixels.SizeT
                        t_idx = len(dims)
                    else:
                        raise ValueError(
                                "Unsupported dimension order for file %s: %s" %
                                (url, pixels.DimensionOrder))
                    dims.append(dim)
                index_order = np.mgrid[0:dims[0], 0:dims[1], 0:dims[2]]
                c_indexes = index_order[c_idx].flatten()
                z_indexes = index_order[z_idx].flatten()
                t_indexes = index_order[t_idx].flatten()
                for index, (c_idx, z_idx, t_idx) in \
                        enumerate(zip(c_indexes, z_indexes, t_indexes)):
                    channel = pixels.Channel(c_idx)
                    exemplar = ImagePlaneDetails(url, series, index, None)
                    metadata = {
                        ImagePlaneDetails.MD_SIZE_C: channel.SamplesPerPixel,
                        ImagePlaneDetails.MD_SIZE_Z: 1,
                        ImagePlaneDetails.MD_SIZE_T: 1,
                        ImagePlaneDetails.MD_COLOR_FORMAT: color_format}
                    channel_name = channel.Name
                    if channel_name is not None and len(channel_name) > 0:
                        metadata[ImagePlaneDetails.MD_CHANNEL_NAME] = \
                            channel_name
                    ipd = self.find_image_plane_details(exemplar)
                    if ipd is None:
                        exemplar.metadata.update(metadata)
                        to_add.append(exemplar)
                    else:
                        ipd.metadata.update(metadata)
                        self.notify_listeners(ImagePlaneDetailsMetadataEvent(ipd))
        if len(to_add) > 0:
            self.add_image_plane_details(to_add, False)

    def test_valid(self):
        """Throw a ValidationError if the pipeline isn't valid

        """
        for module in self.modules():
            module.test_valid(self)

    def notify_listeners(self, event):
        """Notify listeners of an event that happened to this pipeline

        """
        for listener in self.__listeners:
            listener(self, event)

    def add_listener(self, listener):
        self.__listeners.append(listener)

    def remove_listener(self, listener):
        self.__listeners.remove(listener)

    class PipelineListener(object):
        '''A class to wrap add/remove listener for use with "with"

        Usage:
        def my_listener(pipeline, event):
            .....
        with pipeline.PipelineListener(pipeline, my_listener):
            # listener has been added
            .....
        # listener has been removed
        '''

        def __init__(self, pipeline, listener):
            self.pipeline = pipeline
            self.listener = listener

        def __enter__(self):
            self.pipeline.add_listener(self.listener)
            return self

        def __exit__(self, exc_type, exc_value, tb):
            self.pipeline.remove_listener(self.listener)

    def report_prepare_run_error(self, module, message):
        '''Report an error during prepare_run that prevents image set construction

        module - the module that failed

        message - the message for the user

        Report errors due to misconfiguration, such as no files found.
        '''
        event = PrepareRunErrorEvent(module, message)
        self.notify_listeners(event)

    def is_image_from_file(self, image_name):
        """Return True if any module in the pipeline claims to be
        loading this image name from a file."""
        for module in self.modules():
            if module.is_image_from_file(image_name):
                return True
        return False

    def get_measurement_columns(self, terminating_module=None):
        '''Return a sequence describing the measurement columns for this pipeline

        This call returns one element per image or object measurement
        made by each module during image set analysis. The element itself
        is a 3-tuple:
        first entry: either one of the predefined measurement categories,
                     {Image", "Experiment" or "Neighbors" or the name of one
                     of the objects.
        second entry: the measurement name (as would be used in a call
                      to add_measurement)
        third entry: the column data type (for instance, "varchar(255)" or
                     "float")
        fourth entry (optional): attribute dictionary. This tags
                     the column with attributes such as MCA_AVAILABLE_POST_GROUP
                     (column values are only added in post_group).
        '''
        hash = self.settings_hash()
        if hash != self.__measurement_column_hash:
            self.__measurement_columns = {}
            self.__measurement_column_hash = hash

        terminating_module_num = (sys.maxint
                                  if terminating_module is None
                                  else terminating_module.module_num)
        if self.__measurement_columns.has_key(terminating_module_num):
            return self.__measurement_columns[terminating_module_num]
        columns = [
            (cpmeas.EXPERIMENT, M_PIPELINE, cpmeas.COLTYPE_LONGBLOB),
            (cpmeas.EXPERIMENT, M_VERSION, cpmeas.COLTYPE_VARCHAR),
            (cpmeas.EXPERIMENT, M_TIMESTAMP, cpmeas.COLTYPE_VARCHAR),
            (cpmeas.EXPERIMENT, M_MODIFICATION_TIMESTAMP,
             cpmeas.COLTYPE_VARCHAR, {cpmeas.MCA_AVAILABLE_POST_RUN: True}),
            (cpmeas.IMAGE, GROUP_NUMBER, cpmeas.COLTYPE_INTEGER),
            (cpmeas.IMAGE, GROUP_INDEX, cpmeas.COLTYPE_INTEGER)]
        should_write_columns = True
        for module in self.modules():
            if (terminating_module is not None and
                        terminating_module_num <= module.module_num):
                break
            columns += module.get_measurement_columns(self)
            if module.should_stop_writing_measurements():
                should_write_columns = False
            if should_write_columns:
                module_error_measurement = 'ModuleError_%02d%s' % (module.module_num, module.module_name)
                execution_time_measurement = 'ExecutionTime_%02d%s' % (module.module_num, module.module_name)
                columns += [(cpmeas.IMAGE, module_error_measurement, cpmeas.COLTYPE_INTEGER),
                            (cpmeas.IMAGE, execution_time_measurement, cpmeas.COLTYPE_INTEGER)]
        self.__measurement_columns[terminating_module_num] = columns
        return columns

    def get_object_relationships(self):
        '''Return a sequence of five-tuples describing all object relationships

        This returns all relationship categories produced by modules via
        Measurements.add_relate_measurement. The format is:
        [(<module-number>, # the module number of the module that wrote it
          <relationship-name>, # the descriptive name of the relationship
          <object-name-1>, # the subject of the relationship
          <object-name-2>, # the object of the relationship
          <when>)] # cpmeas.MCA_AVAILABLE_{EVERY_CYCLE, POST_GROUP}
        '''
        result = []
        for module in self.modules():
            result += [
                (module.module_num, i1, o1, i2, o2)
                for i1, o1, i2, o2 in module.get_object_relationships(self)]
        return result

    def get_provider_dictionary(self, groupname, module=None):
        '''Get a dictionary of all providers for a given category

        groupname - the name of the category from cellprofiler.settings:
            IMAGE_GROUP for image providers, OBJECT_GROUP for object providers
            or MEASUREMENTS_GROUP for measurement providers.

        module - the module that will subscribe to the names. If None, all
        providers are listed, if a module, only the providers for that module's
        place in the pipeline are listed.

        returns a dictionary where the key is the name and the value is
        a list of tuples of module and setting where the module provides
        the name and the setting is the setting that controls the name (and
        the setting can be None).

        '''
        target_module = module
        result = {}
        #
        # Walk through the modules to find subscriber and provider settings
        #
        for module in self.modules():
            if (target_module is not None and
                        target_module.module_num <= module.module_num):
                break
            #
            # Find "other_providers" - providers that aren't specified
            # by single settings.
            #
            p = module.other_providers(groupname)
            for name in p:
                if (not result.has_key(name)) or target_module is not None:
                    result[name] = []
                result[name].append((module, None))
            if groupname == cps.MEASUREMENTS_GROUP:
                for c in module.get_measurement_columns(self):
                    object_name, feature_name = c[:2]
                    k = (object_name, feature_name)
                    if (not result.has_key(k)) or target_module is not None:
                        result[k] = []
                    result[k].append((module, None))
            for setting in module.visible_settings():
                if (isinstance(setting, cps.NameProvider) and
                            setting.get_group() == groupname):
                    name = setting.value
                    if name == cps.DO_NOT_USE:
                        continue
                    if not result.has_key(name) or target_module is not None:
                        result[name] = []
                    result[name].append((module, setting))
        return result

    def get_dependency_graph(self):
        '''Create a graph that describes the producers and consumers of objects

        returns a list of Dependency objects. These can be used to create a
        directed graph that describes object and image dependencies.
        '''
        #
        # These dictionaries have the following structure:
        # * top level dictionary key indicates whether it is an object, image
        #   or measurement dependency
        # * second level dictionary key is the name of the object or image or
        #   a tuple of (object_name, feature) for a measurement.
        # * the value of the second-level dictionary is a list of tuples
        #   where the first element of the tuple is the module and the
        #   second is either None or the setting.
        #
        all_groups = (cps.OBJECT_GROUP, cps.IMAGE_GROUP, cps.MEASUREMENTS_GROUP)
        providers = dict([(g, self.get_provider_dictionary(g))
                          for g in all_groups])
        #
        # Now match subscribers against providers.
        #
        result = []
        for module in self.modules():
            for setting in module.visible_settings():
                if isinstance(setting, cps.NameSubscriber):
                    group = setting.get_group()
                    name = setting.value
                    if (providers.has_key(group) and
                            providers[group].has_key(name)):
                        for pmodule, psetting in providers[group][name]:
                            if pmodule.module_num < module.module_num:
                                if group == cps.OBJECT_GROUP:
                                    dependency = ObjectDependency(
                                            pmodule, module, name,
                                            psetting, setting)
                                    result.append(dependency)
                                elif group == cps.IMAGE_GROUP:
                                    dependency = ImageDependency(
                                            pmodule, module, name,
                                            psetting, setting)
                                    result.append(dependency)
                                break
                elif isinstance(setting, cps.Measurement):
                    object_name = setting.get_measurement_object()
                    feature_name = setting.value
                    key = (object_name, feature_name)
                    if providers[cps.MEASUREMENTS_GROUP].has_key(key):
                        for pmodule, psetting in providers[cps.MEASUREMENTS_GROUP][key]:
                            if pmodule.module_num < module.module_num:
                                dependency = MeasurementDependency(
                                        pmodule, module, object_name, feature_name,
                                        psetting, setting)
                                result.append(dependency)
                                break
        return result

    def synthesize_measurement_name(self, module, object, category,
                                    feature, image, scale):
        '''Turn a measurement requested by a Matlab module into a measurement name

        Some Matlab modules specify measurement names as a combination
        of category, feature, image name and scale, but not all measurements
        have associated images or scales. This function attempts to match
        the given parts to the measurements available to the module and
        returns the best guess at a measurement. It throws a value error
        exception if it can't find a match

        module - the module requesting the measurement. Only measurements
                 made prior to this module will be considered.
        object - the object name or "Image"
        category - The module's measurement category (e.g. Intensity or AreaShape)
        feature - a descriptive name for the measurement
        image - the measurement should be made on this image (optional)
        scale - the measurement should be made at this scale
        '''
        measurement_columns = self.get_measurement_columns(module)
        measurements = [x[1] for x in measurement_columns
                        if x[0] == object]
        for measurement in ("_".join((category, feature, image, scale)),
                            "_".join((category, feature, image)),
                            "_".join((category, feature, scale)),
                            "_".join((category, feature))):
            if measurement in measurements:
                return measurement
        raise ValueError("No such measurement in pipeline: " +
                         ("Category = %s" % category) +
                         (", Feature = %s" % feature) +
                         (", Image (optional) = %s" % image) +
                         (", Scale (optional) = %s" % scale))

    def loaders_settings_hash(self):
        '''Return a hash for the settings that control image loading, or None
        for legacy pipelines (which can't be hashed)
        '''

        # legacy pipelines can't be cached, because they can load from the
        # Default Image or Output directories.  We could fix this by including
        # them in the hash we use for naming the cache.
        if self.has_legacy_loaders():
            return None

        assert "Groups" in [m.module_name for m in self.modules()]
        return self.settings_hash(until_module="Groups", as_string=True)


def find_image_plane_details(exemplar, ipds):
    '''Find the ImagePlaneDetails instance matching the exemplar

    The point of this function is to retrieve the ImagePlaneDetails from
    the list provided and, in doing so, get the attached metadata and the
    Java IPD object as well.

    exemplar - an IPD with the URL, series, index and channel filled in

    ipds - an ordered list of ImagePlaneDetails instances

    Returns the match or None if not found
    '''
    pos = bisect.bisect_left(ipds, exemplar)
    if (pos == len(ipds) or
            cmp(ipds[pos], exemplar)):
        return None
    return ipds[pos]


class AbstractPipelineEvent(object):
    """Something that happened to the pipeline and was indicated to the listeners
    """

    def __init__(self,
                 is_pipeline_modification=False,
                 is_image_set_modification=False):
        self.is_pipeline_modification = is_pipeline_modification
        self.is_image_set_modification = is_image_set_modification

    def event_type(self):
        raise NotImplementedError("AbstractPipelineEvent does not implement an event type")


class PipelineLoadedEvent(AbstractPipelineEvent):
    """Indicates that the pipeline has been (re)loaded

    """

    def __init__(self):
        super(PipelineLoadedEvent, self).__init__(
                is_pipeline_modification=True,
                is_image_set_modification=True)

    def event_type(self):
        return "PipelineLoaded"


class PipelineClearedEvent(AbstractPipelineEvent):
    """Indicates that all modules have been removed from the pipeline

    """

    def __init__(self):
        super(PipelineClearedEvent, self).__init__(
                is_pipeline_modification=True,
                is_image_set_modification=True)

    def event_type(self):
        return "PipelineCleared"


DIRECTION_UP = "up"
DIRECTION_DOWN = "down"


class ModuleMovedPipelineEvent(AbstractPipelineEvent):
    """A module moved up or down

    """

    def __init__(self, module_num, direction, is_image_set_modification):
        super(ModuleMovedPipelineEvent, self).__init__(
                is_pipeline_modification=True,
                is_image_set_modification=is_image_set_modification)
        self.module_num = module_num
        self.direction = direction

    def event_type(self):
        return "Module moved"


class ModuleAddedPipelineEvent(AbstractPipelineEvent):
    """A module was added to the pipeline

    """

    def __init__(self, module_num, is_image_set_modification=False):
        super(ModuleAddedPipelineEvent, self).__init__(
                is_pipeline_modification=True,
                is_image_set_modification=is_image_set_modification)
        self.module_num = module_num

    def event_type(self):
        return "Module Added"


class ModuleRemovedPipelineEvent(AbstractPipelineEvent):
    """A module was removed from the pipeline

    """

    def __init__(self, module_num, is_image_set_modification=False):
        super(ModuleRemovedPipelineEvent, self).__init__(
                is_pipeline_modification=True,
                is_image_set_modification=is_image_set_modification)
        self.module_num = module_num

    def event_type(self):
        return "Module deleted"


class ModuleEditedPipelineEvent(AbstractPipelineEvent):
    """A module had its settings changed

    """

    def __init__(self, module_num, is_image_set_modification=False):
        super(ModuleEditedPipelineEvent, self).__init__(
                is_pipeline_modification=True,
                is_image_set_modification=is_image_set_modification)
        self.module_num = module_num

    def event_type(self):
        return "Module edited"


class URLsAddedEvent(AbstractPipelineEvent):
    def __init__(self, urls):
        super(self.__class__, self).__init__()
        self.urls = urls

    def event_type(self):
        return "URLs added to file list"


class URLsRemovedEvent(AbstractPipelineEvent):
    def __init__(self, urls):
        super(self.__class__, self).__init__()
        self.urls = urls

    def event_type(self):
        return "URLs removed from file list"


class FileWalkStartedEvent(AbstractPipelineEvent):
    def event_type(self):
        return "File walk started"


class FileWalkEndedEvent(AbstractPipelineEvent):
    def event_type(self):
        return "File walk ended"


class RunExceptionEvent(AbstractPipelineEvent):
    """An exception was caught during a pipeline run

    Initializer:
    error - exception that was thrown
    module - module that was executing
    tb - traceback at time of exception, e.g from sys.exc_info
    """

    def __init__(self, error, module, tb=None):
        self.error = error
        self.cancel_run = True
        self.skip_thisset = False
        self.module = module
        self.tb = tb

    def event_type(self):
        return "Pipeline run exception"


class PrepareRunExceptionEvent(RunExceptionEvent):
    '''An event indicating an uncaught exception during the prepare_run phase'''

    def event_type(self):
        return "Prepare run exception"


class PostRunExceptionEvent(RunExceptionEvent):
    '''An event indicating an uncaught exception during the post_run phase'''

    def event_type(self):
        return "Post run exception"


class PrepareRunErrorEvent(AbstractPipelineEvent):
    """A user configuration error prevented CP from running the pipeline

    Modules use this class to report conditions that prevent construction
    of the image set list. An example would be if the user misconfigured
    LoadImages or NamesAndTypes and no images were matched.
    """

    def __init__(self, module, message):
        super(self.__class__, self).__init__()
        self.module = module
        self.message = message


class LoadExceptionEvent(AbstractPipelineEvent):
    """An exception was caught during pipeline loading

    """

    def __init__(self, error, module, module_name=None, settings=None):
        self.error = error
        self.cancel_run = True
        self.module = module
        self.module_name = module_name
        self.settings = settings

    def event_type(self):
        return "Pipeline load exception"


class IPDLoadExceptionEvent(AbstractPipelineEvent):
    """An exception was cauaght while trying to load the image plane details

    This event is reported when an exception is thrown while loading
    the image plane details from the workspace's file list.
    """

    def __init__(self, error):
        super(self.__class__, self).__init__()
        self.error = error
        self.cancel_run = True

    def event_type(self):
        return "Image load exception"


class CancelledException(Exception):
    '''Exception issued by the analysis worker indicating cancellation by UI

    This is here in order to solve some import dependency problems
    '''
    pass


class PipelineLoadCancelledException(Exception):
    '''Exception thrown if user cancels pipeline load'''
    pass


class EndRunEvent(AbstractPipelineEvent):
    """A run ended"""

    def event_type(self):
        return "Run ended"


class ModuleEnabledEvent(AbstractPipelineEvent):
    """A module was enabled

    module - the module that was enabled.
    """

    def __init__(self, module):
        """Constructor

        module - the module that was enabled
        """
        super(self.__class__, self).__init__(is_pipeline_modification=True)
        self.module = module

    def event_type(self):
        return "Module enabled"


class ModuleDisabledEvent(AbstractPipelineEvent):
    """A module was disabled

    module - the module that was disabled.
    """

    def __init__(self, module):
        """Constructor

        module - the module that was enabled
        """
        super(self.__class__, self).__init__(is_pipeline_modification=True)
        self.module = module

    def event_type(self):
        return "Module disabled"


class ModuleShowWindowEvent(AbstractPipelineEvent):
    """A module had its "show_window" state changed

    module - the module that had its state changed
    """

    def __init__(self, module):
        super(self.__class__, self).__init__(is_pipeline_modification=True)
        self.module = module

    def event_type(self):
        return "Module show_window changed"


class Dependency(object):
    '''This class documents the dependency of one module on another

    A module is dependent on another if the dependent module requires
    data from the producer module. That data can be objects (label matrices),
    a derived image or measurements.
    '''

    def __init__(self, source_module, destination_module,
                 source_setting=None, destination_setting=None):
        '''Constructor

        source_module - the module that produces the data
        destination_module - the module that uses the data
        source_setting - the module setting that names the item (can be None)
        destination_setting - the module setting in the destination that
        picks the setting
        '''
        self.__source_module = source_module
        self.__destination_module = destination_module
        self.__source_setting = source_setting
        self.__destination_setting = destination_setting

    @property
    def source(self):
        '''The source of the data item'''
        return self.__source_module

    @property
    def source_setting(self):
        '''The setting that names the data item

        This can be None if it's ambiguous.
        '''
        return self.__source_setting

    @property
    def destination(self):
        '''The user of the data item'''
        return self.__destination_module

    @property
    def destination_setting(self):
        '''The setting that picks the data item

        This can be None if it's ambiguous.
        '''
        return self.__destination_setting


class ObjectDependency(Dependency):
    '''A dependency on an object labeling'''

    def __init__(self, source_module, destination_module, object_name,
                 source_setting=None, destination_setting=None):
        super(type(self), self).__init__(source_module, destination_module,
                                         source_setting, destination_setting)
        self.__object_name = object_name

    @property
    def object_name(self):
        '''The name of the objects produced by the source and used by the dest'''
        return self.__object_name

    def __str__(self):
        return "Object: %s" % self.object_name


class ImageDependency(Dependency):
    '''A dependency on an image'''

    def __init__(self, source_module, destination_module, image_name,
                 source_setting=None, destination_setting=None):
        super(type(self), self).__init__(source_module, destination_module,
                                         source_setting, destination_setting)
        self.__image_name = image_name

    @property
    def image_name(self):
        '''The name of the image produced by the source and used by the dest'''
        return self.__image_name

    def __str__(self):
        return "Image: %s" % self.image_name


class MeasurementDependency(Dependency):
    '''A dependency on a measurement'''

    def __init__(self, source_module, destination_module, object_name,
                 feature, source_setting=None, destination_setting=None):
        '''Initialize using source, destination and measurement

        source_module - module producing the measurement

        destination_module - module using the measurement

        object_name - the measurement is made on the objects with this name
        (or Image for image measurements)

        feature - the feature name for the measurement, for instance AreaShape_Area

        source_setting - the module setting that controls production of this
        measurement (very typically = None for no such thing)

        destination_setting - the module setting that chooses the measurement
        for the user of the data, for instance a MeasurementSetting
        '''
        super(type(self), self).__init__(source_module, destination_module,
                                         source_setting, destination_setting)
        self.__object_name = object_name
        self.__feature = feature

    @property
    def object_name(self):
        '''The objects / labels used when producing the measurement'''
        return self.__object_name

    @property
    def feature(self):
        '''The name of the measurement'''
        return self.__feature

    def __str__(self):
        return "Measurement: %s.%s" % (self.object_name, self.feature)


def new_string_cell_array(shape):
    """Return a numpy.ndarray that looks like {NxM cell} to Matlab

    Return a numpy.ndarray that looks like {NxM cell} to Matlab.
    Each of the cells looks empty.
    shape - the shape of the array that's generated, e.g. (5,19) for a 5x19 cell array.
            Currently, this must be a 2-d shape.
    The object returned is a numpy.ndarray with dtype=dtype('object') and the given shape
    with each cell in the array filled with a numpy.ndarray with shape = (1,0)
    and dtype=dtype('float64'). This appears to be the form that's created in matlab
    for this sort of object.
    """
    result = numpy.ndarray(shape, dtype=numpy.dtype('object'))
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            result[i, j] = numpy.empty((0, 0))
    return result


def encapsulate_strings_in_arrays(handles):
    """Recursively descend through the handles structure, replacing strings as arrays packed with strings

    This function makes the handles structure loaded through the sandwich compatible with loadmat. It operates on the array in-place.
    """
    if handles.dtype.kind == 'O':
        # cells - descend recursively
        flat = handles.flat
        for i in range(0, len(flat)):
            if isinstance(flat[i], str) or isinstance(flat[i], unicode):
                flat[i] = encapsulate_string(flat[i])
            elif isinstance(flat[i], numpy.ndarray):
                encapsulate_strings_in_arrays(flat[i])
    elif handles.dtype.fields:
        # A structure: iterate over all structure elements.
        for field in handles.dtype.fields.keys():
            if isinstance(handles[field], str) or isinstance(handles[field], unicode):
                handles[field] = encapsulate_string(handles[field])
            elif isinstance(handles[field], numpy.ndarray):
                encapsulate_strings_in_arrays(handles[field])


def encapsulate_string(s):
    """Encapsulate a string in an array of shape 1 of the length of the string
    """
    if isinstance(s, str):
        result = numpy.ndarray((1,), '<S%d' % (len(s)))
    else:
        result = numpy.ndarray((1,), '<U%d' % (len(s)))
    result[0] = s
    return result
