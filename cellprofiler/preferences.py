import cellprofiler
import cellprofiler.utilities.utf16encode
import logging
import os
import os.path
import tempfile
import threading
import time
import weakref

logger = logging.getLogger(__name__)

ABSPATH_OUTPUT = 'abspath_output'
ABSPATH_IMAGE = 'abspath_image'

__python_root = os.path.split(str(cellprofiler.__path__[0]))[0]
__cp_root = os.path.split(__python_root)[0]


class HeadlessConfig(object):
    def __init__(self):
        self.__preferences = {}

    def Read(self, kwd):
        return self.__preferences[kwd]

    def ReadInt(self, kwd, default=0):
        return int(self.__preferences.get(kwd, default))

    def ReadBool(self, kwd, default=False):
        return bool(self.__preferences.get(kwd, default))

    def Write(self, kwd, value):
        self.__preferences[kwd] = value

    # WriteInt = Write
    WriteBool = Write

    def Exists(self, kwd):
        return self.__preferences.has_key(kwd)

    def GetEntryType(self, kwd):
        '''Get the data type of the registry key.

        Returns wx.Config.Type_String = 1
        '''
        return 1


__is_headless = False
__headless_config = HeadlessConfig()
__image_directory_listeners = []
__cached_values = {}


def set_headless():
    global __is_headless
    __is_headless = True


def get_headless():
    return __is_headless


__awt_headless = None


def set_awt_headless(value):
    '''Tell the Javabridge to start Java with AWT headless or not

    value - True to always start Java headless, regardless of headless
            setting or other factors. False to always start Java with
            AWT enabled, despite other factors. None to use the
            default logic.

    If this is not called, Java is started with AWT headless if
    we are headless and the environment variable, CELLPROFILER_USE_XVFB,
    is not present.
    '''
    global __awt_headless
    __awt_headless = value


def get_awt_headless():
    '''Return True if Java is to be started with AWT headless, False to use AWT'''
    global __awt_headless
    if __awt_headless is None:
        return get_headless() and not os.environ.has_key("CELLPROFILER_USE_XVFB")
    return __awt_headless


def get_config():
    global __is_headless, __headless_config

    return __headless_config


def set_preferences_from_dict(d):
    '''Set the preferences by faking the configuration cache'''
    global __cached_values
    __cached_values = d.copy()
    #
    # We also have to defeat value-specific caches.
    #
    global __recent_files
    __recent_files = {}
    for cache_var in (
            "__default_colormap", "__default_image_directory",
            "__default_output_directory", "__allow_output_file_overwrite",
            "__current_pipeline_path", "__has_reported_jvm_error",
            "__show_analysis_complete_dlg",
            "__show_exiting_test_mode_dlg", "__show_report_bad_sizes_dlg",
            "__show_sampling", "__show_workspace_choice_dlg",
            "__use_more_figure_space",
            "__warn_about_old_pipeline", "__write_MAT_files",
            "__workspace_file", "__omero_server", "__omero_port",
            "__omero_user", "__omero_session_id"):
        globals()[cache_var] = None


def config_read(key):
    '''Read the given configuration value

    Only read from the registry once. This is both technically efficient
    and keeps parallel running instances of CellProfiler from overwriting
    each other's values for things like the current output directory.

    Decode escaped config sequences too.
    '''
    global __cached_values

    if __cached_values.has_key(key):
        return __cached_values[key]
    if get_config().Exists(key):
        value = get_config().Read(key)
    else:
        value = None
    if value is not None:
        try:
            value = cellprofiler.utilities.utf16encode.utf16decode(value)
        except:
            logger.warning(
                "Failed to decode preference (%s=%s), assuming 2.0" %
                (key, value))
    __cached_values[key] = value
    return value


def config_write(key, value):
    '''Write the given configuration value

    Encode escaped config sequences.
    '''
    __cached_values[key] = value
    if value is not None:
        value = cellprofiler.utilities.utf16encode.utf16encode(value)
    get_config().Write(key, value)


def config_exists(key):
    '''Return True if the key is defined in the configuration'''
    global __cached_values
    if key in __cached_values and __cached_values[key] is not None:
        return True
    if not get_config().Exists(key):
        return False
    if get_config().GetEntryType(key) == 1:
        return get_config().Read(key) is not None
    return True


def cell_profiler_root_directory():
    if __cp_root:
        return __cp_root
    return '..'


DEFAULT_INPUT_FOLDER_NAME = 'Default Input Folder'
DEFAULT_OUTPUT_FOLDER_NAME = 'Default Output Folder'
ABSOLUTE_FOLDER_NAME = 'Elsewhere...'
DEFAULT_INPUT_SUBFOLDER_NAME = 'Default Input Folder sub-folder'
DEFAULT_OUTPUT_SUBFOLDER_NAME = 'Default Output Folder sub-folder'
URL_FOLDER_NAME = 'URL'
NO_FOLDER_NAME = "None"

'''Please add any new wordings of the above to this dictionary'''
FOLDER_CHOICE_TRANSLATIONS = {
    'Default Input Folder': DEFAULT_INPUT_FOLDER_NAME,
    'Default Output Folder': DEFAULT_OUTPUT_FOLDER_NAME,
    'Absolute path elsewhere': ABSOLUTE_FOLDER_NAME,
    'Default input directory sub-folder': DEFAULT_INPUT_SUBFOLDER_NAME,
    'Default Input Folder sub-folder': DEFAULT_INPUT_SUBFOLDER_NAME,
    'Default output directory sub-folder': DEFAULT_OUTPUT_SUBFOLDER_NAME,
    'Default Output Folder sub-folder': DEFAULT_OUTPUT_SUBFOLDER_NAME,
    'URL': URL_FOLDER_NAME,
    'None': NO_FOLDER_NAME,
    'Elsewhere...': ABSOLUTE_FOLDER_NAME
}

IO_FOLDER_CHOICE_HELP_TEXT = """
You can choose among the following options which are common to all file input/output
modules:
<ul>
<li><i>Default Input Folder</i>: Use the default input folder.</li>
<li><i>Default Output Folder:</i> Use from the default output folder.</li>
<li><i>Elsewhere...</i>: Use a particular folder you specify.</li>
<li><i>Default input directory sub-folder</i>: Enter the name of a subfolder of
the default input folder or a path that starts from the default input folder.</li>
<li><i>Default output directory sub-folder</i>: Enter the name of a subfolder of
the default output folder or a path that starts from the default output folder.</li>
</ul>
<p><i>Elsewhere</i> and the two sub-folder options all require you to enter an additional
path name. You can use an <i>absolute path</i> (such as "C:\imagedir\image.tif" on a PC) or a
<i>relative path</i> to specify the file location relative to a directory):
<ul>
<li>Use one period to represent the current directory. For example, if you choose
<i>Default Input Folder sub-folder</i>, you can enter "./MyFiles" to look in a
folder called "MyFiles" that is contained within the Default Input Folder.</li>
<li>Use two periods ".." to move up one folder level. For example, if you choose
<i>Default Input Folder sub-folder</i>, you can enter "../MyFolder" to look in a
folder called "MyFolder" at the same level as the Default Input Folder.</li>
</ul></p>
"""

IO_WITH_METADATA_HELP_TEXT = """
For <i>%(ABSOLUTE_FOLDER_NAME)s</i>, <i>%(DEFAULT_INPUT_SUBFOLDER_NAME)s</i> and
<i>%(DEFAULT_OUTPUT_SUBFOLDER_NAME)s</i>, if you have metadata associated with your
images via <b>Metadata</b> module, you can name the folder using metadata
tags.""" % globals()

DEFAULT_IMAGE_DIRECTORY = 'DefaultImageDirectory'
DEFAULT_OUTPUT_DIRECTORY = 'DefaultOutputDirectory'
TITLE_FONT_SIZE = 'TitleFontSize'
TITLE_FONT_NAME = 'TitleFontName'
PIXEL_SIZE = 'PixelSize'
COLORMAP = 'Colormap'
MODULEDIRECTORY = 'ModuleDirectory'
FF_RECENTFILES = 'RecentFile%d'
STARTUPBLURB = 'StartupBlurb'
RECENT_FILE_COUNT = 10
PRIMARY_OUTLINE_COLOR = 'PrimaryOutlineColor'
SECONDARY_OUTLINE_COLOR = 'SecondaryOutlineColor'
TERTIARY_OUTLINE_COLOR = 'TertiaryOutlineColor'
ALLOW_OUTPUT_FILE_OVERWRITE = 'AllowOutputFileOverwrite'
PLUGIN_DIRECTORY = 'PluginDirectory'
IJ_PLUGIN_DIRECTORY = 'IJPluginDirectory'
SHOW_ANALYSIS_COMPLETE_DLG = "ShowAnalysisCompleteDlg"
SHOW_EXITING_TEST_MODE_DLG = "ShowExitingTestModeDlg"
SHOW_BAD_SIZES_DLG = "ShowBadSizesDlg"
SHOW_SAMPLING = "ShowSampling"
WRITE_MAT = "WriteMAT"
WRITE_HDF5 = "WriteHDF5"
WORKSPACE_FILE = "WorkspaceFile"
OMERO_SERVER = "OmeroServer"
OMERO_PORT = "OmeroPort"
OMERO_USER = "OmeroUser"
OMERO_SESSION_ID = "OmeroSessionId"
TEMP_DIR = "TempDir"
ERROR_COLOR = "ErrorColor"
INTERPOLATION_MODE = "InterpolationMode"
INTENSITY_MODE = "IntensityMode"
SAVE_PIPELINE_WITH_PROJECT = "SavePipelineWithProject"
BATCHPROFILER_URL = "BatchProfilerURL"
CHOOSE_IMAGE_SET_FRAME_SIZE = "ChooseImageSetFrameSize"
BATCHPROFILER_URL_DEFAULT = "http://imageweb/batchprofiler"
IM_NEAREST = "Nearest"
IM_BILINEAR = "Bilinear"
IM_BICUBIC = "Bicubic"
INTENSITY_MODE_RAW = "raw"
INTENSITY_MODE_NORMAL = "normalized"
INTENSITY_MODE_LOG = "log"
EXT_PIPELINE = "cppipe"
EXT_PIPELINE_CHOICES = [EXT_PIPELINE, "cp", "cpi", "cpproj", "h5", "mat"]
EXT_PROJECT = "cpproj"
EXT_PROJECT_CHOICES = [EXT_PROJECT, "cpi", "h5"]
JVM_HEAP_MB = "JVMHeapMB"
DEFAULT_JVM_HEAP_MB = 512
SPP_NEITHER = "Neither"
SPP_PIPELINE_ONLY = "Pipeline"
SPP_FILE_LIST_ONLY = "File list"
SPP_PIPELINE_AND_FILE_LIST = "Pipeline and file list"

__has_reported_jvm_error = False
__allow_output_file_overwrite = None
__show_analysis_complete_dlg = None
__show_exiting_test_mode_dlg = None
__show_report_bad_sizes_dlg = None
__write_MAT_files = None
__workspace_file = None
__omero_server = None
__omero_port = None
__omero_user = None
__omero_session_id = None
__max_workers = None
__temp_dir = None
__progress_data = threading.local()
__progress_data.last_report = time.time()
__progress_data.callbacks = None
__interpolation_mode = None
__intensity_mode = None
__jvm_heap_mb = None
__save_pipeline_with_project = None
__allow_schema_write = True
__filename_re_guess_file = None
__pathname_re_guess_file = None
__batchprofiler_url = None
__image_set_filename = None
__wants_pony = None
__choose_image_set_frame_size = None
__default_output_directory = None
__output_directory_listeners = []
__error_color = None
__output_filename = None
__output_filename_listeners = []
__default_colormap = None
__current_workspace_path = None
__show_sampling = None
__recent_files = {}
__plugin_directory = None
__ij_plugin_directory = None
__data_file = None
__default_image_directory = None


def recent_file(index, category=""):
    return (FF_RECENTFILES % (index + 1)) + category


def module_directory():
    if not config_exists(MODULEDIRECTORY):
        return os.path.join(cell_profiler_root_directory(), 'Modules')
    return str(config_read(MODULEDIRECTORY))


def get_default_image_directory():
    global __default_image_directory

    if __default_image_directory is not None:
        return __default_image_directory
    # I'm not sure what it means for the preference not to exist.  No read-write preferences file?
    if not config_exists(DEFAULT_IMAGE_DIRECTORY):
        return os.path.abspath(os.path.expanduser('~'))
    # Fetch the default.  Note that it might be None
    default_image_directory = config_read(DEFAULT_IMAGE_DIRECTORY) or ''
    try:
        if os.path.isdir(default_image_directory):
            __default_image_directory = os.path.normcase(default_image_directory)
            return __default_image_directory
    except:
        logger.error("Unknown failure when retrieving the default image directory", exc_info=True)
    logger.warning("Warning: current path of %s is not a valid directory. Switching to home directory." % (
        default_image_directory.encode('ascii', 'replace')))
    # If the user's home directory is not ascii, we're not going to go hunting for one that is.
    # Fail ungracefully.
    default_image_directory = os.path.abspath(os.path.expanduser('~'))
    set_default_image_directory(default_image_directory)
    return str(os.path.normcase(default_image_directory))


def set_default_image_directory(path):
    global __default_image_directory
    __default_image_directory = path
    config_write(DEFAULT_IMAGE_DIRECTORY, path)
    add_recent_file(path, DEFAULT_IMAGE_DIRECTORY)
    fire_image_directory_changed_event()


def fire_image_directory_changed_event():
    '''Notify listeners of a image directory change'''
    global __default_image_directory
    for listener in __image_directory_listeners:
        listener(PreferenceChangedEvent(__default_image_directory))


def add_image_directory_listener(listener):
    """Add a listener that will be notified when the image directory changes

    """
    __image_directory_listeners.append(listener)


def remove_image_directory_listener(listener):
    """Remove a previously-added image directory listener

    """
    if listener in __image_directory_listeners:
        __image_directory_listeners.remove(listener)


class PreferenceChangedEvent:
    def __init__(self, new_value):
        self.new_value = new_value


def get_default_output_directory():
    global __default_output_directory

    if __default_output_directory is not None:
        return __default_output_directory
    if not config_exists(DEFAULT_OUTPUT_DIRECTORY):
        return os.path.abspath(os.path.expanduser('~'))

    # Fetch the default.  Note that it might be None
    default_output_directory = config_read(DEFAULT_OUTPUT_DIRECTORY) or ''
    try:
        if os.path.isdir(default_output_directory):
            __default_output_directory = os.path.normcase(default_output_directory)
            return __default_output_directory
    except:
        logger.error("Unknown failure when retrieving the default output directory", exc_info=True)
    logger.warning("Warning: current path of %s is not a valid directory. Switching to home directory." % (
        default_output_directory.encode('ascii', 'replace')))
    # If the user's home directory is not ascii, we're not going to go hunting for one that is.
    # Fail ungracefully.
    default_output_directory = os.path.abspath(os.path.expanduser('~'))
    set_default_output_directory(default_output_directory)
    return str(os.path.normcase(default_output_directory))


def set_default_output_directory(path):
    global __default_output_directory
    assert os.path.isdir(path), 'Default Output Folder, "%s", is not a directory' % path
    __default_output_directory = path
    config_write(DEFAULT_OUTPUT_DIRECTORY, path)
    add_recent_file(path, DEFAULT_OUTPUT_DIRECTORY)
    for listener in __output_directory_listeners:
        listener(PreferenceChangedEvent(path))


def add_output_directory_listener(listener):
    """Add a listener that will be notified when the output directory changes

    """
    __output_directory_listeners.append(listener)


def remove_output_directory_listener(listener):
    """Remove a previously-added image directory listener

    """
    if listener in __output_directory_listeners:
        __output_directory_listeners.remove(listener)


def get_pixel_size():
    """The size of a pixel in microns"""
    if not config_exists(PIXEL_SIZE):
        return 1.0
    return float(config_read(PIXEL_SIZE))


def set_pixel_size(pixel_size):
    config_write(PIXEL_SIZE, str(pixel_size))


def get_output_file_name():
    global __output_filename
    if __output_filename is None:
        return 'DefaultOUT.mat'
    return __output_filename


def set_output_file_name(filename):
    global __output_filename
    filename = str(filename)
    __output_filename = filename
    for listener in __output_filename_listeners:
        listener(PreferenceChangedEvent(filename))


def add_output_file_name_listener(listener):
    __output_filename_listeners.append(listener)


def remove_output_file_name_listener(listener):
    try:
        __output_filename_listeners.remove(listener)
    except:
        logger.warn("File name listener doubly removed")


def get_absolute_path(path, abspath_mode=ABSPATH_IMAGE):
    """Convert a path into an absolute path using the path conventions

    If a path starts with http:, https: or ftp:, leave it unchanged.
    If a path starts with "./", then make the path relative to the
    Default Output Folder.
    If a path starts with "&/", then make the path relative to the
    Default Input Folder.
    If a "path" has no path component then make the path relative to
    the Default Output Folder.
    """

    if abspath_mode == ABSPATH_OUTPUT:
        osep = '.'
        isep = '&'
    elif abspath_mode == ABSPATH_IMAGE:
        osep = '&'
        isep = '.'
    else:
        raise ValueError("Unknown abspath mode: %s" % abspath_mode)
    if is_url_path(path):
        return path
    if (path.startswith(osep + os.path.sep) or
            ("altsep" in os.path.__all__ and os.path.altsep and
                 path.startswith(osep + os.path.altsep))):
        return os.path.join(get_default_output_directory(), path[2:])
    elif (path.startswith(isep + os.path.sep) or
              ("altsep" in os.path.__all__ and os.path.altsep and
                   path.startswith(isep + os.path.altsep))):
        return os.path.join(get_default_image_directory(), path[2:])
    elif len(os.path.split(path)[0]) == 0:
        return os.path.join(get_default_output_directory(), path)
    else:
        return str(os.path.normpath(os.path.abspath(path)))


def is_url_path(path):
    '''Return True if the path should be treated as a URL'''
    for protocol in ('http', 'https', 'ftp'):
        if path.lower().startswith('%s:' % protocol):
            return True
    return False


def get_default_colormap():
    return "jet"


def get_current_workspace_path():
    global __current_workspace_path
    return __current_workspace_path


def set_current_workspace_path(path):
    global __current_workspace_path
    __current_workspace_path = path


def get_show_sampling():
    global __show_sampling
    if __show_sampling is not None:
        return __show_sampling
    if not config_exists(SHOW_SAMPLING):
        __show_sampling = False
        return False
    return get_config().ReadBool(SHOW_SAMPLING)


def get_recent_files(category=""):
    global __recent_files
    if __recent_files.get(category, None) is None:
        __recent_files[category] = []
        for i in range(RECENT_FILE_COUNT):
            key = recent_file(i, category)
            try:
                if config_exists(key):
                    __recent_files[category].append(config_read(key))
            except:
                pass
    return __recent_files[category]


def add_recent_file(filename, category=""):
    recent_files = get_recent_files(category)
    filename = os.path.abspath(filename)
    if filename in recent_files:
        recent_files.remove(filename)
    recent_files.insert(0, filename)
    if len(recent_files) > RECENT_FILE_COUNT:
        del recent_files[-1]
    for i, filename in enumerate(recent_files):
        config_write(recent_file(i, category), filename)


def get_plugin_directory():
    global __plugin_directory

    if __plugin_directory is not None:
        return __plugin_directory

    if config_exists(PLUGIN_DIRECTORY):
        __plugin_directory = config_read(PLUGIN_DIRECTORY)
    elif get_headless():
        return None
    else:
        pass

    return __plugin_directory


def set_plugin_directory(value, globally=True):
    global __plugin_directory

    __plugin_directory = value
    if globally:
        config_write(PLUGIN_DIRECTORY, value)


def get_data_file():
    '''Get the path to the LoadData data file specified on the command-line'''
    global __data_file
    return __data_file


def set_data_file(path):
    global __data_file
    __data_file = path


def standardize_default_folder_names(setting_values, slot):
    if setting_values[slot] in FOLDER_CHOICE_TRANSLATIONS.keys():
        replacement = FOLDER_CHOICE_TRANSLATIONS[setting_values[slot]]
    elif (setting_values[slot].startswith("Default Image") or
              setting_values[slot].startswith("Default image") or
              setting_values[slot].startswith("Default input")):
        replacement = DEFAULT_INPUT_FOLDER_NAME
    elif setting_values[slot].startswith("Default output"):
        replacement = DEFAULT_OUTPUT_FOLDER_NAME
    else:
        replacement = setting_values[slot]
    setting_values = (setting_values[:slot] +
                      [replacement] +
                      setting_values[slot + 1:])
    return setting_values


__cpfigure_position = (-1, -1)


def get_next_cpfigure_position(update_next_position=True):
    global __cpfigure_position
    pos = __cpfigure_position
    if update_next_position:
        update_cpfigure_position()
    return pos


def reset_cpfigure_position():
    global __cpfigure_position
    __cpfigure_position = (-1, -1)


def update_cpfigure_position():
    '''Called by get_next_cpfigure_position to update the screen position at
    which the next figure frame will be drawn.
    '''
    global __cpfigure_position

    win_size = (800, 600)

    disp = (1280, 800)

    if __cpfigure_position[0] + win_size[0] > disp[0]:
        __cpfigure_position = (-1, __cpfigure_position[1])

    if __cpfigure_position[1] + win_size[1] > disp[1]:
        __cpfigure_position = (-1, -1)
    else:
        __cpfigure_position = (__cpfigure_position[0] + 120, __cpfigure_position[1] + 24)


def get_allow_output_file_overwrite():
    '''Return true if the user wants to allow CP to overwrite the output file

    This is the .MAT output file, typically Default_OUT.mat
    '''
    global __allow_output_file_overwrite
    if __allow_output_file_overwrite is not None:
        return __allow_output_file_overwrite
    if not config_exists(ALLOW_OUTPUT_FILE_OVERWRITE):
        return False
    return config_read(ALLOW_OUTPUT_FILE_OVERWRITE) == "True"


def set_allow_output_file_overwrite(value):
    '''Allow overwrite of .MAT file if true, warn user if false'''
    global __allow_output_file_overwrite
    __allow_output_file_overwrite = value
    config_write(ALLOW_OUTPUT_FILE_OVERWRITE,
                 "True" if value else "False")


def set_show_analysis_complete_dlg(value):
    '''Set the "show analysis complete" flag'''
    global __show_analysis_complete_dlg
    __show_analysis_complete_dlg = value
    config_write(SHOW_ANALYSIS_COMPLETE_DLG,
                 "True" if value else "False")


def get_show_exiting_test_mode_dlg():
    '''Return true if the user wants to see the "exiting test mode" dialog'''
    global __show_exiting_test_mode_dlg
    if __show_exiting_test_mode_dlg is not None:
        return __show_exiting_test_mode_dlg
    if not config_exists(SHOW_EXITING_TEST_MODE_DLG):
        return True
    return config_read(SHOW_EXITING_TEST_MODE_DLG) == "True"


def set_show_exiting_test_mode_dlg(value):
    '''Set the "exiting test mode" flag'''
    global __show_exiting_test_mode_dlg
    __show_exiting_test_mode_dlg = value
    config_write(SHOW_EXITING_TEST_MODE_DLG,
                 "True" if value else "False")


def get_show_report_bad_sizes_dlg():
    '''Return true if the user wants to see the "report bad sizes" dialog'''
    global __show_report_bad_sizes_dlg
    if __show_report_bad_sizes_dlg is not None:
        return __show_report_bad_sizes_dlg
    if not config_exists(SHOW_BAD_SIZES_DLG):
        return True
    return config_read(SHOW_BAD_SIZES_DLG) == "True"


def set_show_report_bad_sizes_dlg(value):
    '''Set the "exiting test mode" flag'''
    global __show_report_bad_sizes_dlg
    __show_report_bad_sizes_dlg = value
    config_write(SHOW_BAD_SIZES_DLG,
                 "True" if value else "False")


def get_write_MAT_files():
    '''Determine whether to write measurements in .MAT files, .h5 files or not at all

    returns True to write .MAT, WRITE_HDF5 to write .h5 files, False to not write
    '''
    global __write_MAT_files
    if __write_MAT_files is not None:
        return __write_MAT_files
    if not config_exists(WRITE_MAT):
        return False
    value = config_read(WRITE_MAT)
    if value == "True":
        return True
    if value == WRITE_HDF5:
        return WRITE_HDF5
    return False


def set_write_MAT_files(value):
    '''Set the "Write MAT files" flag'''
    global __write_MAT_files
    __write_MAT_files = value
    config_write(WRITE_MAT,
                 WRITE_HDF5 if value == WRITE_HDF5
                 else "True" if value else "False")


def set_workspace_file(path, permanently=True):
    '''Set the path to the workspace file

    path - path to the file

    permanently - True to write it to the configuration, False if the file
                  should only be set for the running instance (e.g. as a
                  command-line parameter for a scripted run)
    '''
    global __workspace_file
    __workspace_file = path
    if permanently:
        add_recent_file(path, WORKSPACE_FILE)
        config_write(WORKSPACE_FILE, path)


def get_omero_server():
    '''Get the DNS name of the Omero server'''
    global __omero_server
    if __omero_server is None:
        if not config_exists(OMERO_SERVER):
            return None
        __omero_server = config_read(OMERO_SERVER)
    return __omero_server


def set_omero_server(omero_server, globally=True):
    '''Set the DNS name of the Omero server'''
    global __omero_server
    __omero_server = omero_server
    if globally:
        config_write(OMERO_SERVER, omero_server)


def get_omero_port():
    '''Get the port used to connect to the Omero server'''
    global __omero_port
    if __omero_port is None:
        if not config_exists(OMERO_PORT):
            return 4064
        try:
            __omero_port = int(config_read(OMERO_PORT))
        except:
            return 4064
    return __omero_port


def set_omero_port(omero_port, globally=True):
    '''Set the port used to connect to the Omero server'''
    global __omero_port
    __omero_port = omero_port
    if globally:
        config_write(OMERO_PORT, str(omero_port))


def get_omero_user():
    '''Get the Omero user name'''
    global __omero_user
    if __omero_user is None:
        if not config_exists(OMERO_USER):
            return None
        __omero_user = config_read(OMERO_USER)
    return __omero_user


def set_omero_user(omero_user, globally=True):
    '''Set the Omero user name'''
    global __omero_user
    __omero_user = omero_user
    if globally:
        config_write(OMERO_USER, omero_user)


def get_omero_session_id():
    '''Get the session ID to use to communicate to Omero'''
    global __omero_session_id
    if __omero_session_id is None:
        if not config_exists(OMERO_SESSION_ID):
            return None
        __omero_session_id = config_read(OMERO_SESSION_ID)
    return __omero_session_id


def get_temporary_directory():
    '''Get the directory to be used for temporary files

    The default is whatever is returned by tempfile.gettempdir()
    (see http://docs.python.org/2/library/tempfile.html#tempfile.gettempdir)
    '''
    global __temp_dir
    if __temp_dir is not None:
        pass
    elif config_exists(TEMP_DIR):
        path = config_read(TEMP_DIR)
        if os.path.isdir(path):
            __temp_dir = path
            return __temp_dir
    __temp_dir = tempfile.gettempdir()
    return __temp_dir


def set_temporary_directory(tempdir, globally=False):
    '''Set the directory to be used for temporary files

    tempdir - pathname of the directory
    '''
    global __temp_dir
    if globally:
        config_write(TEMP_DIR, tempdir)
    __temp_dir = tempdir
    tempfile.tempdir = tempdir


def get_interpolation_mode():
    '''Get the interpolation mode for matplotlib

    Returns one of IM_NEAREST, IM_BILINEAR or IM_BICUBIC
    '''
    global __interpolation_mode
    if __interpolation_mode is not None:
        return __interpolation_mode
    if config_exists(INTERPOLATION_MODE):
        __interpolation_mode = config_read(INTERPOLATION_MODE)
    else:
        __interpolation_mode = IM_NEAREST
    return __interpolation_mode


def get_intensity_mode():
    '''Get the intensity scaling mode for matplotlib

    Returns one of INTENSITY_MODE_RAW, INTENSITY_MODE_NORMAL, INTENSITY_MODE_LOG
    '''
    global __intensity_mode
    if __intensity_mode is not None:
        return __intensity_mode
    if config_exists(INTENSITY_MODE):
        __intensity_mode = config_read(INTENSITY_MODE)
    else:
        __intensity_mode = INTENSITY_MODE_NORMAL
    return __intensity_mode


def get_save_pipeline_with_project():
    global __save_pipeline_with_project
    if __save_pipeline_with_project is None:
        if config_exists(SAVE_PIPELINE_WITH_PROJECT):
            __save_pipeline_with_project = \
                config_read(SAVE_PIPELINE_WITH_PROJECT)
        else:
            __save_pipeline_with_project = SPP_NEITHER
    return __save_pipeline_with_project


def get_allow_schema_write():
    '''Returns True if ExportToDatabase is allowed to write the MySQL schema

    For cluster operation without CreateBatchFiles, it's inappropriate to
    have multiple processes overwrite the database schema. Although
    CreateBatchFiles is suggested for this scenario, we put this switch in
    to support disabling schema writes from the command line.
    '''
    return __allow_schema_write


def set_allow_schema_write(value):
    '''Allow or disallow database schema writes

    value - True to allow writes (the default) or False to prevent
            ExportToDatabase from writing the schema.

    For cluster operation without CreateBatchFiles, it's inappropriate to
    have multiple processes overwrite the database schema. Although
    CreateBatchFiles is suggested for this scenario, we put this switch in
    to support disabling schema writes from the command line.
    '''
    global __allow_schema_write
    __allow_schema_write = value


def set_image_set_file(filename):
    '''Record the name of the image set that should be loaded upon startup'''
    global __image_set_filename
    __image_set_filename = filename


def clear_image_set_file():
    '''Remove the recorded image set file name

    Call this after loading the image set file to cancel reloading of the
    file during subsequent operations.
    '''
    global __image_set_filename
    __image_set_filename = None


def get_image_set_file():
    '''Recover the name of the image set file to use to populate the file list

    Returns either None or the name of the file to use. For the UI, the
    file list should be loaded and clear_image_set_file() should be called,
    for headless, the file list should be loaded after the pipeline has been
    loaded.
    '''
    return __image_set_filename


def get_choose_image_set_frame_size():
    '''Return the size (w, h) for the "Choose image set" dialog frame'''
    global __choose_image_set_frame_size
    if __choose_image_set_frame_size is None:
        if config_exists(CHOOSE_IMAGE_SET_FRAME_SIZE):
            s = config_read(CHOOSE_IMAGE_SET_FRAME_SIZE)
            __choose_image_set_frame_size = tuple(
                [int(_.strip()) for _ in s.split(",", 1)])
    return __choose_image_set_frame_size


def set_choose_image_set_frame_size(w, h):
    '''Set the size of the "Choose image set" dialog frame'''
    global __choose_image_set_frame_size
    __choose_image_set_frame_size = (w, h)
    config_write(CHOOSE_IMAGE_SET_FRAME_SIZE, "%d,%d" % (w, h))


def add_progress_callback(callback):
    '''Add a callback function that listens to progress calls

    The progress indicator is designed to monitor progress of operations
    on the user interface thread. The model is that operations are nested
    so that both an operation and sub-operation can report their progress.
    An operation reports its initial progress and is pushed onto the
    stack at that point. When it reports 100% progress, it's popped from
    the stack.

    callback - callback function with signature of
               fn(operation_id, progress, message)
               where operation_id names the instance of the operation being
               performed (e.g. a UUID), progress is a number between 0 and 1
               where 1 indicates that the operation has completed and
               message is the message to show.

               Call the callback with operation_id = None to pop the operation
               stack after an exception.

    Note that the callback must remain in-scope. For example:

    class Foo():
       def callback(operation_id, progress, message):
          ...

    works but

    class Bar():
        def __init__(self):
            def callback(operation_id, progress, message):
                ...

    does not work because the reference is lost when __init__ returns.
    '''
    global __progress_data
    if __progress_data.callbacks is None:
        __progress_data.callbacks = weakref.WeakSet()
    __progress_data.callbacks.add(callback)


def remove_progress_callback(callback):
    global __progress_data
    if (__progress_data.callbacks is not None and
                callback in __progress_data.callbacks):
        __progress_data.callbacks.remove(callback)


def report_progress(operation_id, progress, message):
    '''Report progress to all callbacks registered on the caller's thread

    operation_id - ID of operation being performed

    progress - a number between 0 and 1 indicating the extent of progress.
               None indicates indeterminate operation duration. 0 should be
               reported at the outset and 1 at the end.

    message - an informative message.
    '''
    global __progress_data
    if __progress_data.callbacks is None:
        return
    t = time.time()
    if progress in (None, 0, 1) or t - __progress_data.last_report > 1:
        for callback in __progress_data.callbacks:
            callback(operation_id, progress, message)
        __progress_data.last_report = time.time()


def cancel_progress():
    '''Cancel all progress indicators

    for instance, after an exception is thrown that bubbles to the top.
    '''
    report_progress(None, None, None)
