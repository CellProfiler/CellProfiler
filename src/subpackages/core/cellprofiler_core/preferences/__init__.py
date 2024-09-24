# coding:utf-8

"""Preferences.py - singleton preferences for CellProfiler

   TO-DO - load the default preferences from somewhere.
           Create a function to save the preferences.
           Create a function to populate a handles structure with preferences.
"""
import json
import logging
import os
import os.path
import pathlib
import sys
import tempfile
import threading
import time
import traceback
import uuid
import weakref

import psutil

from ._headless_configuration import HeadlessConfiguration
from ..constants.reader import ALL_READERS
from ..utilities.image import image_resource


LOGGER = logging.getLogger(__name__)

"""get_absolute_path - mode = output. Assume "." is the default output dir"""
ABSPATH_OUTPUT = "abspath_output"

"""get_absolute_path - mode = image. Assume "." is the default input dir"""
ABSPATH_IMAGE = "abspath_image"

__python_root = os.path.split(os.path.dirname(pathlib.Path(__file__).parent))[0]
__cp_root = os.path.split(__python_root)[0]

__is_headless = False
__headless_config = HeadlessConfiguration()


def set_headless():
    global __is_headless
    __is_headless = True


def get_headless():
    return __is_headless


__awt_headless = None


def set_awt_headless(value):
    """Tell the Javabridge to start Java with AWT headless or not

    value - True to always start Java headless, regardless of headless
            setting or other factors. False to always start Java with
            AWT enabled, despite other factors. None to use the
            default logic.

    If this is not called, Java is started with AWT headless if
    we are headless and the environment variable, CELLPROFILER_USE_XVFB,
    is not present.
    """
    global __awt_headless
    __awt_headless = value


def get_awt_headless():
    """Return True if Java is to be started with AWT headless, False to use AWT"""
    global __awt_headless
    if __awt_headless is None:
        return get_headless() and "CELLPROFILER_USE_XVFB" not in os.environ
    return __awt_headless


def get_config():
    global __is_headless, __headless_config
    if __is_headless:
        return __headless_config
    import wx

    # Return simple config on build
    if not wx.App.IsDisplayAvailable():
        return __headless_config

    if wx.App.Get() is None:
        app = wx.App(0)

    config = wx.Config.Get(False)

    if not config:
        wx.Config.Set(
            wx.Config(
                "CellProfiler",
                "BroadInstitute",
                "CellProfilerLocal.cfg",
                "CellProfilerGlobal.cfg",
                wx.CONFIG_USE_LOCAL_FILE,
            )
        )
        config = wx.Config.Get()
        if not config.Exists(PREFERENCES_VERSION):
            for key in ALL_KEYS:
                if config.Exists(key) and config.GetEntryType(key) == 1:
                    if key in BOOL_KEYS:
                        v = config.ReadBool(key)
                    elif key in INT_KEYS:
                        v = config.ReadInt(key)
                    elif key in FLOAT_KEYS:
                        v = config.ReadFloat(key)
                    else:
                        v = config.Read(key)
                    config_write(key, v)
            config_write(PREFERENCES_VERSION, str(PREFERENCES_VERSION_NUMBER))
        else:
            try:
                preferences_version_number = int(config_read(PREFERENCES_VERSION))
                if preferences_version_number != PREFERENCES_VERSION_NUMBER:
                    LOGGER.warning(
                        "Preferences version mismatch: expected %d, at %d"
                        % (PREFERENCES_VERSION_NUMBER, preferences_version_number)
                    )
            except:
                LOGGER.warning(
                    "Preferences version was %s, not a number. Resetting to current version"
                    % preferences_version_number
                )
                config_write(PREFERENCES_VERSION, str(PREFERENCES_VERSION))

    return config


def preferences_as_dict():
    # Decode settings from older CP versions which are stored as bytes.
    pref_dict = dict((k, config_read(k)) for k in ALL_KEYS)
    for key in ALL_KEYS:
        if isinstance(pref_dict[key], bytes):
            pref_dict[key] = pref_dict[key].decode("utf-8")
    # Now get the reader settings
    for reader_name, reader_class in ALL_READERS.items():
        config_stem = f"Reader.{reader_name}."
        enabled_key = config_stem + 'enabled'
        enabled_val = config_read_typed(enabled_key, bool)
        if enabled_val is not None:
            pref_dict[enabled_key] = enabled_val
        for key, _, _, key_type, default in reader_class.get_settings():
            config_key = config_stem + key
            config_val = config_read_typed(config_key, key_type)
            if config_val is not None:
                pref_dict[config_key] = config_val
    return pref_dict


def set_preferences_from_dict(d):
    """Set the preferences by faking the configuration cache"""
    global __cached_values
    __cached_values = d.copy()
    #
    # We also have to defeat value-specific caches.
    #
    global __recent_files
    __recent_files = {}
    for cache_var in (
        "__default_colormap",
        "__default_image_directory",
        "__default_output_directory",
        "__current_pipeline_path",
        "__has_reported_jvm_error",
        "__show_analysis_complete_dlg",
        "__show_exiting_test_mode_dlg",
        "__show_report_bad_sizes_dlg",
        "__show_sampling",
        "__show_workspace_choice_dlg",
        "__use_more_figure_space",
        "__warn_about_old_pipeline",
        "__workspace_file",
        "__omero_server",
        "__omero_port",
        "__omero_user",
        "__omero_session_id",
    ):
        globals()[cache_var] = None


__cached_values: dict = {}


def config_read(key):
    """Read the given configuration value

    Only read from the registry once. This is both technically efficient
    and keeps parallel running instances of CellProfiler from overwriting
    each other's values for things like the current output directory.
    """
    global __cached_values
    if not __is_headless:
        #
        # Keeps popup box from appearing during testing I hope
        #
        import wx

        shutup = wx.LogNull()
    if key in __cached_values:
        return __cached_values[key]
    if get_config().Exists(key):
        if not __is_headless:
            # Fix problems with some 32-bit
            import wx

            if key in BOOL_KEYS:
                return get_config().ReadBool(key)
            elif key in INT_KEYS:
                return get_config().ReadInt(key)
            elif key in FLOAT_KEYS:
                return get_config().ReadFloat(key)
        value = get_config().Read(key)
    else:
        value = None
    __cached_values[key] = value
    return value


def config_write(key, value):
    """Write the given configuration value
    """
    if not __is_headless:
        #
        # Keeps popup box from appearing during testing I hope
        #
        import wx

        shutup = wx.LogNull()
    __cached_values[key] = value
    if key in BOOL_KEYS:
        get_config().WriteBool(key, bool(value))
    elif key in INT_KEYS:
        get_config().WriteInt(key, int(value))
    elif key in FLOAT_KEYS:
        get_config().WriteFloat(key, float(value))
    else:
        get_config().Write(key, value)
    if not __is_headless:
        get_config().Flush()


def config_exists(key):
    """Return True if the key is defined in the configuration"""
    global __cached_values
    if key in __cached_values and __cached_values[key] is not None:
        return True
    if not get_config().Exists(key):
        return False
    # FIXME: Issue reported at https://github.com/wxWidgets/Phoenix/issues/1292, use respective .Read()
    if sys.platform != "win32":
        if get_config().GetEntryType(key) == 1:
            return get_config().Read(key) is not None
    else:
        # Bool keys
        if key in BOOL_KEYS:
            return get_config().ReadBool(key) is not None
        # Int keys
        elif key in INT_KEYS:
            return get_config().ReadInt(key) is not None
        # Double keys
        elif key in FLOAT_KEYS:
            return get_config().ReadFloat(key) is not None
        # String keys
        else:
            return get_config().Read(key) is not None
    return True


def config_read_typed(key, key_type):
    global __cached_values
    config = get_config()
    if key in __cached_values:
        return __cached_values[key]
    if config.Exists(key):
        if key_type == bool:
            value = config.ReadBool(key)
        elif key_type == int:
            value = config.ReadInt(key)
        elif key_type == float:
            value = config.ReadFloat(key)
        else:
            value = config.Read(key)
    else:
        value = None
    __cached_values[key] = value
    return value


def config_write_typed(key, value, key_type=None, flush=True):
    global __cached_values
    if key_type is None:
        # This is less safe, please do specify type.
        key_type = type(value)
    config = get_config()
    if key_type == bool:
        success = config.WriteBool(key, value)
    elif key_type == int:
        success = config.WriteInt(key, value)
    elif key_type == float:
        success = config.WriteFloat(key, value)
    else:
        success = config.Write(key, value)
    if not success:
        LOGGER.error(f"Unable to write preference key {key}")
        return
    if flush:
        config.Flush()
        __cached_values[key] = value


def export_to_json(path):
    global __is_headless
    assert not __is_headless, "wx Config can't be read in headless mode"
    config_dict = preferences_as_dict()
    # Remove recent file keys, these aren't useful.
    bad_keys = [key for key in config_dict.keys() if key.startswith('RecentFile')]
    for key in bad_keys:
        del config_dict[key]
    with open(path, mode='wt') as fd:
        json.dump(config_dict, fd)
        LOGGER.info(f"Wrote config to {path}")


def cell_profiler_root_directory():
    if __cp_root:
        return __cp_root
    return ".."


def python_root_directory():
    return __python_root


def resources_root_directory():
    if hasattr(sys, "frozen"):
        # On Mac, the application runs in CellProfiler2.0.app/Contents/Resources.
        # Not sure where this should be on PC.
        return "."
    else:
        return __python_root


DEFAULT_INPUT_FOLDER_NAME = "Default Input Folder"
DEFAULT_OUTPUT_FOLDER_NAME = "Default Output Folder"
ABSOLUTE_FOLDER_NAME = "Elsewhere..."
DEFAULT_INPUT_SUBFOLDER_NAME = "Default Input Folder sub-folder"
DEFAULT_OUTPUT_SUBFOLDER_NAME = "Default Output Folder sub-folder"
URL_FOLDER_NAME = "URL"
NO_FOLDER_NAME = "None"

"""Please add any new wordings of the above to this dictionary"""
FOLDER_CHOICE_TRANSLATIONS = {
    "Default Input Folder": DEFAULT_INPUT_FOLDER_NAME,
    "Default Output Folder": DEFAULT_OUTPUT_FOLDER_NAME,
    "Absolute path elsewhere": ABSOLUTE_FOLDER_NAME,
    "Default input directory sub-folder": DEFAULT_INPUT_SUBFOLDER_NAME,
    "Default Input Folder sub-folder": DEFAULT_INPUT_SUBFOLDER_NAME,
    "Default output directory sub-folder": DEFAULT_OUTPUT_SUBFOLDER_NAME,
    "Default Output Folder sub-folder": DEFAULT_OUTPUT_SUBFOLDER_NAME,
    "URL": URL_FOLDER_NAME,
    "None": NO_FOLDER_NAME,
    "Elsewhere...": ABSOLUTE_FOLDER_NAME,
}

PREFERENCES_VERSION = "PreferencesVersion"
PREFERENCES_VERSION_NUMBER = 1
DEFAULT_IMAGE_DIRECTORY = "DefaultImageDirectory"
DEFAULT_OUTPUT_DIRECTORY = "DefaultOutputDirectory"
TITLE_FONT_SIZE = "TitleFontSize"
TITLE_FONT_NAME = "TitleFontName"
TABLE_FONT_NAME = "TableFontName"
TABLE_FONT_SIZE = "TableFontSize"
BACKGROUND_COLOR = "BackgroundColor"
PIXEL_SIZE = "PixelSize"
COLORMAP = "Colormap"
CONSERVE_MEMORY = "ConserveMemory"
MODULEDIRECTORY = "ModuleDirectory"
SKIPVERSION = "SkipVersion2.1"
FF_RECENTFILES = "RecentFile%d"
STARTUPBLURB = "StartupBlurb"
TELEMETRY = "Telemetry"
TELEMETRY_PROMPT = "Telemetry prompt"
CHECK_FOR_UPDATES = "CheckForUpdates"
RECENT_FILE_COUNT = 10
PRIMARY_OUTLINE_COLOR = "PrimaryOutlineColor"
SECONDARY_OUTLINE_COLOR = "SecondaryOutlineColor"
TERTIARY_OUTLINE_COLOR = "TertiaryOutlineColor"
JVM_ERROR = "JVMError"
PLUGIN_DIRECTORY = "PluginDirectoryCP4"
SHOW_ANALYSIS_COMPLETE_DLG = "ShowAnalysisCompleteDlg"
SHOW_EXITING_TEST_MODE_DLG = "ShowExitingTestModeDlg"
SHOW_BAD_SIZES_DLG = "ShowBadSizesDlg"
SHOW_SAMPLING = "ShowSampling"
WARN_ABOUT_OLD_PIPELINE = "WarnAboutOldPipeline"
USE_MORE_FIGURE_SPACE = "UseMoreFigureSpace"
WORKSPACE_FILE = "WorkspaceFile"
OMERO_SERVER = "OmeroServer"
OMERO_PORT = "OmeroPort"
OMERO_USER = "OmeroUser"
OMERO_SESSION_ID = "OmeroSessionId"
MAX_WORKERS = "MaxWorkers"
TEMP_DIR = "TempDir"
WORKSPACE_CHOICE = "WorkspaceChoice"
ERROR_COLOR = "ErrorColor"
INTERPOLATION_MODE = "InterpolationMode"
INTENSITY_MODE = "IntensityMode"
NORMALIZATION_FACTOR = "NormalizationFactor"
SAVE_PIPELINE_WITH_PROJECT = "SavePipelineWithProject"
FILENAME_RE_GUESSES_FILE = "FilenameRegularExpressionGuessesFile"
PATHNAME_RE_GUESSES_FILE = "PathnameRegularExpressionGuessesFile"
CHOOSE_IMAGE_SET_FRAME_SIZE = "ChooseImageSetFrameSize"
ALWAYS_CONTINUE = "AlwaysContinue"
WIDGET_INSPECTOR = "WidgetInspector"

"""Default URL root for BatchProfiler"""

IM_NEAREST = "Nearest"
IM_BILINEAR = "Bilinear"
IM_BICUBIC = "Bicubic"

INTENSITY_MODE_RAW = "raw"
INTENSITY_MODE_NORMAL = "normalized"
INTENSITY_MODE_LOG = "log"
INTENSITY_MODE_GAMMA = "gamma"

WC_SHOW_WORKSPACE_CHOICE_DIALOG = "ShowWorkspaceChoiceDlg"
WC_OPEN_LAST_WORKSPACE = "OpenLastWorkspace"
WC_CREATE_NEW_WORKSPACE = "CreateNewWorkspace"
WC_OPEN_OLD_WORKSPACE = "OpenOldWorkspace"

"""The default extension for a CellProfiler pipeline (without the dot)"""
EXT_PIPELINE = "cppipe"

"""Possible CellProfiler pipeline extensions"""
EXT_PIPELINE_CHOICES = [EXT_PIPELINE, "cp", "cpi", "cpproj", "json"]

"""Default project extension"""
EXT_PROJECT = "cpproj"

"""Possible CellProfiler project extensions"""
EXT_PROJECT_CHOICES = [EXT_PROJECT, "cpi"]

"""Preference key for the JVM heap size in megabytes"""
JVM_HEAP_MB = "JVMHeapMB"

"""Default JVM heap size"""
DEFAULT_JVM_HEAP_MB = 512

"""Save neither the pipeline nor the file list when saving the project"""
SPP_NEITHER = "Neither"
SPP_PIPELINE_ONLY = "Pipeline"
SPP_FILE_LIST_ONLY = "File list"
SPP_PIPELINE_AND_FILE_LIST = "Pipeline and file list"
SPP_ALL = [
    SPP_NEITHER,
    SPP_PIPELINE_ONLY,
    SPP_FILE_LIST_ONLY,
    SPP_PIPELINE_AND_FILE_LIST,
]

# Registry Key Types
BOOL_KEYS = {SHOW_SAMPLING, TELEMETRY, TELEMETRY_PROMPT, STARTUPBLURB, 
             CONSERVE_MEMORY, ALWAYS_CONTINUE, WIDGET_INSPECTOR}
INT_KEYS = {SKIPVERSION, OMERO_PORT, MAX_WORKERS, JVM_HEAP_MB}
FLOAT_KEYS = {TITLE_FONT_SIZE, TABLE_FONT_SIZE, PIXEL_SIZE}

#######################
#
# Preferences help text
#
#######################
CONSERVE_MEMORY_HELP = """\
If enabled, CellProfiler will attempt to release unused system memory
after processing each image in an analysis run. Image pixel data will 
also be 'forgotten' if an image is no longer required by later modules 
in the pipeline. This can help to conserve system resources if the 
user is running other tasks in the background. Enabling this setting 
may slightly impact analysis speed, particularly during large runs.\
"""

DEFAULT_COLORMAP_HELP = """\
Specifies the color map that sets the colors for labels and other
elements. See this `page`_ for pictures of available colormaps.

.. _page: http://matplotlib.org/users/colormaps.html\
"""

DEFAULT_IMAGE_FOLDER_HELP = """\
The folder designated as the *Default Input Folder* contains the input
image or data files that you want to analyze. Several File Processing
modules (e.g., **LoadImages** or **LoadData**) provide the
option of retrieving images from this folder on a default basis unless
you specify, within the module, an alternate, specific folder on your
computer. Within modules, we recommend selecting the Default Input
Folder as much as possible, so that your pipeline will work even if you
transfer your images and pipeline to a different computer. If, instead,
you type specific folder path names into a module’s settings, your
pipeline will not work on someone else’s computer until you adjust those
pathnames within each module.

Use the *Browse* button |image0| to specify the folder you would like to
use as the Default Input Folder, or type the full folder path in the
edit box. If you type a folder path that cannot be found, the message
box below will indicate this fact until you correct the problem. If you
want to specify a folder that does not yet exist, type the desired name
and click on the *New folder* button |image1|. The folder will be
created according to the pathname you have typed.

.. |image0| image:: {BROWSE_BUTTON}
.. |image1| image:: {CREATE_BUTTON}\
""".format(
    **{
        "CREATE_BUTTON": image_resource("folder_create.png"),
        "BROWSE_BUTTON": image_resource("folder_browse.png"),
    }
)

DEFAULT_OUTPUT_FOLDER_HELP = """\
The *Default Output Folder* is accessible by pressing the “View output
settings” button at the bottom of the pipeline panel. The *Default Output
Folder* is the folder that CellProfiler uses to store the output file it
creates. Also, several File Processing modules (e.g., **SaveImages** or
**ExportToSpreadsheet**) provide the option of saving analysis results
to this folder on a default basis unless you specify, within the module,
an alternate, specific folder on your computer. Within modules, we
recommend selecting the Default Output Folder as much as possible, so
that your pipeline will work even if you transfer your images and
pipeline to a different computer. If, instead, you type specific folder
path names into a module’s settings, your pipeline will not work on
someone else’s computer until you adjust those pathnames within each
module.

Use the *Browse* button (to the right of the text box) to specify the
folder you would like to use as the Default Output Folder, or type the
full folder path in the edit box. If you type a folder path that cannot
be found, the message box below will indicate this fact until you
correct the problem. If you want to specify a folder that does not yet
exist, type the desired name and click on the *New folder* icon to the
right of the *Browse folder* icon. The folder will be created according
to the pathname you have typed.\
"""

ERROR_COLOR_HELP = "Sets the color used for the error alerts associated with misconfigured settings and other errors."

INTENSITY_MODE_HELP = """\
Sets the way CellProfiler normalizes pixel intensities when displaying.
If you choose “raw”, CellProfiler will display a pixel with a value of
“1” or above with the maximum brightness and a pixel with a value of “0”
or below as black. If you choose “normalize”, CellProfiler will find the
minimum and maximum intensities in the display image and show pixels at
maximum intensity with the maximum brightness and pixels at the minimum
intensity as black. This can be used to view dim images. If you choose
“log”, CellProfiler will use the full brightness range and will use a
log scale to scale the intensities. This can be used to view the image
background in more detail.
"""

INTERPOLATION_MODE_HELP = """\
Sets the way CellProfiler displays image pixels. If you choose
*Nearest*, CellProfiler will display each pixel as a square block of
uniform intensity. This is truest to the data, but the resulting images
look blocky and pixelated. You can choose either *Bilinear* or *Bicubic*
to see images where the a bilinear or bicubic spline model has been used
to interpolate the screen pixel value for screen pixels that do not fall
exactly in the center of the image pixel. The result, for bilinear or
bicubic interpolation is an image that is more visually appealing and
easier to interpret, but obscures the true pixel nature of the real
data.\
"""

JVM_HEAP_HELP = """\
Sets the maximum amount of memory that can be used by the Java virtual
machine. CellProfiler uses Java for loading images and for processing 
image sets. If you load extremely large images or process large image
set lists, you can use this option to start Java with a larger amount of 
memory. By default, CellProfiler starts Java with 512 MB, but you can 
override this by specifying the number of megabytes to load. You can 
also start CellProfiler from the command-line with the –jvm-heap-size
switch to get the same effect.\
"""

MAX_WORKERS_HELP = """\
Controls the maximum number of *workers* (i.e., copies of CellProfiler)
that will be started at the outset of an analysis run. CellProfiler uses
these copies to process multiple image sets in parallel, utilizing the
computer’s CPUs and memory fully. The default value is the number of
CPUs detected on your computer. Use fewer workers for pipelines that
require a large amount of memory. Use more workers for pipelines that
are accessing image data over a slow connection.

If using the **Groups** module, only one worker will be allocated to
handle each group. This means that you may have multiple workers
created, but only a subset of them may actually be active, depending on
the number of groups you have.\
"""

PLUGINS_DIRECTORY_HELP = """\
Chooses the directory that holds dynamically-loaded CellProfiler
modules. You can write your own module and place it in this directory
and CellProfiler will make it available for your pipeline. You must
restart CellProfiler after modifying this setting.\
"""

PRIMARY_OUTLINE_COLOR_HELP = """\
Sets the color used for the outline of the object of interest in the
**IdentifyPrimaryObjects**, **IdentifySecondaryObjects** and
**IdentifyTertiaryObjects** displays.\
"""

REPORT_JVM_ERROR_HELP = """\
Determines whether CellProfiler will display a warning on startup if
CellProfiler can’t locate the Java installation on your computer. Check
this box if you want to be warned. Uncheck this box to hide warnings.\
"""

SAVE_PIPELINE_WITH_PROJECT_HELP = """\
Controls whether a pipeline and/or file list file is saved whenever the
user saves the project file. Users may find it handy to have the
pipeline and/or file list saved in a readable format, for instance, for
version control whenever the project file is saved. Your project can be
restored by importing both the pipeline and file list, and your pipeline
can be run using a different file list, and your file list can be reused
by importing it into a different project. Note: When using LoadData, it
is not recommended to auto-save the file list, as this feature only
saves the file list existing in the Input Modules, not LoadData input
files.

-  *Neither:* Refrain from saving either file.
-  *Pipeline:* Save the pipeline, using the project’s file name and path
   and a .cppipe extension.
-  *File list:* Save the file list, using the project’s file name and
   path and a .txt extension.
-  *Pipeline and file list:* Save both files.\
"""

SECONDARY_OUTLINE_COLOR_HELP = """\
Sets the color used for objects other than the ones of interest. In
**IdentifyPrimaryObjects**, these are the objects that are too small or
too large. In **IdentifySecondaryObjects** and
**IdentifyTertiaryObjects**, this is the color of the secondary objects’
outline.\
"""

SHOW_ANALYSIS_COMPLETE_HELP = """\
Determines whether CellProfiler displays a message box at the end of a
run. Check this preference to show the message box or uncheck it to stop
display.\
"""

SHOW_EXITING_TEST_MODE_HELP = """\
Determines whether CellProfiler displays a message box to inform you
that a change made to the pipeline will cause test mode to end. Check
this preference to show the message box or uncheck it to stop display.\
"""

SHOW_REPORT_BAD_SIZES_DLG_HELP = """\
Determines whether CellProfiler will display a warning dialog if images
of different sizes are loaded together in an image set. Check this
preference to show the message box or uncheck it to stop display.\
"""

SHOW_SAMPLING_MENU_HELP = """\
Show the sampling menu

*Note that CellProfiler must be restarted after setting.*

The sampling menu is an interplace for Paramorama, a plugin for an
interactive visualization program for exploring the parameter space of
image analysis algorithms. will generate a text file, which specifies:
(1) all unique combinations of the sampled parameter values; (2) the
mapping from each combination of parameter values to one or more output
images; and (3) the actual output images.

More information on how to use the plugin can be found `here`_.

**References**

-  Visualization of parameter space for image analysis. Pretorius AJ,
   Bray MA, Carpenter AE and Ruddle RA. (2011) IEEE Transactions on
   Visualization and Computer Graphics, 17(12), 2402-2411.

.. _here: http://www.comp.leeds.ac.uk/scsajp/applications/paramorama2/
"""

SHOW_STARTUP_BLURB_HELP = (
    "Controls whether CellProfiler displays an orientation message on startup."
)

SHOW_TELEMETRY_HELP = """\
Allow limited and anonymous usage statistics and exception reports to be
sent to the CellProfiler team to help improve CellProfiler.\
"""

TABLE_FONT_HELP = "Sets the font used in tables displayed in module figure windows."
TERTIARY_OUTLINE_COLOR_HELP = """\
Sets the color used for the objects touching the image border or image
mask in **IdentifyPrimaryObjects**.\
"""

TEMP_DIR_HELP = """\
Sets the folder that CellProfiler uses when storing temporary files.
CellProfiler will create a temporary measurements file for analyses when
the user specifies that a MATLAB measurements file should be created or
when the user asks that no measurements file should be permanently
saved. CellProfiler will also save images accessed by http URL
temporarily to disk (but will efficiently access OMERO image planes
directly from the server).\
"""

UPDATER_HELP = """\
Allow CellProfiler to automatically check for and notify the user of new versions.
If enabled, CellProfiler will check GitHub once per week for updates.\
"""


NORMALIZATION_FACTOR_HELP = """\
Sets the normalization factor for intensity normalization methods:

-  *{INTENSITY_MODE_LOG}*: Set the gain applied to the each pixel in the image during log normalization. Pixels are
   transformed according to the formula `O = gain * log(1 + I)`, where `I` is the pixel intensity. Increasing the value
   of `gain` makes the displayed image appear brighter.
-  *{INTENSITY_MODE_GAMMA}*: Set the value of gamma. Pixels are transformed according to the formula `O = I ** gamma`,
   where I is the pixel intensity. For `gamma` > 1.0, the output image will appear darker than the original image. For
   `gamma` < 1.0, the output image will appear brighter than the original image.

The normalization factor is ignored when the normalization method is *{INTENSITY_MODE_NORMAL}* or 
*{INTENSITY_MODE_RAW}*.
""".format(
    **{
        "INTENSITY_MODE_GAMMA": INTENSITY_MODE_GAMMA,
        "INTENSITY_MODE_LOG": INTENSITY_MODE_LOG,
        "INTENSITY_MODE_NORMAL": INTENSITY_MODE_NORMAL,
        "INTENSITY_MODE_RAW": INTENSITY_MODE_RAW,
    }
)

ALWAYS_CONTINUE_HELP = """\
Determines whether CellProfiler should always skip images that 
cause errors to be raised vs stopping analysis (in headless mode)
or asking the user what to do (in GUI mode). This may help in 
processing large image sets with a small number of unusual or corrupted
images, but note that the skip of the original image set in question 
may cause issues with modules that look atmultiple image sets across 
groups (CorrectIlluminationCalculate, TrackObjects) or in Export modules,
leading to simply a later failure. Use at your own risk.
"""

WIDGET_INSPECTOR_HELP = """\
Enables wxPython Widget Inspection Tool under the "Test" menu. 
The tool displays a tree of wxWidgets and sizers in CellProfiler. 
Mostly only useful for debugging and development purposes.
"""


def recent_file(index, category=""):
    return (FF_RECENTFILES % (index + 1)) + category


"""All keys saved in the registry"""
ALL_KEYS = [
    BACKGROUND_COLOR,
    COLORMAP,
    DEFAULT_IMAGE_DIRECTORY,
    DEFAULT_OUTPUT_DIRECTORY,
    MODULEDIRECTORY,
    PLUGIN_DIRECTORY,
    PRIMARY_OUTLINE_COLOR,
    SECONDARY_OUTLINE_COLOR,
    SHOW_ANALYSIS_COMPLETE_DLG,
    SHOW_BAD_SIZES_DLG,
    SHOW_EXITING_TEST_MODE_DLG,
    WORKSPACE_CHOICE,
    SHOW_SAMPLING,
    SKIPVERSION,
    STARTUPBLURB,
    TELEMETRY,
    TELEMETRY_PROMPT,
    TABLE_FONT_NAME,
    TABLE_FONT_SIZE,
    TERTIARY_OUTLINE_COLOR,
    TITLE_FONT_NAME,
    TITLE_FONT_SIZE,
    WARN_ABOUT_OLD_PIPELINE,
    USE_MORE_FIGURE_SPACE,
    WORKSPACE_FILE,
    OMERO_SERVER,
    OMERO_PORT,
    OMERO_USER,
    SAVE_PIPELINE_WITH_PROJECT,
] + [
    recent_file(n, category)
    for n in range(RECENT_FILE_COUNT)
    for category in (
        "",
        DEFAULT_IMAGE_DIRECTORY,
        DEFAULT_OUTPUT_DIRECTORY,
        WORKSPACE_FILE,
    )
]


def module_directory():
    if not config_exists(MODULEDIRECTORY):
        return os.path.join(cell_profiler_root_directory(), "Modules")
    return str(config_read(MODULEDIRECTORY))


def set_module_directory(value):
    config_write(MODULEDIRECTORY, value)


def module_extension():
    return ".m"


__default_image_directory = None


def get_default_image_directory():
    global __default_image_directory

    if __default_image_directory is not None:
        return __default_image_directory
    # I'm not sure what it means for the preference not to exist.  No read-write preferences file?
    if not config_exists(DEFAULT_IMAGE_DIRECTORY):
        return os.path.abspath(os.path.expanduser("~"))
    # Fetch the default.  Note that it might be None
    default_image_directory = config_read(DEFAULT_IMAGE_DIRECTORY) or ""
    try:
        if os.path.isdir(default_image_directory):
            __default_image_directory = os.path.normcase(default_image_directory)
            return __default_image_directory
    except:
        LOGGER.error(
            "Unknown failure when retrieving the default image directory", exc_info=True
        )
    LOGGER.warning(
        "Warning: current path of %s is not a valid directory. Switching to home directory."
        % (default_image_directory.encode("ascii", "replace"))
    )
    # If the user's home directory is not ascii, we're not going to go hunting for one that is.
    # Fail ungracefully.
    default_image_directory = os.path.abspath(os.path.expanduser("~"))
    set_default_image_directory(default_image_directory)
    return str(os.path.normcase(default_image_directory))


def set_default_image_directory(path):
    global __default_image_directory
    __default_image_directory = path
    config_write(DEFAULT_IMAGE_DIRECTORY, path)
    add_recent_file(path, DEFAULT_IMAGE_DIRECTORY)
    fire_image_directory_changed_event()


def fire_image_directory_changed_event():
    """Notify listeners of a image directory change"""
    global __default_image_directory
    for listener in __image_directory_listeners:
        listener(PreferenceChangedEvent(__default_image_directory))


__image_directory_listeners = []


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


__default_output_directory = None


def get_default_output_directory():
    global __default_output_directory

    if __default_output_directory is not None:
        return __default_output_directory
    if not config_exists(DEFAULT_OUTPUT_DIRECTORY):
        return os.path.abspath(os.path.expanduser("~"))

    # Fetch the default.  Note that it might be None
    default_output_directory = config_read(DEFAULT_OUTPUT_DIRECTORY) or ""
    try:
        if os.path.isdir(default_output_directory):
            __default_output_directory = os.path.normcase(default_output_directory)
            return __default_output_directory
    except:
        LOGGER.error(
            "Unknown failure when retrieving the default output directory",
            exc_info=True,
        )
    LOGGER.warning(
        "Warning: current path of %s is not a valid directory. Switching to home directory."
        % (default_output_directory.encode("ascii", "replace"))
    )
    # If the user's home directory is not ascii, we're not going to go hunting for one that is.
    # Fail ungracefully.
    default_output_directory = os.path.abspath(os.path.expanduser("~"))
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


__output_directory_listeners = []


def add_output_directory_listener(listener):
    """Add a listener that will be notified when the output directory changes

    """
    __output_directory_listeners.append(listener)


def remove_output_directory_listener(listener):
    """Remove a previously-added image directory listener

    """
    if listener in __output_directory_listeners:
        __output_directory_listeners.remove(listener)


def get_title_font_size():
    if not config_exists(TITLE_FONT_SIZE):
        return 12
    title_font_size = config_read(TITLE_FONT_SIZE)
    return float(title_font_size)


def set_title_font_size(title_font_size):
    config_write(TITLE_FONT_SIZE, str(title_font_size))


def get_title_font_name():
    if not config_exists(TITLE_FONT_NAME):
        return "Tahoma"
    return config_read(TITLE_FONT_NAME)


def set_title_font_name(title_font_name):
    config_write(TITLE_FONT_NAME, title_font_name)


def get_table_font_name():
    if not config_exists(TABLE_FONT_NAME):
        return "Tahoma"
    return config_read(TABLE_FONT_NAME)


def set_table_font_name(title_font_name):
    config_write(TABLE_FONT_NAME, title_font_name)


def get_table_font_size():
    if not config_exists(TABLE_FONT_SIZE):
        return 9
    table_font_size = config_read(TABLE_FONT_SIZE)
    return float(table_font_size)


def set_table_font_size(table_font_size):
    config_write(TABLE_FONT_SIZE, str(table_font_size))


def tuple_to_color(t, default=(0, 0, 0)):
    import wx

    try:
        return wx.Colour(red=int(t[0]), green=int(t[1]), blue=int(t[2]))
    except IndexError as ValueError:
        return tuple_to_color(default)


def get_background_color():
    import wx

    return wx.SystemSettings.GetColour(wx.SYS_COLOUR_BACKGROUND)


def set_background_color(color):
    pass


__error_color = None


def get_error_color():
    """Get the color to be used for error text"""
    global __error_color
    #
    # Red found here:
    # http://www.jankoatwarpspeed.com/css-message-boxes-for-different-message-types/
    # but seems to be widely used.
    #
    default_color = (0xD8, 0x00, 0x0C)
    if __error_color is None:
        if not config_exists(ERROR_COLOR):
            __error_color = tuple_to_color(default_color)
        else:
            color_string = config_read(ERROR_COLOR)
            try:
                __error_color = tuple_to_color(color_string.split(","))
            except:
                print("Failed to parse error color string: " + color_string)
                traceback.print_exc()
                __error_color = default_color
    return __error_color


def set_error_color(color):
    """Set the color to be used for error text

    color - a WX color or ducktyped
    """
    global __error_color
    config_write(ERROR_COLOR, ",".join([str(x) for x in color.Get()]))
    __error_color = tuple_to_color(color.Get())


def get_pixel_size():
    """The size of a pixel in microns"""
    if not config_exists(PIXEL_SIZE):
        return 1.0
    return float(config_read(PIXEL_SIZE))


def set_pixel_size(pixel_size):
    config_write(PIXEL_SIZE, str(pixel_size))


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
        osep = "."
        isep = "&"
    elif abspath_mode == ABSPATH_IMAGE:
        osep = "&"
        isep = "."
    else:
        raise ValueError("Unknown abspath mode: %s" % abspath_mode)
    if is_url_path(path):
        return path
    if path.startswith(osep + os.path.sep) or (
        "altsep" in os.path.__all__
        and os.path.altsep
        and path.startswith(osep + os.path.altsep)
    ):
        return os.path.join(get_default_output_directory(), path[2:])
    elif path.startswith(isep + os.path.sep) or (
        "altsep" in os.path.__all__
        and os.path.altsep
        and path.startswith(isep + os.path.altsep)
    ):
        return os.path.join(get_default_image_directory(), path[2:])
    elif len(os.path.split(path)[0]) == 0:
        return os.path.join(get_default_output_directory(), path)
    else:
        return str(os.path.normpath(os.path.abspath(path)))


def is_url_path(path):
    """Return True if the path should be treated as a URL"""
    for protocol in ("http", "https", "ftp", "s3", "gs"):
        if path.lower().startswith("%s:" % protocol):
            return True
    return False


__default_colormap = None


def get_default_colormap():
    global __default_colormap
    if __default_colormap is None:
        if not config_exists(COLORMAP):
            __default_colormap = "jet"
        else:
            __default_colormap = config_read(COLORMAP)
    return __default_colormap


def set_default_colormap(colormap):
    global __default_colormap
    __default_colormap = colormap
    config_write(COLORMAP, colormap)


__current_workspace_path = None


def get_current_workspace_path():
    global __current_workspace_path
    return __current_workspace_path


def set_current_workspace_path(path):
    global __current_workspace_path
    __current_workspace_path = path


def get_skip_version():
    if not config_exists(SKIPVERSION):
        return 0
    return int(get_config().Read(SKIPVERSION))


def set_skip_version(ver):
    global __is_headless
    get_config().Write(SKIPVERSION, str(ver))
    if not __is_headless:
        get_config().Flush()


__show_sampling = None


def get_show_sampling():
    global __show_sampling
    if __show_sampling is not None:
        return __show_sampling
    if not config_exists(SHOW_SAMPLING):
        __show_sampling = False
        return False
    return get_config().ReadBool(SHOW_SAMPLING)


def set_show_sampling(value):
    global __show_sampling, __is_headless
    get_config().WriteBool(SHOW_SAMPLING, bool(value))
    __show_sampling = bool(value)
    if not __is_headless:
        get_config().Flush()


__recent_files: dict = {}


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


__plugin_directory = None
__conserve_memory = None


def get_plugin_directory():
    global __plugin_directory

    if __plugin_directory is not None:
        return __plugin_directory

    if config_exists(PLUGIN_DIRECTORY):
        __plugin_directory = config_read(PLUGIN_DIRECTORY)
    elif get_headless():
        return None
    elif config_exists("PluginDirectory"):
        # Import and store directory from CP3
        __plugin_directory = config_read("PluginDirectory")
        set_plugin_directory(__plugin_directory)
    else:
        import wx

        if wx.GetApp() is not None:
            __plugin_directory = os.path.join(
                wx.StandardPaths.Get().GetUserDataDir(), "plugins"
            )
    return __plugin_directory


def set_plugin_directory(value, globally=True):
    global __plugin_directory

    __plugin_directory = value
    if globally:
        config_write(PLUGIN_DIRECTORY, value)


__data_file = None


def get_data_file():
    """Get the path to the LoadData data file specified on the command-line"""
    global __data_file
    return __data_file


def set_data_file(path):
    global __data_file
    __data_file = path


def standardize_default_folder_names(setting_values, slot):
    if setting_values[slot] in list(FOLDER_CHOICE_TRANSLATIONS.keys()):
        replacement = FOLDER_CHOICE_TRANSLATIONS[setting_values[slot]]
    elif (
        setting_values[slot].startswith("Default Image")
        or setting_values[slot].startswith("Default image")
        or setting_values[slot].startswith("Default input")
    ):
        replacement = DEFAULT_INPUT_FOLDER_NAME
    elif setting_values[slot].startswith("Default output"):
        replacement = DEFAULT_OUTPUT_FOLDER_NAME
    else:
        replacement = setting_values[slot]
    setting_values = setting_values[:slot] + [replacement] + setting_values[slot + 1 :]
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
    """Called by get_next_cpfigure_position to update the screen position at
    which the next figure frame will be drawn.
    """
    global __cpfigure_position
    import wx

    win_size = (600, 400)
    try:
        disp = wx.GetDisplaySize()
    except:
        disp = (800, 600)
    if __cpfigure_position[0] + win_size[0] > disp[0]:
        __cpfigure_position = (-1, __cpfigure_position[1])
    if __cpfigure_position[1] + win_size[1] > disp[1]:
        __cpfigure_position = (-1, -1)
    else:
        # These offsets could be set in the preferences UI
        __cpfigure_position = (
            __cpfigure_position[0] + 120,
            __cpfigure_position[1] + 24,
        )


def get_telemetry():
    if not config_exists(TELEMETRY):
        return True

    return get_config().ReadBool(TELEMETRY)


def set_telemetry(val):
    global __is_headless
    get_config().WriteBool(TELEMETRY, val)
    if not __is_headless:
        get_config().Flush()


def get_telemetry_prompt():
    if not config_exists(TELEMETRY_PROMPT):
        return True

    return get_config().ReadBool(TELEMETRY_PROMPT)


def set_telemetry_prompt(val):
    global __is_headless
    get_config().WriteBool(TELEMETRY_PROMPT, val)
    if not __is_headless:
        get_config().Flush()


def get_check_update():
    if not config_exists(CHECK_FOR_UPDATES):
        return "Never"
    return get_config().Read(CHECK_FOR_UPDATES)


def get_check_update_bool():
    if not config_exists(CHECK_FOR_UPDATES):
        return True
    update_str = get_config().Read(CHECK_FOR_UPDATES)
    if update_str == "Disabled":
        return False
    else:
        return True


def set_check_update(val):
    if str(val) == "False":
        val = "Disabled"
    elif str(val) == "True":
        val = "Never"
    global __is_headless
    get_config().Write(CHECK_FOR_UPDATES, val)
    if not __is_headless:
        get_config().Flush()


def get_startup_blurb():
    if not config_exists(STARTUPBLURB):
        return True
    return get_config().ReadBool(STARTUPBLURB)


def set_startup_blurb(val):
    global __is_headless
    get_config().WriteBool(STARTUPBLURB, val)
    if not __is_headless:
        get_config().Flush()


def get_primary_outline_color():
    default = (0, 255, 0)
    if not config_exists(PRIMARY_OUTLINE_COLOR):
        return tuple_to_color(default)
    return tuple_to_color(config_read(PRIMARY_OUTLINE_COLOR).split(","))


def set_primary_outline_color(color):
    config_write(PRIMARY_OUTLINE_COLOR, ",".join([str(x) for x in color.Get()]))


def get_secondary_outline_color():
    default = (255, 0, 255)
    if not config_exists(SECONDARY_OUTLINE_COLOR):
        return tuple_to_color(default)
    return tuple_to_color(config_read(SECONDARY_OUTLINE_COLOR).split(","))


def set_secondary_outline_color(color):
    config_write(SECONDARY_OUTLINE_COLOR, ",".join([str(x) for x in color.Get()]))


def get_tertiary_outline_color():
    default = (255, 128, 0)
    if not config_exists(TERTIARY_OUTLINE_COLOR):
        return tuple_to_color(default)
    return tuple_to_color(config_read(TERTIARY_OUTLINE_COLOR).split(","))


def set_tertiary_outline_color(color):
    config_write(TERTIARY_OUTLINE_COLOR, ",".join([str(x) for x in color.Get()]))


__has_reported_jvm_error = False


def get_report_jvm_error():
    """Return true if user still wants to report a JVM error"""
    if __has_reported_jvm_error:
        return False
    if not config_exists(JVM_ERROR):
        return True
    return config_read(JVM_ERROR) == "True"


def set_report_jvm_error(should_report):
    config_write(JVM_ERROR, "True" if should_report else "False")


def set_has_reported_jvm_error():
    """Call this to remember that we showed the user the JVM error"""
    global __has_reported_jvm_error
    __has_reported_jvm_error = True


# "Analysis complete" preference
__show_analysis_complete_dlg = None


def get_show_analysis_complete_dlg():
    """Return true if the user wants to see the "analysis complete" dialog"""
    global __show_analysis_complete_dlg
    if __show_analysis_complete_dlg is not None:
        return __show_analysis_complete_dlg
    if not config_exists(SHOW_ANALYSIS_COMPLETE_DLG):
        return True
    return config_read(SHOW_ANALYSIS_COMPLETE_DLG) == "True"


def set_show_analysis_complete_dlg(value):
    """Set the "show analysis complete" flag"""
    global __show_analysis_complete_dlg
    __show_analysis_complete_dlg = value
    config_write(SHOW_ANALYSIS_COMPLETE_DLG, "True" if value else "False")


# "Existing test mode" preference
__show_exiting_test_mode_dlg = None


def get_show_exiting_test_mode_dlg():
    """Return true if the user wants to see the "exiting test mode" dialog"""
    global __show_exiting_test_mode_dlg
    if __show_exiting_test_mode_dlg is not None:
        return __show_exiting_test_mode_dlg
    if not config_exists(SHOW_EXITING_TEST_MODE_DLG):
        return True
    return config_read(SHOW_EXITING_TEST_MODE_DLG) == "True"


def set_show_exiting_test_mode_dlg(value):
    """Set the "exiting test mode" flag"""
    global __show_exiting_test_mode_dlg
    __show_exiting_test_mode_dlg = value
    config_write(SHOW_EXITING_TEST_MODE_DLG, "True" if value else "False")


# "Report bad sizes" preference
__show_report_bad_sizes_dlg = None


def get_show_report_bad_sizes_dlg():
    """Return true if the user wants to see the "report bad sizes" dialog"""
    global __show_report_bad_sizes_dlg
    if __show_report_bad_sizes_dlg is not None:
        return __show_report_bad_sizes_dlg
    if not config_exists(SHOW_BAD_SIZES_DLG):
        return True
    return config_read(SHOW_BAD_SIZES_DLG) == "True"


def set_show_report_bad_sizes_dlg(value):
    """Set the "exiting test mode" flag"""
    global __show_report_bad_sizes_dlg
    __show_report_bad_sizes_dlg = value
    config_write(SHOW_BAD_SIZES_DLG, "True" if value else "False")


__workspace_file = None


def get_workspace_file():
    """Return the path to the workspace file"""
    global __workspace_file
    if __workspace_file is not None:
        return __workspace_file
    if not config_exists(WORKSPACE_FILE):
        return None
    __workspace_file = config_read(WORKSPACE_FILE)
    return __workspace_file


def set_workspace_file(path, permanently=True):
    """Set the path to the workspace file

    path - path to the file

    permanently - True to write it to the configuration, False if the file
                  should only be set for the running instance (e.g., as a
                  command-line parameter for a scripted run)
    """
    global __workspace_file
    __workspace_file = path
    if permanently:
        add_recent_file(path, WORKSPACE_FILE)
        config_write(WORKSPACE_FILE, path)


###########################################
#
# OMERO logon credentials
#
###########################################

__omero_server = None
__omero_port = None
__omero_user = None
__omero_session_id = None


def get_omero_server():
    """Get the DNS name of the Omero server"""
    global __omero_server
    if __omero_server is None:
        if not config_exists(OMERO_SERVER):
            return None
        __omero_server = config_read(OMERO_SERVER)
    return __omero_server


def set_omero_server(omero_server, globally=True):
    """Set the DNS name of the Omero server"""
    global __omero_server
    __omero_server = omero_server
    if globally:
        config_write(OMERO_SERVER, omero_server)


def get_omero_port():
    """Get the port used to connect to the Omero server"""
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
    """Set the port used to connect to the Omero server"""
    global __omero_port
    __omero_port = omero_port
    if globally:
        config_write(OMERO_PORT, str(omero_port))


def get_omero_user():
    """Get the Omero user name"""
    global __omero_user
    if __omero_user is None:
        if not config_exists(OMERO_USER):
            return None
        __omero_user = config_read(OMERO_USER)
    return __omero_user


def set_omero_user(omero_user, globally=True):
    """Set the Omero user name"""
    global __omero_user
    __omero_user = omero_user
    if globally:
        config_write(OMERO_USER, omero_user)


def get_omero_session_id():
    """Get the session ID to use to communicate to Omero"""
    global __omero_session_id
    if __omero_session_id is None:
        if not config_exists(OMERO_SESSION_ID):
            return None
        __omero_session_id = config_read(OMERO_SESSION_ID)
    return __omero_session_id


def set_omero_session_id(omero_session_id, globally=True):
    """Set the Omero session ID"""
    global __omero_session_id
    __omero_session_id = omero_session_id
    if globally:
        config_write(OMERO_SESSION_ID, omero_session_id)


def default_max_workers():
    return psutil.cpu_count(logical=False)


__max_workers = None


def get_max_workers():
    """Get the maximum number of worker processes allowed during analysis"""
    global __max_workers
    if __max_workers is not None:
        return __max_workers
    default = default_max_workers()
    if config_exists(MAX_WORKERS):
        __max_workers = get_config().ReadInt(MAX_WORKERS, default)
        return __max_workers
    return default


def set_max_workers(value):
    """Set the maximum number of worker processes allowed during analysis"""
    global __max_workers, __is_headless
    get_config().WriteInt(MAX_WORKERS, value)
    __max_workers = value
    if not __is_headless:
        get_config().Flush()


__temp_dir = None


def get_temporary_directory():
    """Get the directory to be used for temporary files

    The default is whatever is returned by tempfile.gettempdir()
    (see http://docs.python.org/2/library/tempfile.html#tempfile.gettempdir)
    """
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
    """Set the directory to be used for temporary files

    tempdir - pathname of the directory
    """
    global __temp_dir
    if globally:
        config_write(TEMP_DIR, tempdir)
    __temp_dir = tempdir
    tempfile.tempdir = tempdir


__progress_data = threading.local()
__progress_data.last_report = time.time()
__progress_data.callbacks = None
__interpolation_mode = None


def get_interpolation_mode():
    """Get the interpolation mode for matplotlib

    Returns one of IM_NEAREST, IM_BILINEAR or IM_BICUBIC
    """
    global __interpolation_mode
    if __interpolation_mode is not None:
        return __interpolation_mode
    if config_exists(INTERPOLATION_MODE):
        __interpolation_mode = config_read(INTERPOLATION_MODE)
    else:
        __interpolation_mode = IM_NEAREST
    return __interpolation_mode


def set_interpolation_mode(value):
    global __interpolation_mode
    __interpolation_mode = value
    config_write(INTERPOLATION_MODE, value)


__intensity_mode = None


def get_intensity_mode():
    """Get the intensity scaling mode for matplotlib

    Returns one of INTENSITY_MODE_RAW, INTENSITY_MODE_NORMAL, INTENSITY_MODE_LOG
    """
    global __intensity_mode
    if __intensity_mode is not None:
        return __intensity_mode
    if config_exists(INTENSITY_MODE):
        __intensity_mode = config_read(INTENSITY_MODE)
    else:
        __intensity_mode = INTENSITY_MODE_NORMAL
    return __intensity_mode


def set_intensity_mode(value):
    """Set the intensity scaling mode for matplotlib"""
    global __intensity_mode
    __intensity_mode = value
    config_write(INTENSITY_MODE, value)


__save_pipeline_with_project = None


def get_save_pipeline_with_project():
    global __save_pipeline_with_project
    if __save_pipeline_with_project is None:
        if config_exists(SAVE_PIPELINE_WITH_PROJECT):
            __save_pipeline_with_project = config_read(SAVE_PIPELINE_WITH_PROJECT)
        else:
            __save_pipeline_with_project = SPP_NEITHER
    return __save_pipeline_with_project


def set_save_pipeline_with_project(value):
    global __save_pipeline_with_project
    __save_pipeline_with_project = value
    config_write(SAVE_PIPELINE_WITH_PROJECT, value)


__allow_schema_write = True


def get_allow_schema_write():
    """Returns True if ExportToDatabase is allowed to write the MySQL schema

    For cluster operation without CreateBatchFiles, it's inappropriate to
    have multiple processes overwrite the database schema. Although
    CreateBatchFiles is suggested for this scenario, we put this switch in
    to support disabling schema writes from the command line.
    """
    return __allow_schema_write


def set_allow_schema_write(value):
    """Allow or disallow database schema writes

    value - True to allow writes (the default) or False to prevent
            ExportToDatabase from writing the schema.

    For cluster operation without CreateBatchFiles, it's inappropriate to
    have multiple processes overwrite the database schema. Although
    CreateBatchFiles is suggested for this scenario, we put this switch in
    to support disabling schema writes from the command line.
    """
    global __allow_schema_write
    __allow_schema_write = value


__filename_re_guess_file = None


def get_filename_re_guess_file():
    """The path to the file that contains filename regular expression guesses

    The file given by this preference is an optional file that contains
    possible regular expression patterns to match against file names.
    """
    global __filename_re_guess_file
    if __filename_re_guess_file is None:
        if config_exists(FILENAME_RE_GUESSES_FILE):
            __filename_re_guess_file = config_read(FILENAME_RE_GUESSES_FILE)
    return __filename_re_guess_file


def set_filename_re_guess_file(value):
    """Set the path to the filename regular expression guess file"""
    global __filename_re_guess_file
    __filename_re_guess_file = value
    config_write(FILENAME_RE_GUESSES_FILE, value)


__pathname_re_guess_file = None


def get_pathname_re_guess_file():
    """The path to the file that contains pathname regular expression guesses

    The file given by this preference is an optional file that contains
    possible regular expression patterns to match against path names.
    """
    global __pathname_re_guess_file
    if __pathname_re_guess_file is None:
        if config_exists(PATHNAME_RE_GUESSES_FILE):
            __pathname_re_guess_file = config_read(PATHNAME_RE_GUESSES_FILE)
    return __pathname_re_guess_file


def set_pathname_re_guess_file(value):
    """Set the path to the pathname regular expression guess file"""
    global __pathname_re_guess_file
    __pathname_re_guess_file = value
    config_write(PATHNAME_RE_GUESSES_FILE, value)


__image_set_filename = None

__wants_pony = None


def get_wants_pony():
    """

    :return:

    """
    global __wants_pony

    if __wants_pony is not None:
        return __wants_pony
    elif config_exists("Pony"):
        return config_read("Pony").lower() == "yes"
    else:
        return False


def set_wants_pony(wants_pony):
    """

    :param wants_pony:

    """
    global __wants_pony

    __wants_pony = wants_pony

    config_write("Pony", "yes" if wants_pony else "no")


def set_image_set_file(filename):
    """Record the name of the image set that should be loaded upon startup"""
    global __image_set_filename
    __image_set_filename = filename


def clear_image_set_file():
    """Remove the recorded image set file name

    Call this after loading the image set file to cancel reloading of the
    file during subsequent operations.
    """
    global __image_set_filename
    __image_set_filename = None


def get_image_set_file():
    """Recover the name of the image set file to use to populate the file list

    Returns either None or the name of the file to use. For the UI, the
    file list should be loaded and clear_image_set_file() should be called,
    for headless, the file list should be loaded after the pipeline has been
    loaded.
    """
    return __image_set_filename


__choose_image_set_frame_size = None


def get_choose_image_set_frame_size():
    """Return the size (w, h) for the "Choose image set" dialog frame"""
    global __choose_image_set_frame_size
    if __choose_image_set_frame_size is None:
        if config_exists(CHOOSE_IMAGE_SET_FRAME_SIZE):
            s = config_read(CHOOSE_IMAGE_SET_FRAME_SIZE)
            __choose_image_set_frame_size = tuple(
                [int(_.strip()) for _ in s.split(",", 1)]
            )
    return __choose_image_set_frame_size


def set_choose_image_set_frame_size(w, h):
    """Set the size of the "Choose image set" dialog frame"""
    global __choose_image_set_frame_size
    __choose_image_set_frame_size = (w, h)
    config_write(CHOOSE_IMAGE_SET_FRAME_SIZE, "%d,%d" % (w, h))


__normalization_factor = None


def get_normalization_factor():
    global __normalization_factor
    if __normalization_factor is not None:
        return __normalization_factor
    if config_exists(NORMALIZATION_FACTOR):
        __normalization_factor = config_read(NORMALIZATION_FACTOR)
    else:
        __normalization_factor = "1.0"
    return __normalization_factor


def set_normalization_factor(normalization_factor):
    global __normalization_factor
    try:
        float(normalization_factor)
    except ValueError:
        print(
            f"Unable to set {NORMALIZATION_FACTOR} to {normalization_factor}, value must be a number."
        )
        return
    __normalization_factor = normalization_factor
    config_write(NORMALIZATION_FACTOR, normalization_factor)


def get_conserve_memory():
    global __conserve_memory
    if __conserve_memory is not None:
        return __conserve_memory in (True, "True")
    if not config_exists(CONSERVE_MEMORY):
        return False
    return get_config().ReadBool(CONSERVE_MEMORY)


def set_conserve_memory(val, globally=True):
    global __conserve_memory
    __conserve_memory = val
    if globally:
        config_write(CONSERVE_MEMORY, val)


def add_progress_callback(callback):
    """Add a callback function that listens to progress calls

    The progress indicator is designed to monitor progress of operations
    on the user interface thread. The model is that operations are nested
    so that both an operation and sub-operation can report their progress.
    An operation reports its initial progress and is pushed onto the
    stack at that point. When it reports 100% progress, it's popped from
    the stack.

    callback - callback function with signature of
               fn(operation_id, progress, message)
               where operation_id names the instance of the operation being
               performed (e.g., a UUID), progress is a number between 0 and 1
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
    """
    global __progress_data
    if __progress_data.callbacks is None:
        __progress_data.callbacks = weakref.WeakSet()
    __progress_data.callbacks.add(callback)


def remove_progress_callback(callback):
    global __progress_data
    if __progress_data.callbacks is not None and callback in __progress_data.callbacks:
        __progress_data.callbacks.remove(callback)


def report_progress(operation_id, progress, message):
    """Report progress to all callbacks registered on the caller's thread

    operation_id - ID of operation being performed

    progress - a number between 0 and 1 indicating the extent of progress.
               None indicates indeterminate operation duration. 0 should be
               reported at the outset and 1 at the end.

    message - an informative message.
    """
    global __progress_data
    if __progress_data.callbacks is None:
        return
    t = time.time()
    if progress in (None, 0, 1) or t - __progress_data.last_report > 1:
        for callback in __progress_data.callbacks:
            callback(operation_id, progress, message)
        __progress_data.last_report = time.time()


def map_report_progress(fn_map, fn_report, sequence, freq=None):
    """Apply a mapping function to a sequence, reporting progress

    fn_map - function that maps members of the sequence to members of the output

    fn_report - function that takes a sequence member and generates an
                informative string

    freq - report on mapping every N items. Default is to report 100 or less
           times.
    """
    n_items = len(sequence)
    if n_items == 0:
        return []
    if freq is None:
        if n_items < 100:
            freq = 1
        else:
            freq = (n_items + 99) / 100
    output = []
    uid = uuid.uuid4()
    for i in range(0, n_items, freq):
        report_progress(uuid, float(i) / n_items, fn_report(sequence[i]))
        output += list(map(fn_map, sequence[i : i + freq]))
    report_progress(uuid, 1, "Done")
    return output


def cancel_progress():
    """Cancel all progress indicators

    for instance, after an exception is thrown that bubbles to the top.
    """
    report_progress(None, None, None)

__always_continue = None
def get_always_continue():
    global __always_continue
    if __always_continue is not None:
        return __always_continue in (True, "True")
    if not config_exists(ALWAYS_CONTINUE):
        return False
    return get_config().ReadBool(ALWAYS_CONTINUE)


def set_always_continue(val, globally=True):
    global __always_continue
    __always_continue = val
    if globally:
        config_write(ALWAYS_CONTINUE, val)

__widget_inspector = None
# global_only - only return True if local is set
# ie ignore config settings
def get_widget_inspector(global_only=False):
    global __widget_inspector
    if not global_only and __widget_inspector is not None:
        return __widget_inspector == True
    if not config_exists(WIDGET_INSPECTOR):
        return False
    return get_config().ReadBool(WIDGET_INSPECTOR)


def set_widget_inspector(val, globally=True):
    global __widget_inspector
    __widget_inspector = val
    if globally:
        config_write(WIDGET_INSPECTOR, val)
