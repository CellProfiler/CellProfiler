"""Preferences.py - singleton preferences for CellProfiler

   TO-DO - load the default preferences from somewhere.
           Create a function to save the preferences.
           Create a function to populate a handles structure with preferences.

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import logging
import random
import cellprofiler
import os
import os.path
import re
import sys
import traceback
from cellprofiler.utilities.utf16encode import utf16encode, utf16decode

logger = logging.getLogger(__name__)

from cellprofiler.utilities.get_proper_case_filename import get_proper_case_filename

'''get_absolute_path - mode = output. Assume "." is the default output dir'''
ABSPATH_OUTPUT = 'abspath_output'

'''get_absolute_path - mode = image. Assume "." is the default input dir'''
ABSPATH_IMAGE = 'abspath_image'

__python_root = os.path.split(str(cellprofiler.__path__[0]))[0]
__cp_root = os.path.split(__python_root)[0]

class HeadlessConfig(object):
    def __init__(self):
        self.__preferences = {}
    
    def Read(self, kwd):
        return self.__preferences[kwd]
    
    def Write(self, kwd, value):
        self.__preferences[kwd] = value
    
    def Exists(self, kwd):
        return self.__preferences.has_key(kwd)

__is_headless = False
__headless_config = HeadlessConfig()

def set_headless():
    global __is_headless
    __is_headless = True
    
def get_headless():
    return __is_headless

def get_config():
    global __is_headless,__headless_config
    if __is_headless:
        return __headless_config
    import wx
    try:
        config = wx.Config.Get(False)
    except wx.PyNoAppError:
        app = wx.App(0)
        config = wx.Config.Get(False)
    if not config:
        wx.Config.Set(wx.Config('CellProfiler','BroadInstitute','CellProfilerLocal.cfg','CellProfilerGlobal.cfg',wx.CONFIG_USE_LOCAL_FILE))
        config = wx.Config.Get()
        if not config.Exists(PREFERENCES_VERSION):
            for key in ALL_KEYS:
                if config.Exists(key):
                    v = config.Read(key)
                    config_write(key, v)
            config_write(PREFERENCES_VERSION, str(PREFERENCES_VERSION_NUMBER))
        else:
            try:
                preferences_version_number = int(config_read(PREFERENCES_VERSION))
                if preferences_version_number != PREFERENCES_VERSION_NUMBER:
                    logger.warning(
                        "Preferences version mismatch: expected %d, at %d" %
                        ( PREFERENCES_VERSION_NUMBER, preferences_version_number))
            except:
                logger.warning(
                    "Preferences version was %s, not a number. Resetting to current version" % preferences_version_number)
                config_write(PREFERENCES_VERSION, str(PREFERENCES_VERSION))
            
    return config

def config_read(key):
    '''Read the given configuration value
    
    Decode escaped config sequences too.
    '''
    value = get_config().Read(key)
    if value is None:
        return None
    return utf16decode(value)

def config_write(key, value):
    '''Write the given configuration value
    
    Encode escaped config sequences.
    '''
    if value is not None:
        value = utf16encode(value)
    get_config().Write(key, value)

def cell_profiler_root_directory():
    if __cp_root:
        return __cp_root
    return '..'

def python_root_directory():
    return __python_root

def resources_root_directory():
    if hasattr(sys, 'frozen'):
        # On Mac, the application runs in CellProfiler2.0.app/Contents/Resources.
        # Not sure where this should be on PC.
        return '.'
    else:
        return __python_root

    
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
images via <b>LoadImages</b> or <b>LoadData</b>, you can name the folder using metadata
tags."""%globals()

PREFERENCES_VERSION = 'PreferencesVersion'
PREFERENCES_VERSION_NUMBER = 1
DEFAULT_IMAGE_DIRECTORY = 'DefaultImageDirectory'
DEFAULT_OUTPUT_DIRECTORY = 'DefaultOutputDirectory'
TITLE_FONT_SIZE = 'TitleFontSize'
TITLE_FONT_NAME = 'TitleFontName'
TABLE_FONT_NAME = 'TableFontName'
TABLE_FONT_SIZE = 'TableFontSize'
BACKGROUND_COLOR = 'BackgroundColor'
PIXEL_SIZE = 'PixelSize'
COLORMAP = 'Colormap'
MODULEDIRECTORY = 'ModuleDirectory'
CHECKFORNEWVERSIONS = 'CheckForNewVersions'
SKIPVERSION = 'SkipVersion'
FF_RECENTFILES = 'RecentFile%d'
STARTUPBLURB = 'StartupBlurb'
RECENT_FILE_COUNT = 10
PRIMARY_OUTLINE_COLOR = 'PrimaryOutlineColor'
SECONDARY_OUTLINE_COLOR = 'SecondaryOutlineColor'
TERTIARY_OUTLINE_COLOR = 'TertiaryOutlineColor'
JVM_ERROR = 'JVMError'
ALLOW_OUTPUT_FILE_OVERWRITE = 'AllowOutputFileOverwrite'
PLUGIN_DIRECTORY = 'PluginDirectory'
IJ_PLUGIN_DIRECTORY = 'IJPluginDirectory'
SHOW_ANALYSIS_COMPLETE_DLG = "ShowAnalysisCompleteDlg"
SHOW_EXITING_TEST_MODE_DLG = "ShowExitingTestModeDlg"
SHOW_BAD_SIZES_DLG = "ShowBadSizesDlg"
SHOW_SAMPLING = "ShowSampling"
WRITE_MAT = "WriteMAT"
RUN_DISTRIBUTED = "RunDistributed"
WARN_ABOUT_OLD_PIPELINE = "WarnAboutOldPipeline"
USE_MORE_FIGURE_SPACE = "UseMoreFigureSpace"

def recent_file(index, category=""):
    return (FF_RECENTFILES % (index + 1)) + category


ALL_KEYS = ([ALLOW_OUTPUT_FILE_OVERWRITE, BACKGROUND_COLOR, CHECKFORNEWVERSIONS,
             COLORMAP, DEFAULT_IMAGE_DIRECTORY, DEFAULT_OUTPUT_DIRECTORY,
             IJ_PLUGIN_DIRECTORY, MODULEDIRECTORY, PLUGIN_DIRECTORY,
             PRIMARY_OUTLINE_COLOR, RUN_DISTRIBUTED, SECONDARY_OUTLINE_COLOR,
             SHOW_ANALYSIS_COMPLETE_DLG, SHOW_BAD_SIZES_DLG, 
             SHOW_EXITING_TEST_MODE_DLG, SHOW_SAMPLING, SKIPVERSION, STARTUPBLURB,
             TABLE_FONT_NAME, TABLE_FONT_SIZE, TERTIARY_OUTLINE_COLOR,
             TITLE_FONT_NAME, TITLE_FONT_SIZE, WARN_ABOUT_OLD_PIPELINE,
             WRITE_MAT, USE_MORE_FIGURE_SPACE] + 
            [recent_file(n, category) for n in range(RECENT_FILE_COUNT)
             for category in ("", DEFAULT_IMAGE_DIRECTORY, DEFAULT_OUTPUT_DIRECTORY)])

def module_directory():
    if not get_config().Exists(MODULEDIRECTORY):
        return os.path.join(cell_profiler_root_directory(), 'Modules')
    return str(get_config().Read(MODULEDIRECTORY))

def set_module_directory(value):
    config_write(MODULEDIRECTORY, value)

def module_extension():
    return '.m'

__default_image_directory = None
def get_default_image_directory():
    global __default_image_directory
    if __default_image_directory is not None:
        return __default_image_directory
    # I'm not sure what it means for the preference not to exist.  No read-write preferences file?
    if not get_config().Exists(DEFAULT_IMAGE_DIRECTORY):
        return os.path.abspath(os.path.expanduser('~'))
    # fetch the default.  Note that it might be None
    default_image_directory = config_read(DEFAULT_IMAGE_DIRECTORY) or ''
    try:
        if os.path.isdir(default_image_directory):
            __default_image_directory = get_proper_case_filename(default_image_directory)
            return __default_image_directory
    except:
        logger.error("Unknown failure when retrieving the default image directory", exc_info=True)
    logger.warning("Warning: current path of %s is not a valid directory. Switching to home directory."%(default_image_directory.encode('ascii', 'replace')))
    # If the user's home directory is not ascii, we're not going to go hunting for one that is.
    # Fail ungracefully.
    default_image_directory = os.path.abspath(os.path.expanduser('~'))
    set_default_image_directory(default_image_directory)
    return str(get_proper_case_filename(default_image_directory))

def set_default_image_directory(path):
    global __default_image_directory
    __default_image_directory = path
    config_write(DEFAULT_IMAGE_DIRECTORY,path)
    add_recent_file(path, DEFAULT_IMAGE_DIRECTORY)
    fire_image_directory_changed_event()
    
def fire_image_directory_changed_event():
    '''Notify listeners of a image directory change'''
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
    __image_directory_listeners.remove(listener)

class PreferenceChangedEvent:
    def __init__(self, new_value):
        self.new_value = new_value

__default_output_directory = None
def get_default_output_directory():
    global __default_output_directory
    if __default_output_directory is not None:
        return __default_output_directory
    if not get_config().Exists(DEFAULT_OUTPUT_DIRECTORY):
        return os.path.abspath(os.path.expanduser('~'))
    __default_output_directory = config_read(DEFAULT_OUTPUT_DIRECTORY)
    __default_output_directory = get_proper_case_filename(__default_output_directory)
    return __default_output_directory

def set_default_output_directory(path):
    global __default_output_directory
    assert os.path.isdir(path),'Default Output Folder, "%s", is not a directory'%(path)
    __default_output_directory = path
    config_write(DEFAULT_OUTPUT_DIRECTORY,path)
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
    __output_directory_listeners.remove(listener)

def get_title_font_size():
    if not get_config().Exists(TITLE_FONT_SIZE):
        return 12
    title_font_size = config_read(TITLE_FONT_SIZE)
    return float(title_font_size)

def set_title_font_size(title_font_size):
    config_write(TITLE_FONT_SIZE,str(title_font_size))

def get_title_font_name():
    if not get_config().Exists(TITLE_FONT_NAME):
        return "Tahoma"
    return config_read(TITLE_FONT_NAME)

def set_title_font_name(title_font_name):
    config_write(TITLE_FONT_NAME, title_font_name)

def get_table_font_name():
    if not get_config().Exists(TABLE_FONT_NAME):
        return "Tahoma"
    return config_read(TABLE_FONT_NAME)

def set_table_font_name(title_font_name):
    config_write(TABLE_FONT_NAME, title_font_name)
    
def get_table_font_size():
    if not get_config().Exists(TABLE_FONT_SIZE):
        return 9
    table_font_size = config_read(TABLE_FONT_SIZE)
    return float(table_font_size)

def set_table_font_size(table_font_size):
    config_write(TABLE_FONT_SIZE,str(table_font_size))

def tuple_to_color(t, default = (0,0,0)):
    import wx
    try:
        return wx.Colour(red=int(t[0]), green = int(t[1]), blue = int(t[2]))
    except IndexError, ValueError:
        return tuple_to_color(default)
    
def get_background_color():
    '''Get the color to be used for window backgrounds
    
    Return wx.Colour that will be applied as
    the background for all frames and dialogs
    '''
    default_color = (143, 188, 143) # darkseagreen
    if not get_config().Exists(BACKGROUND_COLOR):
        return tuple_to_color(default_color)
    else:
        color = config_read(BACKGROUND_COLOR).split(',')
        return tuple_to_color(tuple(color), default_color)

def set_background_color(color):
    '''Set the color to be used for window backgrounds
    
    '''
    config_write(BACKGROUND_COLOR,
                       ','.join([str(x) for x in color.Get()]))

def get_pixel_size():
    """The size of a pixel in microns"""
    if not get_config().Exists(PIXEL_SIZE):
        return 1.0
    return float(config_read(PIXEL_SIZE))

def set_pixel_size(pixel_size):
    config_write(PIXEL_SIZE,str(pixel_size))

__output_filename = 'DefaultOUT.mat'
__output_filename_listeners = []
def get_output_file_name():
    return __output_filename

def set_output_file_name(filename):
    global __output_filename
    filename=str(filename)
    __output_filename = filename
    for listener in __output_filename_listeners:
        listener(PreferenceChangedEvent(filename))

def add_output_file_name_listener(listener):
    __output_filename_listeners.append(listener)

def remove_output_file_name_listener(listener):
    __output_filename_listeners.remove(listener)

def get_absolute_path(path, abspath_mode = ABSPATH_IMAGE):
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
        raise ValueError("Unknown abspath mode: %s"%abspath_mode)
    if is_url_path(path):
        return path
    if (path.startswith(osep+os.path.sep) or
        ("altsep" in os.path.__all__ and os.path.altsep and
         path.startswith(osep+os.path.altsep))):
        return os.path.join(get_default_output_directory(), path[2:])
    elif (path.startswith(isep+os.path.sep) or
          ("altsep" in os.path.__all__ and os.path.altsep and
           path.startswith(isep+os.path.altsep))):
        return os.path.join(get_default_image_directory(), path[2:])
    elif len(os.path.split(path)[0]) == 0:
        return os.path.join(get_default_output_directory(), path)
    else:
        return str(get_proper_case_filename(os.path.abspath(path)))

def is_url_path(path):
    '''Return True if the path should be treated as a URL'''
    for protocol in ('http','https','ftp'):
        if path.lower().startswith('%s:' % protocol):
            return True
    return False

def get_default_colormap():
    if not get_config().Exists(COLORMAP):
        return 'jet'
    return config_read(COLORMAP)

def set_default_colormap(colormap):
    config_write(COLORMAP, colormap)

__current_pipeline_path = None
def get_current_pipeline_path():
    global __current_pipeline_path
    return __current_pipeline_path

def set_current_pipeline_path(path):
    global __current_pipeline_path
    __current_pipeline_path = path

def get_check_new_versions():
    if not get_config().Exists(CHECKFORNEWVERSIONS):
        # should this check for whether we can actually save preferences?
        return True
    return get_config().ReadBool(CHECKFORNEWVERSIONS)
    
def set_check_new_versions(val):
    old_val = get_check_new_versions()
    get_config().WriteBool(CHECKFORNEWVERSIONS, bool(val))
    # If the user turns on version checking, they probably don't want
    # to skip versions anymore.
    if val and (not old_val):
        set_skip_version(0)
    

def get_skip_version():
    if not get_config().Exists(SKIPVERSION):
        return 0
    return get_config().ReadInt(SKIPVERSION)

def set_skip_version(ver):
    get_config().WriteInt(SKIPVERSION, ver)
    

__show_sampling = None
def get_show_sampling():
    global __show_sampling
    if __show_sampling is not None:
        return __show_sampling
    if not get_config().Exists(SHOW_SAMPLING):
        __show_sampling = False
        return False
    return get_config().ReadBool(SHOW_SAMPLING)

def set_show_sampling(value):
    global __show_sampling
    get_config().WriteBool(SHOW_SAMPLING, bool(value))
    __show_sampling = bool(value)
    
__run_distributed = None
__run_distributed_listeners = []

def get_run_distributed():
    global __run_distributed
    if __run_distributed is not None:
        return __run_distributed
    if not get_config().Exists(RUN_DISTRIBUTED):
        __run_distributed = False
        return False
    return get_config().ReadBool(RUN_DISTRIBUTED)

def set_run_distributed(value):
    global __run_distributed
    get_config().WriteBool(RUN_DISTRIBUTED, bool(value))
    __run_distributed = bool(value)
    for listener in __run_distributed_listeners:
        listener(PreferenceChangedEvent(__run_distributed))

def add_run_distributed_listener(listener):
    """Add a listener that will be notified when the image directory changes
    """
    __run_distributed_listeners.append(listener)

def remove_run_distributed_listener(listener):
    """Remove a previously-added image directory listener
    """
    __run_distributed_listeners.remove(listener)

__recent_files = {}
def get_recent_files(category=""):
    global __recent_files
    if __recent_files.get(category, None) is None:
        __recent_files[category] = []
        for i in range(RECENT_FILE_COUNT):
            key = recent_file(i, category)
            if get_config().Exists(key):
                __recent_files[category].append(config_read(key)) 
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
def get_plugin_directory():
    global __plugin_directory
    
    if __plugin_directory is not None:
        return __plugin_directory
    
    if get_config().Exists(PLUGIN_DIRECTORY):
        __plugin_directory = config_read(PLUGIN_DIRECTORY)
    elif get_headless():
        return None
    else:
        import wx
        if wx.GetApp() is not None:
            __plugin_directory = os.path.join(wx.StandardPaths.Get().GetUserDataDir(), 'plugins')
    return __plugin_directory

def set_plugin_directory(value):
    global __plugin_directory
    
    __plugin_directory = value
    config_write(PLUGIN_DIRECTORY, value)

__ij_plugin_directory = None
def get_ij_plugin_directory():
    global __ij_plugin_directory
    
    if __ij_plugin_directory is not None:
        return __ij_plugin_directory
    
    if get_config().Exists(IJ_PLUGIN_DIRECTORY):
        __ij_plugin_directory = config_read(IJ_PLUGIN_DIRECTORY)
    else:
        # The default is the startup directory
        return os.path.abspath(os.path.join(os.curdir, "plugins"))
    return __ij_plugin_directory

def set_ij_plugin_directory(value):
    global __ij_plugin_directory
    
    __ij_plugin_directory = value
    config_write(IJ_PLUGIN_DIRECTORY, value)

__data_file=None

def get_data_file():
    '''Get the path to the LoadData data file specified on the command-line'''
    global __data_file
    return __data_file

def set_data_file(path):
    global __data_file
    __data_file = path

def standardize_default_folder_names(setting_values,slot):
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
                        setting_values[slot+1:])
    return setting_values

__cpfigure_position = (-1,-1)
def get_next_cpfigure_position(update_next_position=True):
    global __cpfigure_position
    pos = __cpfigure_position
    if update_next_position:
        update_cpfigure_position()
    return pos

def reset_cpfigure_position():
    global __cpfigure_position
    __cpfigure_position = (-1,-1)
    
def update_cpfigure_position():
    '''Called by get_next_cpfigure_position to update the screen position at 
    which the next figure frame will be drawn.
    '''
    global __cpfigure_position
    import wx
    win_size = (600,400)
    try:
        disp = wx.GetDisplaySize()
    except:
        disp = (800,600)
    if (__cpfigure_position[0] + win_size[0] > disp[0]):
        __cpfigure_position = (-1, __cpfigure_position[1])
    if (__cpfigure_position[1] + win_size[1] > disp[1]):
        __cpfigure_position = (-1, -1)
    else:
        # These offsets could be set in the preferences UI
        __cpfigure_position = (__cpfigure_position[0] + 120,
                               __cpfigure_position[1] + 24)
    
def get_startup_blurb():
    if not get_config().Exists(STARTUPBLURB):
        return True
    return get_config().ReadBool(STARTUPBLURB)

def set_startup_blurb(val):
    get_config().WriteBool(STARTUPBLURB, val)

def get_primary_outline_color():
    default = (0,255,0)
    if not get_config().Exists(PRIMARY_OUTLINE_COLOR):
        return tuple_to_color(default)
    return tuple_to_color(config_read(PRIMARY_OUTLINE_COLOR).split(","))

def set_primary_outline_color(color):
    config_write(PRIMARY_OUTLINE_COLOR,
                       ','.join([str(x) for x in color.Get()]))

def get_secondary_outline_color():
    default = (255,0,0)
    if not get_config().Exists(SECONDARY_OUTLINE_COLOR):
        return tuple_to_color(default)
    return tuple_to_color(config_read(SECONDARY_OUTLINE_COLOR).split(","))

def set_secondary_outline_color(color):
    config_write(SECONDARY_OUTLINE_COLOR,
                       ','.join([str(x) for x in color.Get()]))

def get_tertiary_outline_color():
    default = (255,255,0)
    if not get_config().Exists(TERTIARY_OUTLINE_COLOR):
        return tuple_to_color(default)
    return tuple_to_color(config_read(TERTIARY_OUTLINE_COLOR).split(","))

def set_tertiary_outline_color(color):
    config_write(TERTIARY_OUTLINE_COLOR,
                       ','.join([str(x) for x in color.Get()]))

__has_reported_jvm_error = False

def get_report_jvm_error():
    '''Return true if user still wants to report a JVM error'''
    if __has_reported_jvm_error:
        return False
    if not get_config().Exists(JVM_ERROR):
        return True
    return config_read(JVM_ERROR) == "True"

def set_report_jvm_error(should_report):
    config_write(JVM_ERROR, "True" if should_report else "False")

def set_has_reported_jvm_error():
    '''Call this to remember that we showed the user the JVM error'''
    global __has_reported_jvm_error
    __has_reported_jvm_error = True
    
__allow_output_file_overwrite = None

def get_allow_output_file_overwrite():
    '''Return true if the user wants to allow CP to overwrite the output file
    
    This is the .MAT output file, typically Default_OUT.mat
    '''
    global __allow_output_file_overwrite
    if __allow_output_file_overwrite is not None:
        return __allow_output_file_overwrite
    if not get_config().Exists(ALLOW_OUTPUT_FILE_OVERWRITE):
        return False
    return config_read(ALLOW_OUTPUT_FILE_OVERWRITE) == "True"

def set_allow_output_file_overwrite(value):
    '''Allow overwrite of .MAT file if true, warn user if false'''
    global __allow_output_file_overwrite
    __allow_output_file_overwrite = value
    config_write(ALLOW_OUTPUT_FILE_OVERWRITE, 
                       "True" if value else "False")

# "Analysis complete" preference
__show_analysis_complete_dlg = None

def get_show_analysis_complete_dlg():
    '''Return true if the user wants to see the "analysis complete" dialog'''
    global __show_analysis_complete_dlg
    if __show_analysis_complete_dlg is not None:
        return __show_analysis_complete_dlg
    if not get_config().Exists(SHOW_ANALYSIS_COMPLETE_DLG):
        return True
    return config_read(SHOW_ANALYSIS_COMPLETE_DLG) == "True"

def set_show_analysis_complete_dlg(value):
    '''Set the "show analysis complete" flag'''
    global __show_analysis_complete_dlg
    __show_analysis_complete_dlg = value
    config_write(SHOW_ANALYSIS_COMPLETE_DLG, 
                       "True" if value else "False")

# "Existing test mode" preference
__show_exiting_test_mode_dlg = None

def get_show_exiting_test_mode_dlg():
    '''Return true if the user wants to see the "exiting test mode" dialog'''
    global __show_exiting_test_mode_dlg
    if __show_exiting_test_mode_dlg is not None:
        return __show_exiting_test_mode_dlg
    if not get_config().Exists(SHOW_EXITING_TEST_MODE_DLG):
        return True
    return config_read(SHOW_EXITING_TEST_MODE_DLG) == "True"

def set_show_exiting_test_mode_dlg(value):
    '''Set the "exiting test mode" flag'''
    global __show_exiting_test_mode_dlg
    __show_exiting_test_mode_dlg = value
    config_write(SHOW_EXITING_TEST_MODE_DLG, 
                       "True" if value else "False")

# "Report bad sizes" preference
__show_report_bad_sizes_dlg = None

def get_show_report_bad_sizes_dlg():
    '''Return true if the user wants to see the "report bad sizes" dialog'''
    global __show_report_bad_sizes_dlg
    if __show_report_bad_sizes_dlg is not None:
        return __show_report_bad_sizes_dlg
    if not get_config().Exists(SHOW_BAD_SIZES_DLG):
        return True
    return config_read(SHOW_BAD_SIZES_DLG) == "True"

def set_show_report_bad_sizes_dlg(value):
    '''Set the "exiting test mode" flag'''
    global __show_report_bad_sizes_dlg
    __show_report_bad_sizes_dlg = value
    config_write(SHOW_BAD_SIZES_DLG, 
                       "True" if value else "False")

# Write .MAT files on output
__write_MAT_files = None

def get_write_MAT_files():
    '''Return true to write measurements as .mat files at end of run'''
    global __write_MAT_files
    if __write_MAT_files is not None:
        return __write_MAT_files
    if not get_config().Exists(WRITE_MAT):
        return True
    return config_read(WRITE_MAT) == "True"

def set_write_MAT_files(value):
    '''Set the "Write MAT files" flag'''
    global __write_MAT_files
    __write_MAT_files = value
    config_write(WRITE_MAT,
                       "True" if value else "False")

__warn_about_old_pipeline = None
def get_warn_about_old_pipeline():
    '''Return True if CP should warn the user about old SVN revision pipelines'''
    global __warn_about_old_pipeline
    if __warn_about_old_pipeline is not None:
        return __warn_about_old_pipeline
    if not get_config().Exists(WARN_ABOUT_OLD_PIPELINE):
        return True
    return config_read(WARN_ABOUT_OLD_PIPELINE) == "True"

def set_warn_about_old_pipeline(value):
    '''Set the "warn about old pipelines" flag'''
    global __warn_about_old_pipeline
    __warn_about_old_pipeline = value
    config_write(WARN_ABOUT_OLD_PIPELINE,
                       "True" if value else "False")

__use_more_figure_space = None
def get_use_more_figure_space():
    '''Return True if CP should use more of the figure space'''
    global __use_more_figure_space
    if __use_more_figure_space is not None:
        return __use_more_figure_space
    if not get_config().Exists(USE_MORE_FIGURE_SPACE):
        return False
    return config_read(USE_MORE_FIGURE_SPACE) == "True"

def set_use_more_figure_space(value):
    '''Set the "use more figure space" flag'''
    global __use_more_figure_space
    __use_more_figure_space = value
    config_write(USE_MORE_FIGURE_SPACE,
                       "True" if value else "False")
