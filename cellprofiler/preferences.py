"""Preferences.py - singleton preferences for CellProfiler

   TO-DO - load the default preferences from somewhere.
           Create a function to save the preferences.
           Create a function to populate a handles structure with preferences.

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2010 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import random
import cellprofiler
import os
import os.path
import re
import sys

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
    return config

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

def recent_file(index, category=""):
    return (FF_RECENTFILES % (index + 1)) + category

def module_directory():
    if not get_config().Exists(MODULEDIRECTORY):
        return os.path.join(cell_profiler_root_directory(), 'Modules')
    return str(get_config().Read(MODULEDIRECTORY))

def set_module_directory(value):
    get_config().Write(MODULEDIRECTORY, value)

def module_extension():
    return '.m'

__default_image_directory = None
def get_default_image_directory():
    global __default_image_directory
    if __default_image_directory is not None:
        return __default_image_directory
    if not get_config().Exists(DEFAULT_IMAGE_DIRECTORY):
        return os.path.abspath(os.path.expanduser('~'))
    default_image_directory = get_config().Read(DEFAULT_IMAGE_DIRECTORY)
    if os.path.isdir(default_image_directory):
        __default_image_directory = str(get_proper_case_filename(default_image_directory))
        return __default_image_directory
    else:
        sys.stderr.write("Warning: current path of %s is not a valid directory. Switching to current directory\n"%
                         (default_image_directory))
        default_image_directory = os.path.abspath(os.path.curdir)
        set_default_image_directory(default_image_directory)
        return str(get_proper_case_filename(default_image_directory))

def set_default_image_directory(path):
    global __default_image_directory
    path = str(path)
    __default_image_directory = path
    get_config().Write(DEFAULT_IMAGE_DIRECTORY,path)
    add_recent_file(path, DEFAULT_IMAGE_DIRECTORY)
    fire_image_directory_changed_event()
    
def fire_image_directory_changed_event():
    '''Notify listeners of a image directory change'''
    global __default_image_directory
    for listener in __image_directory_listeners:
        listener(DirectoryChangedEvent(__default_image_directory))

__image_directory_listeners = []

def add_image_directory_listener(listener):
    """Add a listener that will be notified when the image directory changes
    
    """
    __image_directory_listeners.append(listener)
    
def remove_image_directory_listener(listener):
    """Remove a previously-added image directory listener
    
    """
    __image_directory_listeners.remove(listener)

class DirectoryChangedEvent:
    def __init__(self, path):
        self.image_directory = path

__default_output_directory = None
def get_default_output_directory():
    global __default_output_directory
    if __default_output_directory is not None:
        return __default_output_directory
    if not get_config().Exists(DEFAULT_OUTPUT_DIRECTORY):
        return os.path.abspath(os.path.expanduser('~'))
    __default_output_directory = get_config().Read(DEFAULT_OUTPUT_DIRECTORY)
    __default_output_directory = str(get_proper_case_filename(__default_output_directory))
    return __default_output_directory

def set_default_output_directory(path):
    global __default_output_directory
    path=str(path)
    assert os.path.isdir(path),'Default Output Folder, "%s", is not a directory'%(path)
    __default_output_directory = path
    get_config().Write(DEFAULT_OUTPUT_DIRECTORY,path)
    add_recent_file(path, DEFAULT_OUTPUT_DIRECTORY)
    for listener in __output_directory_listeners:
        listener(DirectoryChangedEvent(path))

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
    title_font_size = get_config().Read(TITLE_FONT_SIZE)
    return float(title_font_size)

def set_title_font_size(title_font_size):
    get_config().Write(TITLE_FONT_SIZE,str(title_font_size))

def get_title_font_name():
    if not get_config().Exists(TITLE_FONT_NAME):
        return "Tahoma"
    return get_config().Read(TITLE_FONT_NAME)

def set_title_font_name(title_font_name):
    get_config().Write(TITLE_FONT_NAME, title_font_name)

def get_table_font_name():
    if not get_config().Exists(TABLE_FONT_NAME):
        return "Tahoma"
    return get_config().Read(TABLE_FONT_NAME)

def set_table_font_name(title_font_name):
    get_config().Write(TABLE_FONT_NAME, title_font_name)
    
def get_table_font_size():
    if not get_config().Exists(TABLE_FONT_SIZE):
        return 9
    table_font_size = get_config().Read(TABLE_FONT_SIZE)
    return float(table_font_size)

def set_table_font_size(table_font_size):
    get_config().Write(TABLE_FONT_SIZE,str(table_font_size))

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
        color = get_config().Read(BACKGROUND_COLOR).split(',')
        return tuple_to_color(tuple(color), default_color)

def set_background_color(color):
    '''Set the color to be used for window backgrounds
    
    '''
    get_config().Write(BACKGROUND_COLOR,
                       ','.join([str(x) for x in color.Get()]))

def get_pixel_size():
    """The size of a pixel in microns"""
    if not get_config().Exists(PIXEL_SIZE):
        return 1.0
    return float(get_config().Read(PIXEL_SIZE))

def set_pixel_size(pixel_size):
    get_config().Write(PIXEL_SIZE,str(pixel_size))

__output_filename = 'DefaultOUT.mat'
__output_filename_listeners = []
def get_output_file_name():
    return __output_filename

class OutputFilenameEvent:
    def __init__(self):
        self.OutputFilename = __output_filename

def set_output_file_name(filename):
    global __output_filename
    filename=str(filename)
    __output_filename = filename
    for listener in __output_filename_listeners:
        listener(OutputFilenameEvent)

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
    return get_config().Read(COLORMAP)

def set_default_colormap(colormap):
    get_config().Write(COLORMAP, colormap)

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
    
__recent_files = {}
def get_recent_files(category=""):
    global __recent_files
    if __recent_files.get(category, None) is None:
        __recent_files[category] = []
        for i in range(RECENT_FILE_COUNT):
            key = recent_file(i, category)
            if get_config().Exists(key):
                __recent_files[category].append(get_config().Read(key)) 
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
        get_config().Write(recent_file(i, category), filename)

__plugin_directory = None
def get_plugin_directory():
    global __plugin_directory
    
    if __plugin_directory is not None:
        return __plugin_directory
    
    if get_config().Exists(PLUGIN_DIRECTORY):
        __plugin_directory = get_config().Read(PLUGIN_DIRECTORY)
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
    get_config().Write(PLUGIN_DIRECTORY, value)

__ij_plugin_directory = None
def get_ij_plugin_directory():
    global __ij_plugin_directory
    
    if __ij_plugin_directory is not None:
        return __ij_plugin_directory
    
    if get_config().Exists(IJ_PLUGIN_DIRECTORY):
        __ij_plugin_directory = get_config().Read(IJ_PLUGIN_DIRECTORY)
    else:
        # The default is the startup directory
        return os.path.abspath(os.path.join(os.curdir, "plugins"))
    return __ij_plugin_directory

def set_ij_plugin_directory(value):
    global __ij_plugin_directory
    
    __ij_plugin_directory = value
    get_config().Write(IJ_PLUGIN_DIRECTORY, value)

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
    return tuple_to_color(get_config().Read(PRIMARY_OUTLINE_COLOR).split(","))

def set_primary_outline_color(color):
    get_config().Write(PRIMARY_OUTLINE_COLOR,
                       ','.join([str(x) for x in color.Get()]))

def get_secondary_outline_color():
    default = (255,0,0)
    if not get_config().Exists(SECONDARY_OUTLINE_COLOR):
        return tuple_to_color(default)
    return tuple_to_color(get_config().Read(SECONDARY_OUTLINE_COLOR).split(","))

def set_secondary_outline_color(color):
    get_config().Write(SECONDARY_OUTLINE_COLOR,
                       ','.join([str(x) for x in color.Get()]))

def get_tertiary_outline_color():
    default = (255,255,0)
    if not get_config().Exists(TERTIARY_OUTLINE_COLOR):
        return tuple_to_color(default)
    return tuple_to_color(get_config().Read(TERTIARY_OUTLINE_COLOR).split(","))

def set_tertiary_outline_color(color):
    get_config().Write(TERTIARY_OUTLINE_COLOR,
                       ','.join([str(x) for x in color.Get()]))

__has_reported_jvm_error = False

def get_report_jvm_error():
    '''Return true if user still wants to report a JVM error'''
    if __has_reported_jvm_error:
        return False
    if not get_config().Exists(JVM_ERROR):
        return True
    return get_config().Read(JVM_ERROR) == "True"

def set_report_jvm_error(should_report):
    get_config().Write(JVM_ERROR, "True" if should_report else "False")

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
    return get_config().Read(ALLOW_OUTPUT_FILE_OVERWRITE) == "True"

def set_allow_output_file_overwrite(value):
    '''Allow overwrite of .MAT file if true, warn user if false'''
    global __allow_output_file_overwrite
    __allow_output_file_overwrite = value
    get_config().Write(ALLOW_OUTPUT_FILE_OVERWRITE, 
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
    return get_config().Read(SHOW_ANALYSIS_COMPLETE_DLG) == "True"

def set_show_analysis_complete_dlg(value):
    '''Set the "show analysis complete" flag'''
    global __show_analysis_complete_dlg
    __show_analysis_complete_dlg = value
    get_config().Write(SHOW_ANALYSIS_COMPLETE_DLG, 
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
    return get_config().Read(SHOW_EXITING_TEST_MODE_DLG) == "True"

def set_show_exiting_test_mode_dlg(value):
    '''Set the "exiting test mode" flag'''
    global __show_exiting_test_mode_dlg
    __show_exiting_test_mode_dlg = value
    get_config().Write(SHOW_EXITING_TEST_MODE_DLG, 
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
    return get_config().Read(SHOW_BAD_SIZES_DLG) == "True"

def set_show_report_bad_sizes_dlg(value):
    '''Set the "exiting test mode" flag'''
    global __show_report_bad_sizes_dlg
    __show_report_bad_sizes_dlg = value
    get_config().Write(SHOW_BAD_SIZES_DLG, 
                       "True" if value else "False")