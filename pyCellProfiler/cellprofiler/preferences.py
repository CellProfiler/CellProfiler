"""Preferences.py - singleton preferences for CellProfiler

   TO-DO - load the default preferences from somewhere.
           Create a function to save the preferences.
           Create a function to populate a handles structure with preferences.

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import random
import cellprofiler
import os
import re
import wx
import sys

from cellprofiler.utilities.get_proper_case_filename import get_proper_case_filename

'''get_absolute_path - mode = output. Assume "." is the default output dir'''
ABSPATH_OUTPUT = 'abspath_output'

'''get_absolute_path - mode = image. Assume "." is the default image dir'''
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
    for listener in __image_directory_listeners:
        listener(DirectoryChangedEvent(path))

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
    assert os.path.isdir(path),'Default output directory, "%s", is not a directory'%(path)
    __default_output_directory = path
    get_config().Write(DEFAULT_OUTPUT_DIRECTORY,path)
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

def get_background_color():
    '''Get the color to be used for window backgrounds
    
    Return wx.Colour that will be applied as
    the background for all frames and dialogs
    '''
    if not get_config().Exists(BACKGROUND_COLOR):
        return wx.Colour(red=143,green=188,blue=143) # darkseagreen
    else:
        try:
            color = [int(x) 
                     for x in get_config().Read(BACKGROUND_COLOR).split(',')]
            return wx.Colour(*tuple(color))
        except:
            return wx.Colour(red=143,green=188,blue=143) # darkseagreen

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
    
    If a path starts with "./", then make the path relative to the
    default output directory.
    If a path starts with "&/", then make the path relative to the
    default image directory.
    If a "path" has no path component then make the path relative to
    the default output directory.
    """
    if abspath_mode == ABSPATH_OUTPUT:
        osep = '.'
        isep = '&'
    elif abspath_mode == ABSPATH_IMAGE:
        osep = '&'
        isep = '.'
    else:
        raise ValueError("Unknown abspath mode: %s"%abspath_mode)
        
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
