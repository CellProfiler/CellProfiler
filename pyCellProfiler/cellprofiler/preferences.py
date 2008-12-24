"""Preferences.py - singleton preferences for CellProfiler

   $Revision$
   TO-DO - load the default preferences from somewhere.
           Create a function to save the preferences.
           Create a function to populate a handles structure with preferences.
"""
import cellprofiler
import os
import re
import wx

__python_root = os.path.split(str(cellprofiler.__path__[0]))[0]
__cp_root = os.path.split(__python_root)[0]

def get_config():
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

def module_directory():
    return os.path.join(cell_profiler_root_directory(), 'Modules')

def module_extension():
    return '.m'

DEFAULT_IMAGE_DIRECTORY = 'DefaultImageDirectory'
DEFAULT_OUTPUT_DIRECTORY = 'DefaultOutputDirectory'

def get_default_image_directory():
    if not get_config().Exists(DEFAULT_IMAGE_DIRECTORY):
        return os.path.abspath(os.path.curdir)
    default_image_directory = get_config().Read(DEFAULT_IMAGE_DIRECTORY)
    return default_image_directory

def set_default_image_directory(path):
    path = str(path)
    get_config().Write(DEFAULT_IMAGE_DIRECTORY,path)
    for listener in __image_directory_listeners:
        if callable(listener):
            listener(ImageDirectoryChangedEvent(path))
        else:
            listener.Notify(ImageDirectoryChangedEvent(path))

__image_directory_listeners = []

def add_image_directory_listener(listener):
    """Add a listener that will be notified when the image directory changes
    
    """
    __image_directory_listeners.append(listener)
    
def remove_image_directory_listener(listener):
    """Remove a previously-added image directory listener
    
    """
    __image_directory_listeners.remove(listener)

class ImageDirectoryChangedEvent:
    def __init__(self, path):
        self.image_directory = path


def get_default_output_directory():
    if not get_config().Exists(DEFAULT_OUTPUT_DIRECTORY):
        return os.path.abspath(os.path.curdir)
    default_output_directory = get_config().Read(DEFAULT_OUTPUT_DIRECTORY)
    return default_output_directory

def set_default_output_directory(path):
    path=str(path)
    assert os.path.isdir(path),'Default output directory, "%s", is not a directory'%(path)
    get_config().Write(DEFAULT_OUTPUT_DIRECTORY,path)

__pixel_size = 1

def get_pixel_size():
    return __pixel_size

def set_pixel_size(pixel_size):
    __pixel_size = pixel_size
    
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

