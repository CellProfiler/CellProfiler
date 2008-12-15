"""Preferences.py - singleton preferences for CellProfiler

   $Revision$
   TO-DO - load the default preferences from somewhere.
           Create a function to save the preferences.
           Create a function to populate a handles structure with preferences.
"""
import CellProfiler
import os
import re
import wx

__python_root = os.path.split(str(CellProfiler.__path__[0]))[0]
__cp_root = os.path.split(__python_root)[0]

def GetConfig():
    try:
        config = wx.Config.Get(False)
    except wx.PyNoAppError:
        app = wx.App(0)
        config = wx.Config.Get(False)
    if not config:
        wx.Config.Set(wx.Config('CellProfiler','BroadInstitute','CellProfilerLocal.cfg','CellProfilerGlobal.cfg',wx.CONFIG_USE_LOCAL_FILE))
        config = wx.Config.Get()
    return config

def CellProfilerRootDirectory():
    if __cp_root:
        return __cp_root
    return '..'

def PythonRootDirectory():
    return __python_root

def ModuleDirectory():
    return os.path.join(CellProfilerRootDirectory(), 'Modules')

def ModuleExtension():
    return '.m'

DEFAULT_IMAGE_DIRECTORY = 'DefaultImageDirectory'
DEFAULT_OUTPUT_DIRECTORY = 'DefaultOutputDirectory'

def GetDefaultImageDirectory():
    if not GetConfig().Exists(DEFAULT_IMAGE_DIRECTORY):
        return os.path.abspath(os.path.curdir)
    default_image_directory = GetConfig().Read(DEFAULT_IMAGE_DIRECTORY)
    return default_image_directory

def SetDefaultImageDirectory(path):
    path = str(path)
    GetConfig().Write(DEFAULT_IMAGE_DIRECTORY,path)
    for listener in __image_directory_listeners:
        if callable(listener):
            listener(ImageDirectoryChangedEvent(path))
        else:
            listener.Notify(ImageDirectoryChangedEvent(path))

__image_directory_listeners = []

def AddImageDirectoryListener(listener):
    """Add a listener that will be notified when the image directory changes
    
    """
    __image_directory_listeners.append(listener)
    
def RemoveImageDirectoryListener(listener):
    """Remove a previously-added image directory listener
    
    """
    __image_directory_listeners.remove(listener)

class ImageDirectoryChangedEvent:
    def __init__(self, path):
        self.ImageDirectory = path


def GetDefaultOutputDirectory():
    if not GetConfig().Exists(DEFAULT_OUTPUT_DIRECTORY):
        return os.path.abspath(os.path.curdir)
    default_output_directory = GetConfig().Read(DEFAULT_OUTPUT_DIRECTORY)
    return default_output_directory

def SetDefaultOutputDirectory(path):
    path=str(path)
    assert os.path.isdir(path),'Default output directory, "%s", is not a directory'%(path)
    GetConfig().Write(DEFAULT_OUTPUT_DIRECTORY,path)

__pixel_size = 1

def GetPixelSize():
    return __pixel_size

def SetPixelSize(pixel_size):
    __pixel_size = pixel_size
    
__output_filename = 'DefaultOUT.mat'
__output_filename_listeners = []
def GetOutputFileName():
    return __output_filename

class OutputFilenameEvent:
    def __init__(self):
        self.OutputFilename = __output_filename

def SetOutputFileName(filename):
    global __output_filename
    filename=str(filename)
    __output_filename = filename
    for listener in __output_filename_listeners:
        listener(OutputFilenameEvent)

def AddOutputFileNameListener(listener):
    __output_filename_listeners.append(listener)

def RemoveOutputFileNameListener(listener):
    __output_filename_listeners.remove(listener)

