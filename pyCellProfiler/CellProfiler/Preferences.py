"""Preferences.py - singleton preferences for CellProfiler

   $Revision$
   TO-DO - load the default preferences from somewhere.
           Create a function to save the preferences.
           Create a function to populate a handles structure with preferences.
"""
import CellProfiler
import os

__python_root = os.path.split(CellProfiler.__path__[0])[0]
__cp_root = os.path.split(__python_root)[0]
__default_module_directory = os.path.join(__cp_root,'Modules') 

def PythonRootDirectory():
    return __python_root

def ModuleDirectory():
    return __default_module_directory

def ModuleExtension():
    return '.m'

__default_image_directory = os.path.abspath(os.path.curdir)
def GetDefaultImageDirectory():
    return __default_image_directory

def SetDefaultImageDirectory(path):
    assert os.path.isdir(path),'Default image directory, "%s", is not a directory'%(path)
    globals()['__default_image_directory'] = path
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

__default_output_directory = os.path.abspath(os.path.curdir)

def GetDefaultOutputDirectory():
    return __default_output_directory

def SetDefaultOutputDirectory(path):
    assert os.path.isdir(path),'Default output directory, "%s", is not a directory'%(path)
    globals()['__default_output_directory']=path

__pixel_size = 1

def GetPixelSize():
    return __pixel_size

def SetPixelSize(pixel_size):
    __pixel_size = pixel_size
 
