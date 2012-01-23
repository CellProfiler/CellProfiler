'''walk_in_background.py - walk a directory tree from a background thread

This module walks a directory tree, incrementally reporting the
results to the UI thread.
'''

import logging
logger = logging.getLogger(__name__)
import os
import threading
import urllib
import wx

pause_lock = threading.Lock()
pause_condition = threading.Condition(pause_lock)
THREAD_RUNNING = "Running"
THREAD_STOP = "Stop"
THREAD_PAUSE = "Pause"
THREAD_RESUME = "Resume"

class InterruptException(Exception):
    def __init__(self, *args):
        super(self.__class__, self).__init__(*args)
        
class Checkpoint(object):
    '''A class that manages pausing and stopping'''
    def __init__(self):
        self.state = THREAD_RUNNING
        
    def set_state(self, state):
        with pause_lock:
            if state == THREAD_RESUME:
                state = THREAD_RUNNING
            self.state = state
            pause_condition.notify_all()
        
    def wait(self):
        with pause_lock:
            if self.state == THREAD_STOP:
                raise InterruptException()
            while self.state == THREAD_PAUSE:
                pause_condition.wait()
                
#
# Some file types won't open with BioFormats unless BioFormats is allowed
# to look at the file contents while determining the appropriate file reader.
# Others will try too hard and will look at associated files, even with
# the grouping option turned off. So here's the list of those that
# absolutely need it.
#
exts_that_need_allow_open_files = ( ".jpg", ".jpeg", ".jpe", 
                                    ".jp2", ".j2k", ".jpf",
                                    ".jpx", ".dic", ".dcm", ".dicom", 
                                    ".j2ki", ".j2kr", ".ome.tif", ".ome.tiff" )

def get_metadata(path):
    import subimager.client as C
    import subimager.omexml as O
    
    if path.lower().endswith(exts_that_need_allow_open_files):
        result = C.get_metadata(path, allowopenfiles="yes")
    else:
        result = C.get_metadata(path)
    if result is not None:
        return O.OMEXML(result)
    return None

def walk_in_background(path, callback_fn, completed_fn=None, metadata_fn=None):
    '''Walk a directory tree in the background
    
    path - path to walk
    
    callback_fn - a function that's called in the UI thread and incrementally
                  reports results. The callback is called with the
                  dirpath, dirnames and filenames for each iteration of walk.
    completed_fn - called when walk has completed
    metadata_fn - if present, call back with metadata. The signature is
                  metadata_fn(path, OMEXML) or metadata_fn(path, None) if
                  the webserver did not find metadata.
                  
    Returns a function that can be called to interrupt the operation.
    To stop, call it like this: fn(THREAD_STOP)
    To pause, call it with THREAD_PAUSE, to resume, call it with
    THREAD_RESUME
    '''
        
    checkpoint = Checkpoint()
        
    def report(dirpath, dirnames, filenames):
        if checkpoint.state != THREAD_STOP:
            callback_fn(dirpath, dirnames, filenames)

    def metadata_report(path, metadata):
        if checkpoint.state != THREAD_STOP:
            metadata_fn(path, metadata)
            
    def complete():
        if checkpoint.state != THREAD_STOP:
            completed_fn()
            
    def fn():
        try:
            path_list = []
            for dirpath, dirnames, filenames in os.walk(path):
                checkpoint.wait()
                wx.CallAfter(report, dirpath, dirnames, filenames)
                if metadata_fn is not None:
                    path_list += [os.path.join(dirpath, filename)
                                  for filename in filenames]
            for subpath in sorted(path_list):
                checkpoint.wait()
                try:
                    metadata = get_metadata("file:" + urllib.pathname2url(subpath))
                    wx.CallAfter(metadata_report, subpath, metadata)
                except:
                    logger.info("Failed to read image metadata for %s" % subpath)
        except InterruptException:
            logger.info("Exiting after request to stop")
        except:
            logger.exception("Exiting background walk after unhandled exception")
        finally:
            if completed_fn is not None:
                wx.CallAfter(complete)
    thread = threading.Thread(target = fn)
    thread.start()
    return checkpoint.set_state

def get_metadata_in_background(pathnames, fn_callback, fn_completed = None):
    '''Get image metadata for each path
    
    pathnames - list of pathnames
    fn_callback - callback with signature fn_callback(pathname, metadata)
    fn_completed - called when operation is complete

    Returns a function that can be called to interrupt the operation.
    '''
    checkpoint = Checkpoint()
    
    def metadata_fn(path, metadata):
        if checkpoint.state != THREAD_STOP:
            fn_callback(path, metadata)
            
    def completion_fn():
        if checkpoint.state != THREAD_STOP:
            fn_completed()
            
    def fn():
        try:
            for path in pathnames:
                checkpoint.wait()
                try:
                    if not path.startswith("file:"):
                        url = "file:" + urllib.pathname2url(path)
                    else:
                        url = path
                    metadata = get_metadata(url)
                    wx.CallAfter(metadata_fn, path, metadata)
                except:
                    logger.info("Failed to read image metadata for %s" % path)
        except InterruptException:
            logger.info("Exiting after request to stop")
        except:
            logger.exception("Exiting background walk after unhandled exception")
        finally:
            if fn_completed is not None:
                wx.CallAfter(completion_fn)
    thread = threading.Thread(target = fn)
    thread.start()
    return checkpoint.set_state
