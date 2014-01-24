"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

'''walk_in_background.py - walk a directory tree from a background thread

This module walks a directory tree, incrementally reporting the
results to the UI thread.
'''

import logging
logger = logging.getLogger(__name__)
import os
import threading
import urllib
import uuid

pause_lock = threading.Lock()
pause_condition = threading.Condition(pause_lock)
THREAD_RUNNING = "Running"
THREAD_STOP = "Stop"
THREAD_STOPPING = "Stopping"
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
                if len(filenames) == 0:
                    continue
                import wx
                wx.CallAfter(report, dirpath, dirnames, filenames)
                if metadata_fn is not None:
                    path_list += [os.path.join(dirpath, filename)
                                  for filename in filenames]
            for subpath in sorted(path_list):
                checkpoint.wait()
                try:
                    metadata = get_metadata("file:" + urllib.pathname2url(subpath))
                    import wx
                    wx.CallAfter(metadata_report, subpath, metadata)
                except:
                    logger.info("Failed to read image metadata for %s" % subpath)
        except InterruptException:
            logger.info("Exiting after request to stop")
        except:
            logger.exception("Exiting background walk after unhandled exception")
        finally:
            if completed_fn is not None:
                import wx
                wx.CallAfter(complete)
    thread = threading.Thread(target = fn)
    thread.setDaemon(True)
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
                    import wx
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

class WalkCollection(object):
    '''A collection of all walks in progress
    
    This class manages a group of walks that are in progress so that they
    can be paused, resumed and stopped in unison.
    '''
    def __init__(self, fn_on_completed):
        self.fn_on_completed = fn_on_completed
        self.stop_functions = {}
        self.paused_tasks = []
        self.state = THREAD_STOP
        
    def on_complete(self, uid):
        if self.stop_functions.has_key(uid):
            del self.stop_functions[uid]
            if len(self.stop_functions) == 0:
                self.state = THREAD_STOP
                self.fn_on_completed()
            
    def walk_in_background(self, path, callback_fn, metadata_fn = None):
        if self.state == THREAD_PAUSE:
            self.paused_tasks.append(
                lambda path, callback_fn, metadata_fn:
                self.walk_in_background(path, callback_fn, metadata_fn))
        else:
            key = uuid.uuid4()
            fn_on_complete = lambda key=key: self.on_complete(key)
            self.stop_functions[key] = walk_in_background(
                path, callback_fn, fn_on_complete, metadata_fn)
            if self.state == THREAD_STOP:
                self.state = THREAD_RUNNING
        
    def get_metadata_in_background(self, pathnames, fn_callback):
        if self.state == THREAD_PAUSE:
            self.paused_tasks.append(
                lambda pathnames, fn_callback: 
                self.get_metadata_in_background(pathnames, fn_callback))
        else:
            key = uuid.uuid4()
            fn_on_complete = lambda key=key: self.on_complete(key)
            self.stop_functions[key] = get_metadata_in_background(
                pathnames, fn_callback, fn_on_complete)
        
    def get_state(self):
        return self.state
    
    def pause(self):
        if self.state == THREAD_RUNNING:
            for stop_fn in self.stop_functions.values():
                stop_fn(THREAD_PAUSE)
            self.state = THREAD_PAUSE
            
    def resume(self):
        if self.state == THREAD_PAUSE:
            for stop_fn in self.stop_functions.values():
                stop_fn(THREAD_RESUME)
            for fn_task in self.paused_tasks:
                fn_task()
            self.paused_tasks = []
            self.state = THREAD_RUNNING
    
    def stop(self):
        if self.state in (THREAD_RUNNING, THREAD_PAUSE):
            for stop_fn in self.stop_functions.values():
                stop_fn(THREAD_STOP)
            self.paused_tasks = []
            self.state = THREAD_STOPPING
