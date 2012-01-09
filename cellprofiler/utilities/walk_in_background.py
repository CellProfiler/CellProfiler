'''walk_in_background.py - walk a directory tree from a background thread

This module walks a directory tree, incrementally reporting the
results to the UI thread.
'''

import logging
logger = logging.getLogger(__name__)
import os
import threading
import wx

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
    '''
    if metadata_fn is not None:
        import subimager.client as C
        import subimager.omexml as O
        import urllib
        
    stop = [False]
    def stop_me():
        stop[0] = True
        
    def report(dirpath, dirnames, filenames):
        if not stop[0]:
            callback_fn(dirpath, dirnames, filenames)

    def metadata_report(metadata):
        if not stop[0]:
            metadata_fn(path, metadata)
            
    def fn():
        path_list = []
        for dirpath, dirnames, filenames in os.walk(path):
            if stop[0]:
                break
            wx.CallAfter(report, dirpath, dirnames, filenames)
            if metadata_fn is not None:
                path_list += [os.path.join(dirpath, filename)
                              for filename in filenames]
        for subpath in path_list:
            try:
                xml = C.get_metadata("file:" + urllib.pathname2url(subpath))
                metadata = O.OMEXML(xml)
                wx.CallAfter(metadata_fn, subpath, metadata)
            except:
                logger.info("Failed to read image metadata for %s" % subpath)
        if completed_fn is not None and not stop[0]:
            wx.CallAfter(completed_fn)
    thread = threading.Thread(target = fn)
    thread.start()
    return stop_me

def get_metadata_in_background(pathnames, fn_callback, fn_completed = None):
    '''Get image metadata for each path
    
    pathnames - list of pathnames
    fn_callback - callback with signature fn_callback(pathname, metadata)
    fn_completed - called when operation is complete

    Returns a function that can be called to interrupt the operation.
    '''
    import subimager.client as C
    import subimager.omexml as O
    import urllib
    
    stop = [False]
    def stop_me():
        stop[0] = True
    
    def metadata_fn(path, metadata):
        if not stop[0]:
            fn_callback(path, metadata)
            
    def completion_fn():
        if not stop[0]:
            fn_completed()
            
    def fn():
        for path in pathnames:
            try:
                if not path.startswith("file:"):
                    url = "file:" + urllib.pathname2url(path)
                else:
                    url = path
                xml = C.get_metadata(url)
                metadata = O.OMEXML(xml)
                wx.CallAfter(metadata_fn, path, metadata)
            except:
                logger.info("Failed to read image metadata for %s" % path)
        if fn_completed is not None:
            wx.CallAfter(completion_fn)
    thread = threading.Thread(target = fn)
    thread.start()
    return stop_me
