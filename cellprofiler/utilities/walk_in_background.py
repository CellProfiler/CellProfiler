'''walk_in_background.py - walk a directory tree from a background thread

This module walks a directory tree, incrementally reporting the
results to the UI thread.
'''

import os
import threading
import wx

def walk_in_background(path, callback_fn, completed_fn=None):
    '''Walk a directory tree in the background
    
    path - path to walk
    
    callback_fn - a function that's called in the UI thread and incrementally
                  reports results. The callback is called with the
                  dirpath, dirnames and filenames for each iteration of walk.
    completed_fn - called when walk has completed
                  
    Returns a function that can be called to interrupt the operation.
    '''
    stop = [False]
    def stop_me():
        stop[0] = True
        
    def report(dirpath, dirnames, filenames):
        if not stop[0]:
            callback_fn(dirpath, dirnames, filenames)
            
    def fn():
        for dirpath, dirnames, filenames in os.walk(path):
            if stop[0]:
                break
            wx.CallAfter(report, dirpath, dirnames, filenames)
        if completed_fn is not None and not stop[0]:
            wx.CallAfter(completed_fn)
    thread = threading.Thread(target = fn)
    thread.start()
    return stop_me

