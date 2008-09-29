"""Preferences.py - singleton preferences for CellProfiler

   $Revision$
"""
import CellProfiler
import os

__python_root = os.path.split(CellProfiler.__path__[0])[0]
__cp_root = os.path.split(__python_root)[0]
__default_module_directory = os.path.join(__cp_root,'Modules') 
def ModuleDirectory():
    return __default_module_directory

def ModuleExtension():
    return '.m'