"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2010

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__ = "$Revision: 8876 $"

import wx
import os.path
import glob
import weakref
import sys

if hasattr(sys, 'frozen'):
    path = os.path.split(os.path.abspath(sys.argv[0]))[0]
    path = os.path.join(path, 'cellprofiler','icons')
else:
    path = __path__[0]

image_cache = weakref.WeakValueDictionary()

def get_icon(name):
    try:
        return image_cache[name]
    except KeyError:
        image_cache[name] = im =  wx.Image(os.path.join(path, name + '.png'))
        return im

def get_icon_path():
    return os.path.join(path, '')
