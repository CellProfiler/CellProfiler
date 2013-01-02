"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2013 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__ = "$Revision: 8876 $"

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

def get_builtin_image(name):
    import wx
    try:
        return image_cache[name]
    except KeyError:
        image_cache[name] = im =  wx.Image(os.path.join(path, name + '.png'))
        return im

def get_builtin_images_path():
    return os.path.join(path, '')
