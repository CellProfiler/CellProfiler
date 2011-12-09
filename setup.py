"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2012 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__ = "$Revision$"

from setuptools import setup
import sys
import os
import os.path
import glob

# fix from
#  http://mail.python.org/pipermail/pythonmac-sig/2008-June/020111.html
import pytz
pytz.zoneinfo = pytz.tzinfo
pytz.zoneinfo.UTC = pytz.UTC

from libtiff.libtiff_ctypes import tiff_h_name

# make sure external dependencies match requirements
import external_dependencies
external_dependencies.fetch_external_dependencies('fail')

if sys.platform == "darwin":
    os.system("svn info | grep Revision | sed -e 's/Revision:/\"Version/' -e 's/^/VERSION = /' -e 's/$/\"/' > version.py")

APPNAME = 'CellProfiler2.0'
APP = ['CellProfiler.py']
DATA_FILES = [('cellprofiler/icons', glob.glob(os.path.join('.', 'cellprofiler', 'icons', '*.png'))),
              ('bioformats', ['bioformats/loci_tools.jar']),
              ('imagej', ['imagej/TCPClient.class', 'imagej/InterProcessIJBridge.class',
                          'imagej/InterProcessIJBridge$1.class', 'imagej/ij.jar']),
              ]
OPTIONS = {'argv_emulation': True,
           'packages': ['cellprofiler', 'contrib', 'bioformats', 'imagej'],
           'includes': ['numpy', 'wx', 'matplotlib','email.iterators', 'smtplib',
                        'sqlite3', 'libtiff', 'wx.lib.intctrl', 'libtiff.'+tiff_h_name,
                        'xml.dom.minidom', 'h5py', 'h5py.defs', 'h5py.utils', 'h5py._proxy'],
           'excludes': ['pylab', 'nose', 'Tkinter', 'Cython', 'scipy.weave'],
           'resources': ['CellProfilerIcon.png', 'cellprofiler/icons'],
           'iconfile' : 'CellProfilerIcon.icns',
           'frameworks' : ['libtiff.dylib'],
           }

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
    name="CellProfiler2.0"
)
