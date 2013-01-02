"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2013 Broad Institute
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
from subprocess import call
import ctypes.util

# fix from
#  http://mail.python.org/pipermail/pythonmac-sig/2008-June/020111.html
import pytz
pytz.zoneinfo = pytz.tzinfo
pytz.zoneinfo.UTC = pytz.UTC
#
# It's necessary to install libtiff and libjpeg explicitly
# so that libtiff can find itself and so that libjpeg
# is the one that we want and not the one that WX thinks
# it wants.
#
from libtiff.libtiff_ctypes import tiff_h_name
tiff_dylib = ctypes.util.find_library('tiff')
jpeg_dylib = ctypes.util.find_library('jpeg')

# make sure external dependencies match requirements
import external_dependencies
external_dependencies.fetch_external_dependencies('fail')

if sys.platform == "darwin":
    import cellprofiler.utilities.version
    f = open("cellprofiler/frozen_version.py", "w")
    f.write("# MACHINE_GENERATED\nversion_string = '%s'" % cellprofiler.utilities.version.version_string)
    f.close()

APPNAME = 'CellProfiler2.0'
APP = ['CellProfiler.py']
DATA_FILES = [('cellprofiler/icons', glob.glob(os.path.join('.', 'cellprofiler', 'icons', '*.png')))]
OPTIONS = {'argv_emulation': True,
           'packages': ['cellprofiler', 'contrib', 'bioformats', 'imagej'],
           'includes': ['numpy', 'wx', 'matplotlib','email.iterators', 'smtplib',
                        'sqlite3', 'libtiff', 'libtiff.'+tiff_h_name,
                        'wx.lib.intctrl', 'wx.lib.filebrowsebutton',
                        'xml.dom.minidom', 'h5py', 'h5py.defs', 'h5py.utils', 'h5py._proxy'],
           'excludes': ['pylab', 'nose', 'Tkinter', 'Cython', 'scipy.weave'],
           'resources': ['CellProfilerIcon.png'],
           'iconfile' : 'CellProfilerIcon.icns',
           'frameworks' : [tiff_dylib, jpeg_dylib],
           }

if sys.argv[-1] == 'py2app':
    assert not os.path.exists("build"), "Remove the build and dist directories before building app!"
    assert not os.path.exists("dist"), "Remove the build and dist directories before building app!"

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
    name="CellProfiler2.0"
)

if sys.argv[-1] == 'py2app':
    # there should be some way to do this within setup's framework, but I don't
    # want to figure it out right now, and our setup is going to be changing
    # significantly soon, anyway.
    call('find dist/CellProfiler2.0.app -name tests -type d | xargs rm -rf', shell=True)
    call('lipo dist/CellProfiler2.0.app/Contents/MacOS/CellProfiler2.0 -thin i386 -output dist/CellProfiler2.0.app/Contents/MacOS/CellProfiler2.0', shell=True)
    call('rm dist/CellProfiler2.0.app/Contents/Resources/lib/python2.7/cellprofiler/icons/*.png', shell=True)
