"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2015 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

from setuptools import setup
import sys
import os
import os.path
import glob
from subprocess import call
import ctypes.util

sys.path.append('.')
import cellprofiler.utilities.version
from external_dependencies import get_cellprofiler_jars

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

APPNAME = 'CellProfiler'
APP = ['CellProfiler.py']
icon_src_path = os.path.join('.', 'cellprofiler', 'icons')
DATA_FILES = [('cellprofiler/icons', 
               glob.glob(os.path.join(icon_src_path, '*.png'))+
               [os.path.join(icon_src_path, "icon_copyrights.txt")])]
from javabridge import JARS
imagej_path = os.path.abspath(os.path.join(".", "imagej", "jars"))
jars = JARS + [os.path.join(imagej_path, jar) for jar in get_cellprofiler_jars()]
jars.append(os.path.join(
    imagej_path, "cellprofiler-java-dependencies-classpath.txt"))
DATA_FILES.append(('imagej/jars', jars))
OPTIONS = {'argv_emulation': True,
           'packages': ['cellprofiler', 'contrib', 'imagej', 'javabridge'],
           'includes': ['objgraph', 'numpy', 'scipy', 'sklearn', 'sklearn.utils.sparsetools.*',
                        'wx', 'matplotlib','email.iterators', 'smtplib', 'zmq',
                        'javabridge', 'bioformats', 
                        'sqlite3', 'libtiff', 'wx.lib.intctrl', 'libtiff.'+tiff_h_name,
                        'xml.dom.minidom', 'h5py', 'h5py.defs', 'h5py.utils', 'h5py._proxy', 'readline'],
           'excludes': ['pylab', 'Tkinter', 'Cython', 'scipy.weave',
                        'virtualenv'],
           'resources': ['CellProfilerIcon.png'],
           'iconfile' : 'CellProfilerIcon.icns',
           'frameworks' : [tiff_dylib, jpeg_dylib],
           'plist': { 
               "LSArchitecturePriority": ["i386"],
               "LSMinimumSystemVersion": "10.6.8", # See #871
               "CFBundleName": "CellProfiler",
               "CFBundleIdentifier": "org.cellprofiler.CellProfiler",
               "CFBundleShortVersionString": cellprofiler.utilities.version.dotted_version,
               "CFBundleDocumentTypes": [{
                   "CFBundleTypeExtensions":["cpproj"],
                   "CFBundleTypeIconFile":"CellProfilerIcon.icns",
                   "CFBundleTypeName":"CellProfiler project",
                   "CFBundleTypeRole":"Editor"
                   }, {
                       "CFBundleTypeExtensions":["cppipe"],
                       "CFBundleTypeIconFile":"CellProfilerIcon.icns",
                       "CFBundleTypeName":"CellProfiler pipeline",
                       "CFBundleTypeRole":"Editor"
                       }]
           }
           }

if sys.argv[-1] == 'py2app':
    assert not os.path.exists("build"), "Remove the build and dist directories before building app!"
    assert not os.path.exists("dist"), "Remove the build and dist directories before building app!"

setup(
    app=APP,
    package_data={'javabridge':['jars/*.jar']},
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
    name="CellProfiler"
)

if sys.argv[-1] == 'py2app':
    # there should be some way to do this within setup's framework, but I don't
    # want to figure it out right now, and our setup is going to be changing
    # significantly soon, anyway.
    call('find dist/CellProfiler.app -name tests -type d | xargs rm -rf', shell=True)
    #call('lipo dist/CellProfiler.app/Contents/MacOS/CellProfiler -thin i386 -output dist/CellProfiler.app/Contents/MacOS/CellProfiler', shell=True)
    call('rm dist/CellProfiler.app/Contents/Resources/lib/python2.7/cellprofiler/icons/*.png', shell=True)
