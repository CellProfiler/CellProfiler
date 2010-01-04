"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2010

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

if sys.platform == "darwin":
    os.system("svn info | grep Revision | sed -e 's/Revision:/\"Version/' -e 's/^/VERSION = /' -e 's/$/\"/' > version.py")

APPNAME = 'CellProfiler2.0'
APP = ['CellProfiler.py']
DATA_FILES = [('cellprofiler/icons', glob.glob(os.path.join('.', 'cellprofiler', 'icons', '*.png')))]
OPTIONS = {'argv_emulation': True,
           'packages': ['cellprofiler', 'cellprofiler.cpmath'],
           'includes': ['numpy', 'wx', 'matplotlib'],
           'excludes': ['pylab', 'nose', 'wx.tools'],
           'resources': ['CellProfilerIcon.png', 'cellprofiler/icons'],
           }

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
    name="CellProfiler2.0"
)
