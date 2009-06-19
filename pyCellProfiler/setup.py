from setuptools import setup
import sys, os

# fix from
#  http://mail.python.org/pipermail/pythonmac-sig/2008-June/020111.html
import pytz
pytz.zoneinfo = pytz.tzinfo
pytz.zoneinfo.UTC = pytz.UTC

if sys.platform == "darwin":
    os.system("svn info | grep Revision | sed -e 's/Revision:/\"Version/' -e 's/^/VERSION = /' -e 's/$/\"/' > version.py")

APPNAME = 'CellProfiler2.0'
APP = ['CellProfiler.py']
DATA_FILES = []
OPTIONS = {'argv_emulation': True,
           'packages': ['numpy', 'cellprofiler', 'cellprofiler.cpmath'],
           'excludes': ['pylab', 'nose', 'wx.tools'],
           'resources': ['CellProfilerIcon.png'],
           }

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
    name = "CellProfiler2.0"
)

