"""cellprofiler.utilities.setup - compiling files in the utilities module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

from distutils.core import setup,Extension
import os
import sys
try:
    from Cython.Distutils import build_ext
    from numpy import get_include
except ImportError:
    import site
    site.addsitedir('../../site-packages')
    from Cython.Distutils import build_ext
    from numpy import get_include

def configuration():
    extensions = []
    if sys.platform.startswith('win') and False:
        extensions += [Extension(name="_get_proper_case_filename",
                                 sources=["get_proper_case_filename.c"],
                                 libraries=["shlwapi", "shell32", "ole32"],
                                 extra_compile_args=['-O3'])]
    dict = { "name":"utilities",
             "description":"utility module for CellProfiler",
             "maintainer":"Lee Kamentsky",
             "maintainer_email":"leek@broad.mit.edu",
             "cmdclass": {'build_ext': build_ext},
             "ext_modules": extensions
            }
    return dict

if __name__ == '__main__':
    if '/' in __file__:
        os.chdir(os.path.dirname(__file__))
    setup(**configuration())
    

