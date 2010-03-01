"""setup.py - setup to build C modules for cpmath

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2010

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

from distutils.core import setup,Extension
import os
try:
    from Cython.Distutils import build_ext
    from numpy import get_include
except ImportError:
    import site
    site.addsitedir('../../site-packages')
    from Cython.Distutils import build_ext
    from numpy import get_include

def configuration():
    extensions = [Extension(name="_cpmorphology",
                            sources=["src/cpmorphology.c"],
                            include_dirs=['src']+[get_include()],
                            extra_compile_args=['-O3']),
                  Extension(name="_cpmorphology2",
                            sources=["_cpmorphology2.pyx"],
                            include_dirs=[get_include()],
                            extra_compile_args=['-O3']),
                  Extension(name="_watershed",
                            sources=["_watershed.pyx", "heap_watershed.pxi"],
                            include_dirs=['src']+[get_include()],
                            extra_compile_args=['-O3']),
                  Extension(name="_propagate",
                            sources=["_propagate.pyx", "heap.pxi"],
                            include_dirs=['src']+[get_include()],
                            extra_compile_args=['-O3']),
                  Extension(name="_filter",
                            sources=["_filter.pyx"],
                            include_dirs=['src']+[get_include()],
                            extra_compile_args=['-O3'])
                  ]
    dict = { "name":"cpmath",
             "description":"algorithms for CellProfiler",
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
    

