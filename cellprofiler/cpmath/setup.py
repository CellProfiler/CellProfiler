"""setup.py - setup to build C modules for cpmath

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

from distutils.core import setup,Extension
import os
import sys
is_win = sys.platform.startswith("win")
try:
    from Cython.Distutils import build_ext
    from numpy import get_include
except ImportError:
    import site
    site.addsitedir('../../site-packages')
    from Cython.Distutils import build_ext
    from numpy import get_include

def configuration():
    if is_win:
        extra_compile_args = None
        extra_link_args = ['/MANIFEST']
    else:
        extra_compile_args = ['-O3']
        extra_link_args = None
    extensions = [Extension(name="_cpmorphology",
                            sources=["src/cpmorphology.c"],
                            include_dirs=['src']+[get_include()],
                            extra_compile_args=extra_compile_args,
                            extra_link_args=extra_link_args),
                  Extension(name="_cpmorphology2",
                            sources=["_cpmorphology2.pyx"],
                            include_dirs=[get_include()],
                            extra_compile_args=extra_compile_args,
                            extra_link_args=extra_link_args),
                  Extension(name="_watershed",
                            sources=["_watershed.pyx", "heap_watershed.pxi"],
                            include_dirs=['src']+[get_include()],
                            extra_compile_args=extra_compile_args,
                            extra_link_args=extra_link_args),
                  Extension(name="_propagate",
                            sources=["_propagate.pyx", "heap.pxi"],
                            include_dirs=['src']+[get_include()],
                            extra_compile_args=extra_compile_args,
                            extra_link_args=extra_link_args),
                  Extension(name="_filter",
                            sources=["_filter.pyx"],
                            include_dirs=['src']+[get_include()],
                            extra_compile_args=extra_compile_args,
                            extra_link_args=extra_link_args),
                  Extension(name="_lapjv",
                            sources=["_lapjv.pyx"],
                            include_dirs=['src']+[get_include()],
                            extra_compile_args=extra_compile_args,
                            extra_link_args=extra_link_args),
                  Extension(name="_convex_hull",
                            sources=["_convex_hull.pyx"],
                            include_dirs=['src']+[get_include()],
                            extra_compile_args=extra_compile_args,
                            extra_link_args=extra_link_args),
                  ]
    dict = { "name":"cpmath",
             "description":"algorithms for CellProfiler",
             "maintainer":"Lee Kamentsky",
             "maintainer_email":"leek@broadinstitute.org",
             "cmdclass": {'build_ext': build_ext},
             "ext_modules": extensions
            }
    return dict

if __name__ == '__main__':
    if '/' in __file__:
        os.chdir(os.path.dirname(__file__))
    setup(**configuration())
    

