"""cellprofiler.utilities.setup - compiling files in the utilities module
"""
__test__ = False

import logging
import os
import sys
import subprocess
import traceback

logger = logging.getLogger(__name__)
is_win = sys.platform.startswith("win")
is_win64 = (is_win and (os.environ["PROCESSOR_ARCHITECTURE"] == "AMD64"))
is_msvc = (is_win and sys.version_info[0] >= 2 and sys.version_info[1] >= 6)
is_mingw = (is_win and not is_msvc)

if not hasattr(sys, 'frozen'):
    from distutils.core import setup, Extension
    from distutils.sysconfig import get_config_var

    try:
        from Cython.Distutils import build_ext
    except ImportError:
        import site

        site.addsitedir('../../site-packages')
        from Cython.Distutils import build_ext


    def configuration():
        extensions = []
        extra_link_args = None
        if is_win:
            extra_link_args = ['/MANIFEST']
            extensions += [Extension(name="_get_proper_case_filename",
                                     sources=["get_proper_case_filename.c"],
                                     libraries=["shlwapi", "shell32", "ole32"],
                                     extra_link_args=extra_link_args)]

        dict = {"name": "utilities",
                "description": "utility module for CellProfiler",
                "maintainer": "Lee Kamentsky",
                "maintainer_email": "leek@broad.mit.edu",
                "cmdclass": {'build_ext': build_ext},
                "ext_modules": extensions
                }
        return dict

if __name__ == '__main__':
    if '/' in __file__:
        os.chdir(os.path.dirname(__file__))
    setup(**configuration())
