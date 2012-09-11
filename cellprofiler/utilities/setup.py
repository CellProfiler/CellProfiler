"""cellprofiler.utilities.setup - compiling files in the utilities module

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2012 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

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
    from distutils.core import setup,Extension
    from distutils.sysconfig import get_config_var

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
        extra_link_args = None
        if is_win:
            extra_link_args = ['/MANIFEST']
            extensions += [Extension(name="_get_proper_case_filename",
                                     sources=["get_proper_case_filename.c"],
                                     libraries=["shlwapi", "shell32", "ole32"],
                                     extra_link_args=extra_link_args)]
            
        dict = { "name":"utilities",
                 "description":"utility module for CellProfiler",
                 "maintainer":"Lee Kamentsky",
                 "maintainer_email":"leek@broad.mit.edu",
                 "cmdclass": {'build_ext': build_ext},
                 "ext_modules": extensions
                }
        return dict

def find_javahome():
    """Find JAVA_HOME if it doesn't exist"""
    if hasattr(sys, 'frozen') and is_win:
        #
        # The standard installation of CellProfiler for Windows comes with a JRE
        #
        path = os.path.split(os.path.abspath(sys.argv[0]))[0]
        path = os.path.join(path, "jre")
        for jvm_folder in ("client", "server"):
            jvm_path = os.path.join(path, "bin", jvm_folder, "jvm.dll")
            if os.path.exists(jvm_path):
                # Problem: have seen JAVA_HOME != jvm_path cause DLL load problems
                if os.environ.has_key("JAVA_HOME"):
                    del os.environ["JAVA_HOME"]
                return path
    
    if os.environ.has_key('JAVA_HOME'):
        return os.environ['JAVA_HOME']
    if sys.platform == 'darwin':
        return "Doesn't matter"
    if is_win:
        import _winreg
        java_key_path = 'SOFTWARE\\JavaSoft\\Java Runtime Environment'
        looking_for = java_key_path
        try:
            kjava = _winreg.OpenKey(_winreg.HKEY_LOCAL_MACHINE, java_key_path)
            looking_for = java_key_path + "\\CurrentVersion"
            kjava_values = dict([_winreg.EnumValue(kjava, i)[:2]
                                 for i in range(_winreg.QueryInfoKey(kjava)[1])])
            current_version = kjava_values['CurrentVersion']
            looking_for = java_key_path + '\\' + current_version
            kjava_current = _winreg.OpenKey(_winreg.HKEY_LOCAL_MACHINE,
                                            looking_for)
            kjava_current_values = dict([_winreg.EnumValue(kjava_current, i)[:2]
                                         for i in range(_winreg.QueryInfoKey(kjava_current)[1])])
            return kjava_current_values['JavaHome']
        except:
            logger.error("Failed to find registry entry: %s\n" %looking_for,
                         exc_info=True)
            return None

def find_jdk():
    """Find the JDK under Windows"""
    if sys.platform == 'darwin':
        return "Doesn't matter"
    if is_win:
        import _winreg
        jdk_key_path = 'SOFTWARE\\JavaSoft\\Java Development Kit'
        kjdk = _winreg.OpenKey(_winreg.HKEY_LOCAL_MACHINE, jdk_key_path)
        kjdk_values = dict([_winreg.EnumValue(kjdk, i)[:2]
                             for i in range(_winreg.QueryInfoKey(kjdk)[1])])
        current_version = kjdk_values['CurrentVersion']
        kjdk_current = _winreg.OpenKey(_winreg.HKEY_LOCAL_MACHINE,
                                       jdk_key_path + '\\' + current_version)
        kjdk_current_values = dict([_winreg.EnumValue(kjdk_current, i)[:2]
                                    for i in range(_winreg.QueryInfoKey(kjdk_current)[1])])
        return kjdk_current_values['JavaHome']
    
if __name__ == '__main__':
    if '/' in __file__:
        os.chdir(os.path.dirname(__file__))
    setup(**configuration())
    

