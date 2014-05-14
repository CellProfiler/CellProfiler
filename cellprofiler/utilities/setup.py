"""cellprofiler.utilities.setup - compiling files in the utilities module

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

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

def get_visual_studio_version():
    try:
        ver = os.environ['VisualStudioVersion']
    except:
        ver = "9.0"
        print 'Warning: Could not find the version of Microsoft Visual Studio. Are you in a Visual Studio command prompt?'
        print 'Assuming version', ver
    return ver

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
        try:
            #
            # Find JAVA_HOME, possibly from Windows registry
            #
            java_home = find_javahome()
            jdk_home = find_jdk()
            logger.debug("Using jdk_home = %s"%jdk_home)
            include_dirs = [get_include()]
            libraries = None
            library_dirs = None
            javabridge_sources = [ "javabridge.pyx" ]
            if is_win:
                if jdk_home is not None:
                    jdk_include = os.path.join(jdk_home, "include")
                    jdk_include_plat = os.path.join(jdk_include, sys.platform)
                    include_dirs += [jdk_include, jdk_include_plat]
                if is_mingw:
                    #
                    # Build libjvm from jvm.dll on Windows.
                    # This assumes that we're using mingw32 for build
                    #
                    cmd = ["dlltool", "--dllname", 
                           os.path.join(jdk_home,"jre\\bin\\client\\jvm.dll"),
                           "--output-lib","libjvm.a",
                           "--input-def","jvm.def",
                           "--kill-at"]
                    p = subprocess.Popen(cmd)
                    p.communicate()
                    library_dirs = [os.path.abspath(".")]
                else:
                    #
                    # Use the MSVC lib in the JDK
                    #
                    jdk_lib = os.path.join(jdk_home, "lib")
                    library_dirs = [jdk_lib]
                    if float(get_visual_studio_version()) < 12:
                        javabridge_sources.append("strtoull.c")
            
                libraries = ["jvm"]
            elif sys.platform == 'darwin':
                javabridge_sources += [ "mac_javabridge_utils.c" ]
                include_dirs += ['/System/Library/Frameworks/JavaVM.framework/Headers']
                extra_link_args = ['-framework', 'JavaVM']
            elif sys.platform.startswith('linux'):

                include_dirs += [os.path.join(java_home,'include'),
                                 os.path.join(java_home,'include','linux')]
                library_dirs = [
                    os.path.join(java_home,'jre','lib',
                                 os.environ.get('HOSTTYPE','amd64'),'server')]
                #
                # Use findlibjvm to find the JVM in case above doesn't
                # work.
                #
                path = os.path.split(__file__)[0]
                p = subprocess.Popen(["java","-cp", path, "findlibjvm"],
                                     stdout=subprocess.PIPE)
                stdout, stderr = p.communicate()
                jvm_dir = stdout.strip()
                library_dirs.append(jvm_dir)
                libraries = ["jvm"]
            extensions += [Extension(name="javabridge",
                                     sources=javabridge_sources,
                                     libraries=libraries,
                                     library_dirs=library_dirs,
                                     include_dirs=include_dirs,
                                     extra_link_args=extra_link_args)]
        except Exception, e:
            print "WARNING: Java and JVM is not installed - Images will be loaded using PIL (%s)"%(str(e))
            
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
        return None
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
        return None
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
    

