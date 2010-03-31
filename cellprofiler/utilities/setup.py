"""cellprofiler.utilities.setup - compiling files in the utilities module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2010

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import os
import sys
import subprocess
if not hasattr(sys, 'frozen'):
    from distutils.core import setup,Extension
    from distutils.sysconfig import get_config_var
    if sys.platform == 'darwin':
        os.environ['LDSHARED'] = get_config_var("LDSHARED").replace("-bundle", "-dynamiclib")

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
        if sys.platform.startswith('win'):
            extensions += [Extension(name="_get_proper_case_filename",
                                     sources=["get_proper_case_filename.c"],
                                     libraries=["shlwapi", "shell32", "ole32"],
                                     extra_compile_args=['-O3'])]
        try:
            #
            # Find JAVA_HOME, possibly from Windows registry
            #
            java_home = find_javahome()
            jdk_home = find_jdk()
            print "Using jdk_home = %s"%jdk_home
            include_dirs = [get_include()]
            extra_link_args = None
            libraries = None
            library_dirs = None
            if sys.platform.startswith('win'):
                #
                # Build libjvm from jvm.dll on Windows
                #
                if sys.platform.startswith('win'):
                    cmd = ["dlltool", "--dllname", 
                           os.path.join(jdk_home,"jre\\bin\\client\\jvm.dll"),
                           "--output-lib","libjvm.a",
                           "--input-def","jvm.def",
                           "--kill-at"]
                    p = subprocess.Popen(cmd)
                    p.communicate()
            
                if jdk_home is not None:
                    jdk_include = os.path.join(jdk_home, "include")
                    jdk_include_plat = os.path.join(jdk_include, sys.platform)
                    include_dirs += [jdk_include, jdk_include_plat]
                library_dirs = [os.path.abspath(".")]
                libraries = ["jvm"]
            elif sys.platform == 'darwin':
                include_dirs += ['/System/Library/Frameworks/JavaVM.framework/Headers']
                extra_link_args = ['-framework', 'JavaVM']
            elif sys.platform.startswith('linux'):
                include_dirs += [os.path.join(java_home,'include'),
                                 os.path.join(java_home,'include','linux')]
                library_dirs = [os.path.join(java_home,'jre','lib','amd64','server')]
                libraries = ["jvm"]
            extensions += [Extension(name="javabridge",
                                     sources=["javabridge.pyx"],
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
    if os.environ.has_key('JAVA_HOME'):
        return os.environ['JAVA_HOME']
    if sys.platform == 'darwin':
        return "Doesn't matter"
    if sys.platform.startswith('win'):
        import _winreg
        java_key_path = 'SOFTWARE\\JavaSoft\\Java Runtime Environment'
        kjava = _winreg.OpenKey(_winreg.HKEY_LOCAL_MACHINE, java_key_path)
        kjava_values = dict([_winreg.EnumValue(kjava, i)[:2]
                             for i in range(_winreg.QueryInfoKey(kjava)[1])])
        current_version = kjava_values['CurrentVersion']
        kjava_current = _winreg.OpenKey(_winreg.HKEY_LOCAL_MACHINE,
                                        java_key_path + '\\' + current_version)
        kjava_current_values = dict([_winreg.EnumValue(kjava_current, i)[:2]
                                     for i in range(_winreg.QueryInfoKey(kjava_current)[1])])
        return kjava_current_values['JavaHome']

def find_jdk():
    """Find the JDK under Windows"""
    if sys.platform == 'darwin':
        return "Doesn't matter"
    if sys.platform.startswith('win'):
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
    

