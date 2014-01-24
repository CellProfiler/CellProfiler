#!/usr/bin/env python
#
# Install.py - install all packages required for CellProfiler
#
"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
import distutils.core
import optparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile

SETUP_TOOLS      = "setuptools-0.6c9.tar.gz"
NUMPY_SPEC       = "numpy>=1.2.1"
NUMPY            = "numpy-1.2.1.tar.gz"
SCIPY_SPEC       = "scipy>=0.7.0rc2"
SCIPY            = "scipy-0.7.0rc2.tar.gz"
CYTHON           = "Cython-0.10.3.zip"
MATPLOTLIB_SPEC  = "matplotlib>=0.98.5.2"
MATPLOTLIB       = "matplotlib-0.98.5.2.tar.gz"
JPYPE            = "JPype-0.5.4.zip"
PYLINT           = "pylint-0.15.2.zip"
NOSE_SPEC        = "nose"
NOSEXUNIT_SPEC   = "NoseXUnit"
LOGILAB_COMMON_SPEC = "logilab-common"
PIL_SPEC         = "PIL"
WXPYTHON_EGG_NT  = "wxPython-2.8.4.0.001-py2.5-win32.egg"
IMAGING          = "Imaging-1.1.6.tar.gz"
PYFFMPEG         = "pyffmpeg.zip"
FFMPEG           = "ffmpeg-export-snapshot.tar.bz2"
DECORATOR        = "decorator"
ZLIB             = "zlib-1.2.3.tar.gz"
JPEGLIB          = "jpegsrc.v6b.tar.gz"
JPEG_PKG         = "jpeg-6b"

default_root = os.path.abspath(os.path.split(__file__)[0])

parser = optparse.OptionParser()
parser.add_option("-v","--verbose",dest="verbose",
                  action="store_true", default=False,
                  help="Blather on about what we're doing")
parser.add_option("-r","--root",dest="root",
                  default= default_root,
                  help="pyCellProfiler directory")
parser.add_option("-d","--dry-run",dest="dry_run",
                  action="store_true", default=False,
                  help="Don't do anything - just talk about what we'd do")
(options,args) = parser.parse_args()
build_path = os.path.join(options.root,"build")
package_path = os.path.join(options.root,"packages")

def make_string_literal(x):
    """Return a string literal that evaluates to the argument passed
    
    Returns a string suitable for inclusion into a Python script, complete
    with open and close quotes and escape sequences for any embedded
    backslashes or quotes.
    """
    escaped_x = x.replace("\\","\\\\").replace("'","\\'")
    return "'%(escaped_x)s'"%(locals())

def windows_check_prerequisites():
    """Make sure we have mingw32, make and some other things available"""
    if options.verbose:
        print "Checking for GCC"
    try:
        run_command("gcc --version")
    except:
        raise EnvironmentError("""Can't find GCC - mingw32 must be installed.
        Please run MinGW-5.1.4.exe, located in the packages directory
        and make sure to install the g++ and g77 compilers as well as
        MinGW make.""")
    if options.verbose:
        print "Checking for make"
    try:
        run_command("make -v")
    except:
        raise EnvironmentError("""Can't find make - Please run MSYS-1.0.10.exe
located in the packages directory and add the MSYS bin directory to your path""")
    try:
        run_command("sh --version")
    except:
        raise EnvironmentError("""Can't find sh - MSYS may not be properly installed""")
    distutils_cfg_path = os.path.join(os.path.split(sys.executable)[0],'Lib','distutils','pydistutils.cfg')
    if not os.path.isfile(distutils_cfg_path):
        if options.verbose:
            print "Configuring distutils to use mingw32"
        if not options.dry_run:
            fid = open(distutils_cfg_path,'w')
            fid.write("[build_ext]\ncompiler=mingw32\n[build]\ncompiler=mingw32\n")
            fid.close()
    if not os.environ.has_key('ATLAS'):
        if options.verbose:
            print "Using Atlas LAPACK library from packages."
        os.environ['ATLAS'] = os.path.join(package_path,'lapack')
    if not os.environ.has_key('BLAS'):
        if options.verbose:
            print "Using BLAS LAPACK library from packages."
        os.environ['BLAS'] = os.path.join(package_path,'lapack')

def unpack_zip(filename, targetpath):
    """Unpack a .zip file"""
    if not zipfile.is_zipfile(filename):
        raise IOError("Could not find %s"%(filename))
    if options.verbose:
        print "Unpacking %s to %s"%(filename, targetpath)
    if not options.dry_run:
        z = zipfile.ZipFile(filename)
        #
        # No extractall until 2.6 - this code is from zipfile._extract_member
        #
        for zip_info in z.infolist():
            # build the destination pathname, replacing
            # forward slashes to platform specific separators.
            if targetpath[-1:] in (os.path.sep, os.path.altsep):
                targetpath = targetpath[:-1]

            # don't include leading "/" from file name if present
            if zip_info.filename[0] == '/':
                filepath = os.path.join(targetpath, zip_info.filename[1:])
            else:
                filepath = os.path.join(targetpath, zip_info.filename)
    
            filepath = os.path.normpath(filepath)
    
            # Create all upper directories if necessary.
            upperdirs = os.path.dirname(filepath)
            if upperdirs and not os.path.exists(upperdirs):
                os.makedirs(upperdirs)
    
            if zip_info.filename[-1] == '/':
                if os.path.isdir(filepath):
                    continue
                os.mkdir(filepath)
            else:
                data = z.read(zip_info.filename)
                target = file(filepath, "wb")
                target.write(data)
                target.close()
        z.close()

def unpack_tar_gz(filename, target_directory, type="gz"):
    """Unpack a .tar.gz file

    filename - path to file to unpack
    target_directory - root directory for the contents
    type - "gz" for gzip files, "bz2" for bzip2 files, None for tar files
    """
    
    if not tarfile.is_tarfile(filename):
        raise IOError("Could not find %s"%(filename))
    if options.verbose:
        print "Unpacking %s to %s"%(filename, target_directory)
    if not options.dry_run:
        tf = tarfile.open(filename,(type and "r:%s"%(type)) or "r")
        tf.extractall(target_directory)
        tf.close()

def unpack_package(package):
    """Unpack a package file to the build directory"""
    if ".tar.gz" in package:
        package_name = package[:-len(".tar.gz")]
        package_dest = os.path.join(build_path, package_name)
        unpack_tar_gz(os.path.join(package_path, package),
                      package_dest)
    elif ".tar.bz2" in package:
        package_name = package[:-len(".tar.bz2")]
        package_dest = os.path.join(build_path, package_name)
        unpack_tar_gz(os.path.join(package_path, package),
                      package_dest,"bz2")
    elif ".zip" in package:
        package_name = package[:-len(".zip")]
        package_dest = os.path.join(build_path, package_name)
        unpack_zip(os.path.join(package_path, package),
                   package_dest)
    else:
        raise IOError("Don't know how to unpack %s"%(package))
    return (package_name,package_dest)

def run_command(command):
    if options.verbose:
        print "Running %s"%(command)
    if not options.dry_run:
        if options.verbose:
            result = subprocess.call(command.split(" "))
        else:
            result = subprocess.call(command.split(" "), subprocess.PIPE)
        if result:
            raise IOError("Command did not complete successfully: %s"%(command))

def install_easy_install(packagespec):
    if options.verbose:
        print "Easy-installing %s"%(packagespec)
    if not options.dry_run:
        run_command("python -m easy_install %s"%(packagespec))

MAKE = "make"
if os.name == 'nt':
    windows_check_prerequisites()
    BUILD = "build --compiler=mingw32"
else:
    BUILD = "build"

try:
    import easy_install
except:
    setup_tools,setup_tools_dir = unpack_package(SETUP_TOOLS)
    os.chdir(os.path.join(setup_tools_dir,setup_tools))
    command = "python setup.py %s install"%(BUILD)
    run_command(command)
    if os.name == 'nt':
        distutils_cfg_path = os.path.join(os.path.split(sys.executable)[0],'Lib','distutils','pydistutils.cfg')
        if not os.path.isfile(distutils_cfg_path):
            if options.verbose:
                print "Configuring distutils to use mingw32"
            if not options.dry_run:
                fid = open(distutils_cfg_path,'w')
                fid.write("[build_ext]\ncompiler=mingw32\n[build]\ncompiler=mingw32\n")
                fid.close()

try:
    import Cython
except:
    cython_pkg,cython_dir = unpack_package(CYTHON)
    os.chdir(os.path.join(cython_dir,cython_pkg))
    command = "python setup.py %s install"%(BUILD)
    run_command(command)

try:
    import wx
except:
    if os.name == 'nt' and sys.version_info[:2]==(2,5):
        # We have an egg for this... really not ideal, it's best to
        # install it yourself
        wx_egg = os.path.join(package_path,WXPYTHON_EGG_NT)
        command = "python -m easy_install %s"%(wx_egg)
        run_command(command)
    else:
        # This doesn't work so hot under Windows - hope your luck is better
        try:
            install_easy_install('wxPython')
        except:
            sys.stderr.write("""Easy-install didn't do so well installing wxPython
on your system and you're getting this message because that's not such a
surprise. You should go to http://www.wxpython.org/download.php and install
the package using whatever installer is appropriate for your system.""")
            sys.exit(-1)

try:
    import decorator
except:
    install_easy_install(DECORATOR)

# Can't install PIL if imported "successfully", but later part
# of script fails, so run it in another Python instance
script = """
try:
    from sys import exit
    import PIL.Image
    #make sure that we have PIL installed with the libraries we want
    ignore = PIL.Image.core.zip_decoder
    ignore = PIL.Image.core.jpeg_decoder
    print "Python imaging library installed"
except:
    print "Python imaging library not correctly installed"
    exit(-1)
    """
p = subprocess.Popen("python",stdin=subprocess.PIPE,stdout=subprocess.PIPE)
output = p.communicate(script)
if p.returncode:
    if os.name == 'nt':
        # Build the zip and JPEG packages
        zip_pkg,zip_dir = unpack_package(ZLIB)
        zip_path = os.path.join(zip_dir,zip_pkg)
        os.chdir(zip_path)
        run_command("sh ./configure")
        run_command(MAKE)
        
        ignore, jpeg_dir = unpack_package(JPEGLIB)
        jpeg_path = os.path.join(jpeg_dir,JPEG_PKG)
        os.chdir(jpeg_path)
        run_command("sh ./configure")
        run_command(MAKE)
    pil_pkg,pil_dir = unpack_package(IMAGING)
    os.chdir(os.path.join(pil_dir,pil_pkg))
    if os.name == 'nt':
        # We hack the setup file here to point at the zip and jpeg libraries
        fd = open('setup.py','r')
        setup_lines = fd.readlines()
        fd.close()
        fd = open('setup.py','w')
        jpeg_literal_path = make_string_literal(jpeg_path)
        zlib_literal_path = make_string_literal(zip_path)  
        for line in setup_lines:
            if re.search('^JPEG_ROOT\\s*=\\s*None',line):
                line = ("JPEG_ROOT = (%(jpeg_literal_path)s,%(jpeg_literal_path)s)\n" %
                        (globals()))
            elif re.search('^ZLIB_ROOT\\s*=\\s*None',line):
                line = ("ZLIB_ROOT = (%(zlib_literal_path)s,%(zlib_literal_path)s)\n" % 
                        (globals()))
            fd.write(line)
        fd.close()
    command = "python setup.py %s install"%(BUILD)
    run_command(command)

try:
    import numpy
except:
    try:
	install_easy_install(NUMPY_SPEC)
    except:
	numpy_pkg, numpy_dir = unpack_package(NUMPY)
        os.chdir(os.path.join(numpy_dir, numpy_pkg))
        command = "python setup.py %s install"%(BUILD)
        run_command(command)

try:
    import scipy
except:
    try:
        install_easy_install(SCIPY_SPEC)
    except:
        scipy_pkg, scipy_dir = unpack_package(SCIPY)
        os.chdir(os.path.join(scipy_dir, scipy_pkg))
        command = "python setup.py %s install"%(BUILD)
        run_command(command)

try:
    install_easy_install(MATPLOTLIB_SPEC)
except:
    matplotlib_pkg, matplotlib_dir = unpack_package(MATPLOTLIB)
    os.chdir(os.path.join(matplotlib_dir, matplotlib_pkg))
    command = "python setup.py %s install"%(BUILD)
    try:
        run_command(command)
    except:
        print """
-----------------------------------------------------------------------------
So so sorry - if you're on Windows, please manually install
matplotlib-0.98.5.2.win32-py2.5.exe from the packages directory or download
an equivalent for your system from http://sourceforge.net/project/showfiles.php?group_id=80706&package_id=278194&release_id=646644
Then start the install script again.
-----------------------------------------------------------------------------
"""
        exit(0)
#
# Removing support for jpype for now
#
if False:
    #
    # Need to install jpype semi-manually because it assumes MSVC build tools
    # and needed mods. Also no easy_install of it.
    #    
    try:
        import jpype
    except:
        if not os.environ.has_key('JAVA_HOME'):
            if options.verbose:
                print "Using includes from Java JRE 1.6"
            os.environ['JAVA_HOME'] = os.path.join(package_path,"java")
    
        jpype_pkg,jpype_dir = unpack_package(JPYPE)
        os.chdir(os.path.join(jpype_dir,jpype_pkg))
        command = "python setup.py %s install"%(BUILD)
        run_command(command)

#
# Need to install pylint separately from Nose because the windows
# install is flawed and fixed here. Pylint seems to be moribund...?
#
try:
    import pylint
except:
    pylint_pkg,pylint_dir = unpack_package(PYLINT)
    os.chdir(os.path.join(pylint_dir,pylint_pkg))
    command = "python setup.py %s install"%(BUILD)
    run_command(command)
install_easy_install(LOGILAB_COMMON_SPEC)
install_easy_install(NOSE_SPEC)
install_easy_install(NOSEXUNIT_SPEC)
#
# Install the prebuilt pyffmpeg package for Windows or build ffmpeg
# for unixishes and setup pyffmpeg
#
try:
    import pyffmpeg
except:
    pyffmpeg_pkg, pyffmpeg_dir = unpack_package(PYFFMPEG)
    if os.name == 'nt':
        os.environ["FFMPEG_ROOT"]=os.path.join(package_path,'ffmpeg-windows')
        os.chdir(pyffmpeg_dir)
        os.chdir("pyffmpeg")
        command = "python ..\\setup.py %s install"%(BUILD)
        run_command(command)
    else:
        ffmpeg_pkg, ffmpeg_dir = unpack_package(FFMPEG)
        os.chdir(os.path.join(ffmpeg_dir,ffmpeg_pkg))
        run_command("sh ./configure")
        run_command("make")
        run_command("make install")
        os.chdir(pyffmpeg_dir)
        os.chdir("pyffmpeg")
        command = "python ../setup.py %s install"%(BUILD)
        run_command(command)
    
