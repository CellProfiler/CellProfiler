# Install.py - install all packages required for CellProfiler
#
import distutils.core
import optparse
import os
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
JPYPE            = "JPype-0.5.4.zip"
MLABWRAP         = "mlabwrap-1.0.tar.gz"
PYLINT           = "pylint-0.15.2.zip"
NOSE_SPEC        = "nose>=0.10.4"
NOSEXUNIT_SPEC   = "NoseXUnit"
PIL_SPEC         = "PIL"
WXPYTHON_EGG_NT  = "wxPython-2.8.4.0.001-py2.5-win32.egg"
IMAGING          = "Imaging-1.1.6.tar.gz"

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

def windows_check_prerequisites():
    """Make sure we have mingw32, make and some other things available"""
    if options.verbose:
        print "Checking for GCC"
    if run_command("gcc --version"):
        raise EnvironmentError("""Can't find GCC - mingw32 must be installed.
        Please run MinGW-5.1.4.exe, located in the packages directory
        and make sure to install the g++ and g77 compilers as well as
        MinGW make.""")
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
            result = subprocess.call(command)
        else:
            result = subprocess.call(command, subprocess.PIPE)
        if result:
            raise IOError("Command did not complete successfully: %s"%(command))

def install_easy_install(packagespec):
    if options.verbose:
        print "Easy-installing %s"%(packagespec)
    if not options.dry_run:
        run_command("python -m easy_install %s"%(packagespec))
    
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

install_easy_install(MATPLOTLIB_SPEC)
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
try:
    import mlabwrap
except:
    mlabwrap_pkg, mlabwrap_dir = unpack_package(MLABWRAP)
    # Copy a specially modded setup.py over
    shutil.copy(os.path.join(package_path,'mlabwrap','setup.py'),
                os.path.join(mlabwrap_dir,mlabwrap_pkg,'setup.py'))
    os.chdir(os.path.join(mlabwrap_dir,mlabwrap_pkg))
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
install_easy_install(NOSE_SPEC)
install_easy_install(NOSEXUNIT_SPEC)
try:
    import PIL
except:
    pil_pkg,pil_dir = unpack_package(IMAGING)
    os.chdir(os.path.join(pil_dir,pil_pkg))
    command = "python setup.py %s install"%(BUILD)
    run_command(command)
