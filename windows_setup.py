"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2013 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

"""Windows setup file
To invoke, from the command-line type:
python windows_setup.py py2exe msi

This script will create three subdirectories
build: contains the collection of files needed during packaging
dist:  the contents that need to be given to the user to run WormProfiler.
output: contains the .msi if you did the msi commmand
"""
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2013 Broad Institute
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org

from distutils.core import setup
import distutils.core
import distutils.errors
import py2exe
import sys
import glob
import subprocess
import re
import os
import _winreg
import matplotlib
import tempfile
import xml.dom.minidom

# make sure external dependencies match requirements
import external_dependencies
external_dependencies.fetch_external_dependencies('fail')

from cellprofiler.utilities.version import version_number, dotted_version, version_string
f = open("cellprofiler/frozen_version.py", "w")
f.write("# MACHINE_GENERATED\nversion_string = '%s'" % version_string)
f.close()

is_win64 = (os.environ["PROCESSOR_ARCHITECTURE"] == "AMD64")
is_2_6 = sys.version_info[0] >= 2 and sys.version_info[1] >= 6
vcredist = os.path.join("windows",
                        "vcredist_x64.exe" if is_win64
                        else "vcredist_x86.exe")
do_modify = is_2_6 and not os.path.exists(vcredist)

class CellProfilerMSI(distutils.core.Command):
    description = "Make CellProfiler.msi using the CellProfiler.iss InnoSetup compiler"
    user_options = []
    
    def initialize_options(self):
        pass
    
    def finalize_options(self):
        pass
    
    def run(self):
        revision = str(version_number)
        if is_2_6 and do_modify:
            self.modify_manifest("CellProfiler.exe")
            self.modify_manifest("python26.dll;#2")
            self.modify_manifest("_imaging.pyd;#2")
            self.modify_manifest("wx._stc.pyd;#2")
            self.modify_manifest("wx._controls_.pyd;#2")
            self.modify_manifest("wx._aui.pyd;#2")
            self.modify_manifest("wx._misc_.pyd;#2")
            self.modify_manifest("wx._windows_.pyd;#2")
            self.modify_manifest("wx._gdi_.pyd;#2")
            self.modify_manifest("wx._html.pyd;#2")
            self.modify_manifest("wx._grid.pyd;#2")
            self.modify_manifest("PIL._imaging.pyd;#2")
            self.modify_manifest("wx._core_.pyd;#2")
        fd = open("version.iss", "w")
        fd.write("""
AppVerName=CellProfiler %s r%s
OutputBaseFilename=CellProfiler_%s_win%d_r%s
""" % (dotted_version, revision, dotted_version, (64 if is_win64 else 32), revision))
        fd.close()
        if is_win64:
            cell_profiler_iss = "CellProfiler64.iss"
            cell_profiler_setup = "CellProfiler_%s_win64_r%s.exe" % (dotted_version, revision)
        else:
            cell_profiler_iss = "CellProfiler.iss"
            cell_profiler_setup = "CellProfiler_%s_win32_r%s.exe" % (dotted_version, revision)
        required_files = ["dist\\CellProfiler.exe",cell_profiler_iss]
        compile_command = self.__compile_command()
        compile_command = compile_command.replace("%1",cell_profiler_iss)
        self.make_file(required_files,"Output\\"+cell_profiler_setup, 
                       subprocess.check_call,([compile_command]),
                       "Compiling %s" % cell_profiler_iss)
        os.remove("version.iss")
        
    def modify_manifest(self, resource_name):
        '''Change the manifest of a resource to match the CRT
        
        resource_name - the name of the executable or DLL maybe + ;#2
                        to pick up the manifest in the assembly
        Read the manifest using "mt", hack the XML to change the version
        and reinsert it into the resource.
        '''
        #
        # Find mt
        #
        try:
            key_path = r"SOFTWARE\Microsoft\Microsoft SDKs\Windows\v7.0"
            key = _winreg.OpenKey(_winreg.HKEY_LOCAL_MACHINE, key_path)
            path = _winreg.QueryValueEx(key, "InstallationFolder")[0]
            mt = os.path.join(path,"bin", "mt.exe")
            key.Close()
        except:
            print "Using MT from path"
            mt = "mt"
        directory = "dist"
        msvcrt = xml.dom.minidom.parse(os.path.join(
            directory, "Microsoft.VC90.CRT.manifest"))
        msvcrt_assembly = msvcrt.getElementsByTagName("assembly")[0]
        msvcrt_assembly_identity = msvcrt.getElementsByTagName("assemblyIdentity")[0]
        msvcrt_version = msvcrt_assembly_identity.getAttribute("version")
        
        manifest_file_name = tempfile.mktemp()
        pipe = subprocess.Popen(
            (mt,
             "-inputresource:%s" % os.path.join(directory, resource_name),
             "-out:%s" % manifest_file_name))
        pipe.communicate()
        if not os.path.exists(manifest_file_name):
            return
        
        manifest = xml.dom.minidom.parse(manifest_file_name)
        manifest_assembly = manifest.getElementsByTagName("assembly")[0]
        manifest_dependencies = manifest_assembly.getElementsByTagName("dependency")
        for dependency in manifest_dependencies:
            dependent_assemblies = dependency.getElementsByTagName("dependentAssembly")
            for dependent_assembly in dependent_assemblies:
                assembly_identity = dependent_assembly.getElementsByTagName("assemblyIdentity")[0]
                if assembly_identity.getAttribute("name") == "Microsoft.VC90.CRT":
                    assembly_identity.setAttribute("version", msvcrt_version)
        fd = open(manifest_file_name, "wt")
        fd.write(manifest.toprettyxml())
        fd.close()
        
        pipe = subprocess.Popen(
            (mt,
             "-outputresource:%s" % os.path.join(directory, resource_name),
             "-manifest",
             manifest_file_name))
        pipe.communicate()
        os.remove(manifest_file_name)
    
    def __compile_command(self):
        """Return the command to use to compile an .iss file
        """
        try:
            key = _winreg.OpenKey(_winreg.HKEY_CLASSES_ROOT, 
                                   "InnoSetupScriptFile\\shell\\Compile\\command")
            result = _winreg.QueryValueEx(key,None)[0]
            key.Close()
            return result
        except WindowsError:
            if key:
                key.Close()
            raise distutils.errors.DistutilsFileError, "Inno Setup does not seem to be installed properly. Specifically, there is no entry in the HKEY_CLASSES_ROOT for InnoSetupScriptFile\\shell\\Compile\\command"

opts = {
    'py2exe': { "includes" : ["numpy", "scipy","PIL","wx",
                              "matplotlib", "matplotlib.numerix.random_array",
                              "email.iterators",
                              "cellprofiler.modules.*"],
                'excludes': ['pylab', 'Tkinter', 'Cython', 'IPython'],
                'dll_excludes': ["jvm.dll"]
              },
    'msi': {}
       }

data_files = []
####################################
#
# Ilastik fixups
#
####################################
try:
    import vigra
    import ilastik
    vigranumpy_path = os.path.join(vigra.__path__[0],"vigranumpycore.pyd")
    if os.path.exists(vigranumpy_path):
        data_files += [(".",[vigranumpy_path])]
    opts['py2exe']['includes'] += [
        "vigra", "vigra.impex",
        "h5py", "h5py._stub", "h5py._conv", "h5py.utils", "h5py._proxy",
        "PyQt4", "PyQt4.QtOpenGL", "PyQt4.uic", "sip",
        "zmq", "zmq.utils", "zmq.utils.jsonapi", "zmq.utils.strtypes"]
    opts['py2exe']['excludes'] += ["ilastik"]
    #
    # Put path to QT dlls in PATH environment variable
    #
    import PyQt4
    pyqt4_path = os.path.split(PyQt4.__file__)[0]
    os.environ["PATH"] = os.environ["PATH"] + ";" + pyqt4_path
    il_path = ilastik.__path__[0]
    for root, subFolders, files in os.walk(il_path):
        #
        # Do not include experimental modules
        #
        relative_path = root[(len(il_path)+1):]
        if any([relative_path.startswith(os.path.join('modules',x))
                for x in (
            'automatic_segmentation','object_picking',
            'connected_components')]):
            continue
        dest = os.path.join('site-packages','ilastik')
        if root != il_path:
            dest = os.path.join(dest, relative_path)
        ilastik_files = [os.path.join(root, f) for f in files 
                         if f.endswith(".ui") or f.endswith(".png") or
                         f.endswith(".py")]
        if len(ilastik_files) > 0:
            data_files += [(dest, ilastik_files)]
                                     
    #
    # port_v3 is for Python 3.1
    #
    opts['py2exe']['excludes'] += ['PyQt4.uic.port_v3']
    try:
        import OpenGL.platform.win32
        opts['py2exe']['includes'] += ['OpenGL.platform.win32', 
                                       'OpenGL.arrays.*']
    except:
        print "This installation will not supply OpenGL support for Ilastik"
except:
    print "This installation will not include Ilastik"

##################################
#
# Scipy fixups
#
##################################
try:
    # Include this package if present
    import scipy.io.matlab.streams
    opts['py2exe']['includes'] += [ "scipy.io.matlab.streams"]
except:
    pass

##############################################
#
# 0MQ fixups
#
# libzmq.dll is not found by py2exe, so we
# semi-manually copy it to the right place
# and ask py2exe to ignore not being able
# to find it.
##############################################
try:
    import zmq
    zmq_loc = os.path.split(zmq.__file__)[0]
    os.environ["PATH"] = os.environ["PATH"] + ";"+zmq_loc
    #
    # 2.2 added this and distutils did not find it
    # Not present prior to 2.2
    #
    if zmq.__version__ >= "2.2.0":
        opts['py2exe']['includes'] += ["zmq.core.pysocket"]
except:
    print "This installation will not include 0MQ"

##############################################
#
# Visual Studio DLL fixups - Much better to use the official installer
#                            than to hand-patch the manifest and DLLs.
#
##############################################

if do_modify:
    # A trick to load the dlls
    if is_win64:
        path = r"SOFTWARE\WOW6432node\Microsoft\VisualStudio\9.0\Setup\VC"
    else:
        path = r"SOFTWARE\Microsoft\VisualStudio\9.0\Setup\VC"
    key = _winreg.OpenKey(_winreg.HKEY_LOCAL_MACHINE, path)
    product_dir = _winreg.QueryValueEx(key, "ProductDir")[0]
    key.Close()
    redist = os.path.join(product_dir, r"redist\x86\Microsoft.VC90.CRT")
    data_files += [(".",[os.path.join(redist, x)
                         for x in ("Microsoft.VC90.CRT.manifest", 
                                   "msvcr90.dll",
                                   "msvcm90.dll",
                                   "msvcp90.dll")])]
else:
    opts['py2exe']['dll_excludes'] += ["msvcr90.dll", "msvcm90.dll", "msvcp90.dll"]

data_files += [('cellprofiler\\icons',
               ['cellprofiler\\icons\\%s'%(x) 
                for x in os.listdir('cellprofiler\\icons')
                if x.endswith(".png") 
                or x.endswith(".psd") or x.endswith(".txt")]),
              ('imagej\\jars', 
               ['imagej\\jars\\%s' % x for x in os.listdir('imagej\\jars')])]
data_files += matplotlib.get_py2exe_datafiles()
################################
#
# Collect the JVM
#
################################

from cellprofiler.utilities.setup import find_jdk
jdk_dir = find_jdk()
def add_jre_files(path):
    files = []
    directories = []
    local_path = os.path.join(jdk_dir, path)
    for filename in os.listdir(local_path):
        if filename.startswith("."):
            continue
        local_file = os.path.join(jdk_dir, path, filename)
        relative_path = os.path.join(path, filename) 
        if os.path.isfile(local_file):
            files.append(local_file)
        elif os.path.isdir(local_file):
            directories.append(relative_path)
    if len(files):
        data_files.append([path, files])
    for subdirectory in directories:
        add_jre_files(subdirectory)
    
add_jre_files("jre")
data_files += [("jre\\ext", [os.path.join(jdk_dir, "lib", "tools.jar")])]
#
# Call setup
#
try:
    setup(console=[{'script':'CellProfiler.py',
                    'icon_resources':[(1,'CellProfilerIcon.ico')]},
                   {'script':'cellprofiler\\analysis_worker.py'}],
          name='Cell Profiler',
          data_files = data_files,
          cmdclass={'msi':CellProfilerMSI
                    },
          options=opts)
finally:
    try:
        import cellprofiler.utilities.jutil as jutil
        jutil.kill_vm()
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(0)
    except:
        import traceback
        traceback.print_exc()
        print "Caught exception while killing VM"
