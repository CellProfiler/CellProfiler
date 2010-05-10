"""Windows setup file
To invoke, from the command-line type:
python windows_setup.py py2exe msi

This script will create three subdirectories
build: contains the collection of files needed during packaging
dist:  the contents that need to be given to the user to run WormProfiler.
output: contains the .msi if you did the msi commmand
"""
from distutils.core import setup
import distutils.core
import py2exe
import sys
import glob
import subprocess
import re
import os
import _winreg
import matplotlib

is_win64 = (os.environ["PROCESSOR_ARCHITECTURE"] == "AMD64")
class CellProfilerMSI(distutils.core.Command):
    description = "Make CellProfiler.msi using the CellProfiler.iss InnoSetup compiler"
    user_options = []
    
    def initialize_options(self):
        pass
    
    def finalize_options(self):
        pass
    
    def run(self):
        if is_win64:
            cell_profiler_iss = "CellProfiler64.iss"
            cell_profiler_setup = "CellProfiler64Setup.exe"
        else:
            cell_profiler_iss = "CellProfiler.iss"
            cell_profiler_setup = "CellProfilerSetup.exe"
        required_files = ["dist\\CellProfiler.exe",cell_profiler_iss]
        compile_command = self.__compile_command()
        compile_command = compile_command.replace("%1",cell_profiler_iss)
        self.make_file(required_files,"Output\\"+cell_profiler_setup, 
                       subprocess.check_call,([compile_command]),
                       "Compiling %s" % cell_profiler_iss)
    
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
            raise DistutilsFileError, "Inno Setup does not seem to be installed properly. Specifically, there is no entry in the HKEY_CLASSES_ROOT for InnoSetupScriptFile\\shell\\Compile\\command"

opts = {
    'py2exe': { "includes" : ["numpy", "scipy","PIL","wx",
                              "matplotlib", "matplotlib.numerix.random_array",
                              "email.iterators",
                              "cellprofiler.modules.*"],
                'excludes': ['pylab','Tkinter','Cython'],
                'dll_excludes': ["jvm.dll"]
              },
    'msi': {}
       }

if is_win64:
    opts['py2exe']['includes'] += [ "scipy.io.matlab.streams"]
    
data_files = [('cellprofiler\\icons',
               ['cellprofiler\\icons\\%s'%(x) 
                for x in os.listdir('cellprofiler\\icons')
                if x.endswith(".png") or x.endswith(".psd")]),
              ('bioformats', ['bioformats\\loci_tools.jar'])]
data_files += matplotlib.get_py2exe_datafiles()
# Collect the JVM
#
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
#
# Call setup
#
setup(console=[{'script':'CellProfiler.py',
                'icon_resources':[(1,'CellProfilerIcon.ico')]}],
      name='Cell Profiler',
      #setup_requires=['py2exe'],
      data_files = data_files,
      cmdclass={'msi':CellProfilerMSI
                },
      options=opts)
