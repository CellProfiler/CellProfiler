import distutils
import glob
import os
import setuptools
import setuptools.command.build_ext
import setuptools.command.install
import setuptools.dist
import sys

try:
    import matplotlib
    import numpy # for proper discovery of its libraries by distutils
    import scipy.sparse.csgraph._validation
    import zmq   # for proper discovery of its libraries by distutils
    import zmq.libzmq
except ImportError:
    pass

from cellprofiler.utilities.version import version_string

with open("cellprofiler/frozen_version.py", "w") as fd:
    fd.write("version_string='%s'\n" % version_string)

if sys.platform.startswith("win"):
    import _winreg
    try:
        import py2exe
        has_py2exe = True
        #
        # See http://www.py2exe.org/index.cgi/Py2exeAndzmq
        # Recipe needed for py2exe to package libzmq.dll
        os.environ["PATH"] += os.path.pathsep + os.path.split(zmq.__file__)[0]
    except:
        has_py2exe = False
else:
    has_py2exe = False    

# Recipe needed to get real distutils if virtualenv.
# Error message is "ImportError: cannot import name dist"
# when running app.
# See http://sourceforge.net/p/py2exe/mailman/attachment/47C45804.9030206@free.fr/1/
#
if hasattr(sys, 'real_prefix'):
    # Running from a virtualenv
    assert hasattr(distutils, 'distutils_path'), \
           "Can't get real distutils path"
    libdir = os.path.dirname(distutils.distutils_path)
    sys.path.insert(0, libdir)
    #
    # Get the system "site" package, not the virtualenv one. This prevents
    # site.virtual_install_main_packages from being called, resulting in
    # "IOError: [Errno 2] No such file or directory: 'orig-prefix.txt'
    #
    del sys.modules["site"]
    import site
    assert not hasattr(site, "virtual_install_main_packages")

class Install(setuptools.command.install.install):
    def run(self):
        try:
            import clint.textui
            import requests
        except ImportError:
            raise ImportError

        version = "1.0.3"

        directory = os.path.join(self.build_lib, "imagej", "jars")

        if not os.path.exists(directory):
            os.makedirs(directory)

        prokaryote = "{}/prokaryote-{}.jar".format(os.path.abspath(directory), version)

        resource = "https://github.com/CellProfiler/prokaryote/" + "releases/download/{tag}/prokaryote-{tag}.jar".format(tag=version)

        request = requests.get(resource, stream=True)

        if not os.path.isfile(prokaryote):
            with open(prokaryote, "wb") as f:
                total_length = int(request.headers.get("content-length"))

                chunks = clint.textui.progress.bar(request.iter_content(chunk_size=32768), expected_size=(total_length / 32768) + 1, hide=not self.verbose)

                for chunk in chunks:
                    if chunk:
                        f.write(chunk)

                        f.flush()

        dependencies = os.path.abspath(os.path.join(
            self.build_lib, 'imagej', 'jars', 
            'cellprofiler-java-dependencies-classpath.txt'))

        if not os.path.isfile(dependencies):
            dependency = open(dependencies, "w")

            dependency.write(prokaryote)

            dependency.close()

        setuptools.command.install.install.run(self)


class Test(setuptools.Command):
    user_options = [
        ("pytest-args=", "a", "arguments to pass to py.test")
    ]

    def initialize_options(self):
        self.pytest_args = []

    def finalize_options(self):
        pass

    def run(self):
        try:
            import pytest
            import unittest
        except ImportError:
            raise ImportError

        import cellprofiler.__main__
        import cellprofiler.utilities.cpjvm

        #
        # Monkey-patch pytest.Function
        # See https://github.com/pytest-dev/pytest/issues/1169
        #
        try:
            from _pytest.unittest import TestCaseFunction

            def runtest(self):
                setattr(self._testcase, "__name__", self.name)
                self._testcase(result=self)

            TestCaseFunction.runtest = runtest
        except:
            pass

        try:
            import ilastik.core.jobMachine

            ilastik.core.jobMachine.GLOBAL_WM.set_thread_count(1)
        except ImportError:
            pass

        cellprofiler.utilities.cpjvm.cp_start_vm()

        errno = pytest.main(self.pytest_args)

        cellprofiler.__main__.stop_cellprofiler()

        sys.exit(errno)

if has_py2exe:        
    class CPPy2Exe(py2exe.build_exe.py2exe):
        def run(self):
            #
            # py2exe runs install_data a second time. We want to inject some
            # data files into the dist but we do it here so that if the user
            # does a straight "install", they won't end up dumped into their
            # Python directory.
            #
            import javabridge
            from cellprofiler.utilities.cpjvm import get_path_to_jars
            
            if self.distribution.data_files is None:
                self.distribution.data_files = []
            self.distribution.data_files += matplotlib.get_py2exe_datafiles()
            self.distribution.data_files.append(
                ("javabridge/jars", javabridge.JARS))
            self.distribution.data_files.append(
                ("imagej/jars", 
                 glob.glob(os.path.join(get_path_to_jars(), "prokaryote*.jar")) +
                 [os.path.join(get_path_to_jars(), 
                               "cellprofiler-java-dependencies-classpath.txt")]))
            self.distribution.data_files.append(
                ("artwork", glob.glob("artwork/*")))
            py2exe.build_exe.py2exe.run(self)
        
class CPNSIS(setuptools.Command):
    description = "Use NSIS to create an MSI for windows installation"
    user_options = [('nsis-exe', None, "Path to the NSIS executable")]
    
    def initialize_options(self):
        self.nsis_exe = None
        
    def finalize_options(self):
        if self.nsis_exe is None:
            for path in ((_winreg.HKEY_LOCAL_MACHINE, "SOFTWARE", "NSIS"), 
                         (_winreg.HKEY_LOCAL_MACHINE, "SOFTWARE", 
                          "Wow6432Node", "NSIS")):
                key = path[0]
                try:
                    for subkey in path[1:]:
                        key = _winreg.OpenKey(key, sub_key)
                    break
                except WindowsError:
                    continue
            else:
                raise distutils.errors.DistutilsExecError(
                    "The NSIS installer is not installed on your system")
            self.nsis_exe = os.path.join(_winreg.QueryValue(key, None), 
                                         "NSIS.exe")
    
    def run(self):
        pass
            
    sub_commands = setuptools.Command.sub_commands + [ ("py2exe", None) ]
        
packages = setuptools.find_packages(exclude=[
        "*.tests",
        "*.tests.*",
        "tests.*",
        "tests",
        "tutorial"
    ])

setuptools.setup(
    author="cellprofiler-dev",
    author_email="cellprofiler-dev@broadinstitute.org",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: C",
        "Programming Language :: C++",
        "Programming Language :: Cython",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering"
    ],
    cmdclass={
        "install": Install,
        "py2exe": CPPy2Exe,
        "test": Test
    },
    console = [ 
        {
        "icon_resources": [
            (1, "artwork/CellProfilerIcon.ico")
            ],
        "script" : "CellProfiler.py"
        },
        {
            "icon_resources": [
                (1, "artwork/CellProfilerIcon.ico")
                ],
            "script" : "cellprofiler/analysis_worker.py"
            }],
    description="",
    entry_points={
        'console_scripts': [
            "cellprofiler=cellprofiler.__main__:main"
        ],
        'gui_scripts': [

        ]
    },
    include_package_data=True,
    install_requires=[
        "cellh5",
        "centrosome",
        "h5py",
        "javabridge",
        "libtiff",
        "matplotlib",
        "MySQL-python",
        "numpy",
        "pytest",
        "python-bioformats",
        "pyzmq==13.1.0",
        "scipy"
    ],
    keywords="",
    license="BSD",
    long_description="",
    name="cellprofiler",
    options = {
        "py2exe": {
            "dll_excludes": [
                "iphlpapi.dll",
                "jvm.dll",
                "msvcr90.dll",
                "msvcm90.dll",
                "msvcp90.dll",
                "nsi.dll",
                "winnsi.dll"
                ],
            "excludes": [
                "Cython",
                "IPython",
                "pylab",
                "Tkinter"
                ],
            "includes": [
                "h5py", "h5py.*",
                "lxml", "lxml.*",
                "scipy.io.matlab.streams", "scipy.special", "scipy.special.*",
                "scipy.sparse.csgraph._validation",
                "skimage.draw", "skimage._shared.geometry", 
                "skimage.filters.rank.*",
                "sklearn.*", "sklearn.neighbors", "sklearn.neighbors.*",
                "sklearn.utils.*", "sklearn.utils.sparsetools.*",
                "zmq", "zmq.utils", "zmq.utils.*", "zmq.utils.strtypes"
                ],
            "packages": packages,
            "skip_archive": True
            }
    },
    package_data = {
        "artwork": glob.glob(os.path.join("artwork", "*"))
    },
    packages = packages + ["artwork"],
    setup_requires=[
        "clint",
        "matplotlib",
        "numpy",
        "pytest",
        "requests",
        "scipy",
        "pyzmq"
    ],
    url="https://github.com/CellProfiler/CellProfiler",
    version="2.2.0"
)
