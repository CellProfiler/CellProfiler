import glob
import importlib
import os
import sys

import setuptools.dist

# We need some packages in order to properly prepare for setup, but
# setuptools.dist.Distribution seems to upgrade them willy-nilly
# So try importing and only ask for ones that are not present.
packages = []
for package_name, import_name in [
    ("clint", "clint"),
    ("javabridge", "javabridge"),
    ("matplotlib", "matplotlib"),
    ("numpy", "numpy"),
    ("pytest", "pytest"),
    ("pyzmq", "zmq"),
    ("requests", "requests"),
    ("scipy", "scipy")]:
    try:
        importlib.import_module(import_name)
    except ImportError:
        packages.append(package_name)

setuptools.dist.Distribution({
    "setup_requires": packages
})

try:
    import matplotlib
    import scipy.sparse.csgraph._validation
    import scipy.linalg
    import zmq  # for proper discovery of its libraries by distutils
    import zmq.libzmq
except ImportError:
    pass

#
# Recipe for ZMQ
#
if sys.platform.startswith("win"):
    #
    # See http://www.py2exe.org/index.cgi/Py2exeAndzmq
    # Recipe needed for py2exe to package libzmq.dll
    os.environ["PATH"] += os.path.pathsep + os.path.split(zmq.__file__)[0]


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
            import unittest

            import pytest

            import cellprofiler.__main__
            import cellprofiler.preferences
            import cellprofiler.utilities.cpjvm
        except ImportError:
            raise ImportError

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

        cellprofiler.preferences.set_headless()

        cellprofiler.utilities.cpjvm.cp_start_vm()

        errno = pytest.main(self.pytest_args)

        cellprofiler.__main__.stop_cellprofiler()

        sys.exit(errno)

version_file = open(os.path.join(os.path.dirname(__file__), "cellprofiler", "VERSION"))
version = version_file.read().strip()

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
            "test": Test
        },
        description="",
        entry_points={
            "console_scripts": [
                "cellprofiler=cellprofiler.__main__:main"
            ]
        },
        include_package_data=True,
        install_requires=[
            "centrosome",
            "h5py",
            "inflect",
            "javabridge",
            "joblib",
            "libtiff",
            "mahotas",
            "matplotlib",
            "MySQL-python",
            "numpy",
            "prokaryote>=1.0.11",
            "pyamg==3.1.1",
            "pytest",
            "python-bioformats",
            "pyzmq",
            "raven",
            "requests",
            "scikit-image",
            "scikit-learn",
            "scipy"
        ],
        keywords="",
        license="BSD",
        long_description="",
        name="CellProfiler",
        package_data={
            "artwork": glob.glob(os.path.join("artwork", "*"))
        },
        packages=setuptools.find_packages(exclude=[
            "*.tests",
            "*.tests.*",
            "tests.*",
            "tests",
            "tutorial"
        ]) + ["artwork"],
        setup_requires=[
            "pytest"
        ],
        url="https://github.com/CellProfiler/CellProfiler",
        version="3.0.0rc1"
)
