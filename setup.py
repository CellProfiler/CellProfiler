import distutils
import glob
import importlib
import math
import os
import shlex
import setuptools
import setuptools.command.build_py
import setuptools.command.sdist
import setuptools.dist
import sys


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

        cellprofiler.utilities.cpjvm.cp_start_vm()

        errno = pytest.main(self.pytest_args)

        cellprofiler.__main__.stop_cellprofiler()

        sys.exit(errno)


cmdclass = {
    "test": Test
}

version_file = open(os.path.join(os.path.dirname(__file__), "cellprofiler", "VERSION"))
version = version_file.read().strip()

setuptools.setup(
        app=[
            "CellProfiler.py"
        ],
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
        cmdclass=cmdclass,
        console=[
            {
                "icon_resources": [
                    (1, "artwork/CellProfilerIcon.ico")
                ],
                "script": "CellProfiler.py"
            }
        ],
        description="",
        dependency_links=[
            "git+https://github.com/scikit-image/scikit-image.git#egg=scikit-image-0.13.0dev"
        ],
        entry_points={
            "console_scripts": [
                "cellprofiler=cellprofiler.__main__:main"
            ]
        },
        include_package_data=True,
        install_requires=[
            "cellh5",
            "centrosome>=1.0.4",
            "h5py",
            "inflect",
            "javabridge",
            "libtiff",
            "mahotas",
            "matplotlib",
            "MySQL-python",
            "numpy",
            "prokaryote>=1.0.11",
            "pyamg!=3.2.0",
            "pytest",
            "python-bioformats",
            "pyzmq<15.4",
            "scikit-image==0.13.0dev",
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
