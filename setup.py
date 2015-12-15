import glob
import os
import setuptools
import setuptools.command.build_ext
import setuptools.dist
import sys
import setuptools.command.install
import cellprofiler.utilities.version

setuptools.dist.Distribution({
    "setup_requires": [
        "pytest"
    ]
})


class Install(setuptools.command.install.install):
    def run(self):
        with open("cellprofiler/frozen_version.py", "w") as fd:
            fd.write("version_string='%s'\n" % cellprofiler.utilities.version.version_string)

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
    cmdclass={
        "install": Install,
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
        "cellh5",
        "centrosome",
        "h5py",
        "javabridge",
        "libtiff",
        "matplotlib",
        "MySQL-python",
        "numpy",
        "prokaryote",
        "pytest",
        "python-bioformats",
        "pyzmq==13.1.0",
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
    ]),
    setup_requires=[
        "pytest"
    ],
    url="https://github.com/CellProfiler/CellProfiler",
    version="2.2.0rc1"
)
