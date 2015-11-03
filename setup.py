import clint.textui
import glob
import os
import requests
import setuptools
import setuptools.command.build_ext
import sys


class Prokaryote(setuptools.Command):
    user_options = [
        ("version=", "v", "version to fetch")
    ]

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        VERSION = "1.0.3"

        directory = os.path.join("imagej", "jars")

        if not os.path.exists(directory):
            os.makedirs(directory)

        prokaryote = "{}/prokaryote-{}.jar".format(os.path.abspath(directory), VERSION)

        resource = "https://github.com/CellProfiler/prokaryote/" + "releases/download/{tag}/prokaryote-{tag}.jar".format(tag=VERSION)

        request = requests.get(resource, stream=True)

        if not os.path.isfile(prokaryote):
            with open(prokaryote, "wb") as f:
                total_length = int(request.headers.get("content-length"))

                for chunk in clint.textui.progress.bar(
                    request.iter_content(chunk_size=32768), 
                    expected_size=(total_length / 32768) + 1,
                    hide=not self.verbose):
                    if chunk:
                        f.write(chunk)

                        f.flush()

        dependencies = os.path.abspath("imagej/jars/cellprofiler-java-dependencies-classpath.txt")

        if not os.path.isfile(dependencies):
            file = open(dependencies, "w")

            file.write(prokaryote)

            file.close()


class Test(setuptools.Command):
    user_options = [
        ("pytest-args=", "a", "arguments to pass to py.test")
    ]

    def initialize_options(self):
        self.pytest_args = []

    def finalize_options(self):
        pass

    def run(self):
        import pytest
        import unittest
        from cellprofiler.utilities.cpjvm import cp_start_vm
        from cellprofiler.main import stop_cellprofiler
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

        # Tell Ilastik to run on only a single thread during testing
        try:
            from ilastik.core.jobMachine import GLOBAL_WM
            GLOBAL_WM.set_thread_count(1)
        except:
            pass
        cp_start_vm()
        errno = pytest.main(self.pytest_args)
        stop_cellprofiler()

        sys.exit(errno)


setuptools.setup(
    author="Allen Goodman",
    author_email="allen.goodman@icloud.com",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: C",
        "Programming Language :: C++",
        "Programming Language :: Cython",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 2",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering",
    ],
    cmdclass={
        "prokaryote": Prokaryote,
        "test": Test
    },
    data_files=[
        (
            os.path.join("artwork"),
            glob.glob(os.path.join("artwork", '*'))
        ),
        (
            os.path.join("imagej", "jars"),
            glob.glob(os.path.join("imagej", "jars", "*.jar"))
        )
    ],
    description="",
    entry_points={
        'console_scripts': [
            "cellprofiler=cellprofiler.main:main"
        ],
        'gui_scripts': [

        ]
    },
    install_requires=[
        'h5py',
        'pyzmq',
        'matplotlib',
        'python-bioformats',
        'centrosome',
        'MySQL-python',
        'libtiff',
        'scikit-learn',
    ],
    keywords="",
    license="BSD",
    long_description="",
    name="cellprofiler",
    packages=[
        "cellprofiler", "cellprofiler.modules", "cellprofiler.modules.plugins",
        "cellprofiler.utilities", "cellprofiler.cpmath",
        "cellprofiler.gui", "cellprofiler.gui.html", 
        "cellprofiler.icons", "cellprofiler.matlab", "contrib", "imagej"
    ],
    setup_requires=[
        "clint",
        "requests",
        "pytest>=2.8"
    ],
    url="https://github.com/CellProfiler/CellProfiler",
    version="2.2.0"
)
