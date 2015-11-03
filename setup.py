import clint.textui
import glob
import os
import requests
import setuptools
import setuptools.command.build_ext
import setuptools.command.test
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
        VERSION = "1.0.2"

        directory = os.path.join("imagej", "jars")

        if not os.path.exists(directory):
            os.makedirs(directory)

        prokaryote = "{}/prokaryote-{}.jar".format(os.path.abspath(directory), VERSION)

        resource = "https://github.com/CellProfiler/prokaryote/" + "releases/download/{tag}/prokaryote-{tag}.jar".format(tag=VERSION)

        request = requests.get(resource, stream=True)

        if not os.path.isfile(prokaryote):
            with open(prokaryote, "wb") as f:
                total_length = int(request.headers.get("content-length"))

                for chunk in clint.textui.progress.bar(request.iter_content(chunk_size=1024), expected_size=(total_length / 1024) + 1):
                    if chunk:
                        f.write(chunk)

                        f.flush()

        dependencies = os.path.abspath("imagej/jars/cellprofiler-java-dependencies-classpath.txt")

        if not os.path.isfile(dependencies):
            file = open(dependencies, "w")

            file.write(prokaryote)

            file.close()


class Test(setuptools.command.test.test):
    user_options = [
        ("pytest-args=", "a", "arguments to pass to py.test")
    ]

    def initialize_options(self):
        setuptools.command.test.test.initialize_options(self)

        self.pytest_args = []

    def finalize_options(self):
        setuptools.command.test.test.finalize_options(self)

        self.test_args = []

        self.test_suite = True

    def run_tests(self):
        import pytest

        errno = pytest.main(self.pytest_args)

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
        "cellprofiler"
    ],
    setup_requires=[
        "clint",
        "requests",
        "pytest"
    ],
    tests_require=[
        "pytest"
    ],
    url="https://github.com/CellProfiler/CellProfiler",
    version="2.2.0"
)
