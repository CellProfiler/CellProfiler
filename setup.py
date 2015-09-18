import glob
import setuptools.command.test
import sys


class Test(setuptools.command.test.test):
    user_options = [
        ("pytest-args=", "a", "py.test arguments")
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
    author="Broad Institute of MIT and Harvard",
    author_email="agoodman@broadinstitute.org",
    cmdclass={
        "test": Test,
    },
    data_files=[
        ("artwork", glob.glob("artwork/*.png")),
    ],
    description="quantitatively measure phenotypes",
    entry_points={
        "console_scripts": [

        ],
        "gui_scripts": [

        ],
    },
    install_requires=[
        "cython",
        "h5py",
        "lxml",
        "matplotlib",
        "numpy",
        "pandas",
        "Pillow",
        "pytest",
        "python-bioformats",
        "pyzmq",
        "scikit-learn",
        "scipy",
    ],
    license="MIT",
    name="CellProfiler",
    tests_require=[
        "pytest",
        "tox",
    ],
    url="http://cellprofiler.org",
    version="2.1.1",
)
