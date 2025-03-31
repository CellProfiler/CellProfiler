from __future__ import absolute_import

import glob
import os.path
import sys

import setuptools
import setuptools.command.build_ext
import setuptools.command.test
import numpy

try:
    import Cython.Build

    __cython = True
except ImportError:
    __cython = False


class Test(setuptools.command.test.test):
    user_options = [("pytest-args=", "a", "Arguments to pass to py.test")]

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


if __cython:
    __suffix = "pyx"
    __extkwargs = {
        "language": "c++",
        # TODO: needed for cython 3.0
        # "define_macros": [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    }
else:
    __suffix = "cpp"
    __extkwargs = {}

__extensions = [
    setuptools.Extension(
        name="centrosome._propagate",
        sources=[
            "centrosome/_propagate.{}".format("c" if __suffix == "cpp" else __suffix)
        ],
        # TODO: needed for cython 3.0
        # define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")] if __suffix == "pyx" else None,
        include_dirs=["centrosome/include", numpy.get_include()],
    )
]

for pyxfile in glob.glob(os.path.join("centrosome", "*.pyx")):
    name = os.path.splitext(os.path.basename(pyxfile))[0]

    if name == "_propagate":
        continue

    __extensions += [
        setuptools.Extension(
            name="centrosome.{}".format(name),
            sources=["centrosome/{}.{}".format(name, __suffix)],
            include_dirs=["centrosome/include", numpy.get_include()],
            **__extkwargs
        )
    ]

if __suffix == "pyx":
    __extensions = Cython.Build.cythonize(__extensions, compiler_directives={'language_level': "3"})

setuptools.setup(
    author="Nodar Gogoberidze",
    author_email="ngogober@broadinstitute.org",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: C",
        "Programming Language :: C++",
        "Programming Language :: Cython",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering",
    ],
    cmdclass={"test": Test},
    description="An open source image processing library",
    ext_modules=__extensions,
    extras_require={
        "dev": ["black==19.10b0", "pre-commit==1.20.0"],
        "test": ["pytest"],
    },
    install_requires=[
        "deprecation",
        "matplotlib>=3.1.3,<4",
        "numpy>=1.18.2,<2",
        "pillow>=7.1.0,<12",
        "scikit-image>=0.17.2,<=0.24",
        "scipy>=1.4.1,!=1.11.0,<2",
    ],
    tests_require=[
        "pytest",
    ],
    keywords="",
    license="BSD",
    long_description="",
    name="centrosome",
    packages=["centrosome"],
    setup_requires=["cython", "numpy", "pytest", ],
    url="https://github.com/CellProfiler/centrosome",
    version="1.3.1",
)
