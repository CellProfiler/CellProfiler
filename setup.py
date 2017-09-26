import codecs
import glob
import os
import re

import setuptools.dist


def read(*directories):
    pathname = os.path.abspath(os.path.dirname(__file__))

    return codecs.open(os.path.join(pathname, *directories), "r").read()


def find_version(*pathnames):
    data = read(*pathnames)

    matched = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", data, re.M)

    if matched:
        return matched.group(1)

    raise RuntimeError("Unable to find version string.")

setuptools.setup(
    author="cellprofiler-dev",
    author_email="cellprofiler-dev@broadinstitute.org",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering"
    ],
    entry_points={
        "console_scripts": [
            "cellprofiler=cellprofiler.__main__:main"
        ]
    },
    extras_require={
        "build": [
            "pyinstaller",
            "twine"
        ],
        "test": [
            "pytest"
        ]
    },
    install_requires=[
        "centrosome",
        "docutils",
        "h5py",
        "inflect",
        "javabridge",
        "joblib",
        "mahotas",
        "matplotlib",
        "MySQL-python",
        "numpy",
        "prokaryote==2.3.1",
        "python-bioformats==1.3.1",
        "pyzmq==15.3.0",
        "raven",
        "requests",
        "scikit-image",
        "scikit-learn",
        "scipy"
    ],
    license="BSD",
    name="CellProfiler",
    package_data={
        "images": glob.glob(os.path.join("data", "**", "*"))
    },
    packages=setuptools.find_packages(exclude=[
        "tests*"
    ]),
    python_requires=">=2.7, <3",
    setup_requires=[
        "pytest"
    ],
    url="https://github.com/CellProfiler/CellProfiler",
    version=find_version("cellprofiler", "__init__.py")
)
