import codecs
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


def find_resources(directory, subdirectory):
    resources = []
    for root, _, filenames in os.walk(os.path.join(directory, subdirectory)):
        resources += [os.path.relpath(os.path.join(root, filename), directory) for filename in filenames]

    return resources


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
        "boto3",
        "centrosome",
        "docutils",
        "h5py",
        "inflect",
        "javabridge",
        "joblib",
        "mahotas",
        "matplotlib>=2.0.0, !=2.1.0",
        "MySQL-python",
        "numpy",
        "prokaryote==2.3.3",
        "python-bioformats==1.4.0",
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
        "cellprofiler": find_resources("cellprofiler", "data")
    },
    include_package_data=True,
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
