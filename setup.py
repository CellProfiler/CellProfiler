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


def package_data():
    resources = []

    for root, _, filenames in os.walk(os.path.join("cellprofiler", "data")):
        resources += [
            os.path.relpath(os.path.join(root, filename), "cellprofiler")
            for filename in filenames
        ]

    for root, _, filenames in os.walk(os.path.join("cellprofiler", "gui")):
        resources += [
            os.path.relpath(os.path.join(root, filename), "cellprofiler")
            for filename in filenames
            if ".html" in filename
        ]

    return {"cellprofiler": resources}


setuptools.setup(
    author="cellprofiler-dev",
    author_email="cellprofiler-dev@broadinstitute.org",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering",
    ],
    entry_points={"console_scripts": ["cellprofiler=cellprofiler.__main__:main"]},
    extras_require={
        "build": ["black", "pre-commit", "pyinstaller", "twine"],
        "docs": ["Sphinx>=3.1.1", "sphinx-rtd-theme>=0.5.0"],
        # pooch needed for scikit-image's example dataset,
        # e.g. skimage.data.human_mitosis
        # version pin should match scikit-image test requirements
        "test": ["pytest~=7.4.1", "pooch>=1.3.0"],
    },
    install_requires=[
        "boto3~=1.28.41",
        "cellprofiler-core==5.0.0",
        "centrosome==1.2.2",
        "docutils==0.15.2",
        "h5py~=3.6.0",
        "imageio~=2.31.3",
        "inflect~=7.0.0",
        "Jinja2~=3.1.2",
        "joblib~=1.3.2",
        "mahotas~=1.4.13",
        "matplotlib~=3.1.3",
        "mysqlclient~=2.0.0",
        "numpy~=1.24.4",
        "Pillow~=10.0.0",
        "pyzmq~=22.3.0",
        "sentry-sdk>=0.18.0,<=1.31.0",
        "requests~=2.31.0",
        "scikit-image~=0.20.0",
        "scikit-learn~=1.3.0",
        "scipy>=1.9.1,<1.11",
        "scyjava>=1.9.1",
        "six~=1.16.0",
        "tifffile>=2022.4.8,<2022.4.22",
        "wxPython==4.2.0",

    ],
    license="BSD",
    name="CellProfiler",
    package_data=package_data(),
    include_package_data=True,
    packages=setuptools.find_packages(exclude=["tests*"]),
    python_requires=">=3.8",
    setup_requires=["pytest"],
    url="https://github.com/CellProfiler/CellProfiler",
    version=find_version("cellprofiler", "__init__.py"),
)
