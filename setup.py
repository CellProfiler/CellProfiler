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
        "test": ["pytest>=3.3.2,<4"],
    },
    install_requires=[
        "boto3>=1.12.28",
        "cellprofiler-core",
        "centrosome==1.1.7",
        "docutils==0.15.2",
        "h5py>=2.9",
        "imageio>=2.5",
        "inflect>=2.1",
        "Jinja2>=2.11.2",
        "joblib>=0.13",
        "mahotas>=1.4",
        "matplotlib==3.1.3",
        "mysqlclient==1.4.6",
        "numpy>=1.18.2",
        "pillow==7.0.0",
        "prokaryote==2.4.1",
        "python-bioformats>=1.5.2",
        "python-javabridge",
        "pyzmq==18.0.1",
        "sentry-sdk>=0.16.0",
        "requests>=2.22",
        "scikit-image==0.16.2",
        "scikit-learn>=0.20",
        "scipy>=1.4.1",
        "six",
        "wxPython==4.0.7.post2",
    ],
    license="BSD",
    name="CellProfiler",
    package_data=package_data(),
    include_package_data=True,
    packages=setuptools.find_packages(exclude=["tests*"]),
    python_requires=">=3.8, <4.0",
    setup_requires=["pytest"],
    url="https://github.com/CellProfiler/CellProfiler",
    version=find_version("cellprofiler", "__init__.py"),
)
