import os
import setuptools


version_file = open(os.path.join(os.path.dirname(__file__), "cellprofiler", "data", "VERSION"))

version = version_file.read().strip()

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
        "application": [

        ],
        "test": [
            "pytest"
        ]
    },
    include_package_data=True,
    install_requires=[
        "cellh5",
        "centrosome",
        "h5py",
        "inflect",
        "javabridge",
        "libtiff",
        "mahotas",
        "matplotlib",
        "MySQL-python",
        "numpy",
        "prokaryote",
        "pyamg",
        "pytest",
        "python-bioformats",
        "pyzmq",
        "scikit-image",
        "scipy"
    ],
    license="BSD",
    name="CellProfiler",
    package_data={
        "cellprofiler": [
            "data/*.png"
        ],
    },
    packages=setuptools.find_packages(
        exclude=[
            "cellprofiler.plugins.*",
            "tests.*",
            "tests"
        ]
    ),
    url="https://github.com/CellProfiler/CellProfiler",
    version="3.0.0rc1"
)
